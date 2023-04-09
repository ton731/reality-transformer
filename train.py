import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from datetime import datetime
import os
import logging
import numpy as np
import random
from pathlib import Path

from utils.dataset import SkecthRenderDataset
from utils.metrics import CycleGANMetric
from utils.plot import plot_metric
from model.generator import Generator
from model.discriminator import Discriminator



# A_name = "reality"
# B_name = "animation"

# A_name = "photo"
# B_name = "monet"

A_name = "sketch"
B_name = "render"




def parse_args() -> Namespace:
	parser = ArgumentParser()

	# data path
	parser.add_argument("--data_dir", type=Path, default="./data/")
	parser.add_argument("--project_name", type=str, default="sketch2render", help="reality2animation, monet2photo, sketch2render")	

	# checkpoint
	parser.add_argument("--ckpt_dir", type=Path, default="./results/")

	# output image size
	parser.add_argument("--img_size", type=int, default=256)

	# cycleGAN
	parser.add_argument("--use_identity", action="store_true", default=False)
	parser.add_argument("--lambda_identity", type=float, default=1)
	parser.add_argument("--lambda_cycle", type=float, default=10)

	# training
	parser.add_argument("--batch_size", type=int, default=3)
	parser.add_argument("--lr", type=float, default=1e-5)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--num_epoch", type=int, default=50)
	parser.add_argument("--device", type=torch.device, default="cuda")

	args = parser.parse_args()
	return args



def set_random_seed(SEED: int):
	random.seed(SEED)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = True
	torch.autograd.set_detect_anomaly(True)



def get_loggings(ckpt_dir):
	logger = logging.getLogger(name='CycleGAN')
	logger.setLevel(level=logging.INFO)
	# set formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	# console handler
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)
	# file handler
	file_handler = logging.FileHandler(ckpt_dir / "record.log")
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	return logger



def train_fn(disc_A, disc_B, gen_A, gen_B, 
			 train_loader, 
			 opt_disc, opt_gen,
			 use_identity, 
			 l1, mse, 
			 epoch,
			 train_metric, logger, visualization_paths,
			 **kwargs):

	disc_A.train()
	disc_B.train()
	gen_A.train()
	gen_B.train()

	loop = tqdm(train_loader, leave=True)

	for idx, (A, B) in enumerate(loop):
		A = A.to(args.device)
		B = B.to(args.device)

		# Train Discriminator A and B
		fake_A = gen_A(B)
		D_A_real = disc_A(A)
		D_A_fake = disc_A(fake_A.detach())
		D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
		D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_real))
		D_A_loss = D_A_real_loss + D_A_fake_loss

		fake_B = gen_B(A)
		D_B_real = disc_B(B)
		D_B_fake = disc_B(fake_B.detach())
		D_B_real_loss = mse(D_B_real, torch.ones_like(D_B_real))
		D_B_fake_loss = mse(D_B_fake, torch.zeros_like(D_B_fake))
		D_B_loss = D_B_real_loss + D_B_fake_loss

		# put it together
		D_loss = (D_A_loss + D_B_loss) / 2
		
		opt_disc.zero_grad()
		D_loss.backward()
		opt_disc.step()


		# Train Generators H and Z
		# advesarial loss for both generators
		D_A_fake = disc_A(fake_A)
		D_B_fake = disc_B(fake_B)
		loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))
		loss_G_B = mse(D_B_fake, torch.ones_like(D_B_fake))

		# cycle loss
		cycle_A = gen_A(fake_B)
		cycle_B = gen_B(fake_A)
		cycle_A_loss = l1(A, cycle_A)
		cycle_B_loss = l1(B, cycle_B)

		# add all together
		G_loss = (
			loss_G_B
			+ loss_G_A
			+ cycle_A_loss * args.lambda_cycle
			+ cycle_B_loss * args.lambda_cycle
		)

		# identity loss (remove these for efficiency if you set lambda_identity=0)
		identity_A, identity_B = None, None
		if use_identity:
			identity_A = gen_A(A)
			identity_B = gen_B(B)
			identity_A_loss = l1(A, identity_A)
			identity_B_loss = l1(B, identity_B)
			G_loss += identity_B_loss * args.lambda_identity + identity_A_loss * args.lambda_identity
		
		opt_gen.zero_grad()
		G_loss.backward()
		opt_gen.step()

		# metric update
		train_metric.batch_update(epoch, D_A_real, D_A_fake, D_B_real, D_B_fake,
								  cycle_A, cycle_B, A, B, identity_A, identity_B)

		# visualize
		if idx % 400 == 0:
			batch_size = A.shape[0]
			nrow = batch_size
			
			# 1. fake A, fake B
			save_image(torch.cat([fake_A, B], dim=0) * 0.5 + 0.5, visualization_paths["train_fake"] / f"fake_{A_name}_epoch{epoch}_idx{idx}.png", nrow=nrow)
			save_image(torch.cat([fake_B, A], dim=0) * 0.5 + 0.5, visualization_paths["train_fake"] / f"fake_{B_name}_epoch{epoch}_idx{idx}.png", nrow=nrow)
		
			# 2. cycle A, cycle B
			save_image(torch.cat([cycle_A, A], dim=0) * 0.5 + 0.5, visualization_paths["train_cycle"] / f"cycle_{A_name}_epoch{epoch}_idx{idx}.png", nrow=nrow)
			save_image(torch.cat([cycle_B, B], dim=0) * 0.5 + 0.5, visualization_paths["train_cycle"] / f"cycle_{B_name}_epoch{epoch}_idx{idx}.png", nrow=nrow)

			# 3. identity A, identity B
			if use_identity:
				save_image(torch.cat([identity_A, A], dim=0) * 0.5 + 0.5, visualization_paths["train_identity"] / f"identity_{A_name}_epoch{epoch}_idx{idx}.png", nrow=nrow)
				save_image(torch.cat([identity_B, B], dim=0) * 0.5 + 0.5, visualization_paths["train_identity"] / f"identity_{B_name}_epoch{epoch}_idx{idx}.png", nrow=nrow)

	# update metric for whole epoch
	train_metric.epoch_update(epoch)
	logger.info(train_metric.info(epoch))


@torch.no_grad()
def valid_fn(disc_A, disc_B, gen_A, gen_B, 
			 valid_loader, 
			 use_identity, 
			 epoch,
			 valid_metric, logger, visualization_paths,
			 **kwargs):

	disc_A.eval()
	disc_B.eval()
	gen_A.eval()
	gen_B.eval()

	loop = tqdm(valid_loader, leave=True)

	for idx, (A, B) in enumerate(loop):
		A = A.to(args.device)
		B = B.to(args.device)

		# Train Discriminator A and B
		fake_A = gen_A(B)
		D_A_real = disc_A(A)
		D_A_fake = disc_A(fake_A.detach())

		fake_B = gen_B(A)
		D_B_real = disc_B(B)
		D_B_fake = disc_B(fake_B.detach())

		# Train Generators H and Z
		# advesarial loss for both generators
		D_A_fake = disc_A(fake_A)
		D_B_fake = disc_B(fake_B)

		# cycle loss
		cycle_A = gen_A(fake_B)
		cycle_B = gen_B(fake_A)

		# identity loss (remove these for efficiency if you set lambda_identity=0)
		identity_A, identity_B = None, None
		if use_identity:
			identity_A = gen_A(A)
			identity_B = gen_B(B)

		# metric update
		valid_metric.batch_update(epoch, D_A_real, D_A_fake, D_B_real, D_B_fake,
								  cycle_A, cycle_B, A, B, identity_A, identity_B)

		# visualize
		if idx % 100 == 0:
			batch_size = A.shape[0]
			nrow = batch_size
			
			# 1. fake A, fake B
			save_image(torch.cat([fake_A, B], dim=0) * 0.5 + 0.5, visualization_paths["valid_fake"] / f"fake_{A_name}_epoch{epoch}_idx{idx}.png", nrow=nrow)
			save_image(torch.cat([fake_B, A], dim=0) * 0.5 + 0.5, visualization_paths["valid_fake"] / f"fake_{B_name}_epoch{epoch}_idx{idx}.png", nrow=nrow)
		
			# 2. cycle A, cycle B
			save_image(torch.cat([cycle_A, A], dim=0) * 0.5 + 0.5, visualization_paths["valid_cycle"] / f"cycle_{A_name}_epoch{epoch}_idx{idx}.png", nrow=nrow)
			save_image(torch.cat([cycle_B, B], dim=0) * 0.5 + 0.5, visualization_paths["valid_cycle"] / f"cycle_{B_name}_epoch{epoch}_idx{idx}.png", nrow=nrow)

			# 3. identity A, identity B
			if use_identity:
				save_image(torch.cat([identity_A, A], dim=0) * 0.5 + 0.5, visualization_paths["valid_identity"] / f"identity_{A_name}_epoch{epoch}_idx{idx}.png", nrow=nrow)
				save_image(torch.cat([identity_B, B], dim=0) * 0.5 + 0.5, visualization_paths["valid_identity"] / f"identity_{B_name}_epoch{epoch}_idx{idx}.png", nrow=nrow)

	# update metric for whole epoch
	valid_metric.epoch_update(epoch)
	logger.info(valid_metric.info(epoch))




def main(args):

	# set random seed
	set_random_seed(731)


	# set checkpoint directory
	args.ckpt_dir = args.ckpt_dir / datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
	args.ckpt_dir.mkdir(parents=True, exist_ok=True)


	# save visualization
	visualization_dir = args.ckpt_dir / "visualization"
	visualization_dir.mkdir(parents=True, exist_ok=True)
	vis_train_fake = visualization_dir / "train" / "fake"
	vis_train_cycle = visualization_dir / "train" / "cycle"
	vis_train_identity = visualization_dir / "train" / "identity"
	vis_valid_fake = visualization_dir / "valid" / "fake"
	vis_valid_cycle = visualization_dir / "valid" / "cycle"
	vis_valid_identity = visualization_dir / "valid" / "identity"
	visualization_paths = {"train_fake": vis_train_fake, "train_cycle": vis_train_cycle, "train_identity": vis_train_identity,
						   "valid_fake": vis_valid_fake, "valid_cycle": vis_valid_cycle, "valid_identity": vis_valid_identity}
	for vis_path in visualization_paths.keys():
		visualization_paths[vis_path].mkdir(parents=True, exist_ok=True)


	# set logger
	logger = get_loggings(args.ckpt_dir)
	logger.critical(args.ckpt_dir)
	logger.critical(args)

	# initialize generator, discriminator
	disc_A = Discriminator(in_channels=3).to(args.device) # discriminate A
	disc_B = Discriminator(in_channels=3).to(args.device)
	gen_B = Generator(img_channels=3, num_residuals=9).to(args.device)
	gen_A = Generator(img_channels=3, num_residuals=9).to(args.device)

	# optimizer
	opt_disc = optim.Adam(
		list(disc_A.parameters()) + list(disc_B.parameters()),
		lr=args.lr,
		betas=(0.5, 0.999),
	)
	opt_gen = optim.Adam(
		list(gen_B.parameters()) + list(gen_A.parameters()),
		lr=args.lr,
		betas=(0.5, 0.999),
	)

	# loss functions
	l1 = nn.L1Loss()
	mse = nn.MSELoss()

	# dataset & dataloader
	dataset = SkecthRenderDataset(
		root_A=args.data_dir / f"{args.project_name}/{A_name}_images",
		root_B=args.data_dir / f"{args.project_name}/{B_name}_images",
		img_size=args.img_size,
		logger=logger,
	)
	train_len = int(len(dataset) * 0.8)
	valid_len = len(dataset) - train_len
	train_dataset, valid_dataset = random_split(dataset, lengths=[train_len, valid_len])
	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=True,
	)
	valid_loader = DataLoader(
		valid_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True
	)

	# metrics
	train_metric = CycleGANMetric(name="train", use_identity=args.use_identity, num_epoch=args.num_epoch)
	valid_metric = CycleGANMetric(name="valid", use_identity=args.use_identity, num_epoch=args.num_epoch)

	# training (currently no validation)
	for epoch in range(args.num_epoch):
		print(f"Epoch: {epoch+1}/{args.num_epoch}")
		train_kwargs = {"disc_A": disc_A, "disc_B": disc_B, "gen_A": gen_A, "gen_B": gen_B,
						"train_loader": train_loader, "valid_loader": valid_loader,
						"opt_disc": opt_disc, "opt_gen": opt_gen, "use_identity": args.use_identity,
						"l1": l1, "mse": mse, "epoch": epoch,
						"train_metric": train_metric, "valid_metric": valid_metric,
						"logger": logger, "visualization_paths": visualization_paths}
		train_fn(**train_kwargs)
		valid_fn(**train_kwargs)

	# plot
	plot_metric(train_metric, valid_metric, A_name, B_name, args.ckpt_dir)



if __name__ == "__main__":
	args = parse_args()
	main(args)
