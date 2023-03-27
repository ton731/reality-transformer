import torch
import torch.nn as nn
import torch.optim as optim
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
from model.generator import Generator
from model.discriminator import Discriminator



def parse_args() -> Namespace:
	parser = ArgumentParser()

	# data path
	parser.add_argument("--data_dir", type=Path, default="./data/")

	# checkpoint
	parser.add_argument("--ckpt_dir", type=Path, default="./results/")

	# output image size
	parser.add_argument("--img_size", type=int, default=256)

	# cycleGAN
	parser.add_argument("--lambda_identity", type=float, default=0.0)
	parser.add_argument("--lambda_cycle", type=float, default=10)

	# training
	parser.add_argument("--batch_size", type=int, default=1)
	parser.add_argument("--lr", type=float, default=1e-5)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--num_epoch", type=int, default=150)
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




# A for domain A
# B for domain B
def train_fn(disc_A, disc_B, gen_A, gen_B, loader, opt_disc, opt_gen, l1, mse, epoch, visualization_dir):
	loop = tqdm(loader, leave=True)
	A_reals = 0
	A_fakes = 0

	for idx, (B, A) in enumerate(loop):
		B = B.to(args.device)
		A = A.to(args.device)

		# Train Discriminator H and Z
		fake_A = gen_A(B)
		D_A_real = disc_A(A)
		D_A_fake = disc_A(fake_A.detach())
		A_reals += D_A_real.mean().item()
		A_fakes += D_A_fake.mean().item()
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
		cycle_B = gen_B(fake_A)
		cycle_A = gen_A(fake_B)
		cycle_B_loss = l1(B, cycle_B)
		cycle_A_loss = l1(A, cycle_A)

		# identity loss (remove these for efficiency if you set lambda_identity=0)
		# identity_B = gen_B(B)
		# identity_A = gen_A(A)
		# identity_B_loss = l1(B, identity_B)
		# identity_A_loss = l1(A, identity_A)

		# add all together
		G_loss = (
			loss_G_B
			+ loss_G_A
			+ cycle_B_loss * args.lambda_cycle
			+ cycle_A_loss * args.lambda_cycle
			# + identity_B_loss * args.lambda_identity
			# + identity_A_loss * args.lambda_identity
		)
		
		opt_gen.zero_grad()
		G_loss.backward()
		opt_gen.step()

		if idx % 50 == 0:
			save_image(fake_A * 0.5 + 0.5, visualization_dir / f"fake_reality_epoch{epoch}_idx{idx}.png")
			save_image(fake_B * 0.5 + 0.5, visualization_dir / f"fake_animation_epoch{epoch}_idx{idx}.png")
		
		loop.set_postfix(A_real=A_reals / (idx + 1), A_fake=A_fakes / (idx + 1))





def main(args):

	# set random seed
	set_random_seed(731)

	# set checkpoint directory
	args.ckpt_dir = args.ckpt_dir / datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
	args.ckpt_dir.mkdir(parents=True, exist_ok=True)

	# save visualization
	visualization_dir = args.ckpt_dir / "visualization"
	visualization_dir.mkdir(parents=True, exist_ok=True)

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
	L1 = nn.L1Loss()
	mse = nn.MSELoss()

	# dataset & dataloader
	dataset = SkecthRenderDataset(
		root_A=args.data_dir / "reality_images",
		root_B=args.data_dir / "animation_images",
		img_size=args.img_size,
		logger=logger,
	)
	loader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=True,
	)

	# training (currently no validation)
	for epoch in range(args.num_epoch):
		print(f"Epoch: {epoch+1}/{args.num_epoch}")
		train_fn(
			disc_A,
			disc_B,
			gen_A,
			gen_B,
			loader,
			opt_disc,
			opt_gen,
			L1,
			mse,
			epoch,
			visualization_dir,
		)



if __name__ == "__main__":
	args = parse_args()
	main(args)
