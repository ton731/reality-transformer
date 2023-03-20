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

from utils.dataset import RealityAnimationDataset
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
	parser.add_argument("--batch_size", type=int, default=3)
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




# R for Reality
# A for Animation
def train_fn(disc_R, disc_A, gen_R, gen_A, loader, opt_disc, opt_gen, l1, mse, epoch, visualization_dir):
	loop = tqdm(loader, leave=True)
	R_reals = 0
	R_fakes = 0

	for idx, (animation, reality) in enumerate(loop):
		animation = animation.to(args.device)
		reality = reality.to(args.device)

		# Train Discriminator H and Z
		with torch.cuda.amp.autocast():
			fake_reality = gen_R(animation)
			D_R_real = disc_R(reality)
			D_R_fake = disc_R(fake_reality.detach())
			R_reals += D_R_real.mean().item()
			R_fakes += D_R_fake.mean().item()
			D_R_real_loss = mse(D_R_real, torch.ones_like(D_R_real))
			D_R_fake_loss = mse(D_R_fake, torch.zeros_like(D_R_real))
			D_R_loss = D_R_real_loss + D_R_fake_loss

			fake_animation = gen_A(reality)
			D_A_real = disc_A(animation)
			D_A_fake = disc_A(fake_animation.detach())
			D_A_real_loss = mse(D_A_real, torch.ones_like(D_A_real))
			D_A_fake_loss = mse(D_A_fake, torch.zeros_like(D_A_fake))
			D_A_loss = D_A_real_loss + D_A_fake_loss

			# put it together
			D_loss = (D_R_loss + D_A_loss) / 2
		
		opt_disc.zero_grad()
		D_loss.backward()
		opt_disc.step()

		# Train Generators H and Z
		with torch.cuda.amp.autocast():
			# advesarial loss for both generators
			D_R_fake = disc_R(fake_reality)
			D_A_fake = disc_A(fake_animation)
			loss_G_R = mse(D_R_fake, torch.ones_like(D_R_fake))
			loss_G_A = mse(D_A_fake, torch.ones_like(D_A_fake))

			# cycle loss
			cycle_animation = gen_A(fake_reality)
			cycle_reality = gen_R(fake_animation)
			cycle_animation_loss = l1(animation, cycle_animation)
			cycle_reality_loss = l1(reality, cycle_reality)

			# identity loss (remove these for efficiency if you set lambda_identity=0)
			# identity_animation = gen_A(animation)
			# identity_reality = gen_R(reality)
			# identity_animation_loss = l1(animation, identity_animation)
			# identity_reality_loss = l1(reality, identity_reality)

			# add all together
			G_loss = (
				loss_G_A
				+ loss_G_R
				+ cycle_animation_loss * args.lambda_cycle
				+ cycle_reality_loss * args.lambda_cycle
				# + identity_animation_loss * args.lambda_identity
				# + identity_reality_loss * args.lambda_identity
			)
		
		opt_gen.zero_grad()
		G_loss.backward()
		opt_gen.step()

		if idx % 200 == 0:
			save_image(fake_reality * 0.5 + 0.5, visualization_dir / f"fake_reality_epoch{epoch}_idx{idx}.png")
			save_image(fake_animation * 0.5 + 0.5, visualization_dir / f"fake_animation_epoch{epoch}_idx{idx}.png")
		
		loop.set_postfix(R_real=R_reals / (idx + 1), R_fake=R_fakes / (idx + 1))





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
	disc_R = Discriminator(in_channels=3).to(args.device) # discriminate reality
	disc_A = Discriminator(in_channels=3).to(args.device)
	gen_A = Generator(img_channels=3, num_residuals=9).to(args.device)
	gen_R = Generator(img_channels=3, num_residuals=9).to(args.device)

	# optimizer
	opt_disc = optim.Adam(
		list(disc_R.parameters()) + list(disc_A.parameters()),
		lr=args.lr,
		betas=(0.5, 0.999),
	)
	opt_gen = optim.Adam(
		list(gen_A.parameters()) + list(gen_R.parameters()),
		lr=args.lr,
		betas=(0.5, 0.999),
	)

	# loss functions
	L1 = nn.L1Loss()
	mse = nn.MSELoss()

	# dataset & dataloader
	dataset = RealityAnimationDataset(
		root_reality=args.data_dir / "reality_images",
		root_animation=args.data_dir / "animation_images",
		img_size=args.img_size,
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
			disc_R,
			disc_A,
			gen_R,
			gen_A,
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
