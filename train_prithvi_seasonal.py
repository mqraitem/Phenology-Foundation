import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml

import os

# Limit threading libraries BEFORE importing torch/numpy
os.environ["OMP_NUM_THREADS"] = "4"  # OpenMP threads
os.environ["MKL_NUM_THREADS"] = "4"  # Intel MKL threads
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # OpenBLAS threads
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # vecLib threads
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # NumExpr threads
os.environ["GDAL_NUM_THREADS"] = "4"
os.environ["GDAL_CACHEMAX"] = "512"  # Limit cache to 512MB

import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import path_config

from lib.models.prithvi_seasonal import PrithviSeasonal

from lib.utils import segmentation_loss, segmentation_loss_mae, eval_data_loader, get_masks_paper, save_checkpoint, str2bool
from lib.utils import get_data_paths, print_trainable_parameters

from lib.dataloaders.dataloaders import CycleDataset
from arg_configs import get_core_parser

#######################################################################################

def main():

	parser = get_core_parser()

	# Prithvi args
	parser.add_argument("--model_size", type=str, default="300m",
	                   help="Model size to use (300m or 600m)")
	parser.add_argument("--load_prithvi_checkpoint", type=str2bool, default=True,
	                   help="Whether to load pretrained Prithvi checkpoint")
	parser.add_argument("--load_feature_checkpoint", type=str2bool, default=True,
	                   help="Whether to load pretrained feature extractor checkpoint")
	parser.add_argument("--feature_ckpt_path", type=str, default="data/checkpoints_feature/feature.pth",
	                   help="Path to pretrained feature extractor checkpoint")
	parser.add_argument("--using_sampler", type=str2bool, default=False,
	                   help="Whether to use weighted sampler for imbalanced data")

	# PixelShuffle args
	parser.add_argument("--c_per_t", type=int, default=4,
	                   help="Channels per timestep after PixelShuffle")

	# Feature extractor args
	parser.add_argument("--fe_variant", type=str, default="conv3",
	                   choices=["se", "conv1", "conv3", "transformer"],
	                   help="Temporal block variant for feature extractor")
	parser.add_argument("--fe_emb_dim", type=int, default=128,
	                   help="Embedding dimension for feature extractor")
	parser.add_argument("--fe_n_layers", type=int, default=4,
	                   help="Number of temporal blocks in feature extractor")
	parser.add_argument("--fe_dropout", type=float, default=0.1,
	                   help="Dropout rate for feature extractor")

	parser.add_argument("--loss", type=str, default="mae", choices=["mse", "mae"],
	                   help="Loss function: mse (mean squared error) or mae (mean absolute error)")

	args = parser.parse_args()

	wandb_config = {
		"learningrate": args.learning_rate,
		"model_size": args.model_size,
		"load_prithvi_checkpoint": args.load_prithvi_checkpoint,
		"load_feature_checkpoint": args.load_feature_checkpoint,
		"batch_size": args.batch_size,
		"data_percentage": args.data_percentage,
		"c_per_t": args.c_per_t,
		"fe_variant": args.fe_variant,
		"fe_emb_dim": args.fe_emb_dim,
		"fe_n_layers": args.fe_n_layers,
		"loss": args.loss,
	}

	wandb_name = args.wandb_name

	with open(f'configs/prithvi_{args.model_size}.yaml', 'r') as f:
		config = yaml.safe_load(f)

	config["training"]["n_iteration"] = 200
	config["pretrained_cfg"]["img_size"] = 336

	# Key modification: set num_frames=4 for Prithvi (seasonal tokens)
	config["pretrained_cfg"]["num_frames"] = 4

	config["training"]["batch_size"] = args.batch_size
	config["validation"]["batch_size"] = args.batch_size
	config["test"]["batch_size"] = args.batch_size

	group_name = args.group_name

	if args.logging:
		wandb.init(
				project=f"phenology_mae_{args.data_percentage}",
				group=group_name,
				config=wandb_config,
				name=wandb_name,
				)
		wandb.run.log_code(".")

	path_train=get_data_paths("training", args.data_percentage)
	path_val=get_data_paths("validation", args.data_percentage)
	path_test=get_data_paths("testing", args.data_percentage)

	print(len(path_train), len(path_val), len(path_test))

	# Use config normalization if flag is set
	if args.use_config_normalization:
		means = config["pretrained_cfg"]["mean"]
		stds = config["pretrained_cfg"]["std"]
		print(f"Using config normalization - means: {means}, stds: {stds}")
	else:
		means = None
		stds = None
		print("Computing means/stds from dataset")

	cycle_dataset_train=CycleDataset(path_train,split="training", data_percentage=args.data_percentage, means=means, stds=stds)
	cycle_dataset_val=CycleDataset(path_val,split="validation", data_percentage=args.data_percentage, means=means, stds=stds)
	cycle_dataset_test=CycleDataset(path_test,split="testing", data_percentage=args.data_percentage, means=means, stds=stds)

	from torch.utils.data import WeightedRandomSampler

	sampler = WeightedRandomSampler(
		weights=cycle_dataset_train.sample_weights,
		num_samples=len(cycle_dataset_train),
		replacement=True
	)

	if args.using_sampler:
		train_dataloader=DataLoader(cycle_dataset_train,batch_size=config["training"]["batch_size"],num_workers=2, sampler=sampler)
	else:
		train_dataloader=DataLoader(cycle_dataset_train,batch_size=config["training"]["batch_size"],shuffle=config["training"]["shuffle"],num_workers=4)

	val_dataloader=DataLoader(cycle_dataset_val,batch_size=config["validation"]["batch_size"],shuffle=config["validation"]["shuffle"],num_workers=1)
	test_dataloader=DataLoader(cycle_dataset_test,batch_size=config["test"]["batch_size"],shuffle=config["validation"]["shuffle"],num_workers=1)

	device = "cuda"
	prithvi_weights = path_config.get_model_weights(args.model_size) if args.load_prithvi_checkpoint else None
	feature_weights = args.feature_ckpt_path if args.load_feature_checkpoint else None

	model = PrithviSeasonal(
		config["pretrained_cfg"], prithvi_weights, feature_weights,
		n_classes=4, model_size=args.model_size,
		fe_variant=args.fe_variant, fe_emb_dim=args.fe_emb_dim,
		fe_n_layers=args.fe_n_layers, fe_dropout=args.fe_dropout,
		c_per_t=args.c_per_t,
	)
	model=model.to(device)

	n_epochs = config["training"]["n_iteration"]
	unfreeze_epoch = max(1, int(0.1 * n_epochs)) if args.load_prithvi_checkpoint else 0

	print_trainable_parameters(model, detailed=True)

	# Freeze pretrained components initially
	if args.load_prithvi_checkpoint:
		# Freeze Prithvi transformer blocks
		for param in model.prithvi.encoder.blocks.parameters():
			param.requires_grad = False
		for param in model.prithvi.encoder.norm.parameters():
			param.requires_grad = False
		print(f"Prithvi transformer blocks frozen for first {unfreeze_epoch} epoch(s)")

	if args.load_feature_checkpoint:
		for param in model.feature_extractor.parameters():
			param.requires_grad = False
		print("Feature extractor frozen")

	group_name_checkpoint = f"{group_name}_{args.data_percentage}"
	checkpoint_dir = path_config.get_checkpoint_root() + f"/{group_name_checkpoint}"
	os.makedirs(checkpoint_dir, exist_ok=True)
	checkpoint = f"{checkpoint_dir}/{wandb_name}.pth"

	optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=1e-4)
	scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, verbose=True)

	loss_fn = segmentation_loss_mae if args.loss == "mae" else segmentation_loss
	print(f"Using loss function: {args.loss}")

	best_acc_val=100
	for epoch in range(config["training"]["n_iteration"]):

		loss_i=0.0

		print("iteration started")
		model.train()

		for j,batch_data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

			input = batch_data["image"]
			mask = batch_data["gt_mask"]

			mask=mask.to(device)

			optimizer.zero_grad()

			out=model(input)
			out = out[:, :, :330, :330]

			loss=loss_fn(mask=mask,pred=out,device=device)
			loss_i += loss.item() * mask.size(0)

			loss.backward()
			optimizer.step()

			if j%10==0:
				to_print = f"Epoch: {epoch}, iteration: {j}, loss: {loss.item()} \n "
				print(to_print)

		epoch_loss_train = loss_i / len(train_dataloader.dataset)

		# Validation Phase
		acc_dataset_val, _, epoch_loss_val = eval_data_loader(val_dataloader, model, device, get_masks_paper("train"), loss_fn=loss_fn)
		acc_dataset_test, _, epoch_loss_test = eval_data_loader(test_dataloader, model, device, get_masks_paper("test"), loss_fn=loss_fn)

		if args.logging:
			to_log = {}
			to_log["epoch"] = epoch + 1
			to_log["val_loss"] = epoch_loss_val
			to_log["test_loss"] = epoch_loss_test
			to_log["train_loss"] = epoch_loss_train
			to_log["learning_rate"] = optimizer.param_groups[0]['lr']
			for idx in range(4):
				to_log[f"acc_val_{idx}"] = acc_dataset_val[idx]
				to_log[f"acc_test_{idx}"] = acc_dataset_test[idx]

			wandb.log(to_log)

		print("="*100)
		to_print = f"Epoch: {epoch}, val_loss: {epoch_loss_val} \n "
		for idx in range(4):
			to_print += f"acc_val_{idx}: {acc_dataset_val[idx]} \n "

		for idx in range(4):
			to_print += f"acc_test_{idx}: {acc_dataset_test[idx]} \n "

		print(to_print)
		print("="*100)

		scheduler.step(epoch_loss_val)
		acc_dataset_val_mean = np.mean(list(acc_dataset_val.values()))

		if acc_dataset_val_mean<best_acc_val:
			save_checkpoint(model, optimizer, epoch, epoch_loss_train, epoch_loss_val, checkpoint)
			best_acc_val=acc_dataset_val_mean

		if epoch + 1 == unfreeze_epoch:
			for param in model.parameters():
				param.requires_grad = True
			print(f"Unfreezing all parameters at epoch {epoch + 1}")
			print("=" * 100)

	model.load_state_dict(torch.load(checkpoint)["model_state_dict"])

	acc_dataset_val, _, epoch_loss_val = eval_data_loader(val_dataloader, model, device, get_masks_paper("train"), loss_fn=loss_fn)
	acc_dataset_test, _, _ = eval_data_loader(test_dataloader, model, device, get_masks_paper("test"), loss_fn=loss_fn)

	if args.logging:
		for idx in range(4):
			wandb.run.summary[f"best_acc_val_{idx}"] = acc_dataset_val[idx]
			wandb.run.summary[f"best_acc_test_{idx}"] = acc_dataset_test[idx]
		wandb.run.summary[f"best_avg_acc_val"] = np.mean(list(acc_dataset_val.values()))
		wandb.run.summary[f"best_avg_acc_test"] = np.mean(list(acc_dataset_test.values()))

	if args.logging:
		wandb.finish()


if __name__ == "__main__":
	main()
