import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
import path_config


def day_of_year_to_decimal_month(day_of_year):
	decimal_month = day_of_year / 547
	return decimal_month

def compute_or_load_means_stds(
	data_dir,
	split,
	data_percentage,
	num_bands,
	load_raster_fn,
	file_suffix="",
):
	means_stds_path = f"{path_config.get_mean_stds_dir()}/means_stds_{data_percentage}{file_suffix}.pkl"
	os.makedirs(os.path.dirname(means_stds_path), exist_ok=True)

	# Load precomputed means and stds if available
	if os.path.exists(means_stds_path):
		with open(means_stds_path, 'rb') as f:
			means, stds = pickle.load(f)
		return means, stds

	# Don't compute stats for test split
	if split in ["test", "testing"]:
		raise ValueError("Cannot compute mean and std for test split")

	# Initialize accumulators for Chan et al. algorithm
	global_mean = np.zeros(num_bands, dtype=np.float64)
	global_var = np.zeros(num_bands, dtype=np.float64)
	global_count = np.zeros(num_bands, dtype=np.float64)

	for i in tqdm(range(len(data_dir)), desc=f"Computing stats for {num_bands} band(s)"):
		image_path = data_dir[i][0]
		gt_path = data_dir[i][1]

		# Load all time steps for this sample
		images = []
		for path in image_path:
			images.append(load_raster_fn(path)[:, np.newaxis])

		img = np.concatenate(images, axis=1)  # shape: (num_bands, time_steps, H, W)

		# Create mask for dead pixels (zeros in both time and bands dimensions)
		# A pixel is dead if it's zero across all bands and all time steps
		dead_pixel_mask = np.all(img == 0, axis=(0, 1))  # shape: (H, W)

		# Expand mask to match image shape: (num_bands, time_steps, H, W)
		time_steps = img.shape[1]
		expanded_mask = np.repeat(dead_pixel_mask[np.newaxis, :, :], time_steps, axis=0)  # (time_steps, H, W)
		expanded_mask = np.repeat(expanded_mask[np.newaxis, :, :, :], num_bands, axis=0)  # (num_bands, time_steps, H, W)

		# Zero out dead pixels
		img[expanded_mask] = 0
		img_flat = img.reshape(num_bands, -1)
		mask_flat = ~expanded_mask.reshape(num_bands, -1)

		# Compute statistics per band using Chan et al. algorithm
		for b in range(num_bands):
			valid_values = img_flat[b][mask_flat[b]]
			# valid_values = img_flat[b]

			n = len(valid_values)
			if n == 0:
				continue

			batch_mean = valid_values.mean()
			batch_var = valid_values.var(ddof=1)  # sample variance

			m = global_count[b]
			mu1 = global_mean[b]
			mu2 = batch_mean
			v1 = global_var[b]
			v2 = batch_var

			# Combine means and variances
			combined_mean = (m / (m + n)) * mu1 + (n / (m + n)) * mu2 if (m + n) > 0 else mu2
			combined_var = (
				(m / (m + n)) * v1
				+ (n / (m + n)) * v2
				+ (m * n / (m + n) ** 2) * (mu1 - mu2) ** 2
				if (m + n) > 0
				else v2
			)

			global_mean[b] = combined_mean
			global_var[b] = combined_var
			global_count[b] = m + n

	means = global_mean
	stds = np.sqrt(global_var)

	print("Mean: ", means)
	print("Stds: ", stds)

	# Cache the computed statistics
	with open(means_stds_path, 'wb') as f:
		pickle.dump([means, stds], f)

	return means, stds


def print_trainable_parameters(model, detailed=False):
	"""
	Prints the number of trainable parameters in the model.
	If detailed=True, also prints breakdown by top-level module.
	"""
	trainable_params = 0
	all_param = 0
	module_params = {}  # Track params per top-level module

	for name, param in model.named_parameters():
		num_params = param.numel()
		all_param += num_params

		# Get top-level module name (e.g., 'backbone', 'head')
		top_level = name.split('.')[0] if '.' in name else name

		if top_level not in module_params:
			module_params[top_level] = {'trainable': 0, 'total': 0}
		module_params[top_level]['total'] += num_params

		if param.requires_grad:
			trainable_params += num_params
			module_params[top_level]['trainable'] += num_params

	print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}%")

	if detailed:
		print("\nParameter breakdown by module:")
		print("-" * 60)
		for module_name, counts in sorted(module_params.items()):
			trainable = counts['trainable']
			total = counts['total']
			pct = 100 * trainable / total if total > 0 else 0
			print(f"  {module_name}: {trainable:,} / {total:,} ({pct:.2f}% trainable)")


def segmentation_loss_pixels(targets, preds, device, ignore_index=-1):
	"""
	Compute regression loss for pixel dataset.
	
	Args:
		targets: (B,) tensor of ground truth labels (float)
		preds:   (B,) or (B, num_outputs) tensor of predictions
		device:  torch device
		ignore_index: value in targets to ignore (default -1)
	"""
	criterion = nn.MSELoss(reduction="sum").to(device)

	# valid mask = targets not equal to ignore_index
	valid_mask = targets != ignore_index

	if valid_mask.sum() > 0:
		valid_pred = preds[valid_mask]
		valid_target = targets[valid_mask]   # normalize like before
		loss = criterion(valid_pred, valid_target)
		return loss / valid_mask.sum().item()
	else:
		return torch.tensor(0.0, device=device)


def segmentation_loss_pixels_mae(targets, preds, device, ignore_index=-1):
	"""
	Compute MAE regression loss for pixel dataset.

	Args:
		targets: (B,) tensor of ground truth labels (float)
		preds:   (B,) or (B, num_outputs) tensor of predictions
		device:  torch device
		ignore_index: value in targets to ignore (default -1)
	"""
	criterion = nn.L1Loss(reduction="sum").to(device)

	valid_mask = targets != ignore_index

	if valid_mask.sum() > 0:
		valid_pred = preds[valid_mask]
		valid_target = targets[valid_mask]
		loss = criterion(valid_pred, valid_target)
		return loss / valid_mask.sum().item()
	else:
		return torch.tensor(0.0, device=device)


def segmentation_loss(mask, pred, device, ignore_index=-1):
	mask = mask.float()  # Convert mask to float for regression loss

	criterion = nn.MSELoss(reduction="sum").to(device)

	loss = 0
	num_channels = pred.shape[1]  # Number of output channels
	total_valid_pixels = 0  # Counter for valid pixels

	for idx in range(num_channels):
		valid_mask = mask[:, idx] != ignore_index

		if valid_mask.sum() > 0:  # Ensure there are valid pixels to compute loss

			valid_pred = pred[:, idx][valid_mask]  # Apply mask to predictions
			valid_target = mask[:, idx][valid_mask]  # Apply mask to ground truth

			loss += criterion(valid_pred, valid_target)
			total_valid_pixels += valid_mask.sum().item()

	# Normalize by total valid pixels to avoid division by zero
	return loss / total_valid_pixels if total_valid_pixels > 0 else torch.tensor(0.0, device=device)


def segmentation_loss_mae(mask, pred, device, ignore_index=-1):
	mask = mask.float()

	criterion = nn.L1Loss(reduction="sum").to(device)

	loss = 0
	num_channels = pred.shape[1]
	total_valid_pixels = 0

	for idx in range(num_channels):
		valid_mask = mask[:, idx] != ignore_index

		if valid_mask.sum() > 0:
			valid_pred = pred[:, idx][valid_mask]
			valid_target = mask[:, idx][valid_mask]

			loss += criterion(valid_pred, valid_target)
			total_valid_pixels += valid_mask.sum().item()

	return loss / total_valid_pixels if total_valid_pixels > 0 else torch.tensor(0.0, device=device)


def get_masks_paper(data="train", device="cuda"):

	data_name = "train" if data in ["train", "val"] else "test"
	test_file = f"data/LSP_{data_name}_samples.csv"
	
	data_paper_df = pd.read_csv(test_file)
	data_paper_df = data_paper_df[data_paper_df["version"] == "v1"]

	tiles_paper_masks = {}

	# Group by (year, tile)
	for (year, site_id, tile), group in data_paper_df.groupby(['years', "SiteID", 'tile']):
		# Initialize 320x320 mask with False
		mask = np.zeros((330, 330), dtype=bool)
		
		#subtract by 1 except 0 
		# Set True where (row, col) is mentioned in the group
		mask[group['row'].values, group['col'].values] = True
		
		# Store the mask with key (year, tile)

		mask = torch.Tensor(mask).bool().to(device)
		tiles_paper_masks[f"{year}_{site_id}_{tile}"] = mask
	
	return tiles_paper_masks

def compute_accuracy(gt_hls_tile, pred_hls_tile_avg, all_errors_hls_tile, hls_tile_n, paper_mask):

	for idx in range(4): 

		pred_idx = pred_hls_tile_avg[idx]
		gt_idx = gt_hls_tile[idx]

		pred_idx = pred_idx.flatten()
		gt_idx = gt_idx.flatten()

		mask = (gt_idx != -1) & paper_mask.flatten()
		pred_idx = pred_idx[mask]
		gt_idx = gt_idx[mask]

		errors = (pred_idx - gt_idx).detach().cpu().numpy() * 547
		all_errors_hls_tile[hls_tile_n][idx] = np.mean(np.abs(errors))

	return all_errors_hls_tile


def eval_data_loader(data_loader,model, device, tiles_paper_masks, loss_fn=None):

	if loss_fn is None:
		loss_fn = segmentation_loss

	model.eval()

	all_errors_hls_tile = {}

	eval_loss = 0.0
	with torch.no_grad():
		for _,data in tqdm(enumerate(data_loader), total=len(data_loader)):


			input = data["image"]
			ground_truth = data["gt_mask"].to(device)
			predictions=model(input)

			predictions = predictions[:, :, :330, :330]
			eval_loss += loss_fn(mask=data["gt_mask"].to(device),pred=predictions,device=device).item() * ground_truth.size(0)  # Multiply by batch size

			pred_hls_tile_all = predictions  # Average over the last dimension	

			for gt_hls_tile, pred_hls_tile_avg, hls_tile_n in zip(ground_truth, pred_hls_tile_all, data["hls_tile_name"]): 

				assert hls_tile_n not in all_errors_hls_tile, f"Tile {hls_tile_n} already exists in all_errors_hls_tile"
				all_errors_hls_tile[hls_tile_n] = {i:0 for i in range(4)}  # Initialize errors for each of the 4 predicted dates
				all_errors_hls_tile = compute_accuracy(gt_hls_tile, pred_hls_tile_avg, all_errors_hls_tile, hls_tile_n, tiles_paper_masks[hls_tile_n])

	all_errors_time = {i:[] for i in range(4)}
	for tile in all_errors_hls_tile:
		for i in range(4):
			all_errors_time[i].append(all_errors_hls_tile[tile][i])

	acc_dataset_val = {i:np.mean(all_errors_time[i]) for i in range(4)}
	epoch_loss_val = eval_loss / len(data_loader.dataset)

	return acc_dataset_val, all_errors_hls_tile, epoch_loss_val

def eval_data_loader_df(data_loader,model, device, tiles_paper_masks, feats_path=None):

	model.eval()

	#G_pred_DOY  M_pred_DOY  S_pred_DOY  D_pred_DOY
	data_df = { 
		"index":[],
		"years":[],
		"HLStile":[], 
		"SiteID": [],
		"row":[],
		"col":[],
		"version":[],
		"G_pred_DOY":[],
		"M_pred_DOY":[],
		"S_pred_DOY":[],
		"D_pred_DOY":[],
		"G_truth_DOY":[],
		"M_truth_DOY":[],
		"S_truth_DOY":[],
		"D_truth_DOY":[]
	}


	eval_loss = 0.0
	with torch.no_grad():
		for _,data in tqdm(enumerate(data_loader), total=len(data_loader)):
			
			input = data["image"]
			ground_truth = data["gt_mask"].to(device)
			hls_tile_name = data["hls_tile_name"]

			if feats_path is not None:
				feats = [] 
				for hls_tile_name in hls_tile_name:
					feats.append(np.load(feats_path + hls_tile_name + ".npz")["Z"])
				feats = np.array(feats)
				feats = torch.from_numpy(feats).to(device).float()
				assert feats.shape[0] == input.shape[0], f"Feats shape {feats.shape} does not match input shape {input.shape}"
				predictions=model(input, z_ctx=feats)
			else:
				predictions=model(input)

			predictions = predictions[:, :, :330, :330]
			pred_hls_tile_all = predictions

			eval_loss += segmentation_loss(mask=data["gt_mask"].to(device),pred=predictions,device=device).item() * ground_truth.size(0)  # Multiply by batch size

			for gt_hls_tile, pred_hls_tile_avg, hls_tile_n in zip(ground_truth, pred_hls_tile_all, data["hls_tile_name"]): 
				
				mask_tilen = tiles_paper_masks[hls_tile_n] 
				year, siteid, hlstile = hls_tile_n.split("_")
				#get the row and col from the mask
				row, col = np.where(mask_tilen.cpu().numpy())
				#add all data for row/col, don't worry about -1
				#do it in parralel 

				for r, c in zip(row, col):
					data_df["index"].append(len(data_df["index"]))
					data_df["years"].append(year)
					data_df["HLStile"].append(hlstile)
					data_df["SiteID"].append(siteid)
					data_df["row"].append(r)
					data_df["col"].append(c)
					data_df["version"].append("v1")
					data_df["G_pred_DOY"].append(pred_hls_tile_avg[0, r, c].item()*547)
					data_df["M_pred_DOY"].append(pred_hls_tile_avg[1, r, c].item()*547)
					data_df["S_pred_DOY"].append(pred_hls_tile_avg[2, r, c].item()*547)
					data_df["D_pred_DOY"].append(pred_hls_tile_avg[3, r, c].item()*547)
					data_df["G_truth_DOY"].append(gt_hls_tile[0, r, c].item()*547)
					data_df["M_truth_DOY"].append(gt_hls_tile[1, r, c].item()*547)
					data_df["S_truth_DOY"].append(gt_hls_tile[2, r, c].item()*547)
					data_df["D_truth_DOY"].append(gt_hls_tile[3, r, c].item()*547)

	data_df = pd.DataFrame(data_df)
	return data_df	


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filename):

	checkpoint = {
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'train_loss': train_loss,
		'val_loss':val_loss
	}
	torch.save(checkpoint, filename)
	print(f"Checkpoint saved at {filename}")
	
def get_data_paths(mode, data_percentage=1.0):

	data_dir_name = f"{data_percentage}"
	checkpoint_data = f"{path_config.get_data_paths_dir()}/{data_dir_name}/"

	if os.path.exists(f'{checkpoint_data}/data_pths_{mode}.pkl'):
		with open(f'{checkpoint_data}/data_pths_{mode}.pkl', 'rb') as f:
			data_dir = pickle.load(f)

		return data_dir

	hls_path = path_config.get_data_hls_composites()
	lsp_path = path_config.get_data_lsp_ancillary()

	hls_tiles = [x for x in os.listdir(hls_path) if x.endswith('.tif')]
	lsp_tiles = []
	lsp_tiles.extend([x for x in os.listdir(f"{lsp_path}/A2019") if x.endswith('.tif')])
	lsp_tiles.extend([x for x in os.listdir(f"{lsp_path}/A2020") if x.endswith('.tif')])

	title_hls = ['_'.join(x.split('_')[3:5]).split(".")[0] for x in hls_tiles]
	title_hls = set(title_hls)

	title_hls_lsp = ["_".join(x.split('_')[3:5]) for x in lsp_tiles]
	title_hls_lsp = set(title_hls_lsp)

	hls_tiles_time = [] 
	lsp_tiles_time = []
	hls_tiles_name = []

	for year in ["2019", "2020"]:
		past_months = range(1, 13)
		timesteps = [f"{year}-{str(x).zfill(2)}" for x in past_months]

		for hls_tile in tqdm(title_hls):
			temp_ordered = [] 

			for timestep in timesteps:

				hls_tile_location = hls_tile.split("_")[0]
				hls_tile_name = hls_tile.split("_")[1]

				temp_ordered.append(f"{hls_path}/HLS_composite_{timestep}_{hls_tile_location}_{hls_tile_name}.tif")

			temp_lsp = f"{lsp_path}/A{year}/HLS_PhenoCam_A{year}_{hls_tile_location}_{hls_tile_name}_LSP_Date.tif" if f"HLS_PhenoCam_A{year}_{hls_tile_location}_{hls_tile_name}_LSP_Date.tif" in lsp_tiles else None

			hls_tiles_time.append(temp_ordered)
			lsp_tiles_time.append(temp_lsp)
			hls_tiles_name.append(f"{year}_{hls_tile}")

	#open training file 
	with open(f"{lsp_path}/HP-LSP_train_ids.csv", 'r') as f:
		train_ids = f.readlines()[0].replace("'", "").split(",")
		train_ids = [x.strip() for x in train_ids]

	with open(f"{lsp_path}/HP-LSP_test_ids.csv", 'r') as f:
		test_ids = f.readlines()[0].replace("'", "").split(",")
		test_ids = [x.strip() for x in test_ids]


	hls_tiles_val = [
		"2019_ME-1_T19TEL",
		"2019_FL-3_T17RML",
		"2020_WI-2_T15TYL",
		"2019_AZ-5_T12SVE",
		"2020_CO-2_T13TDE",
		"2020_OR-1_T10TEQ",
		"2019_MD-1_T18SUJ",
		"2020_ND-1_T14TLS"
	]

	hls_tiles_train = [x for x in hls_tiles_name if x.split("_")[1] in train_ids]
	hls_tiles_train = [x for x in hls_tiles_train if x not in hls_tiles_val]

	# Apply data_percentage globally to all training data
	num_to_keep = int(len(hls_tiles_train) * data_percentage)
	hls_tiles_train = hls_tiles_train[:num_to_keep]
	hls_tiles_test = [x for x in hls_tiles_name if x.split("_")[1] in test_ids]
	data_dir_train = [(x, y, z) for (x,y,z) in zip(hls_tiles_time, lsp_tiles_time, hls_tiles_name) if z in hls_tiles_train]

	data_dir_val = [(x, y, z) for (x,y,z) in zip(hls_tiles_time, lsp_tiles_time, hls_tiles_name) if z in hls_tiles_val]
	data_dir_test = [(x, y, z) for (x,y,z) in zip(hls_tiles_time, lsp_tiles_time, hls_tiles_name) if z in hls_tiles_test]

	os.makedirs(checkpoint_data, exist_ok=True)
	with open(f'{checkpoint_data}/data_pths_training.pkl', 'wb') as f:
		pickle.dump(data_dir_train, f)
	
	with open(f'{checkpoint_data}/data_pths_validation.pkl', 'wb') as f:
		pickle.dump(data_dir_val, f)
	
	with open(f'{checkpoint_data}/data_pths_testing.pkl', 'wb') as f:
		pickle.dump(data_dir_test, f)

	if mode == 'training':	
		return data_dir_train
	elif mode == 'validation':
		return data_dir_val
	elif mode == "testing":
		return data_dir_test
	else: 
		raise ValueError(f"Unknown mode: {mode}. Expected 'training', 'validation', or 'testing'.")


def get_ndvi_data_paths(mode, data_percentage=1.0):

	data_dir_name = f"{data_percentage}_ndvi"
	checkpoint_data = f"{path_config.get_data_paths_dir()}/{data_dir_name}/"

	if os.path.exists(f'{checkpoint_data}/data_pths_{mode}.pkl'):
		with open(f'{checkpoint_data}/data_pths_{mode}.pkl', 'rb') as f:
			data_dir = pickle.load(f)

		return data_dir

	hls_path = path_config.get_data_ndvi_composites()
	lsp_path = path_config.get_data_lsp_ancillary()

	hls_tiles = [x for x in os.listdir(hls_path) if x.endswith('.tif')]
	lsp_tiles = []
	lsp_tiles.extend([x for x in os.listdir(f"{lsp_path}/A2019") if x.endswith('.tif')])
	lsp_tiles.extend([x for x in os.listdir(f"{lsp_path}/A2020") if x.endswith('.tif')])

	title_hls = ['_'.join(x.split('_')[3:5]).split(".")[0] for x in hls_tiles]
	title_hls = set(title_hls)

	title_hls_lsp = ["_".join(x.split('_')[3:5]) for x in lsp_tiles]
	title_hls_lsp = set(title_hls_lsp)

	hls_tiles_time = [] 
	lsp_tiles_time = []
	hls_tiles_name = []

	for year in ["2019", "2020"]:
		past_months = range(1, 13)
		timesteps = [f"{year}-{str(x).zfill(2)}" for x in past_months]

		for hls_tile in tqdm(title_hls):
			temp_ordered = [] 

			for timestep in timesteps:

				hls_tile_location = hls_tile.split("_")[0]
				hls_tile_name = hls_tile.split("_")[1]

				temp_ordered.append(f"{hls_path}/HLS_composite_{timestep}_{hls_tile_location}_{hls_tile_name}_NDVI.tif")

			temp_lsp = f"{lsp_path}/A{year}/HLS_PhenoCam_A{year}_{hls_tile_location}_{hls_tile_name}_LSP_Date.tif" if f"HLS_PhenoCam_A{year}_{hls_tile_location}_{hls_tile_name}_LSP_Date.tif" in lsp_tiles else None

			hls_tiles_time.append(temp_ordered)
			lsp_tiles_time.append(temp_lsp)
			hls_tiles_name.append(f"{year}_{hls_tile}")

	#open training file 
	with open(f"{lsp_path}/HP-LSP_train_ids.csv", 'r') as f:
		train_ids = f.readlines()[0].replace("'", "").split(",")
		train_ids = [x.strip() for x in train_ids]

	with open(f"{lsp_path}/HP-LSP_test_ids.csv", 'r') as f:
		test_ids = f.readlines()[0].replace("'", "").split(",")
		test_ids = [x.strip() for x in test_ids]


	hls_tiles_val = [
		"2019_ME-1_T19TEL",
		"2019_FL-3_T17RML",
		"2020_WI-2_T15TYL",
		"2019_AZ-5_T12SVE",
		"2020_CO-2_T13TDE",
		"2020_OR-1_T10TEQ",
		"2019_MD-1_T18SUJ",
		"2020_ND-1_T14TLS"
	]

	hls_tiles_train = [x for x in hls_tiles_name if x.split("_")[1] in train_ids]
	hls_tiles_train = [x for x in hls_tiles_train if x not in hls_tiles_val]

	# Apply data_percentage globally to all training data
	num_to_keep = int(len(hls_tiles_train) * data_percentage)
	hls_tiles_train = hls_tiles_train[:num_to_keep]
	hls_tiles_test = [x for x in hls_tiles_name if x.split("_")[1] in test_ids]
	data_dir_train = [(x, y, z) for (x,y,z) in zip(hls_tiles_time, lsp_tiles_time, hls_tiles_name) if z in hls_tiles_train]

	data_dir_val = [(x, y, z) for (x,y,z) in zip(hls_tiles_time, lsp_tiles_time, hls_tiles_name) if z in hls_tiles_val]
	data_dir_test = [(x, y, z) for (x,y,z) in zip(hls_tiles_time, lsp_tiles_time, hls_tiles_name) if z in hls_tiles_test]

	os.makedirs(checkpoint_data, exist_ok=True)
	with open(f'{checkpoint_data}/data_pths_training.pkl', 'wb') as f:
		pickle.dump(data_dir_train, f)
	
	with open(f'{checkpoint_data}/data_pths_validation.pkl', 'wb') as f:
		pickle.dump(data_dir_val, f)
	
	with open(f'{checkpoint_data}/data_pths_testing.pkl', 'wb') as f:
		pickle.dump(data_dir_test, f)

	if mode == 'training':	
		return data_dir_train
	elif mode == 'validation':
		return data_dir_val
	elif mode == "testing":
		return data_dir_test
	else: 
		raise ValueError(f"Unknown mode: {mode}. Expected 'training', 'validation', or 'testing'.")

