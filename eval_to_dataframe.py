import torch
from torch.utils.data import DataLoader
import yaml
import os
import argparse
import pandas as pd
import path_config

from lib.utils import get_masks_paper, eval_data_loader_df, get_data_paths, str2bool
from lib.dataloaders.dataloaders import CycleDataset

#######################################################################################

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
						help="Dataset split to evaluate on (train/val/test)")
	args = parser.parse_args()

	device = "cuda"
	best_param_df = pd.read_csv(f"results/best_params.csv")
	model_names = best_param_df["Model Name"].values

	# Set output directory based on split
	output_dir = "results"
	os.makedirs(output_dir, exist_ok=True)

	for model_name in model_names:

		data_percentage = model_name.split("_")[-1]

		# Map split name for data loading
		if args.split == "test":
			data_split = "testing"
		elif args.split == "val":
			data_split = "validation"
		else:  # train
			data_split = "training"


		data_path = get_data_paths(data_split, data_percentage)
		cycle_dataset = CycleDataset(data_path, split=data_split, data_percentage=data_percentage)

		data_loader = DataLoader(cycle_dataset, batch_size=2, shuffle=False, num_workers=2)
		data_loader_name = args.split

		if os.path.exists(f"{output_dir}/{model_name}_{data_loader_name}.csv"):
			print(f"Results for {model_name} on {data_loader_name} already exist, skipping...")
			continue

		config_dir = f"{path_config.get_checkpoint_root()}/{model_name}/"
		best_param = best_param_df[best_param_df["Model Name"] == model_name]["Best Param"].values[0] 

		print(f"Best parameters: {best_param}")
		weights_path = None

		if "lora" in model_name:

			with open(f'configs/prithvi_300m.yaml', 'r') as f:
				prithvi_config = yaml.safe_load(f)
			prithvi_config["pretrained_cfg"]["img_size"] = 336

			from lib.models.prithvi_lora import PrithviSegLora
			r_param = int(best_param.split("_r-")[1].split("_")[0])
			alpha_param = int(best_param.split("alpha-")[1].split("_")[0].replace(".pth", ""))
			lora_dict = {
				"Lora_peft_layer_name_pre": prithvi_config["Lora_peft_layer_name"][0],
				"Lora_peft_layer_name_suffix": prithvi_config["Lora_peft_layer_name"][1],
				"LP_layer_no_start": prithvi_config["Lora_peft_layer_no"][0],
				"LP_layer_no_end": prithvi_config["Lora_peft_layer_no"][1]
			}
			model = PrithviSegLora(prithvi_config["pretrained_cfg"], lora_dict, None, True, n_classes=4, model_size="300m", r=r_param, alpha=alpha_param)


		elif "shallow_transformer_patch" in model_name:
			from lib.models.lsp_transformer_patches import TemporalTransformerPerPatch
			patch_size = int(model_name.split("patch")[1].split("_")[0])
			d_model = 128 * patch_size
			model = TemporalTransformerPerPatch(
				input_channels=6,
				seq_len=12,
				num_classes=4,
				d_model=d_model,
				nhead=4,
				num_layers=3,
				dropout=0.1,
				patch_size=(patch_size, patch_size),
			)

		elif "shallow_transformer_pixels" in model_name:
			from lib.models.lsp_transformer_pixels import TemporalTransformer
			model = TemporalTransformer(
				input_channels=6,
				seq_len=12,
				num_classes=4,
				d_model=128,
				nhead=4,
				num_layers=3,
				dropout=0.1
			)

		elif ("prithvi" in model_name) and ("conv3d" in model_name):

			with open(f'configs/prithvi_300m.yaml', 'r') as f:
				prithvi_config = yaml.safe_load(f)
			prithvi_config["pretrained_cfg"]["img_size"] = 336

			if "_feed_timeloc" in best_param:
				feed_timeloc = str2bool(best_param.split("_feed_timeloc-")[1].split("_")[0].replace(".pth", ""))
			else:
				feed_timeloc = False

			n_layers = int(best_param.split("_n_layers-")[1].split("_")[0].replace(".pth", ""))

			from lib.models.prithvi_conv3d import PrithviSegConv3D
			model=PrithviSegConv3D(prithvi_config["pretrained_cfg"], weights_path, True, n_classes=4, model_size="300m", feed_timeloc=feed_timeloc, n_layers=n_layers)

		else: 
			raise


		model=model.to(device)
		model.load_state_dict(torch.load(os.path.join(config_dir, best_param))["model_state_dict"])
		out_df = eval_data_loader_df(data_loader, model, device, get_masks_paper(data_loader_name))

		out_df.to_csv(f"{output_dir}/{model_name}_{data_loader_name}.csv", index=False)


if __name__ == "__main__":
	main()


