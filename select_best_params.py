import torch
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
from tqdm import tqdm
import pandas as pd
import path_config
import sys
sys.path.append("../")

from lib.utils import get_data_paths, eval_data_loader, get_masks_paper
from lib.dataloaders.dataloaders import CycleDataset

#######################################################################################

def main():

	device = "cuda"
	groups_dir = f"{path_config.get_checkpoint_root()}"

	all_groups = os.listdir(groups_dir)

	if os.path.exists(f"results/best_params.csv"):
		param_df = pd.read_csv(f"results/best_params.csv")
		best_param_df_cached = param_df.to_dict(orient="list")

		already_done_groups = best_param_df_cached["Model Name"]
		all_groups = [group for group in all_groups if group not in already_done_groups]

		best_param_df = {}
		best_param_df["Model Name"] = best_param_df_cached["Model Name"]
		best_param_df["Best Param"] = best_param_df_cached["Best Param"]
	else: 
		best_param_df = {}
		best_param_df["Model Name"] = []
		best_param_df["Best Param"] = []
		all_groups = all_groups


	for group in tqdm(all_groups):
		data_percentage = group.split("_")[-1]

		batch_size = 2 if "shallow_transformer" not in group else 4

		path_val = get_data_paths("validation", data_percentage)
		cycle_dataset_val = CycleDataset(path_val, split="validation", data_percentage=data_percentage)
		val_dataloader = DataLoader(cycle_dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)

		best_param = None 
		best_acc = 1000 

		for params in os.listdir(os.path.join(groups_dir, group)):

			if not params.endswith(".pth"):
				continue
			checkpoint = os.path.join(groups_dir, group, params)
			print(f"Loading checkpoint: {checkpoint}")

			# Load the model
			weights_path = None

			if "lora" in group:

				with open(f'configs/prithvi_300m.yaml', 'r') as f:
					prithvi_config = yaml.safe_load(f)
				prithvi_config["pretrained_cfg"]["img_size"] = 336

				from lib.models.prithvi_lora import PrithviSegLora
				# r_param = int(params.split("_")[3].replace("r-", ""))
				r_param = int(params.split("_r-")[1].split("_")[0])
				alpha_param = int(params.split("alpha-")[1].split("_")[0].replace(".pth", ""))
				lora_dict = {
					"Lora_peft_layer_name_pre": prithvi_config["Lora_peft_layer_name"][0],
					"Lora_peft_layer_name_suffix": prithvi_config["Lora_peft_layer_name"][1],
					"LP_layer_no_start": prithvi_config["Lora_peft_layer_no"][0],
					"LP_layer_no_end": prithvi_config["Lora_peft_layer_no"][1]
				}
				model = PrithviSegLora(prithvi_config["pretrained_cfg"], lora_dict, None, True, n_classes=4, model_size="300m", r=r_param, alpha=alpha_param)

			elif "miniprithvi" in group:
				from lib.models.prithvi_mini import TinyPrithviSeg
				model = TinyPrithviSeg(
					in_ch=6,
					T=12,                 # match baseline seq_len
					img_size=336,
					patch=(1,16,16),      # keeps tokens small; 12×21×21 tokens
					d_model=132,           # modest width
					depth=3,              # 3 encoder layers
					nhead=4,              # 80 / 4 = 20 per head
					num_classes=4,
					up_depth=4,           # /16 -> /8 -> /4 -> /2 -> /1
				)

			elif "shallow_transformer_patch" in group:
				from lib.models.lsp_transformer_patches import TemporalTransformerPerPatch
				patch_size = int(group.split("patch")[1].split("_")[0])
				model = TemporalTransformerPerPatch(
					input_channels=6,
					seq_len=12,
					num_classes=4,
					d_model=128,
					nhead=4,
					num_layers=3,
					dropout=0.1,
					patch_size=(patch_size, patch_size),
				)

			elif "shallow_transformer_pixels" in group:
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

			else: 

				with open(f'configs/prithvi_300m.yaml', 'r') as f:
					prithvi_config = yaml.safe_load(f)
				prithvi_config["pretrained_cfg"]["img_size"] = 336

				from lib.models.prithvi import PrithviSeg
				model=PrithviSeg(prithvi_config["pretrained_cfg"], weights_path, True, n_classes=4, model_size="300m")

			model=model.to(device)
			model.load_state_dict(torch.load(checkpoint)["model_state_dict"])

			acc_dataset_val, _, _ = eval_data_loader(val_dataloader, model, device, get_masks_paper("train"))

			print(f"Parameters: {params}")
			print(f"Val avg acc: {np.mean(list(acc_dataset_val.values()))}")	

			if np.mean(list(acc_dataset_val.values())) < best_acc:
				best_acc = np.mean(list(acc_dataset_val.values()))
				best_param = params

		print(f"Best parameters: {best_param}")
		best_param_df["Model Name"].append(group)
		best_param_df["Best Param"].append(best_param)

		os.makedirs("results", exist_ok=True)
		best_param_df_to_save = pd.DataFrame(best_param_df)
		best_param_df_to_save.to_csv(f"results/best_params.csv", index=False)

if __name__ == "__main__":
	main()
