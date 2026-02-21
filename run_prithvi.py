import os

records_dir = "records"

for name in os.listdir(records_dir):
    file_content = open(f"{records_dir}/{name}", "r", encoding='latin-1').readlines()
    last_line = file_content[-1]
    if "wandb: Find logs" in last_line: 
        continue

    print(name)

# ###############################################################
# # * Prithvi Pretrained Conv3d 
# ###############################################################
load_checkpoint = True
for feed_timeloc in [True]:
    for batch_size in [2]:
        for data_percentage in [1.0]:
            for use_config_normalization in [True]:
                group_name = f"prithvi_pretrained_conv3d"
                for learning_rate in [0.0001, 0.0005, 0.00001]:
                    for n_layers in [2]:

                        name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}_feed_timeloc-{feed_timeloc}_loss-mse_n_layers-{n_layers}"
                        if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
                            file_content = open(f"{records_dir}/{name}", "r").readlines()
                            last_line = file_content[-1]
                            if "wandb: Find logs" in last_line:
                                continue

                        command = f"qsub -v args=' --n_layers {n_layers} --loss mse --wandb_name {name} --feed_timeloc {feed_timeloc} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi_conv3d.sh"
                        os.system(command)
# # ###############################################################

# ###############################################################
# # * Shallow Transformer
# ###############################################################
for learning_rate in [0.0001, 0.0005, 0.00001]:
    for batch_size in [512]:
        for data_percentage in [1.0]:
                
            group_name = f"shallow_transformer_pixels"

            name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_loss-mse"
            
            if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
                file_content = open(f"{records_dir}/{name}", "r").readlines()
                last_line = file_content[-1]
                if "wandb: Find logs" in last_line: 
                    continue


            command = f"qsub -v args='--loss mse --wandb_name {name}   --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name}  --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_lsp_pixels.sh"
            os.system(command)
# # ###############################################################



# ###############################################################
# # * Prithvi Random Conv3d 
# ###############################################################
load_checkpoint = True
feed_timeloc = False
use_config_normalization = False
for batch_size in [2]:
    for data_percentage in [1.0]:
        group_name = f"prithvi_random_conv3d"
        for learning_rate in [0.0001, 0.0005, 0.00001]:
            for n_layers in [2]:

                name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}_feed_timeloc-{feed_timeloc}_loss-mse_n_layers-{n_layers}"
                if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
                    file_content = open(f"{records_dir}/{name}", "r").readlines()
                    last_line = file_content[-1]
                    if "wandb: Find logs" in last_line:
                        continue

                command = f"qsub -v args=' --n_layers {n_layers} --loss mse --wandb_name {name} --feed_timeloc {feed_timeloc} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi_conv3d.sh"
                os.system(command)
# # ###############################################################

###############################################################
# * Shallow Transformer Patch 
###############################################################
for learning_rate in [0.0001, 0.0005, 0.00001]:
    for patch_size in [2, 4, 8, 16]:
        for batch_size in [int(512/patch_size)]:
            for data_percentage in [1.0]:
                group_name = f"shallow_transformer_patch-{patch_size}"

                name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}"

                if os.path.exists(f"{records_dir}/{name}"):
                    file_content = open(f"{records_dir}/{name}", "r").readlines()
                    last_line = file_content[-1]
                    if "wandb: Find logs" in last_line:
                        continue

                    print("haha: ", name)

                command = f"qsub -v args='  --loss mse  --wandb_name {name}  --patch_size {patch_size} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name}  --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_lsp_patch.sh"
                os.system(command)
###############################################################
