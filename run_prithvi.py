import os

records_dir = "records"

# for name in os.listdir(records_dir):
#     file_content = open(f"{records_dir}/{name}", "r").readlines()
#     last_line = file_content[-1]
#     if "wandb: Find logs" in last_line: 
#         continue

#     print(name)

# # Region filtering removed - all training uses ALL data
# # # ###############################################################
# # # # * Prithvi Pretrained LORA
# # # ###############################################################

# # # 8, 8
# # # 16, 32
# # # 64, 32
# # # 32, 64

# load_checkpoint = True
# for batch_size in [2]:
#     for data_percentage in [1.0]:
#         group_name = f"prithvi_lora"
        
#         for learning_rate in [0.00001]:
#             for r, alpha in [(8, 8), (16, 32), (64, 32), (32, 64)]:
#                 name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_r-{r}_alpha-{alpha}"
#                 if os.path.exists(f"{records_dir}/{name}"):
#                     file_content = open(f"{records_dir}/{name}", "r").readlines()
#                     last_line = file_content[-1]
#                     if "wandb: Find logs" in last_line: 
#                         continue

#                 command = f"qsub -v args=' --wandb_name {name} --r {r} --alpha {alpha} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi_lora.sh"
#                 os.system(command)
# # ###############################################################

# ###############################################################
# # * Prithvi Pretrained Temporal Only
# ###############################################################
# load_checkpoint = False
# for batch_size in [2]:
#     for data_percentage in [1.0]:
#         group_name = f"prithvi_pretrained_temopralonly"
        
#         for learning_rate in [0.0001, 0.00001, 0.000001]:
            
#             name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}"
#             if os.path.exists(f"{records_dir}/{name}"):
#                 file_content = open(f"{records_dir}/{name}", "r").readlines()
#                 last_line = file_content[-1]
#                 if "wandb: Find logs" in last_line: 
#                     continue

#             command = f"qsub -v args=' --wandb_name {name}  --temporal_only True --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate} --freeze False'  -o {records_dir}/{name} run_scripts/train_prithvi.sh"
#             os.system(command)
# ###############################################################



# # ###############################################################
# # # * Prithvi Random
# # ###############################################################
# load_checkpoint = False
# for freeze in [False]:
#     for batch_size in [2]:
#         for data_percentage in [1.0]:
#             group_name = f"prithvi_random"
            
#             for learning_rate in [0.0001, 0.00001, 0.000001]:
                
#                 name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}"
#                 if os.path.exists(f"{records_dir}/{name}"):
#                     file_content = open(f"{records_dir}/{name}", "r").readlines()
#                     last_line = file_content[-1]
#                     if "wandb: Find logs" in last_line: 
#                         continue

#                 command = f"qsub -v args='  --wandb_name {name}  --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate} --freeze {freeze}'  -o {records_dir}/{name} run_scripts/train_prithvi.sh"
#                 os.system(command)
# # ###############################################################


# # ###############################################################
# # # * Prithvi Pretrained
# # ###############################################################
# load_checkpoint = True
# for freeze in [False]:
#     for batch_size in [2]:
#         for data_percentage in [1.0]:
#             for use_config_normalization in [False, True]:
#                 group_name = "prithvi_pretrained"
#                 for learning_rate in [0.0001, 0.00001, 0.000001]:
                    
#                     name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}"
#                     if os.path.exists(f"{records_dir}/{name}"):
#                         file_content = open(f"{records_dir}/{name}", "r").readlines()
#                         last_line = file_content[-1]
#                         if "wandb: Find logs" in last_line: 
#                             continue

#                     command = f"qsub -v args='  --wandb_name {name}   --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate} --freeze {freeze}'  -o {records_dir}/{name} run_scripts/train_prithvi.sh"
#                     os.system(command)
# # # ###############################################################

# # ###############################################################
# # # * Shallow Transformer 
# # ###############################################################
# for learning_rate in [0.0001, 0.00001, 0.000001]:
#     for batch_size in [264, 512]:
#         for data_percentage in [1.0]:
                
#             group_name = f"shallow_transformer_pixels"

#             name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}"
            
#             if os.path.exists(f"{records_dir}/{name}"):
#                 file_content = open(f"{records_dir}/{name}", "r").readlines()
#                 last_line = file_content[-1]
#                 if "wandb: Find logs" in last_line: 
#                     continue

#                 # if "Disk quota exceeded" not in last_line:
#                 #     continue
#                 print("haha: ", name)

#             command = f"qsub -v args='--wandb_name {name}   --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name}  --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_lsp_pixels.sh"
#             os.system(command)
# # # ###############################################################



# ###############################################################
# # * Shallow Transformer Patch 
# ###############################################################
# for learning_rate in [0.0001, 0.00001]:
#     for patch_size in [2, 4, 8, 16]:
#         for batch_size in [int(264/patch_size), int(512/patch_size)]:
#             for data_percentage in [1.0]:
#                 group_name = f"shallow_transformer_patch{patch_size}"

#                 name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}"

#                 if os.path.exists(f"{records_dir}/{name}"):
#                     file_content = open(f"{records_dir}/{name}", "r").readlines()
#                     last_line = file_content[-1]
#                     if "wandb: Find logs" in last_line:
#                         continue

#                     print("haha: ", name)

#                 command = f"qsub -v args=' --wandb_name {name}  --patch_size {patch_size} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name}  --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_lsp_patch.sh"
#                 os.system(command)



# ###############################################################
# ###############################################################
# ###############################################################
# ###############################################################
# ###############################################################



# ###############################################################
# # * Shallow Transformer DATA 
# ###############################################################
for learning_rate in [0.00001]:
    for batch_size in [264]:
        for data_percentage in [0.05]:
                
            group_name = f"shallow_transformer_pixels"

            name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}"
            
            if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
                file_content = open(f"{records_dir}/{name}_{data_percentage}", "r").readlines()
                last_line = file_content[-1]
                if "wandb: Find logs" in last_line: 
                    continue

                # if "Disk quota exceeded" not in last_line:
                #     continue
                print("haha: ", name)

            command = f"qsub -v args='--wandb_name {name}   --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name}  --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name}_{data_percentage} run_scripts/train_lsp_pixels.sh"
            os.system(command)
# # ###############################################################


# ###############################################################
# # * Prithvi Pretrained DATA
# ###############################################################
load_checkpoint = True
for freeze in [False]:
    for batch_size in [2]:
        for data_percentage in [0.05]:
            for use_config_normalization in [False]:
                group_name = "prithvi_pretrained"
                for learning_rate in [0.00001]:
                    
                    name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}"
                    if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
                        file_content = open(f"{records_dir}/{name}_{data_percentage}", "r").readlines()
                        last_line = file_content[-1]
                        if "wandb: Find logs" in last_line: 
                            continue

                    command = f"qsub -v args='  --wandb_name {name}   --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate} --freeze {freeze}'  -o {records_dir}/{name}_{data_percentage} run_scripts/train_prithvi.sh"
                    os.system(command)
# # ###############################################################