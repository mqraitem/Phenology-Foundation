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
for feed_timeloc in [False]:
    for batch_size in [2]:
        for data_percentage in [1.0]:
            for use_config_normalization in [True, False]:
                group_name = f"prithvi_pretrained_conv3d"
                for learning_rate in [0.00001]:
                    # for n_layers, hidden_dim in [[4, 768]]:
                    for n_layers, hidden_dim in [[1, 768], [1, 128], [4, 128]]:
                        
                        name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}_feed_timeloc-{feed_timeloc}_loss-mae_n_layers-{n_layers}_hidden_dim-{hidden_dim}"
                        if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
                            file_content = open(f"{records_dir}/{name}", "r").readlines()
                            last_line = file_content[-1]
                            if "wandb: Find logs" in last_line: 
                                continue

                        command = f"qsub -v args=' --hidden_dim {hidden_dim} --n_layers {n_layers} --loss mae --wandb_name {name} --feed_timeloc {feed_timeloc} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi_conv3d.sh"
                        os.system(command)
# # ###############################################################

# # ###############################################################
# # # * Shallow Transformer
# # ###############################################################
# for learning_rate in [0.00001]:
#     for batch_size in [512]:
#         for data_percentage in [1.0]:
                
#             group_name = f"shallow_transformer_pixels"

#             name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_loss-mae"
            
#             if os.path.exists(f"{records_dir}/{name}"):
#                 file_content = open(f"{records_dir}/{name}", "r").readlines()
#                 last_line = file_content[-1]
#                 if "wandb: Find logs" in last_line: 
#                     continue


#             command = f"qsub -v args='--loss mae --wandb_name {name}   --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name}  --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_lsp_pixels.sh"
#             os.system(command)
# # # ###############################################################

# # ###############################################################
# # # * Prithvi Pretrained Upshuffle
# # ###############################################################
# load_checkpoint = True
# for feed_timeloc in [False]:
#     for batch_size in [2]:
#         for data_percentage in [1.0]:
#             for use_config_normalization in [True, False]:
#                 for n_temporal_layers in [4]:
#                     for c_per_t, hidden_dim in [(8, 768)]:
#                         group_name = f"prithvi_pretrained_upshuffle"
#                         for learning_rate in [0.00001]:
                            
#                             name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}_feed_timeloc-{feed_timeloc}_cpert-{c_per_t}_hiddendim-{hidden_dim}_ntemporallayers-{n_temporal_layers}_loss-mae"
#                             if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
#                                 file_content = open(f"{records_dir}/{name}", "r").readlines()
#                                 last_line = file_content[-1]
#                                 if "wandb: Find logs" in last_line: 
#                                     continue

#                             command = f"qsub -v args=' --loss mae --n_temporal_layers {n_temporal_layers} --hidden_dim {hidden_dim} --c_per_t {c_per_t} --wandb_name {name} --feed_timeloc {feed_timeloc} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi_upshuffle.sh"
#                             os.system(command)
# # # ###############################################################


# # # * Prithvi Pretrained 
# # ###############################################################
# load_checkpoint = True
# for feed_timeloc in [False]:
#     for batch_size in [2]:
#         for data_percentage in [1.0]:
#             for use_config_normalization in [True, False]:
#                 group_name = f"prithvi_pretrained"
#                 for learning_rate in [0.00001]:
                    
#                     name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}_feed_timeloc-{feed_timeloc}_loss-mae"
#                     if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
#                         file_content = open(f"{records_dir}/{name}", "r").readlines()
#                         last_line = file_content[-1]
#                         if "wandb: Find logs" in last_line: 
#                             continue

#                     command = f"qsub -v args='  --loss mae --wandb_name {name} --feed_timeloc {feed_timeloc} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi.sh"
#                     os.system(command)
# # # ###############################################################




# # ###############################################################
# # # * Prithvi Random Conv3d 
# # ###############################################################
# load_checkpoint = True
# for feed_timeloc in [False]:
#     for batch_size in [2]:
#         for data_percentage in [1.0]:
#             for use_config_normalization in [False]:
#                 group_name = f"prithvi_random_conv3d"
#                 for learning_rate in [0.00001]:
#                     for n_layers, hidden_dim in [[4, 768]]:
                        
#                         name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}_feed_timeloc-{feed_timeloc}_loss-mae_n_layers-{n_layers}_hidden_dim-{hidden_dim}"
#                         if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
#                             file_content = open(f"{records_dir}/{name}", "r").readlines()
#                             last_line = file_content[-1]
#                             if "wandb: Find logs" in last_line: 
#                                 continue

#                         command = f"qsub -v args=' --hidden_dim {hidden_dim} --n_layers {n_layers} --loss mae --wandb_name {name} --feed_timeloc {feed_timeloc} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi_conv3d.sh"
#                         os.system(command)
# # # ###############################################################


# # ###############################################################
# # # * Prithvi Random Upshuffle
# # ###############################################################
# load_checkpoint = False
# for feed_timeloc in [False]:
#     for batch_size in [2]:
#         for data_percentage in [1.0]:
#             for use_config_normalization in [False]:
#                 for n_temporal_layers in [4]:
#                     for c_per_t, hidden_dim in [(8, 768)]:
#                         group_name = f"prithvi_random_upshuffle"
#                         for learning_rate in [0.00001]:
                            
#                             name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}_feed_timeloc-{feed_timeloc}_cpert-{c_per_t}_hiddendim-{hidden_dim}_ntemporallayers-{n_temporal_layers}_loss-mae"
#                             if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
#                                 file_content = open(f"{records_dir}/{name}", "r").readlines()
#                                 last_line = file_content[-1]
#                                 if "wandb: Find logs" in last_line: 
#                                     continue

#                             command = f"qsub -v args=' --loss mae --n_temporal_layers {n_temporal_layers} --hidden_dim {hidden_dim} --c_per_t {c_per_t} --wandb_name {name} --feed_timeloc {feed_timeloc} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi_upshuffle.sh"
#                             os.system(command)

# # ###############################################################


# # ###############################################################
# # # * Prithvi Random 
# # ###############################################################
# load_checkpoint = False
# for feed_timeloc in [False]:
#     for batch_size in [2]:
#         for data_percentage in [1.0]:
#             for use_config_normalization in [False]:
#                 group_name = f"prithvi_random"
#                 for learning_rate in [0.00001]:
                    
#                     name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}_feed_timeloc-{feed_timeloc}_loss-mae"
#                     if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
#                         file_content = open(f"{records_dir}/{name}", "r").readlines()
#                         last_line = file_content[-1]
#                         if "wandb: Find logs" in last_line: 
#                             continue

#                     command = f"qsub -v args='  --loss mae --wandb_name {name} --feed_timeloc {feed_timeloc} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi.sh"
#                     os.system(command)
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



# # ###############################################################
# # # * Prithvi Pretrained Simple
# # ###############################################################
# load_checkpoint = True
# for feed_timeloc in [False]:
#     for batch_size in [2]:
#         for data_percentage in [1.0]:
#             for use_config_normalization in [True]:
#                 for proj_dim in ["512", "768", "1600", "3200", "6400"]:
#                     group_name = f"prithvi_pretrained_simple_final"
#                     for learning_rate in [0.00001]:
                        
#                         name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}_feed_timeloc-{feed_timeloc}_projdim-{proj_dim}"
#                         if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
#                             file_content = open(f"{records_dir}/{name}", "r").readlines()
#                             last_line = file_content[-1]
#                             if "wandb: Find logs" in last_line: 
#                                 continue

#                         command = f"qsub -v args=' --proj_dim {proj_dim} --wandb_name {name} --feed_timeloc {feed_timeloc} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi_simple.sh"
#                         os.system(command)
# # # ###############################################################