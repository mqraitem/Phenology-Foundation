import os

records_dir = "records"

# for name in os.listdir(records_dir):
#     file_content = open(f"{records_dir}/{name}", "r", encoding='latin-1').readlines()
#     last_line = file_content[-1]
#     if "wandb: Find logs" in last_line: 
#         continue

#     print(name)

# quit()


# # ##############################################################
# # # * Prithvi Pretrained Blowup Final Improved No Shuffle
# # ###############################################################
# load_checkpoint = True
# for feed_timeloc in [False]:
#     for batch_size in [2]:
#         for data_percentage in [1.0]:
#             for use_config_normalization in [True]:
#                 for n_temporal_layers in [8]:
#                     for c_per_t, hidden_dim in [(8, 768)]:
#                         group_name = f"prithvi_pretrained_blowup_final3_improved_noshuffle"
#                         for temporal_attention in [False, True]:
#                             for separate_heads in [False, True]:
#                                 for learning_rate in [0.00001]:
                                    
#                                     name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}_feed_timeloc-{feed_timeloc}_cpert-{c_per_t}_hiddendim-{hidden_dim}_ntemporallayers-{n_temporal_layers}"
#                                     name += f"_separate_heads-{separate_heads}_temporal_attention-{temporal_attention}"
#                                     if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
#                                         file_content = open(f"{records_dir}/{name}", "r").readlines()
#                                         last_line = file_content[-1]
#                                         if "wandb: Find logs" in last_line: 
#                                             continue

#                                     command = f"qsub -v args=' --temporal_attention {temporal_attention} --separate_heads {separate_heads} --n_temporal_layers {n_temporal_layers} --hidden_dim {hidden_dim} --c_per_t {c_per_t} --wandb_name {name} --feed_timeloc {feed_timeloc} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi_blowup_improved_noshuffle.sh"
#                                     os.system(command)
# # # ###############################################################


# ###############################################################
# # * Prithvi Pretrained Blowup Final HLS
# ###############################################################
load_checkpoint = True
for feed_timeloc in [False]:
    for batch_size in [2]:
        for data_percentage in [1.0]:
            for use_config_normalization in [True]:
                for n_temporal_layers in [4]:
                    for c_per_t, hidden_dim in [(16, 768), (16, 512)]:
                        group_name = f"prithvi_pretrained_blowup_hls_final3"
                        for learning_rate in [0.00001]:
                            
                            name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}_feed_timeloc-{feed_timeloc}_cpert-{c_per_t}_hiddendim-{hidden_dim}_ntemporallayers-{n_temporal_layers}"
                            if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
                                file_content = open(f"{records_dir}/{name}", "r").readlines()
                                last_line = file_content[-1]
                                if "wandb: Find logs" in last_line: 
                                    continue

                            command = f"qsub -v args=' --n_temporal_layers {n_temporal_layers} --hidden_dim {hidden_dim} --c_per_t {c_per_t} --wandb_name {name} --feed_timeloc {feed_timeloc} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi_blowup_hls.sh"
                            os.system(command)
# # ###############################################################


# # ###############################################################
# # # * Prithvi Pretrained Blowup Final Improved
# # ###############################################################
# load_checkpoint = True
# for feed_timeloc in [False]:
#     for batch_size in [2]:
#         for data_percentage in [1.0]:
#             for use_config_normalization in [True]:
#                 for n_temporal_layers in [8]:
#                     for c_per_t, hidden_dim in [(8, 768)]:
#                         group_name = f"prithvi_pretrained_blowup_final3_improved"
#                         for temporal_attention in [False, True]:
#                             for separate_heads in [False, True]:
#                                 for learning_rate in [0.00001]:
                                    
#                                     name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}_feed_timeloc-{feed_timeloc}_cpert-{c_per_t}_hiddendim-{hidden_dim}_ntemporallayers-{n_temporal_layers}"
#                                     name += f"_separate_heads-{separate_heads}_temporal_attention-{temporal_attention}"
#                                     if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
#                                         file_content = open(f"{records_dir}/{name}", "r").readlines()
#                                         last_line = file_content[-1]
#                                         if "wandb: Find logs" in last_line: 
#                                             continue

#                                     command = f"qsub -v args=' --temporal_attention {temporal_attention} --separate_heads {separate_heads} --n_temporal_layers {n_temporal_layers} --hidden_dim {hidden_dim} --c_per_t {c_per_t} --wandb_name {name} --feed_timeloc {feed_timeloc} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi_blowup_improved.sh"
#                                     os.system(command)
# # # ###############################################################


# # ###############################################################
# # # * Shallow Transformer final
# # ###############################################################
# for learning_rate in [0.0001, 0.00001, 0.000001]:
#     for batch_size in [264, 512]:
#         for data_percentage in [1.0]:
                
#             group_name = f"shallow_transformer_pixels_mae"

#             name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}"
            
#             if os.path.exists(f"{records_dir}/{name}"):
#                 file_content = open(f"{records_dir}/{name}", "r").readlines()
#                 last_line = file_content[-1]
#                 if "wandb: Find logs" in last_line: 
#                     continue

#                 # if "Disk quota exceeded" not in last_line:
#                 #     continue
#                 print("haha: ", name)

#             command = f"qsub -v args='--loss mae --wandb_name {name}   --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name}  --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_lsp_pixels.sh"
#             os.system(command)
# # # ###############################################################

# # ###############################################################
# # # * Prithvi Pretrained Blowup Final
# # ###############################################################
# load_checkpoint = True
# for feed_timeloc in [False]:
#     for batch_size in [2]:
#         for data_percentage in [1.0]:
#             for use_config_normalization in [True]:
#                 for n_temporal_layers in [12, 16]:
#                     # for c_per_t, hidden_dim in [(8, 768), (16, 768), (8, 1024), (16, 1024)]:
#                     # for c_per_t, hidden_dim in [(8, 128), (8, 256), (4, 128)]:
#                     # for c_per_t, hidden_dim in [(8, 512), (4, 512)]:
#                     for c_per_t, hidden_dim in [(8, 768)]:
#                         group_name = f"prithvi_pretrained_blowup_final3"
#                         for learning_rate in [0.00001]:
                            
#                             name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}_feed_timeloc-{feed_timeloc}_cpert-{c_per_t}_hiddendim-{hidden_dim}_ntemporallayers-{n_temporal_layers}"
#                             if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
#                                 file_content = open(f"{records_dir}/{name}", "r").readlines()
#                                 last_line = file_content[-1]
#                                 if "wandb: Find logs" in last_line: 
#                                     continue

#                             command = f"qsub -v args=' --n_temporal_layers {n_temporal_layers} --hidden_dim {hidden_dim} --c_per_t {c_per_t} --wandb_name {name} --feed_timeloc {feed_timeloc} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi_blowup.sh"
#                             os.system(command)
# # # ###############################################################


# # ###############################################################
# # # * Prithvi Pretrained Simple Final
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



# # ###############################################################
# # # * Prithvi Pretrained Final mae
# # ###############################################################
# load_checkpoint = True
# for feed_timeloc in [False]:
#     for batch_size in [2]:
#         for data_percentage in [1.0]:
#             for use_config_normalization in [True]:
#                 group_name = f"prithvi_pretrained_mae_final"
#                 for learning_rate in [0.00001]:
                    
#                     name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}_feed_timeloc-{feed_timeloc}"
#                     if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
#                         file_content = open(f"{records_dir}/{name}", "r").readlines()
#                         last_line = file_content[-1]
#                         if "wandb: Find logs" in last_line: 
#                             continue

#                     command = f"qsub -v args='  --loss mae --wandb_name {name} --feed_timeloc {feed_timeloc} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi.sh"
#                     os.system(command)
# # # ###############################################################


# # ###############################################################
# # # * Prithvi Pretrained Final
# # ###############################################################
# load_checkpoint = True
# for feed_timeloc in [False]:
#     for batch_size in [2]:
#         for data_percentage in [0.05, 0.2, 0.4, 0.6, 0.8]:
#         # for data_percentage in [1.0]:
#             for use_config_normalization in [True]:
#                 group_name = f"prithvi_pretrained_final"
#                 # for learning_rate in [0.0001, 0.00001, 0.000001]:
#                 for learning_rate in [0.00001]:
                    
#                     name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}_feed_timeloc-{feed_timeloc}"
#                     if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
#                         file_content = open(f"{records_dir}/{name}", "r").readlines()
#                         last_line = file_content[-1]
#                         if "wandb: Find logs" in last_line: 
#                             continue

#                     command = f"qsub -v args='  --wandb_name {name} --feed_timeloc {feed_timeloc} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi.sh"
#                     os.system(command)
# # # ###############################################################


# # ###############################################################
# # # * Prithvi Random Final
# # ###############################################################
# load_checkpoint = False
# for batch_size in [2]:
#     for data_percentage in [1.0]:
#         group_name = f"prithvi_random_final"
        
#         for learning_rate in [0.0001, 0.00001, 0.000001]:
            
#             name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}"
#             if os.path.exists(f"{records_dir}/{name}"):
#                 file_content = open(f"{records_dir}/{name}", "r").readlines()
#                 last_line = file_content[-1]
#                 if "wandb: Find logs" in last_line: 
#                     continue

#             command = f"qsub -v args='  --wandb_name {name}  --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi.sh"
#             os.system(command)
# # ###############################################################

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





# # # * Prithvi Pretrained Upsample Final
# # ###############################################################
# load_checkpoint = True
# for feed_timeloc in [False]:
#     for batch_size in [2]:
#         for data_percentage in [1.0]:
#             for use_config_normalization in [True]:
#                 for conv_k in [1, 3]:
#                     group_name = f"prithvi_pretrained_upsample_final"
#                     for learning_rate in [0.00001]:
                        
#                         name = f"{group_name}_lr-{learning_rate}_batch_size-{batch_size}_confignorm-{use_config_normalization}_feed_timeloc-{feed_timeloc}_convk-{conv_k}"
#                         if os.path.exists(f"{records_dir}/{name}_{data_percentage}"):
#                             file_content = open(f"{records_dir}/{name}", "r").readlines()
#                             last_line = file_content[-1]
#                             if "wandb: Find logs" in last_line: 
#                                 continue

#                         command = f"qsub -v args=' --conv_k {conv_k} --wandb_name {name} --feed_timeloc {feed_timeloc} --data_percentage {data_percentage} --batch_size {batch_size} --group_name {group_name} --load_checkpoint {load_checkpoint} --logging True --learning_rate {learning_rate}'  -o {records_dir}/{name} run_scripts/train_prithvi_upsample.sh"
#                         os.system(command)
# # # ###############################################################
