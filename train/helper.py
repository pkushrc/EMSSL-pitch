import os
listdir = os.listdir(r'/home/linan/EMSSL-master/train/log_dir/a2t_model_experiment_name_20240626-01:43:55/iter_train/iter_1/train')
sorted(listdir)
for name in listdir:
    print(name)