# Run distribute train

## description

The number of Ascend accelerators can be automatically allocated based on the device_num set in hccl config file, You don not need to specify that.

## how to use

For example, if we want to generate the launch command of the distributed training of CenterNet model on Ascend accelerators, we can run the following command in `/centernet/` dir:

```python
python ./scripts/ascend_distributed_launcher/get_distribute_pretrain_cmd.py --run_script_dir ./train.py --hyper_parameter_config_dir ./scripts/ascend_distributed_launcher/hyper_parameter_config.ini --data_dir /path/dataset/ --mindrecord_dir /path/mindrecord_dataset/ --hccl_config_dir model_zoo/utils/hccl_tools/hccl_2p_56_x.x.x.x.json
```

output:

```text
hccl_config_dir: model_zoo/utils/hccl_tools/hccl_2p_56_x.x.x.x.json
the number of logical core: 192
avg_core_per_rank: 96
rank_size: 2

start training for rank 0, device 5:
rank_id: 0
device_id: 5
core nums: 0-95
epoch_size: 350
data_dir: /path/dataset/
mindrecord_dir: /path/mindrecord_dataset/
log file dir: ./LOG5/training_log.txt

start training for rank 1, device 6:
rank_id: 1
device_id: 6
core nums: 96-191
epoch_size: 350
data_dir: /path/dataset/
mindrecord_dir: /path/mindrecord_dataset/
log file dir: ./LOG6/training_log.txt
```

## Note

1. Note that `hccl_2p_56_x.x.x.x.json` can use [hccl_tools.py](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools) to generate.

2. For hyper parameter, please note that you should customize the scripts `hyper_parameter_config.ini`. Please note that these two hyper parameters are not allowed to be configured here:
    - device_id
    - device_num
    - data_dir

3. For Other Model, please note that you should customize the option `run_script` and Corresponding `hyper_parameter_config.ini`.
