# description

mindspore distributed training launch helper utilty that will generate hccl config file.

# use

```
python hccl_tools.py --device_num [1,8]
```

output:
```
hccl_[device_num]p_[which device]_[server_ip].json
```