# Description

MindSpore distributed training launch helper utility that will generate hccl config file.

## Usage

```python
python hccl_tools.py --device_num "[0,8)"
```

output:

```python
hccl_[device_num]p_[which device]_[server_ip].json
```

## Note

Please note that the Ascend accelerators used must be continuous, such [0,4) means to use four chips 0，1，2，3; [0,1) means to use chip 0; The first four chips are a group, and the last four chips are a group. In addition to the [0,8) chips are allowed, other cross-group such as [3,6) are prohibited.
