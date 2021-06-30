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

`--visible_devices` means the visible devices according to the software system. Usually used in the virtual system or docker container that makes the device_id dismatch logic_id. `--device_num` uses logic_id. For example "4,5,6,7" means the system has 4 logic chips which are actually the last 4 chips in hardware while `--device_num` could only be set to "[0, 4)" instead of "[4, 8)"

`hccl_tools` used `/etc/hccn.conf` to generate rank_table_file. `/etc/hccn.conf` is the configuration file about ascend accelerator resources. If you are using an entirely new server without setting up NIC ip for device, you could refer to this [Chinese guide](https://support.huaweicloud.com/instg-9000-A800_9000_9010/atlastrain_03_0049.html) or this [English guide](https://support.huaweicloud.com/intl/en-us/instg-cli-cann202/atlasrun_03_0051.html) to generate `hccn.conf`.
