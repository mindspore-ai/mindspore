#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" File Description
Details
"""

import os
import shutil
import subprocess
import time
import re
from mindspore import log as logger

rank_table_path = "/home/workspace/mindspore_config/hccl/rank_table_8p.json"
data_root = "/home/workspace/mindspore_dataset/"
ckpt_root = "/home/workspace/mindspore_dataset/checkpoint"
cur_path = os.path.split(os.path.realpath(__file__))[0]
geir_root = os.path.join(cur_path, "mindspore_geir")
arm_main_path = os.path.join(cur_path, "mindir_310infer_exe")
model_zoo_path = os.path.join(cur_path, "../../../tests/models")


def copy_files(from_, to_, model_name):
    if not os.path.exists(os.path.join(from_, model_name)):
        raise ValueError("There is no file or path", os.path.join(from_, model_name))
    if os.path.exists(os.path.join(to_, model_name)):
        shutil.rmtree(os.path.join(to_, model_name))
    return os.system("cp -r {0} {1}".format(os.path.join(from_, model_name), to_))


def exec_sed_command(old_list, new_list, file):
    if isinstance(old_list, str):
        old_list = [old_list]
    if isinstance(new_list, str):
        old_list = [new_list]
    if len(old_list) != len(new_list):
        raise ValueError("len(old_list) should be equal to len(new_list)")
    for old, new in zip(old_list, new_list):
        ret = os.system('sed -i "s#{0}#{1}#g" {2}'.format(old, new, file))
        if ret != 0:
            raise ValueError('exec `sed -i "s#{0}#{1}#g" {2}` failed.'.format(old, new, file))
    return ret


def process_check(cycle_time, cmd, wait_time=5):
    for i in range(cycle_time):
        time.sleep(wait_time)
        sub = subprocess.Popen(args="{}".format(cmd), shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)
        stdout_data, _ = sub.communicate()
        if not stdout_data:
            logger.info("process execute success.")
            return True
        logger.warning("process is running, please wait {}".format(i))
    logger.error("process execute execute timeout.")
    return False


def get_perf_data(log_path, search_str="per step time", cmd=None):
    if cmd is None:
        get_step_times_cmd = r"""grep -a "{0}" {1}|egrep -v "loss|\]|\["|awk '{{print $(NF-1)}}'""" \
            .format(search_str, log_path)
    else:
        get_step_times_cmd = cmd
    sub = subprocess.Popen(args="{}".format(get_step_times_cmd), shell=True,
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True)
    stdout, _ = sub.communicate()
    if sub.returncode != 0:
        raise RuntimeError("exec {} failed".format(cmd))
    logger.info("execute {} success".format(cmd))
    stdout = stdout.strip().split("\n")
    step_time_list = list(map(float, stdout[1:]))
    if not step_time_list:
        cmd = "cat {}".format(log_path)
        os.system(cmd)
        raise RuntimeError("step_time_list is empty")
    per_step_time = sum(step_time_list) / len(step_time_list)
    return per_step_time


def get_loss_data_list(log_path, search_str="loss is", cmd=None):
    if cmd is None:
        loss_value_cmd = """ grep -a '{}' {}| awk '{{print $NF}}' """.format(search_str, log_path)
    else:
        loss_value_cmd = cmd
    sub = subprocess.Popen(args="{}".format(loss_value_cmd), shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True)
    stdout, _ = sub.communicate()
    if sub.returncode != 0:
        raise RuntimeError("get loss from {} failed".format(log_path))
    logger.info("execute {} success".format(cmd))
    stdout = stdout.strip().split("\n")
    loss_list = list(map(float, stdout))
    if not loss_list:
        cmd = "cat {}".format(log_path)
        os.system(cmd)
        raise RuntimeError("loss_list is empty")
    return loss_list


def parse_log_file(pattern, log_path):
    value_list = []
    with open(log_path, "r") as file:
        for line in file.readlines():
            match_result = re.search(pattern, line)
            if match_result is not None:
                value_list.append(float(match_result.group(1)))
    if not value_list:
        print("pattern is", pattern)
        cmd = "cat {}".format(log_path)
        os.system(cmd)
    return value_list
