# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""train and evaluate resnet50 network on imagenet dataset"""
import os
import shutil
import pytest


def get_env_info():
    print("================== CPU ======================")
    os.system("top -bi -n 2 -d 0.02")
    print("================= IO ====================")
    os.system("iostat")
    print("================= Memory =====================")
    os.system("free -h")
    print("================= Process ====================")
    os.system("ps -ef | grep python")
    print("================= NPU ====================")
    os.system("npu-smi info")


def resnet_end():
    acc = 0
    cost = 0
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    for i in range(4):
        with open(os.path.join(sh_path, f"train_parallel{i}", f"resnet_{i}.txt")) as f:
            lines = f.readlines()
            acc += float(lines[0].strip().split(": ")[1])
            cost += float(lines[1].strip().split(": ")[1])
    acc /= 4
    cost /= 4
    print(f"resnet acc: {acc}, cost: {cost}")
    assert acc > 0.1
    assert cost < 26
    for i in range(4):
        shutil.rmtree(os.path.join(sh_path, f"train_parallel{i}"))


def thor_end():
    thor_cost = 0
    thor_loss = 0
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    for i in range(4):
        with open(os.path.join(sh_path, f"train_parallel{i+4}", f"thor_{i}.txt")) as f:
            lines = f.readlines()
            thor_loss += float(lines[0].strip().split(": ")[1])
            thor_cost += float(lines[1].strip().split(": ")[1])
    thor_loss /= 4
    thor_cost /= 4
    print(f"resnet thor_loss: {thor_loss}, thor_cost: {thor_cost}")
    assert thor_loss < 7
    assert thor_cost < 30
    for i in range(4):
        shutil.rmtree(os.path.join(sh_path, f"train_parallel{i+4}"))


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_resnet_imagenet_and_thor_4p():
    """
    Feature: Resnet50 network.
    Description: Train and evaluate resnet50 network on imagenet dataset.
    Expectation: accuracy > 0.1, time cost < 26.
    """
    get_env_info()
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(f"sh {sh_path}/scripts/run_train.sh")
    assert ret == 0
    resnet_end()
    thor_end()
