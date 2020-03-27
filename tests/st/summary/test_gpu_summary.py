# Copyright 2019 Huawei Technologies Co., Ltd
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
import pytest
import os
import time
import shutil
import random
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.train.summary.summary_record import SummaryRecord

'''
    This testcase is used for save summary data only. You need install MindData first and uncomment the commented 
    packages to analyse summary data.
    Using "minddata start --datalog='./test_me_summary_event_file/' --host=0.0.0.0" to make data visible. 
'''
# from minddata.datavisual.data_transform.data_manager import DataManager
# from minddata.datavisual.visual.train_visual.train_task_manager import TrainTaskManager
# from minddata.datavisual.visual.train_visual.scalars_processor import ScalarsProcessor
# from minddata.datavisual.common.enums import PluginNameEnum
# from minddata.datavisual.common.enums import DataManagerStatus


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


CUR_DIR = os.getcwd()
SUMMARY_DIR_ME = CUR_DIR + "/test_me_summary_event_file/"
SUMMARY_DIR_ME_TEMP = CUR_DIR + "/test_me_temp_summary_event_file/"

def clean_environment_file(srcDir):
    if os.path.exists(srcDir):
        ls = os.listdir(srcDir)
        for line in ls:
            filePath = os.path.join(srcDir, line)
            os.remove(filePath)
        os.removedirs(srcDir)
def save_summary_events_file(srcDir, desDir):
    if not os.path.exists(desDir):
        print("-- create desDir")
        os.makedirs(desDir)

    ls = os.listdir(srcDir)
    for line in ls:
        filePath = os.path.join(srcDir, line)
        if os.path.isfile(filePath):
            print("-- move events file : {}".format(filePath))
            shutil.copy(filePath, desDir)
        os.remove(filePath)
    os.removedirs(srcDir)

class SummaryNet(nn.Cell):
    def __init__(self, tag_tuple=None, scalar=1):
        super(SummaryNet, self).__init__()
        self.summary_s = P.ScalarSummary()
        self.summary_i = P.ImageSummary()
        self.summary_t = P.TensorSummary()
        self.add = P.TensorAdd()
        self.tag_tuple = tag_tuple
        self.scalar = scalar

    def construct(self, x, y):
        self.summary_i("image", x)
        self.summary_s("x1", x)
        z = self.add(x, y)
        self.summary_t("z1", z)
        return z

def train_summary_record_scalar_for_1(test_writer, steps, fwd_x, fwd_y):
    net = SummaryNet()
    out_me_dict = {}
    for i in range(0, steps):
        x = Tensor(np.array([1.1 + random.uniform(1, 10)]).astype(np.float32))
        y = Tensor(np.array([1.2 + random.uniform(1, 10)]).astype(np.float32))
        out_put = net(x, y)
        test_writer.record(i)
        print("-----------------output: %s-------------\n", out_put.asnumpy())
        out_me_dict[i] = out_put.asnumpy()
    return out_me_dict

def me_scalar_summary(steps, tag=None, value=None):
    test_writer = SummaryRecord(SUMMARY_DIR_ME_TEMP)

    x = Tensor(np.array([1.1]).astype(np.float32))
    y = Tensor(np.array([1.2]).astype(np.float32))

    out_me_dict = train_summary_record_scalar_for_1(test_writer, steps, x, y)

    test_writer.close()
    return out_me_dict

def print_scalar_data():
    print("============start print_scalar_data\n")
    data_manager = DataManager()
    data_manager.start_load_data(path=SUMMARY_DIR_ME)
    while data_manager.get_status() != DataManagerStatus.DONE:
        time.sleep(0.1)
    task_manager = TrainTaskManager(data_manager)
    train_jobs = task_manager.get_all_train_tasks(PluginNameEnum.scalar)
    print(train_jobs)
    """
        train_jobs
        ['train_jobs': {
	    'id': '12-123',
	    'name': 'train_job_name',
	    'tags': ['x1', 'y1']
	                    }]
    """
    scalar_processor = ScalarsProcessor(data_manager)
    metadata = scalar_processor.get_metadata_list(train_job_ids=train_jobs['train_jobs'][0]['id'], tag=train_jobs['train_jobs'][0]['tags'][0])
    print(metadata)
    '''
    metadata
    {
        'scalars' : [
            {
                'train_job_id' : '12-12',
                'metadatas' : [
                    {
                        'wall_time' : 0.1,
                        'step' : 1,
                        'value' : 0.1
                    }
                ]
            }
        ]
    }
    '''
    print("============end print_scalar_data\n")

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scalarsummary_scalar1_step10_summaryrecord1():
    clean_environment_file(SUMMARY_DIR_ME_TEMP)
    output_dict = me_scalar_summary(10)
    print("test_scalarsummary_scalar1_step10_summaryrecord1 \n",output_dict)
    save_summary_events_file(SUMMARY_DIR_ME_TEMP, SUMMARY_DIR_ME)
    clean_environment_file(SUMMARY_DIR_ME)
