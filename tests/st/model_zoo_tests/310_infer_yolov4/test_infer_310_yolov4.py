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

import os
import pytest

from tests.st.model_zoo_tests import utils


def check_log_file(infer_ret_path):
    """
    Feature：check the log file and get the 310infer results.
    Description: infer_ret_path -- the path of 310 inference.
    Expectation: accuracy_data, fps_data.
    """

    build_log = "{0}/../ascend310_infer/src/build.log".format(infer_ret_path)
    infer_log = os.path.join(infer_ret_path, "infer.log")
    acc_log = os.path.join(infer_ret_path, "acc.log")
    log_file_list = [build_log, infer_log, acc_log]
    check_type = True
    for log_file in log_file_list:
        if not os.path.exists(log_file):
            print(f"{log_file} is not exists.")
            check_type = False
        os.system(r""" grep "ERROR" {} > log_error.txt""".format(log_file))
        with open("./log_error.txt", "r") as f:
            check_err_info = f.read()
            if check_err_info != '':
                print(check_err_info)
                check_type = False
        os.remove("./log_error.txt")
    assert check_type, "Exist 'ERROR' in log files."

    # verify accuracy and performance
    get_accuracy_shell = "cd %s; grep 'Average Precision' acc.log | grep 'IoU=0.50:0.95' \
                          | grep 'area=   all' | awk -F ' ] = ' '{print $2}' >acc_data.txt" % infer_ret_path
    get_perf_shell = "cd %s; grep 'cost average time' infer.log | awk -F : '{ print $2 }'\
                          | awk -F 'ms' '{ print $1 }' > pref_data.txt" % infer_ret_path
    os.system(get_accuracy_shell)
    os.system(get_perf_shell)
    accuracy_data, fps_data = 0, 0
    with open(f"{infer_ret_path}/acc_data.txt", "r") as acc:
        accuracy_data = float(acc.read())
    with open(f"{infer_ret_path}/pref_data.txt", "r") as pref:
        perf_data = float(pref.read())
        fps_data = 4 * 1000 / perf_data
    os.remove(f"{infer_ret_path}/acc_data.txt")
    os.remove(f"{infer_ret_path}/pref_data.txt")

    return accuracy_data, fps_data


@pytest.mark.level0
@pytest.mark.platform_x86_ascend310_inference
@pytest.mark.platform_arm_ascend310_inference
@pytest.mark.env_onecard
def test_infer_310_yolov4():
    """
    Feature：Verify the yolov4 310 infer process.
    Description: export DATASET_PATH(option), CKPT_FILE(option), DEVICE_ID(option).
    Expectation: success or accuracy/fps less than standard.
    """

    dataset_path = os.path.abspath(os.getenv('DATASET_PATH', "/home/workspace/mindspore_dataset"))
    ckpt_file = os.path.abspath(os.getenv('CKPT_FILE',\
        "/home/workspace/mindspore_ckpt/ckpt/yolov4_ascend_v130_coco2017_official_cv_bs8_acc44.ckpt"))
    device_id = int(os.getenv('DEVICE_ID', '0'))
    print("Specifying path by setting environment variables 'DEVICE_ID', 'DATASET_PATH' and 'CKPT_FILE'.")
    assert os.path.isfile(ckpt_file), f"Ckpt_File:{ckpt_file} is not exist."
    assert os.path.isdir(dataset_path), f"Dataset_Path:{dataset_path} is not exist."

    # export to MindIR
    cur_path = os.path.dirname(os.path.abspath(__file__))
    model_path = "{}/../../../../tests/models/official/cv".format(cur_path)
    model_name = "yolov4"
    assert os.path.isdir(os.path.join(model_path, model_name)), "models dir is not exist."
    utils.copy_files(model_path, cur_path, model_name)
    cur_model_path = os.path.join(cur_path, model_name)
    exec_export_shell = "cd {0}; python export.py --ckpt_file {1} --file_name yolov4 --file_format MINDIR \
                         --batch_size 1 --device_id {2}".format(model_name, ckpt_file, device_id)
    os.system(exec_export_shell)
    assert os.path.exists(os.path.join(cur_model_path, "yolov4.mindir")), "MINDIR_File is not exist."

    # do 310 infer
    old_list, new_list = ["&> acc.log &"], ["> acc.log"]
    infer_ret_path = os.path.join(cur_model_path, "scripts")
    utils.exec_sed_command(old_list, new_list, os.path.join(infer_ret_path, "run_infer_310.sh"))
    exec_infer_shell = "cd {0}; bash run_infer_310.sh ../yolov4.mindir {1} {2} {3}".format(infer_ret_path, \
        os.path.join(dataset_path, "coco", "coco2017_500", "val2017_500"), device_id, \
        os.path.join(dataset_path, "coco", "coco2017_500", "annotations", "instances_val2017_500.json"))
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib:' + os.environ['LD_LIBRARY_PATH']
    print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')}")
    os.system(exec_infer_shell)

    accuracy_data, fps_data = check_log_file(infer_ret_path)
    standard_value = {'acc': 0.470, 'fps': 55}
    print(f"accuracy:{accuracy_data}, fps:{fps_data}")
    assert accuracy_data > standard_value.get('acc')
    assert fps_data > standard_value.get('fps')
