# Copyright 2021 Huawei Technologies Co., Ltd
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

'''
Inference script of ONNX exported from the Bert classification model.
'''

import os
from src.dataset import create_classification_dataset
from src.assessment_method import Accuracy, F1, MCC, Spearman_Correlation
from src.model_utils.config import config as args_opt
from mindspore import Tensor, dtype
import onnxruntime as rt


def eval_result_print(assessment_method="accuracy", callback=None):
    """ print eval result """
    if assessment_method == "accuracy":
        print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                                  callback.acc_num / callback.total_num))
    elif assessment_method == "f1":
        print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
        print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
        print("F1 {:.6f} ".format(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN)))
    elif assessment_method == "mcc":
        print("MCC {:.6f} ".format(callback.cal()))
    elif assessment_method == "spearman_correlation":
        print("Spearman Correlation is {:.6f} ".format(callback.cal()[0]))
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")


def do_eval_onnx(dataset=None, num_class=15, assessment_method="accuracy"):
    """ do eval for onnx model"""
    if assessment_method == "accuracy":
        callback = Accuracy()
    elif assessment_method == "f1":
        callback = F1(False, num_class)
    elif assessment_method == "mcc":
        callback = MCC()
    elif assessment_method == "spearman_correlation":
        callback = Spearman_Correlation()
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")

    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    onnx_file_name = args_opt.export_file_name
    if not args_opt.export_file_name.endswith('.onnx'):
        onnx_file_name = onnx_file_name + '.onnx'
    if not os.path.isabs(onnx_file_name):
        onnx_file_name = os.getcwd() + '/' + onnx_file_name
    if not os.path.exists(onnx_file_name):
        raise ValueError("ONNX file not exists, please check onnx file has been saved and whether the "
                         "export_file_name is correct.")
    sess = rt.InferenceSession(onnx_file_name)
    input_name_0 = sess.get_inputs()[0].name
    input_name_1 = sess.get_inputs()[1].name
    input_name_2 = sess.get_inputs()[2].name
    output_name_0 = sess.get_outputs()[0].name

    for data in dataset.create_dict_iterator(num_epochs=1):
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data

        x0 = input_ids.asnumpy()
        x1 = input_mask.asnumpy()
        x2 = token_type_id.asnumpy()

        result = sess.run([output_name_0], {input_name_0: x0, input_name_1: x1, input_name_2: x2})
        logits = Tensor(result[0], dtype.float32)
        callback.update(logits, label_ids)

    print("==============================================================")
    eval_result_print(assessment_method, callback)
    print("==============================================================")


def run_classifier_onnx():
    """run classifier task for onnx model"""
    if args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do onnx evaluation task")
    assessment_method = args_opt.assessment_method.lower()
    ds = create_classification_dataset(batch_size=args_opt.eval_batch_size, repeat_count=1,
                                       assessment_method=assessment_method,
                                       data_file_path=args_opt.eval_data_file_path,
                                       schema_file_path=args_opt.schema_file_path,
                                       do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))
    do_eval_onnx(ds, args_opt.num_class, assessment_method)


if __name__ == "__main__":
    run_classifier_onnx()
