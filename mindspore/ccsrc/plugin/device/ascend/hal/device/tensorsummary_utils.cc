/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "plugin/device/ascend/hal/device/tensorsummary_utils.h"
#include <string>
#include <map>
#include <utility>
#include "pybind11/pybind11.h"
#include "ir/tensor.h"
#include "ir/dtype/type.h"
#include "plugin/device/ascend/hal/device/ascend_data_queue.h"
#include "include/common/utils/python_adapter.h"
#include "include/transform/graph_ir/types.h"
#include "plugin/device/ascend/hal/device/mbuf_receive_manager.h"

namespace py = pybind11;

namespace mindspore::device::ascend {

namespace {
const char PYTHON_MOD_CALLBACK_MODULE[] = "mindspore.train.callback._callback";
const char PYTHON_FUN_PROCESS_SUMMARY[] = "summary_cb_for_save_op";
const std::map<string, string> channel_name_suffix = {{"ms_tensor_summary", "[:Tensor]"},
                                                      {"ms_image_summary", "[:Image]"},
                                                      {"ms_scalar_summary", "[:Scalar]"},
                                                      {"ms_histogram_summary", "[:Histogram]"}};
}  // namespace

void SummaryReceiveData(acltdtDataset *acl_dataset, const string &channel_name) {
  //  Acquire Python GIL
  py::gil_scoped_acquire gil_acquire;
  std::string tensor_name = std::string{acltdtGetDatasetName(acl_dataset)};
  auto suffix = channel_name_suffix.find(channel_name)->second;
  std::string summary_name = tensor_name + suffix;
  MS_LOG(INFO) << "For " << channel_name << "channel, acltdt received Tensor name is " << tensor_name;
  size_t acl_dataset_size = acltdtGetDatasetSize(acl_dataset);
  for (size_t i = 0; i < acl_dataset_size; i++) {
    acltdtDataItem *item = acltdtGetDataItem(acl_dataset, i);
    MS_EXCEPTION_IF_NULL(item);
    if (acltdtGetTensorTypeFromItem(item) == ACL_TENSOR_DATA_END_OF_SEQUENCE) {
      MS_LOG(INFO) << "end of sequence" << std::endl;
      break;
    }
    auto tensor_ptr = acltdtDataItemToTensorPtr(item);
    if (tensor_ptr == nullptr) {
      MS_LOG(ERROR) << "Invalid tensor data item.";
      continue;
    }
    py::list summary_list = py::list();
    py::dict summary_value_dict;
    summary_value_dict["name"] = summary_name;
    summary_value_dict["data"] = tensor_ptr;
    summary_list.append(summary_value_dict);
    py::bool_ ret = python_adapter::CallPyFn(PYTHON_MOD_CALLBACK_MODULE, PYTHON_FUN_PROCESS_SUMMARY, summary_list);

    auto bool_ret = py::cast<bool>(ret);
    if (!bool_ret) {
      MS_LOG(ERROR) << "Python checkpoint return false during callback";
    }
  }
}

}  // namespace mindspore::device::ascend
