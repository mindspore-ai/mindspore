/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "include/utils/callbacks.h"
#include <map>
#include <string>
#include <memory>
#include "pybind11/pybind11.h"
#include "include/utils/python_adapter.h"
#include "include/utils/tensor_py.h"

namespace mindspore {
namespace callbacks {
const char PYTHON_MOD_CALLBACK_MODULE[] = "mindspore.train.callback._callback";
const char PYTHON_FUN_PROCESS_CHECKPOINT[] = "checkpoint_cb_for_save_op";
const char PYTHON_FUN_PROCESS_SUMMARY[] = "summary_cb_for_save_op";

const int ONE_SHAPE = 1;

// Cache the summary callback data from ME session
// Remove the GE module on new architecture
// Output Format: [{"name": tag_name, "data": tensor}, {"name": tag_name, "data": tensor},...]
uint32_t SummarySaveCallback(uint32_t graph_id, const std::map<std::string, TensorPtr> &params_list) {
  // Acquire GIL before calling Python code
  py::gil_scoped_acquire acquire;
  py::list summary_list = py::list();

  MS_LOG(INFO) << "The Summary save callback function for graph " << graph_id
               << ", Param list size = " << params_list.size() << ".";
  for (auto &item : params_list) {
    std::string tag_name = item.first;
    auto tensor_ptr = item.second;
    if (tensor_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Summary tensor is null";
    }
    py::object tensorpyObject = PackTensorToPyObject(tensor_ptr);
    py::dict summary_value_dict;
    summary_value_dict["name"] = tag_name;
    summary_value_dict["data"] = tensorpyObject;
    summary_list.append(summary_value_dict);
  }

  py::bool_ ret = python_adapter::CallPyFn(PYTHON_MOD_CALLBACK_MODULE, PYTHON_FUN_PROCESS_SUMMARY, summary_list);
  auto bool_ret = py::cast<bool>(ret);
  if (!bool_ret) {
    MS_LOG(ERROR) << "Python checkpoint return false during callback";
    return kCallbackFalied;
  }
  MS_LOG(DEBUG) << "End the summary save callback function.";
  return kCallbackOk;
}
}  // namespace callbacks
}  // namespace mindspore
