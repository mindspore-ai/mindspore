/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "utils/callbacks_ge.h"
#include "pybind11/pybind11.h"
#include "transform/df_graph_manager.h"
#include "transform/util.h"
#include "pipeline/parse/data_converter.h"
#include "pipeline/parse/python_adapter.h"
#include "utils/visible.h"

namespace mindspore {
namespace callbacks {

const char PYTHON_MOD_CALLBACK_MODULE[] = "mindspore.train.callback";
const char PYTHON_FUN_PROCESS_CHECKPOINT[] = "_checkpoint_cb_for_save_op";
const char PYTHON_FUN_PROCESS_SUMMARY[] = "_summary_cb_for_save_op";
const char kSummary[] = "Summary";
const char kCheckPoint[] = "Save";
const int ONE_SHAPE = 1;

using mindspore::transform::Status;
using mindspore::transform::TransformUtil;

bool GetParameterShape(const FuncGraphPtr &graph, const std::string &param_name,
                       const std::shared_ptr<std::vector<int>> &shape) {
  if (graph == nullptr) {
    MS_LOG(ERROR) << "Graph is null, can not get graph parameter";
    return false;
  }

  auto parameter_nodes = graph->parameters();
  for (auto &node : parameter_nodes) {
    ParameterPtr param_node = std::static_pointer_cast<Parameter>(node);
    if (param_node == nullptr) {
      MS_LOG(ERROR) << "Parameter node is null, can not get graph parameter";
      return false;
    }
    if (param_node->name() == param_name) {
      py::object parameter = param_node->default_param();
      ValuePtr value = parse::data_converter::PyDataToValue(parameter);
      TensorPtr tensor = std::dynamic_pointer_cast<tensor::Tensor>(value);
      if (tensor == nullptr) {
        shape->push_back(ONE_SHAPE);
      } else {
        *shape = tensor->shape();
      }
      return true;
    }
  }
  MS_LOG(ERROR) << "Can not find parameter of name:" << param_name;
  return false;
}

static TensorPtr GetMeTensorTransformed(uint32_t graph_id, const std::string &parameter_name,
                                        const std::shared_ptr<ge::Tensor> &ge_tensor_ptr) {
  FuncGraphPtr anf_graph = transform::DfGraphManager::GetInstance().GetAnfGraph(graph_id);
  if (anf_graph == nullptr) {
    MS_LOG(ERROR) << "Get anf graph failed during callback";
    return nullptr;
  }

  std::shared_ptr<std::vector<int>> parameter_shape_ptr = std::make_shared<std::vector<int>>();
  if (!GetParameterShape(anf_graph, parameter_name, parameter_shape_ptr)) {
    MS_LOG(ERROR) << "Can not get parameter shape during callback";
    return nullptr;
  }

  return TransformUtil::ConvertGeTensor(ge_tensor_ptr, *parameter_shape_ptr);
}

uint32_t CheckpointSaveCallback(uint32_t graph_id, const std::map<std::string, ge::Tensor> &params_list) {
  // Acquire GIL before calling Python code
  py::gil_scoped_acquire acquire;

  MS_LOG(DEBUG) << "Start the checkpoint save callback function in checkpoint save process.";
  py::list parameter_list = py::list();
  for (auto &item : params_list) {
    std::string name = item.first;
    std::shared_ptr<ge::Tensor> ge_tensor_ptr = std::make_shared<ge::Tensor>(item.second);
    TensorPtr tensor_ptr = GetMeTensorTransformed(graph_id, name, ge_tensor_ptr);
    if (tensor_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Transform ge tensor to me tensor failed";
    }
    py::dict param_dict;
    param_dict["name"] = name;
    param_dict["data"] = tensor_ptr;
    parameter_list.append(param_dict);
  }
  py::bool_ ret =
    parse::python_adapter::CallPyFn(PYTHON_MOD_CALLBACK_MODULE, PYTHON_FUN_PROCESS_CHECKPOINT, parameter_list);
  auto bool_ret = py::cast<bool>(ret);

  uint32_t status = Status::SUCCESS;
  if (!bool_ret) {
    status = Status::FAILED;
    MS_LOG(ERROR) << "Python checkpoint return false during callback";
  }
  return status;
}

static TensorPtr GetMeTensorForSummary(const std::string &name, const std::shared_ptr<ge::Tensor> &ge_tensor_ptr) {
  // confirm the type by name
  // Format: xxx[:Scalar] xxx[:Image] xxx[:Tensor]
  if (name.empty()) {
    MS_LOG(EXCEPTION) << "The summary name is empty.";
  }
  auto bpos = name.rfind("[:");
  if (bpos >= name.size()) {
    MS_LOG(EXCEPTION) << "The summary name(" << name << ") is invalid.";
  }
  auto tname = name.substr(bpos);
  if (tname == "[:Scalar]") {
    MS_LOG(DEBUG) << "The summary(" << name << ") is Scalar";
    // process the scalar type summary
    // Because the ge tensor is dim = 4, so set the (1,1,1,1)-->(1,)
    // We do the (1,) shape is scalar
    auto shape = std::vector<int>({ONE_SHAPE});
    return TransformUtil::ConvertGeTensor(ge_tensor_ptr, shape);
  }
  if (tname == "[:Tensor]" || tname == "[:Histogram]") {
    MS_LOG(DEBUG) << "The summary(" << name << ") is Tensor";
    // process the tensor summary
    // Now we can't get the real shape, so we keep same shape with GE
    return TransformUtil::ConvertGeTensor(ge_tensor_ptr);
  }
  if (tname == "[:Image]") {
    MS_LOG(DEBUG) << "The summary(" << name << ") is Image";
    // process the Image summary
    // Image dim = 4, is same with ge, so we keep same shape with GE
    return TransformUtil::ConvertGeTensor(ge_tensor_ptr);
  }

  MS_LOG(EXCEPTION) << "The summary name(" << name << ") is invalid.";
}

// Cache the summary callback data
// Output Format: [{"name": tag_name, "data": tensor}, {"name": tag_name, "data": tensor},...]
uint32_t MS_EXPORT SummarySaveCallback(uint32_t graph_id, const std::map<std::string, ge::Tensor> &params_list) {
  // Acquire GIL before calling Python code
  py::gil_scoped_acquire acquire;

  MS_LOG(DEBUG) << "Start the summary save callback function for graph " << graph_id << ".";
  py::list summary_list = py::list();
  MS_LOG(DEBUG) << "Param list size = " << params_list.size();
  for (auto &item : params_list) {
    std::string tag_name = item.first;
    std::shared_ptr<ge::Tensor> ge_tensor_ptr = std::make_shared<ge::Tensor>(item.second);
    TensorPtr tensor_ptr = GetMeTensorForSummary(tag_name, ge_tensor_ptr);
    if (tensor_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "ConvertGeTensor return tensor is null";
    }
    py::dict summary_value_dict;
    summary_value_dict["name"] = tag_name;
    summary_value_dict["data"] = tensor_ptr;
    summary_list.append(summary_value_dict);
  }

  py::bool_ ret = parse::python_adapter::CallPyFn(PYTHON_MOD_CALLBACK_MODULE, PYTHON_FUN_PROCESS_SUMMARY, summary_list);
  auto bool_ret = py::cast<bool>(ret);
  if (!bool_ret) {
    MS_LOG(ERROR) << "Python checkpoint return false during callback";
    return Status::FAILED;
  }
  MS_LOG(DEBUG) << "End the summary save callback function.";
  return Status::SUCCESS;
}
}  // namespace callbacks
}  // namespace mindspore
