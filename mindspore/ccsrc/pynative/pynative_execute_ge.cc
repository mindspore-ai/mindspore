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

#include "pynative/pynative_execute_ge.h"

#include <typeinfo>
#include <map>
#include <set>
#include <unordered_set>

#include "utils/any.h"
#include "utils/utils.h"
#include "utils/context/ms_context.h"
#include "operator/ops.h"
#include "pipeline/parse/data_converter.h"
#include "pipeline/static_analysis/prim.h"
#include "session/session_factory.h"
#include "ir/tensor_py.h"

const char SINGLE_OP_GRAPH[] = "single_op_graph";

using mindspore::tensor::TensorPy;

namespace mindspore {
namespace pynative {
using MeTensor = mindspore::tensor::Tensor;
using MeTensorPtr = mindspore::tensor::TensorPtr;
using GeOperator = ge::Operator;
using GeOperatorPtr = std::shared_ptr<GeOperator>;

using transform::GraphRunner;
using transform::GraphRunnerOptions;
using transform::OperatorPtr;
static std::shared_ptr<session::SessionBasic> session = nullptr;
inline ValuePtr PyAttrValue(const py::object &obj) {
  ValuePtr converted_ret = nullptr;
  bool converted = parse::ConvertData(obj, &converted_ret);
  if (!converted) {
    MS_LOG(EXCEPTION) << "Attribute convert error with type:" << std::string(py::str(obj));
  }
  return converted_ret;
}

MeTensorPtr ConvertPyObjToTensor(const py::object &obj) {
  MeTensorPtr me_tensor_ptr = nullptr;
  if (py::isinstance<MeTensor>(obj)) {
    me_tensor_ptr = py::cast<MeTensorPtr>(obj);
  } else if (py::isinstance<py::tuple>(obj)) {
    me_tensor_ptr = TensorPy::MakeTensor(py::array(py::cast<py::tuple>(obj)), nullptr);
  } else if (py::isinstance<py::float_>(obj)) {
    me_tensor_ptr = TensorPy::MakeTensor(py::array(py::cast<py::float_>(obj)), nullptr);
  } else if (py::isinstance<py::int_>(obj)) {
    me_tensor_ptr = TensorPy::MakeTensor(py::array(py::cast<py::int_>(obj)), nullptr);
  } else if (py::isinstance<py::list>(obj)) {
    me_tensor_ptr = TensorPy::MakeTensor(py::array(py::cast<py::list>(obj)), nullptr);
  } else if (py::isinstance<py::array>(obj)) {
    me_tensor_ptr = TensorPy::MakeTensor(py::cast<py::array>(obj), nullptr);
  } else {
    MS_LOG(EXCEPTION) << "Run op inputs type is invalid!";
  }
  return me_tensor_ptr;
}

bool SetInputsForSingleOpGraph(const OpExecInfoPtr &op_exec_info, const std::vector<GeTensorPtr> &inputs,
                               const OperatorPtr &op, std::vector<GeOperator> *graph_input_nodes) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(graph_input_nodes);
  auto op_inputs = op_exec_info->op_inputs;
  std::string op_name = op_exec_info->op_name;
  transform::OpAdapterPtr adapter = transform::DfGraphConvertor::FindAdapter(op_name, true);
  if (adapter == nullptr) {
    return false;
  }

  int op_input_idx = 1;
  size_t size = inputs.size();
  for (size_t i = 0; i < size; i++) {
    if (inputs[i] == nullptr) {
      continue;
    }
    auto const_op = std::make_shared<transform::Constant>();
    MS_EXCEPTION_IF_NULL(const_op);
    (void)const_op->set_attr_value(*inputs[i]);
    MeTensorPtr me_tensor_ptr = ConvertPyObjToTensor(op_inputs[i]);
    MS_EXCEPTION_IF_NULL(me_tensor_ptr);
    auto const_op_desc =
      transform::TransformUtil::GetGeTensorDesc(me_tensor_ptr->shape_c(), me_tensor_ptr->data_type(), kOpFormat_NCHW);
    if (const_op_desc == nullptr) {
      MS_LOG(ERROR) << "Create variable " << op_name << " output descriptor failed!";
      return false;
    }
    auto pointer_cast_const_op = std::static_pointer_cast<transform::Constant>(const_op);
    MS_EXCEPTION_IF_NULL(pointer_cast_const_op);
    (void)pointer_cast_const_op->update_output_desc_y(*const_op_desc);
    auto &input_map = adapter->getInputMap();
    if (input_map.find(op_input_idx) == input_map.end()) {
      continue;
    }
    if (adapter->setInput(op, op_input_idx++, const_op)) {
      MS_LOG(ERROR) << "Failed to set params, index is " << op_input_idx;
      return false;
    }
    graph_input_nodes->push_back(*const_op);
  }
  return true;
}

bool BuildSingleOpGraph(const OpExecInfoPtr &op_exec_info, const std::vector<GeTensorPtr> &inputs,
                        const std::unordered_map<std::string, ValuePtr> &attrs, const GeGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  std::string op_name = op_exec_info->op_name;
  auto op_inputs = op_exec_info->op_inputs;
  transform::OpAdapterPtr adapter = transform::DfGraphConvertor::FindAdapter(op_name, true);
  if (adapter == nullptr) {
    MS_LOG(ERROR) << "Unable to find Adapter for " << ((std::string)py::str(op_name));
    return false;
  }
  OperatorPtr op = adapter->generate(op_name);
  MS_EXCEPTION_IF_NULL(op);

  std::vector<GeOperator> graph_input_nodes;
  // hold param nodes after setting input and output for the graph
  // set input
  if (!SetInputsForSingleOpGraph(op_exec_info, inputs, op, &graph_input_nodes)) {
    return false;
  }
  // set attributes
  for (auto attr : attrs) {
    (void)adapter->setAttr(op, attr.first, attr.second);
  }
  // set default attributes
  auto extra_attrs = adapter->GetExtraAttr();
  for (auto attr : extra_attrs) {
    (void)adapter->setAttr(op, attr.first, attr.second);
  }
  // set input attributes
  auto &input_attr_map = adapter->getInputAttrMap();
  for (auto &it : input_attr_map) {
    if (op_inputs.size() < it.first) {
      continue;
    }
    auto const_value = PyAttrValue(op_inputs[it.first - 1]);
    if (const_value->isa<None>()) {
      continue;
    }
    it.second.set_attr(op, const_value);
  }
  // construct output data nodes
  std::vector<GeOperator> graph_outputs{*op};
  // set input and output nodes for the graph
  MS_EXCEPTION_IF_NULL(graph);
  (void)graph->SetInputs(graph_input_nodes).SetOutputs(graph_outputs);
  MS_LOG(INFO) << "BuildSingleOpGraph done";
  return true;
}

void ToTensorPtr(const OpExecInfoPtr op_exec_info, std::vector<GeTensorPtr> *const inputs) {
  MS_EXCEPTION_IF_NULL(inputs);
  MS_EXCEPTION_IF_NULL(op_exec_info);
  auto op_inputs = op_exec_info->op_inputs;
  size_t size = op_inputs.size();
  for (size_t i = 0; i < size; i++) {
    if (py::isinstance<py::none>(op_inputs[i])) {
      inputs->emplace_back(nullptr);
      continue;
    }
    MeTensorPtr me_tensor_ptr = ConvertPyObjToTensor(op_inputs[i]);
    auto ge_tensor_ptr = transform::TransformUtil::ConvertTensor(me_tensor_ptr, kOpFormat_NCHW);
    if (ge_tensor_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Convert inputs to GE tensor failed in op " << op_exec_info->op_name << ".";
    }
    // set inputs for operator to build single node graph
    inputs->push_back(ge_tensor_ptr);
  }
}

PynativeStatusCode ConvertAttributes(const OpExecInfoPtr &op_exec_info, const std::vector<GeTensorPtr> &inputs) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  auto op_attrs = op_exec_info->op_attrs;
  std::unordered_map<std::string, ValuePtr> attrs{};

  for (auto &item : op_attrs) {
    if (!py::isinstance<py::str>(item.first)) {
      MS_LOG(ERROR) << "Type error in py dict convert";
      return PYNATIVE_OP_ATTRS_ERR;
    }
    std::string name = py::cast<std::string>(item.first);
    auto attr_value = PyAttrValue(py::cast<py::object>(item.second));
    (void)attrs.emplace(name, attr_value);
  }

  // build graph
  GeGraphPtr graph = std::make_shared<GeGraph>(op_exec_info->op_name);
  if (BuildSingleOpGraph(op_exec_info, inputs, attrs, graph) == false) {
    MS_LOG(ERROR) << "Failed to BuildSingleOpGraph";
    return PYNATIVE_GRAPH_GE_BUILD_ERR;
  }

  // add the single op graph into the graph manager, which will be iterated by session.
  transform::Status ret =
    transform::DfGraphManager::GetInstance().AddGraph(SINGLE_OP_GRAPH, std::shared_ptr<transform::DfGraph>(graph));
  if (ret != transform::SUCCESS) {
    MS_LOG(ERROR) << "Failed to AddGraph into graph manager";
    return PYNATIVE_GRAPH_MANAGER_ERR;
  }

  return PYNATIVE_SUCCESS;
}

std::vector<MeTensorPtr> ConvertOutputTensors(const OpExecInfoPtr &op_exec_info,
                                              const std::vector<GeTensorPtr> &ge_tensors) {
  std::vector<MeTensorPtr> outputs;
  AbstractBasePtr abs_base = op_exec_info->abstract;
  std::vector<std::vector<int>> shapes;
  if (abs_base != nullptr && abs_base->isa<abstract::AbstractTensor>()) {
    auto arg_tensor = dyn_cast<abstract::AbstractTensor>(abs_base);
    shapes.emplace_back(arg_tensor->shape()->shape());
    outputs = transform::TransformUtil::ConvertGeTensors(ge_tensors, shapes);
    return outputs;
  }
  if (abs_base != nullptr && abs_base->isa<abstract::AbstractTuple>()) {
    auto arg_tuple = dyn_cast<abstract::AbstractTuple>(abs_base);
    size_t len = arg_tuple->size();

    for (size_t i = 0; i < len; i++) {
      if (arg_tuple->elements()[i]->isa<abstract::AbstractTensor>()) {
        auto arg_tensor = dyn_cast<abstract::AbstractTensor>(arg_tuple->elements()[i]);
        shapes.emplace_back(arg_tensor->shape()->shape());
      }
    }
    outputs = transform::TransformUtil::ConvertGeTensors(ge_tensors, shapes);
    return outputs;
  }
  for (auto &it : ge_tensors) {
    auto tensor = transform::TransformUtil::ConvertGeTensor(it);
    if (tensor != nullptr) {
      outputs.emplace_back(tensor);
    }
  }
  return outputs;
}

py::object RunOpInGE(const OpExecInfoPtr &op_exec_info, PynativeStatusCode *status) {
  MS_LOG(INFO) << "RunOpInGe start";
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(status);

  // returns a null py::tuple on error
  py::tuple err_ret(0);
  auto op_name = op_exec_info->op_name;
  transform::OpAdapterPtr adapter = transform::DfGraphConvertor::FindAdapter(op_name, true);
  if (adapter == nullptr) {
    MS_LOG(ERROR) << "Unable to find GE Adapter for " << ((std::string)py::str(op_name));
    *status = PYNATIVE_OP_NOT_IMPLEMENTED_ERR;
    return std::move(err_ret);
  }

  std::vector<GeTensorPtr> inputs{};
  ToTensorPtr(op_exec_info, &inputs);
  // convert me attr to ge AttrValue
  PynativeStatusCode ret = ConvertAttributes(op_exec_info, inputs);
  if (ret != PYNATIVE_SUCCESS) {
    *status = ret;
    return std::move(err_ret);
  }
  // run graph
  transform::RunOptions run_options;
  run_options.name = SINGLE_OP_GRAPH;
  std::vector<GeTensorPtr> ge_inputs;
  std::vector<GeTensorPtr> ge_outputs;
  transform::GraphRunnerOptions graph_runner_options;
  graph_runner_options.options["ge.trainFlag"] = "1";
  auto graph_runner = std::make_shared<transform::GraphRunner>(graph_runner_options);
  transform::Status run_ret;
  {
    // Release GIL before calling into (potentially long-running) C++ code
    py::gil_scoped_release release;
    run_ret = graph_runner->RunGraph(run_options, ge_inputs, &ge_outputs);
  }
  if (run_ret != transform::Status::SUCCESS) {
    MS_LOG(ERROR) << "GraphRunner fails to run graph";
    *status = PYNATIVE_GRAPH_GE_RUN_ERR;
    return std::move(err_ret);
  }

  std::vector<MeTensorPtr> graph_outputs = ConvertOutputTensors(op_exec_info, ge_outputs);
  size_t output_size = graph_outputs.size();
  py::tuple result(output_size);
  for (size_t i = 0; i < output_size; i++) {
    MS_EXCEPTION_IF_NULL(graph_outputs[i]);
    result[i] = *graph_outputs[i];
  }

  *status = PYNATIVE_SUCCESS;
  MS_LOG(INFO) << "RunOpInGe end";
  return std::move(result);
}
}  // namespace pynative
}  // namespace mindspore
