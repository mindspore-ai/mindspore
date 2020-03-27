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

#include "pynative/pynative_execute.h"

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

const char SINGLE_OP_GRAPH[] = "single_op_graph";
// primitive unable to infer value for constant input in pynative mode
const std::unordered_set<std::string> ignore_infer_prim = {"partial"};
const std::unordered_set<std::string> vm_operators = {"partial", "depend"};

namespace mindspore {
namespace pynative {
using transform::GraphRunner;
using transform::GraphRunnerOptions;
using transform::OperatorPtr;
inline ValuePtr PyAttrValue(const py::object& obj) {
  ValuePtr converted_ret = nullptr;
  bool converted = parse::ConvertData(obj, &converted_ret);
  if (!converted) {
    MS_LOG(EXCEPTION) << "attribute convert error with type:" << std::string(py::str(obj));
  }
  return converted_ret;
}

MeTensorPtr ConvertPyObjToTensor(const py::object& obj) {
  MeTensorPtr me_tensor_ptr = nullptr;
  if (py::isinstance<MeTensor>(obj)) {
    me_tensor_ptr = py::cast<MeTensorPtr>(obj);
  } else if (py::isinstance<py::tuple>(obj)) {
    me_tensor_ptr = std::make_shared<MeTensor>(py::cast<py::tuple>(obj), nullptr);
  } else if (py::isinstance<py::float_>(obj)) {
    me_tensor_ptr = std::make_shared<MeTensor>(py::cast<py::float_>(obj), nullptr);
  } else if (py::isinstance<py::int_>(obj)) {
    me_tensor_ptr = std::make_shared<MeTensor>(py::cast<py::int_>(obj), nullptr);
  } else if (py::isinstance<py::list>(obj)) {
    me_tensor_ptr = std::make_shared<MeTensor>(py::cast<py::list>(obj), nullptr);
  } else if (py::isinstance<py::array>(obj)) {
    me_tensor_ptr = std::make_shared<MeTensor>(py::cast<py::array>(obj), nullptr);
  } else {
    MS_LOG(EXCEPTION) << "run op inputs type is invalid!";
  }
  return me_tensor_ptr;
}

void PynativeInfer(const PrimitivePyPtr& prim, const py::tuple& py_args, OpExecInfo* const op_exec_info) {
  size_t size = py_args.size();
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < size; i++) {
    ValuePtr input_value = PyAttrValue(py_args[i]);
    if (py::isinstance<MeTensor>(py_args[i])) {
      args_spec_list.emplace_back(abstract::FromValueInside(input_value, true));
    } else {
      args_spec_list.emplace_back(abstract::FromValueInside(input_value, false));
    }
  }
  AbstractBasePtr infer_res = InferOnePrim(prim, args_spec_list);
  op_exec_info->abstract = infer_res;
}

OpExecInfoPtr GenerateOpExecInfo(const py::args& args) {
  if (args.size() != PY_ARGS_NUM) {
    MS_LOG(ERROR) << "four args are needed by RunOp";
    return nullptr;
  }
  auto op_exec_info = std::make_shared<OpExecInfo>();
  MS_EXCEPTION_IF_NULL(op_exec_info);
  op_exec_info->op_name = py::cast<std::string>(args[PY_NAME]);
  if (py::isinstance<py::none>(args[PY_PRIM])) {
    py::module ops_mod = py::module::import("mindspore.ops.operations");
    py::object py_primitive = ops_mod.attr(op_exec_info->op_name.c_str())();
    op_exec_info->py_primitive = py::cast<PrimitivePyPtr>(py_primitive);
    py::dict none_attrs = py::dict();
    op_exec_info->op_attrs = none_attrs;
  } else {
    PrimitivePyPtr prim = py::cast<PrimitivePyPtr>(args[PY_PRIM]);
    auto pyobj = prim->GetPyObj();
    if (pyobj == nullptr) {
      MS_LOG(EXCEPTION) << "pyobj is empty";
    }
    py::tuple py_args = args[PY_INPUTS];
    // use python infer method
    if (ignore_infer_prim.find(op_exec_info->op_name) == ignore_infer_prim.end()) {
      PynativeInfer(prim, py_args, op_exec_info.get());
    }
    op_exec_info->py_primitive = prim;
    op_exec_info->op_attrs = py::getattr(args[PY_PRIM], "attrs");
  }
  op_exec_info->op_inputs = args[PY_INPUTS];
  op_exec_info->inputs_mask = args[PY_INPUT_MASK];
  if (op_exec_info->op_inputs.size() != op_exec_info->inputs_mask.size()) {
    MS_LOG(ERROR) << "" << op_exec_info->op_name << " op_inputs size not equal op_mask";
    return nullptr;
  }
  return op_exec_info;
}

std::string GetSingleOpGraphInfo(const OpExecInfoPtr& op_exec_info) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  std::string graph_info;
  MS_EXCEPTION_IF_NULL(op_exec_info->abstract);
  // get input tensor info
  size_t input_num = op_exec_info->op_inputs.size();
  for (size_t index = 0; index < input_num; ++index) {
    if (py::isinstance<tensor::Tensor>(op_exec_info->op_inputs[index])) {
      auto tensor_ptr = py::cast<tensor::TensorPtr>(op_exec_info->op_inputs[index]);
      MS_EXCEPTION_IF_NULL(tensor_ptr);
      (void)graph_info.append(tensor_ptr->GetShapeAndDataTypeInfo() + "_");
    }
  }
  // get prim and abstract info
  (void)graph_info.append(std::to_string((uintptr_t)(op_exec_info->py_primitive.get())) + "_" +
                          op_exec_info->abstract->ToString());
  MS_LOG(INFO) << "graph info [" << graph_info << "]";
  return graph_info;
}

bool SetInputsForSingleOpGraph(const OpExecInfoPtr& op_exec_info, const std::vector<GeTensorPtr>& inputs,
                               const OperatorPtr& op, std::vector<GeOperator>* graph_input_nodes) {
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
      MS_LOG(ERROR) << "Create variable " << op_name << " ouptut descriptor failed!";
      return false;
    }
    auto pointer_cast_const_op = std::static_pointer_cast<transform::Constant>(const_op);
    MS_EXCEPTION_IF_NULL(pointer_cast_const_op);
    (void)pointer_cast_const_op->update_output_desc_y(*const_op_desc);
    auto& input_map = adapter->getInputMap();
    if (input_map.find(op_input_idx) == input_map.end()) {
      continue;
    }
    if (adapter->setInput(op, op_input_idx++, const_op)) {
      MS_LOG(ERROR) << "fail to set params, index is " << op_input_idx;
      return false;
    }
    graph_input_nodes->push_back(*const_op);
  }
  return true;
}

bool BuildSingleOpGraph(const OpExecInfoPtr& op_exec_info, const std::vector<GeTensorPtr>& inputs,
                        const std::unordered_map<std::string, ValuePtr>& attrs, const GeGraphPtr& graph) {
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
  auto& input_attr_map = adapter->getInputAttrMap();
  for (auto& it : input_attr_map) {
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

void ToTensorPtr(const OpExecInfoPtr op_exec_info, std::vector<GeTensorPtr>* const inputs) {
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
      MS_LOG(EXCEPTION) << "convert inputs to GE tensor failed in op " << op_exec_info->op_name << ".";
    }
    // set inputs for operator to build single node graph
    inputs->push_back(ge_tensor_ptr);
  }
}

PynativeStatusCode ConvertAttributes(const OpExecInfoPtr& op_exec_info, const std::vector<GeTensorPtr>& inputs) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  auto op_attrs = op_exec_info->op_attrs;
  std::unordered_map<std::string, ValuePtr> attrs{};

  for (auto& item : op_attrs) {
    if (!py::isinstance<py::str>(item.first)) {
      MS_LOG(ERROR) << "type error in py dict convert";
      return PYNATIVE_OP_ATTRS_ERR;
    }
    std::string name = py::cast<std::string>(item.first);
    auto attr_value = PyAttrValue(py::cast<py::object>(item.second));
    (void)attrs.emplace(name, attr_value);
  }

  // build graph
  GeGraphPtr graph = std::make_shared<GeGraph>(op_exec_info->op_name);
  if (BuildSingleOpGraph(op_exec_info, inputs, attrs, graph) == false) {
    MS_LOG(ERROR) << "Fail to BuildSingleOpGraph";
    return PYNATIVE_GRAPH_GE_BUILD_ERR;
  }

  // add the single op graph into the graph manager, which will be iterated by session.
  transform::Status ret =
    transform::DfGraphManager::GetInstance().AddGraph(SINGLE_OP_GRAPH, std::shared_ptr<transform::DfGraph>(graph));
  if (ret != transform::SUCCESS) {
    MS_LOG(ERROR) << "Fail to AddGraph into graph manager";
    return PYNATIVE_GRAPH_MANAGER_ERR;
  }

  return PYNATIVE_SUCCESS;
}

std::vector<MeTensorPtr> ConvertOutputTensors(const OpExecInfoPtr& op_exec_info,
                                              const std::vector<GeTensorPtr>& ge_tensors) {
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
  for (auto& it : ge_tensors) {
    auto tensor = transform::TransformUtil::ConvertGeTensor(it);
    if (tensor != nullptr) {
      outputs.emplace_back(tensor);
    }
  }
  return outputs;
}

py::object RunOpInGE(const OpExecInfoPtr& op_exec_info, PynativeStatusCode* status) {
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
    MS_LOG(ERROR) << "GraphRunner Fails to Run Graph";
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

py::object RunOpInVM(const OpExecInfoPtr& op_exec_info, PynativeStatusCode* status) {
  MS_LOG(INFO) << "RunOpInVM start";

  MS_EXCEPTION_IF_NULL(status);
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(op_exec_info->py_primitive);
  auto func = op_exec_info->py_primitive->GetComputeFunction();
  if (py::isinstance<py::none>(func)) {
    MS_LOG(ERROR) << "VM failed to get func";
    *status = PYNATIVE_OP_NOT_IMPLEMENTED_ERR;
    py::tuple err_ret(0);
    return std::move(err_ret);
  }

  // execute op
  py::tuple result = py::make_tuple(func(*op_exec_info->op_inputs));
  *status = PYNATIVE_SUCCESS;
  MS_LOG(INFO) << "RunOpInVM end";
  return std::move(result);
}

py::object RunOpInMs(const OpExecInfoPtr& op_exec_info, PynativeStatusCode* status) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_LOG(INFO) << "start run op[" << op_exec_info->op_name << "] with backend policy ms";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_enable_pynative_infer(true);
  std::string device_target = ms_context->device_target();
  if (device_target != kAscendDevice && device_target != kGPUDevice) {
    MS_EXCEPTION(ArgumentError) << "device target [" << device_target << "] is not supported in Pynative mode";
  }
  std::shared_ptr<session::SessionBasic> session = session::SessionFactory::Get().Create(device_target);
  MS_EXCEPTION_IF_NULL(session);
  session->Init(ms_context->device_id());

  std::string graph_info = GetSingleOpGraphInfo(op_exec_info);
  session->BuildOp(*op_exec_info, graph_info);
  py::tuple result = session->RunOp(*op_exec_info, graph_info);
  ms_context->set_enable_pynative_infer(false);
  *status = PYNATIVE_SUCCESS;
  return result;
}

py::object RunOpWithBackendPolicy(MsBackendPolicy backend_policy, const OpExecInfoPtr op_exec_info,
                                  PynativeStatusCode* const status) {
  MS_EXCEPTION_IF_NULL(status);
  py::object result;
  switch (backend_policy) {
    case kMsBackendGeOnly: {
      // use GE only
      MS_LOG(INFO) << "RunOp use GE only backend";
      result = RunOpInGE(op_exec_info, status);
      break;
    }
    case kMsBackendVmOnly: {
      // use vm only
      MS_LOG(INFO) << "RunOp use VM only backend";
      result = RunOpInVM(op_exec_info, status);
      break;
    }
    case kMsBackendGePrior: {
      // use GE first, use vm when GE fails
      MS_LOG(INFO) << "RunOp use GE first backend";
      result = RunOpInGE(op_exec_info, status);
      if (*status != PYNATIVE_SUCCESS) {
        result = RunOpInVM(op_exec_info, status);
      }
      break;
    }
    case kMsBackendVmPrior: {
      // GE_VM_SILENT
      // (should not use this policy) use vm first, use GE when vm fails
      MS_LOG(INFO) << "RunOp use VM first backend";
      result = RunOpInVM(op_exec_info, status);
      if (*status != PYNATIVE_SUCCESS) {
        result = RunOpInGE(op_exec_info, status);
      }
      break;
    }
    case kMsBackendMsPrior: {
      // use Ms fisrt,use others when ms failed
      MS_LOG(INFO) << "RunOp use Ms first backend";
      result = RunOpInMs(op_exec_info, status);
      if (*status != PYNATIVE_SUCCESS) {
        MS_LOG(ERROR) << "RunOp use Ms backend failed!!!";
      }
      break;
    }
    default:
      MS_LOG(ERROR) << "No backend configed for run op";
  }
  return result;
}

py::tuple RunOp(const py::args& args) {
  py::object result;
  // returns a null py::tuple on error
  py::tuple err_ret(0);
  PynativeStatusCode status = PYNATIVE_UNKNOWN_STATE;

  OpExecInfoPtr op_exec_info = GenerateOpExecInfo(args);
  MS_EXCEPTION_IF_NULL(op_exec_info);
  if (op_exec_info->abstract != nullptr) {
    py::dict output = abstract::ConvertAbstractToPython(op_exec_info->abstract);
    if (!output["value"].is_none()) {
      py::tuple value_ret(1);
      value_ret[0] = output["value"];
      return value_ret;
    }
  }
  MS_LOG(INFO) << "RunOp start, op name is: " << op_exec_info->op_name;
  mindspore::parse::python_adapter::set_python_env_flag(true);
  MsBackendPolicy backend_policy;
#if (!defined ENABLE_GE)
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->backend_policy() == "ms") {
    backend_policy = kMsBackendMsPrior;
  } else {
    backend_policy = kMsBackendVmOnly;
  }
#else
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->PynativeInitGe();
  backend_policy = kMsBackendGeOnly;
#endif
  if (vm_operators.find(op_exec_info->op_name) != vm_operators.end()) {
    backend_policy = kMsBackendVmOnly;
  }
  result = RunOpWithBackendPolicy(backend_policy, op_exec_info, &status);
  if (status != PYNATIVE_SUCCESS) {
    MS_LOG(ERROR) << "Fail to run " << op_exec_info->op_name;
    return err_ret;
  }

  MS_LOG(INFO) << "RunOp end";
  return result;
}
}  // namespace pynative
}  // namespace mindspore
