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

#include "pynative/base.h"

#ifdef ENABLE_GE
#include "pynative/pynative_execute_ge.h"
#endif

const char SINGLE_OP_GRAPH[] = "single_op_graph";
// primitive unable to infer value for constant input in pynative mode
const std::unordered_set<std::string> vm_operators = {"partial", "depend"};

namespace mindspore {
namespace pynative {
inline ValuePtr PyAttrValue(const py::object& obj) {
  ValuePtr converted_ret = nullptr;
  bool converted = parse::ConvertData(obj, &converted_ret);
  if (!converted) {
    MS_LOG(EXCEPTION) << "attribute convert error with type:" << std::string(py::str(obj));
  }
  return converted_ret;
}

void PynativeInfer(const PrimitivePyPtr& prim, const py::tuple& py_args, OpExecInfo* const op_exec_info) {
  size_t size = py_args.size();
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < size; i++) {
    ValuePtr input_value = PyAttrValue(py_args[i]);
    if (py::isinstance<tensor::Tensor>(py_args[i])) {
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
    case kMsBackendVmOnly: {
      // use vm only
      MS_LOG(INFO) << "RunOp use VM only backend";
      result = RunOpInVM(op_exec_info, status);
      break;
    }
    case kMsBackendGePrior: {
#ifdef ENABLE_GE
      // use GE first, use vm when GE fails
      MS_LOG(INFO) << "RunOp use GE first backend";
      result = RunOpInGE(op_exec_info, status);
      if (*status != PYNATIVE_SUCCESS) {
        result = RunOpInVM(op_exec_info, status);
      }
#endif
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
