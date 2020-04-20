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
#include <algorithm>

#include "utils/any.h"
#include "utils/utils.h"
#include "utils/context/ms_context.h"
#include "operator/ops.h"
#include "operator/composite/do_signature.h"
#include "pipeline/parse/data_converter.h"
#include "pipeline/static_analysis/prim.h"
#include "session/session_factory.h"
#include "pre_activate/pass/const_input_to_attr_registry.h"
#include "pre_activate/common/helper.h"
#include "pynative/base.h"

#ifdef ENABLE_GE
#include "pynative/pynative_execute_ge.h"
#endif

const char SINGLE_OP_GRAPH[] = "single_op_graph";
// primitive unable to infer value for constant input in PyNative mode
const std::unordered_set<std::string> vm_operators = {"partial", "depend", "make_ref"};

namespace mindspore {
namespace pynative {
inline ValuePtr PyAttrValue(const py::object &obj) {
  ValuePtr converted_ret = nullptr;
  bool converted = parse::ConvertData(obj, &converted_ret);
  if (!converted) {
    MS_LOG(EXCEPTION) << "Attribute convert error with type:" << std::string(py::str(obj));
  }
  return converted_ret;
}

py::tuple ConvertInputs(const PrimitivePyPtr &prim, const py::tuple &py_args) {
  auto signature = prim->signatures();
  std::vector<SignatureEnumDType> dtypes;
  (void)std::transform(signature.begin(), signature.end(), std::back_inserter(dtypes),
                       [](const Signature &sig) { return sig.dtype; });
  int empty_dtype_count = std::count(dtypes.begin(), dtypes.end(), SignatureEnumDType::kDTypeEmptyDefaultValue);
  if (dtypes.size() == 0 || static_cast<int>(dtypes.size()) == empty_dtype_count) {
    return py_args;
  }
  std::map<SignatureEnumDType, std::vector<size_t>> type_indexs;
  for (size_t i = 0; i < dtypes.size(); ++i) {
    auto it = type_indexs.find(dtypes[i]);
    if (it == type_indexs.end()) {
      (void)type_indexs.insert(std::make_pair(dtypes[i], std::vector<size_t>{i}));
    } else {
      it->second.push_back(i);
    }
  }
  std::map<SignatureEnumDType, size_t> dst_type;
  for (auto it = type_indexs.begin(); it != type_indexs.end(); (void)++it) {
    auto type = it->first;
    auto indexs = it->second;
    if (indexs.size() < 2) {
      continue;
    }
    size_t m_index = indexs[0];
    for (size_t i = 1; i < indexs.size(); ++i) {
      if (py::isinstance<tensor::Tensor>(py_args[indexs[i]])) {
        m_index = indexs[i];
      }
    }
    (void)dst_type.insert(std::make_pair(type, m_index));
  }
  py::tuple py_inputs(py_args.size());
  for (size_t i = 0; i < py_args.size(); ++i) {
    auto it = dst_type.find(dtypes[i]);
    if (it != dst_type.end() && it->second != i &&
        (py::isinstance<py::int_>(py_args[i]) || py::isinstance<py::float_>(py_args[i]))) {
      auto tensor_ptr = py::cast<tensor::TensorPtr>(py_args[it->second]);
      if (py::isinstance<py::int_>(py_args[i])) {
        py_inputs[i] = std::make_shared<tensor::Tensor>(py::cast<py::int_>(py_args[i]), tensor_ptr->Dtype());
      } else {
        py_inputs[i] = std::make_shared<tensor::Tensor>(py::cast<py::float_>(py_args[i]), tensor_ptr->Dtype());
      }
      continue;
    }
    py_inputs[i] = py_args[i];
  }
  return py_inputs;
}

void PynativeInfer(const PrimitivePyPtr &prim, const py::tuple &py_args, OpExecInfo *const op_exec_info) {
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

OpExecInfoPtr GenerateOpExecInfo(const py::args &args) {
  if (args.size() != PY_ARGS_NUM) {
    MS_LOG(ERROR) << "Four args are needed by RunOp";
    return nullptr;
  }
  auto op_exec_info = std::make_shared<OpExecInfo>();
  MS_EXCEPTION_IF_NULL(op_exec_info);
  op_exec_info->op_name = py::cast<std::string>(args[PY_NAME]);
  auto prim = py::cast<PrimitivePyPtr>(args[PY_PRIM]);
  auto pyobj = prim->GetPyObj();
  if (pyobj == nullptr) {
    MS_LOG(EXCEPTION) << "pyobj is empty";
  }
  py::tuple py_args = ConvertInputs(prim, args[PY_INPUTS]);
  // use python infer method
  if (ignore_infer_prim.find(op_exec_info->op_name) == ignore_infer_prim.end()) {
    PynativeInfer(prim, py_args, op_exec_info.get());
  }
  op_exec_info->py_primitive = prim;
  op_exec_info->op_attrs = py::getattr(args[PY_PRIM], "attrs");
  op_exec_info->op_inputs = py_args;
  op_exec_info->inputs_mask = args[PY_INPUT_MASK];
  if (op_exec_info->op_inputs.size() != op_exec_info->inputs_mask.size()) {
    MS_LOG(ERROR) << "Op:" << op_exec_info->op_name << " inputs size not equal op_mask";
    return nullptr;
  }
  return op_exec_info;
}

std::string GetSingleOpGraphInfo(const OpExecInfoPtr &op_exec_info) {
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
  MS_LOG(INFO) << "Graph info [" << graph_info << "]";
  return graph_info;
}

py::object RunOpInVM(const OpExecInfoPtr &op_exec_info, PynativeStatusCode *status) {
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

bool RunOpConvertConstInputToAttr(const py::object &input_object, size_t input_index, const PrimitivePtr &op_prim,
                                  const std::unordered_set<size_t> &input_attrs) {
  MS_EXCEPTION_IF_NULL(op_prim);
  auto input_names_value = op_prim->GetAttr(kAttrInputNames);
  if (input_names_value == nullptr) {
    return false;
  }
  auto input_names_vec = GetValue<std::vector<std::string>>(input_names_value);
  if (input_index >= input_names_vec.size()) {
    MS_LOG(EXCEPTION) << "The input index: " << input_index << " is large than the input names vector size!";
  }

  if (input_attrs.find(input_index) != input_attrs.end()) {
    ValuePtr value = parse::data_converter::PyDataToValue(input_object);
    MS_EXCEPTION_IF_NULL(value);
    auto input_name = input_names_vec[input_index];
    op_prim->set_attr(input_name, value);
    return true;
  }
  return false;
}

void PlantTensorTupleToVector(const py::tuple &tuple_inputs, const PrimitivePtr &op_prim,
                              std::vector<tensor::TensorPtr> *input_tensor) {
  MS_EXCEPTION_IF_NULL(op_prim);
  MS_EXCEPTION_IF_NULL(input_tensor);
  for (const auto &input_object : tuple_inputs) {
    if (!py::isinstance<tensor::Tensor>(input_object)) {
      MS_LOG(EXCEPTION) << "The input object is not a tensor!";
    }
    auto tensor = py::cast<tensor::TensorPtr>(input_object);
    MS_EXCEPTION_IF_NULL(tensor);
    input_tensor->push_back(tensor);
  }
  op_prim->set_attr(kAttrDynInputSizes, MakeValue(std::vector<int>{SizeToInt(tuple_inputs.size())}));
}

void ConvertValueTupleToTensor(const py::object &input_object, std::vector<tensor::TensorPtr> *input_tensor) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  ValuePtr input_value = parse::data_converter::PyDataToValue(input_object);
  MS_EXCEPTION_IF_NULL(input_value);
  if (!input_value->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "The input object is not a value tuple!";
  }
  auto value_tuple = input_value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple);
  tensor::TensorPtr tensor_ptr = opt::CreateTupleTensor(value_tuple);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  input_tensor->push_back(tensor_ptr);
}

void ConvertPyObjectToTensor(const py::object &input_object, const PrimitivePtr &op_prim,
                             std::vector<tensor::TensorPtr> *input_tensor) {
  MS_EXCEPTION_IF_NULL(op_prim);
  MS_EXCEPTION_IF_NULL(input_tensor);
  tensor::TensorPtr tensor_ptr = nullptr;
  if (py::isinstance<tensor::Tensor>(input_object)) {
    tensor_ptr = py::cast<tensor::TensorPtr>(input_object);
  } else if (py::isinstance<py::float_>(input_object)) {
    tensor_ptr = std::make_shared<tensor::Tensor>(py::cast<py::float_>(input_object), kFloat32);
  } else if (py::isinstance<py::int_>(input_object)) {
    tensor_ptr = std::make_shared<tensor::Tensor>(py::cast<py::int_>(input_object), nullptr);
  } else if (py::isinstance<py::list>(input_object)) {
    tensor_ptr = std::make_shared<tensor::Tensor>(py::cast<py::list>(input_object), nullptr);
  } else if (py::isinstance<py::array>(input_object)) {
    tensor_ptr = std::make_shared<tensor::Tensor>(py::cast<py::array>(input_object), nullptr);
  } else if (py::isinstance<py::tuple>(input_object)) {
    auto tuple_inputs = py::cast<py::tuple>(input_object);
    if (py::isinstance<tensor::Tensor>(tuple_inputs[0])) {
      PlantTensorTupleToVector(tuple_inputs, op_prim, input_tensor);
    } else {
      ConvertValueTupleToTensor(input_object, input_tensor);
    }
    return;
  } else {
    MS_LOG(EXCEPTION) << "Run op inputs type is invalid!";
  }
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  input_tensor->push_back(tensor_ptr);
}

void ConstructInputTensor(const OpExecInfoPtr &op_run_info, std::vector<bool> *tensors_mask,
                          std::vector<tensor::TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(tensors_mask);
  MS_EXCEPTION_IF_NULL(input_tensors);
  PrimitivePtr op_prim = op_run_info->py_primitive;
  MS_EXCEPTION_IF_NULL(op_prim);

  if (op_run_info->op_inputs.size() != op_run_info->inputs_mask.size()) {
    MS_LOG(EXCEPTION) << "Op input size " << op_run_info->op_inputs.size() << " should be equal to op input mask size "
                      << op_run_info->inputs_mask.size();
  }
  opt::ConstInputToAttrInfoRegister reg;
  bool reg_exist = opt::ConstInputToAttrInfoRegistry::Instance().GetRegisterByOpName(op_run_info->op_name, &reg);
  size_t input_num = op_run_info->op_inputs.size();
  MS_LOG(INFO) << "py input size: " << input_num;
  for (size_t index = 0; index < input_num; ++index) {
    // convert const input to attr
    if (reg_exist &&
        RunOpConvertConstInputToAttr(op_run_info->op_inputs[index], index, op_prim, reg.GetConstInputAttrInfo())) {
      continue;
    }
    // convert const and tuple input to tensor
    ConvertPyObjectToTensor(op_run_info->op_inputs[index], op_prim, input_tensors);
    // make tensors, weight : 1, data : 0
    std::vector<bool> new_mask(input_tensors->size() - tensors_mask->size(),
                               py::cast<bool>(op_run_info->inputs_mask[index]));
    tensors_mask->insert(tensors_mask->end(), new_mask.begin(), new_mask.end());
  }
}

py::object RunOpInMs(const OpExecInfoPtr &op_exec_info, PynativeStatusCode *status) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_LOG(INFO) << "Start run op[" << op_exec_info->op_name << "] with backend policy ms";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_enable_pynative_infer(true);
  std::string device_target = ms_context->device_target();
  if (device_target != kAscendDevice && device_target != kGPUDevice) {
    MS_EXCEPTION(ArgumentError) << "Device target [" << device_target << "] is not supported in Pynative mode";
  }
  std::shared_ptr<session::SessionBasic> session = session::SessionFactory::Get().Create(device_target);
  MS_EXCEPTION_IF_NULL(session);
  session->Init(ms_context->device_id());

  std::string graph_info = GetSingleOpGraphInfo(op_exec_info);
  std::vector<tensor::TensorPtr> input_tensors;
  std::vector<bool> tensors_mask;
  ConstructInputTensor(op_exec_info, &tensors_mask, &input_tensors);
  session->BuildOp(*op_exec_info, graph_info, input_tensors, tensors_mask);
  py::tuple result = session->RunOp(*op_exec_info, graph_info, input_tensors);
  ms_context->set_enable_pynative_infer(false);
  *status = PYNATIVE_SUCCESS;
  return result;
}

py::object RunOpWithBackendPolicy(MsBackendPolicy backend_policy, const OpExecInfoPtr op_exec_info,
                                  PynativeStatusCode *const status) {
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
      MS_LOG(ERROR) << "No backend configured for run op";
  }
  return result;
}

py::tuple RunOp(const py::args &args) {
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
    MS_LOG(ERROR) << "Failed to run " << op_exec_info->op_name;
    return err_ret;
  }

  MS_LOG(INFO) << "RunOp end";
  return result;
}
}  // namespace pynative
}  // namespace mindspore
