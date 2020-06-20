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

#include "ir/param_value_py.h"
#include "utils/any.h"
#include "utils/utils.h"
#include "utils/context/ms_context.h"
#include "operator/ops.h"
#include "operator/composite/composite.h"
#include "operator/composite/do_signature.h"
#include "pipeline/parse/data_converter.h"
#include "pipeline/parse/parse_base.h"
#include "pipeline/parse/resolve.h"
#include "pipeline/static_analysis/prim.h"
#include "session/session_factory.h"
#include "pre_activate/pass/const_input_to_attr_registry.h"
#include "pre_activate/common/helper.h"
#include "pipeline/action.h"

#include "pynative/base.h"
#include "pybind_api/api_register.h"
#include "vm/transform.h"

#include "optimizer/ad/grad.h"
#include "pipeline/resource.h"
#include "pipeline/pipeline.h"
#include "pipeline/pass.h"

#ifdef ENABLE_GE
#include "pynative/pynative_execute_ge.h"
#endif

const char SINGLE_OP_GRAPH[] = "single_op_graph";
// primitive unable to infer value for constant input in PyNative mode
const std::set<std::string> vm_operators = {"make_ref", "HookBackward"};

namespace mindspore {
namespace pynative {

static std::shared_ptr<session::SessionBasic> session = nullptr;
PynativeExecutorPtr PynativeExecutor::executor_ = nullptr;
std::mutex PynativeExecutor::instance_lock_;
ResourcePtr PynativeExecutor::resource_;

inline ValuePtr PyAttrValue(const py::object &obj) {
  ValuePtr converted_ret = parse::data_converter::PyDataToValue(obj);
  if (!converted_ret) {
    MS_LOG(EXCEPTION) << "Attribute convert error with type:" << std::string(py::str(obj));
  }
  return converted_ret;
}

std::string GetId(const py::object &obj) {
  py::object to_process = obj;
  std::string prefix = "";
  if (py::isinstance<py::tuple>(to_process)) {
    auto p_list = py::cast<py::tuple>(to_process);
    if (p_list.size() == 0) {
      return "empty";
    }
    to_process = p_list[0];
    prefix = "tuple:";
    if (!py::isinstance<tensor::Tensor>(to_process)) {
      std::string key = "";
      for (size_t i = 0; i < p_list.size(); ++i) {
        key += std::string(py::str(p_list[i])) + ":";
      }
      return prefix + key;
    }
  }
  if (py::isinstance<py::int_>(to_process)) {
    return prefix + std::string(py::str(to_process));
  }
  if (py::isinstance<py::float_>(to_process)) {
    return prefix + std::string(py::str(to_process));
  }
  if (py::isinstance<tensor::Tensor>(to_process)) {
    auto tensor_ptr = py::cast<tensor::TensorPtr>(to_process);
    return prefix + tensor_ptr->id();
  }

  py::object ret = parse::python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_MOD_GET_OBJ_ID, obj);
  return py::cast<std::string>(ret);
}

py::object GetTupleObj(const py::object &obj) {
  py::module mod = parse::python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  py::object obj_tuple = parse::python_adapter::CallPyModFn(mod, parse::PYTHON_MOD_GET_DEFAULT_INPUT, obj);
  return obj_tuple;
}

std::map<SignatureEnumDType, std::vector<size_t>> GetTypeIndex(const std::vector<SignatureEnumDType> &dtypes) {
  std::map<SignatureEnumDType, std::vector<size_t>> type_indexes;
  for (size_t i = 0; i < dtypes.size(); ++i) {
    auto it = type_indexes.find(dtypes[i]);
    if (it == type_indexes.end()) {
      (void)type_indexes.insert(std::make_pair(dtypes[i], std::vector<size_t>{i}));
    } else {
      it->second.push_back(i);
    }
  }
  return type_indexes;
}

std::map<SignatureEnumDType, size_t> GetDstType(const py::tuple &py_args,
                                                const std::map<SignatureEnumDType, std::vector<size_t>> &type_indexes) {
  std::map<SignatureEnumDType, size_t> dst_type;
  for (auto it = type_indexes.begin(); it != type_indexes.end(); (void)++it) {
    auto type = it->first;
    auto indexes = it->second;
    if (indexes.size() < 2) {
      continue;
    }
    size_t m_index = indexes[0];
    for (size_t i = 1; i < indexes.size(); ++i) {
      if (py::isinstance<tensor::Tensor>(py_args[indexes[i]])) {
        m_index = indexes[i];
      }
    }
    (void)dst_type.insert(std::make_pair(type, m_index));
  }
  return dst_type;
}

py::tuple ConvertInputs(const PrimitivePyPtr &prim, const py::list &args, py::tuple *const out_args) {
  auto &py_args = *out_args;
  py::tuple input_mask(args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    if (py::hasattr(args[i], "__parameter__")) {
      input_mask[i] = true;
    } else {
      input_mask[i] = false;
    }
    py_args[i] = GetTupleObj(args[i]);
  }
  auto signature = prim->signatures();
  std::vector<SignatureEnumDType> dtypes;
  (void)std::transform(signature.begin(), signature.end(), std::back_inserter(dtypes),
                       [](const Signature &sig) { return sig.dtype; });
  int empty_dtype_count = std::count(dtypes.begin(), dtypes.end(), SignatureEnumDType::kDTypeEmptyDefaultValue);
  if (dtypes.size() == 0 || static_cast<int>(dtypes.size()) == empty_dtype_count) {
    return input_mask;
  }
  auto type_indexes = GetTypeIndex(dtypes);
  auto dst_type = GetDstType(py_args, type_indexes);
  for (size_t i = 0; i < py_args.size(); ++i) {
    auto it = dst_type.find(dtypes[i]);
    if (it != dst_type.end() && it->second != i &&
        (py::isinstance<py::int_>(py_args[i]) || py::isinstance<py::float_>(py_args[i]))) {
      auto tensor_ptr = py::cast<tensor::TensorPtr>(py_args[it->second]);
      if (py::isinstance<py::int_>(py_args[i])) {
        py_args[i] = std::make_shared<tensor::Tensor>(py::cast<py::int_>(py_args[i]), tensor_ptr->Dtype());
      } else {
        py_args[i] = std::make_shared<tensor::Tensor>(py::cast<py::float_>(py_args[i]), tensor_ptr->Dtype());
      }
      continue;
    }
  }
  return input_mask;
}

void PynativeInfer(const PrimitivePyPtr &prim, const py::list &py_args, OpExecInfo *const op_exec_info) {
  size_t size = py_args.size();
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < size; i++) {
    ValuePtr input_value = PyAttrValue(py_args[i]);
    if (!py::hasattr(prim->GetPyObj(), "const_value") && input_value->isa<tensor::Tensor>()) {
      args_spec_list.emplace_back(abstract::FromValueInside(input_value, true));
    } else {
      args_spec_list.emplace_back(abstract::FromValueInside(input_value, false));
    }
  }
  AbstractBasePtr infer_res = EvalOnePrim(prim, args_spec_list)->abstract();
  op_exec_info->abstract = infer_res;
}

OpExecInfoPtr GenerateOpExecInfo(const py::args &args) {
  if (args.size() != PY_ARGS_NUM) {
    MS_LOG(ERROR) << "Three args are needed by RunOp";
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

  py::list a = args[PY_INPUTS];
  size_t input_num = a.size();
  op_exec_info->op_inputs = py::tuple(input_num);

  op_exec_info->inputs_mask = ConvertInputs(prim, args[PY_INPUTS], &op_exec_info->op_inputs);
  // use python infer method
  if (ignore_infer_prim.find(op_exec_info->op_name) == ignore_infer_prim.end()) {
    PynativeInfer(prim, op_exec_info->op_inputs, op_exec_info.get());
  }
  op_exec_info->py_primitive = prim;
  op_exec_info->op_attrs = py::getattr(args[PY_PRIM], "attrs");
  if (op_exec_info->op_inputs.size() != op_exec_info->inputs_mask.size()) {
    MS_LOG(ERROR) << "Op:" << op_exec_info->op_name << " inputs size not equal op_mask";
    return nullptr;
  }
  return op_exec_info;
}

std::string GetSingleOpGraphInfo(const OpExecInfoPtr &op_exec_info,
                                 const std::vector<tensor::TensorPtr> &input_tensors) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  std::string graph_info;
  // get input tensor info
  size_t input_num = op_exec_info->op_inputs.size();
  for (size_t index = 0; index < input_num; ++index) {
    auto input = op_exec_info->op_inputs[index];
    if (py::isinstance<tensor::Tensor>(input)) {
      auto tensor_ptr = py::cast<tensor::TensorPtr>(input);
      (void)graph_info.append(tensor_ptr->GetShapeAndDataTypeInfo() + "_");
    }
  }
  // get prim and abstract info
  MS_EXCEPTION_IF_NULL(op_exec_info->abstract);
  (void)graph_info.append(std::to_string((uintptr_t)(op_exec_info->py_primitive.get())) + "_" +
                          op_exec_info->abstract->ToString());
  return graph_info;
}

py::object RunOpInVM(const OpExecInfoPtr &op_exec_info, PynativeStatusCode *status) {
  MS_LOG(INFO) << "RunOpInVM start";

  MS_EXCEPTION_IF_NULL(status);
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(op_exec_info->py_primitive);
  if (op_exec_info->op_name == "HookBackward") {
    auto op_inputs = op_exec_info->op_inputs;
    py::tuple result(op_inputs.size());
    for (size_t i = 0; i < op_inputs.size(); i++) {
      py::object input = op_inputs[i];
      if (py::hasattr(input, "__parameter__")) {
        result[i] = py::getattr(input, "data");
      } else {
        auto tensor = py::cast<tensor::TensorPtr>(op_inputs[i]);
        auto new_tensor = std::make_shared<tensor::Tensor>(tensor->data());
        result[i] = new_tensor;
      }
    }
    *status = PYNATIVE_SUCCESS;
    MS_LOG(INFO) << "RunOpInVM end";
    return std::move(result);
  }
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
                              std::vector<tensor::TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(op_prim);
  MS_EXCEPTION_IF_NULL(input_tensors);
  for (const auto &input_object : tuple_inputs) {
    if (!py::isinstance<tensor::Tensor>(input_object)) {
      MS_LOG(EXCEPTION) << "The input object is not a tensor!";
    }
    auto tensor = py::cast<tensor::TensorPtr>(input_object);
    MS_EXCEPTION_IF_NULL(tensor);
    input_tensors->push_back(tensor);
  }
  op_prim->set_attr(kAttrDynInputSizes, MakeValue(std::vector<int>{SizeToInt(tuple_inputs.size())}));
}

void ConvertValueTupleToTensor(const py::object &input_object, std::vector<tensor::TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  ValuePtr input_value = parse::data_converter::PyDataToValue(input_object);
  MS_EXCEPTION_IF_NULL(input_value);
  if (!input_value->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "The input object is not a value tuple!";
  }
  auto value_tuple = input_value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple);
  tensor::TensorPtr tensor_ptr = opt::CreateTupleTensor(value_tuple);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  input_tensors->push_back(tensor_ptr);
}

void ConvertMultiPyObjectToTensor(const py::object &input_object, const PrimitivePtr &op_prim,
                                  std::vector<tensor::TensorPtr> *input_tensors, int *tensor_mask) {
  MS_EXCEPTION_IF_NULL(op_prim);
  MS_EXCEPTION_IF_NULL(input_tensors);
  MS_EXCEPTION_IF_NULL(tensor_mask);

  if (!py::isinstance<py::tuple>(input_object)) {
    MS_LOG(EXCEPTION) << "The input should be a tuple!";
  }
  auto tuple_inputs = py::cast<py::tuple>(input_object);
  if (tuple_inputs.size() == 0) {
    MS_LOG(EXCEPTION) << "The size of input list or tuple is 0!";
  }
  if (py::isinstance<tensor::Tensor>(tuple_inputs[0])) {
    PlantTensorTupleToVector(tuple_inputs, op_prim, input_tensors);
  } else {
    ConvertValueTupleToTensor(input_object, input_tensors);
    *tensor_mask = kValueNodeTensorMask;
  }
}

void ConvertPyObjectToTensor(const py::object &input_object, const PrimitivePtr &op_prim,
                             std::vector<tensor::TensorPtr> *input_tensors, int *tensor_mask) {
  MS_EXCEPTION_IF_NULL(op_prim);
  MS_EXCEPTION_IF_NULL(input_tensors);
  MS_EXCEPTION_IF_NULL(tensor_mask);
  tensor::TensorPtr tensor_ptr = nullptr;
  if (py::isinstance<tensor::Tensor>(input_object)) {
    tensor_ptr = py::cast<tensor::TensorPtr>(input_object);
  } else if (py::isinstance<py::float_>(input_object)) {
    tensor_ptr = std::make_shared<tensor::Tensor>(py::cast<py::float_>(input_object), kFloat32);
    *tensor_mask = kValueNodeTensorMask;
  } else if (py::isinstance<py::int_>(input_object)) {
    tensor_ptr = std::make_shared<tensor::Tensor>(py::cast<py::int_>(input_object), kInt32);
    *tensor_mask = kValueNodeTensorMask;
  } else if (py::isinstance<py::array>(input_object)) {
    tensor_ptr = std::make_shared<tensor::Tensor>(py::cast<py::array>(input_object), nullptr);
  } else if (py::isinstance<py::list>(input_object)) {
    auto list_inputs = py::cast<py::list>(input_object);
    py::tuple tuple_inputs(list_inputs.size());
    for (size_t i = 0; i < tuple_inputs.size(); ++i) {
      tuple_inputs[i] = list_inputs[i];
    }
    ConvertMultiPyObjectToTensor(tuple_inputs, op_prim, input_tensors, tensor_mask);
    return;
  } else if (py::isinstance<py::tuple>(input_object)) {
    ConvertMultiPyObjectToTensor(input_object, op_prim, input_tensors, tensor_mask);
    return;
  } else if (py::isinstance<py::none>(input_object)) {
    return;
  } else {
    MS_LOG(EXCEPTION) << "Run op inputs type is invalid!";
  }
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  input_tensors->push_back(tensor_ptr);
}

void ConstructInputTensor(const OpExecInfoPtr &op_run_info, std::vector<int> *tensors_mask,
                          std::vector<tensor::TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(op_run_info);
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
  for (size_t index = 0; index < input_num; ++index) {
    // convert const input to attr
    if (reg_exist &&
        RunOpConvertConstInputToAttr(op_run_info->op_inputs[index], index, op_prim, reg.GetConstInputAttrInfo())) {
      continue;
    }
    // convert const and tuple input to tensor
    int tensor_mask = py::cast<int>(op_run_info->inputs_mask[index]);
    ConvertPyObjectToTensor(op_run_info->op_inputs[index], op_prim, input_tensors, &tensor_mask);
    // mark tensors, data : 0, weight : 1, valuenode: 2
    std::vector<int> new_mask(input_tensors->size() - tensors_mask->size(), tensor_mask);
    tensors_mask->insert(tensors_mask->end(), new_mask.begin(), new_mask.end());
  }
}

void EraseValueNodeTensor(const std::vector<int> &tensors_mask, std::vector<tensor::TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  if (input_tensors->size() != tensors_mask.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors->size() << " should be equal to tensors mask size "
                      << tensors_mask.size();
  }
  std::vector<tensor::TensorPtr> new_input_tensors;
  for (size_t index = 0; index < tensors_mask.size(); ++index) {
    if (tensors_mask[index] != kValueNodeTensorMask) {
      new_input_tensors.push_back(input_tensors->at(index));
    }
  }
  *input_tensors = new_input_tensors;
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

  if (session == nullptr) {
    session = session::SessionFactory::Get().Create(device_target);
  }
  MS_EXCEPTION_IF_NULL(session);
  session->Init(ms_context->device_id());

  std::vector<tensor::TensorPtr> input_tensors;
  std::vector<int> tensors_mask;
  ConstructInputTensor(op_exec_info, &tensors_mask, &input_tensors);
  // get graph info for checking it whether existing in the cache
  std::string graph_info = GetSingleOpGraphInfo(op_exec_info, input_tensors);
  session->BuildOp(*op_exec_info, graph_info, input_tensors, tensors_mask);
  EraseValueNodeTensor(tensors_mask, &input_tensors);
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

AnfNodePtr PynativeExecutor::MakeCNode(const OpExecInfoPtr &op_exec_info, const py::args &args, const py::tuple &out) {
  if (!grad_flag_ || graph_info_map_.size() == 0) {
    return nullptr;
  }
  std::vector<AnfNodePtr> inputs;
  auto prim = op_exec_info->py_primitive;
  inputs.push_back(NewValueNode(prim));
  py::tuple op_masks = op_exec_info->inputs_mask;
  py::list op_args = args[PY_INPUTS];
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < op_args.size(); i++) {
    auto node = GetInput(op_args[i], op_masks[i]);
    args_spec_list.push_back(node->abstract());
    inputs.push_back(node);
  }

  auto cnode = curr_g_->NewCNode(inputs);
  MS_LOG(DEBUG) << "MakeCnode set node " << cnode->DebugString();
  py::object out_real = out;
  if (out.size() == 1) {
    MS_LOG(DEBUG) << "MakeCnode out size is one.";
    out_real = out[0];
  }
  std::string obj_id = GetId(out_real);
  if (py::isinstance<py::tuple>(out_real)) {
    auto value = py::cast<py::tuple>(out_real);
    if (value.size() > 1) {
      for (int i = 0; i < static_cast<int>(value.size()); i++) {
        auto value_id = GetId(value[i]);
        set_obj_node_map(curr_g_, value_id, cnode, i);
      }
    }
  }
  set_obj_node_map(curr_g_, obj_id, cnode);
  set_pyobj(curr_g_, obj_id);
  return cnode;
}

AnfNodePtr PynativeExecutor::GetObjNode(const py::object &obj) {
  auto &out = graph_info_map_[curr_g_].obj_node_map[GetId(obj)];
  if (out.second == -1) {
    return out.first;
  }
  std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), out.first,
                                                NewValueNode(out.second)};
  return curr_g_->NewCNode(tuple_get_item_inputs);
}

py::tuple RunOp(const OpExecInfoPtr &op_exec_info, const py::args &args) {
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
  PynativeStatusCode status = PYNATIVE_UNKNOWN_STATE;
  // returns a null py::tuple on error
  py::tuple err_ret(0);
  py::object result = RunOpWithBackendPolicy(backend_policy, op_exec_info, &status);
  if (status != PYNATIVE_SUCCESS) {
    MS_LOG(ERROR) << "Failed to run " << op_exec_info->op_name;
    return err_ret;
  }

  auto node = PynativeExecutor::GetInstance()->MakeCNode(op_exec_info, args, result);
  if (node != nullptr) {
    node->set_abstract(op_exec_info->abstract);
    MS_LOG(DEBUG) << "RunOp MakeCnode,new node is: " << node->DebugString();
  }
  MS_LOG(DEBUG) << "RunOp end";
  return result;
}

py::tuple RunOp(const py::args &args) {
  MS_LOG(DEBUG) << "RunOp start" << args.size();
  OpExecInfoPtr op_exec_info = GenerateOpExecInfo(args);
  MS_EXCEPTION_IF_NULL(op_exec_info);
  if (op_exec_info->abstract != nullptr) {
    py::dict output = abstract::ConvertAbstractToPython(op_exec_info->abstract);
    if (!output["value"].is_none()) {
      py::tuple value_ret(1);
      value_ret[0] = output["value"];
      return value_ret;
    }
    if (py::hasattr(op_exec_info->py_primitive->GetPyObj(), "const_value")) {
      py::tuple value_ret(1);
      value_ret[0] = "";
      return value_ret;
    }
  }
  return RunOp(op_exec_info, args);
}

void ClearPyNativeSession() { session = nullptr; }

PynativeExecutor::~PynativeExecutor() { ClearRes(); }

PynativeExecutor::PynativeExecutor() { grad_flag_ = false; }

void PynativeExecutor::NewGraph(const py::object &cell, const py::args &args) {
  auto cell_id = GetId(cell);
  if (cell_graph_map_.count(cell_id) != 0) {
    MS_LOG(DEBUG) << "Newgraph already compiled";
    return;
  }

  auto g = std::make_shared<FuncGraph>();

  if (top_g_ == nullptr) {
    top_g_ = curr_g_ = g;
    df_builder_ = std::make_shared<FuncGraph>();
    MS_LOG(DEBUG) << "First new graph" << top_g_.get();
    Pushp();
  } else {
    Pushp();
    curr_g_ = g;
  }
  if (graph_info_map_.count(g) == 0) {
    graph_info_map_[g] = GraphInfo();
  }
  for (size_t i = 0; i < args.size(); i++) {
    auto new_param = g->add_parameter();
    std::string param_obj = GetId(args[i]);
    graph_info_map_[g].param_map[param_obj] = new_param;
  }
}

AnfNodePtr PynativeExecutor::GetInput(const py::object &obj, const py::object &op_mask) {
  AnfNodePtr node = nullptr;
  std::string obj_id = GetId(obj);

  if (op_mask != nullptr && py::cast<bool>(op_mask)) {
    MS_LOG(DEBUG) << "Topgraph free parameter";
    // get the parameter name from parameter object
    auto name_attr = mindspore::parse::python_adapter::GetPyObjAttr(obj, "name");
    if (py::isinstance<py::none>(name_attr)) {
      MS_LOG(EXCEPTION) << "Parameter object should have name attribute";
    }
    std::string param_name = py::cast<std::string>(name_attr);
    if (graph_info_map_[df_builder_].param_map.count(obj_id) == 0) {
      auto free_param = df_builder_->add_parameter();
      free_param->set_name(param_name);
      auto free_param_new = std::make_shared<ParamValuePy>(obj);
      free_param->set_default_param(free_param_new);
      free_param->debug_info()->set_name(param_name);
      MS_LOG(DEBUG) << "Top graph set free parameter " << obj_id;
      graph_info_map_[df_builder_].param_map[obj_id] = free_param;
      return free_param;
    }
    return graph_info_map_[df_builder_].param_map[obj_id];
  }

  // if input is graph output
  if (graph_info_map_[curr_g_].param_map.count(obj_id) != 0) {
    // op(x, y)
    node = graph_info_map_[curr_g_].param_map[obj_id];
  } else if (graph_info_map_[curr_g_].obj_node_map.count(obj_id) != 0) {
    // out = op(op1(x, y))
    // out = op(cell1(x, y))
    // out = op(cell1(x, y)[0])
    node = GetObjNode(obj);
  } else if (py::isinstance<py::tuple>(obj)) {
    // out = op((x, y))
    // out = cell((x, y))
    std::vector<AnfNodePtr> args;
    args.push_back(NewValueNode(prim::kPrimMakeTuple));

    auto tuple = obj.cast<py::tuple>();
    auto tuple_size = static_cast<int>(tuple.size());
    for (int i = 0; i < tuple_size; i++) {
      args.push_back(GetInput(tuple[i], py::object()));
    }
    auto cnode = curr_g_->NewCNode(args);
    set_obj_node_map(curr_g_, GetId(obj), cnode);
    node = cnode;
  } else {
    // out = op(x, 1)
    ValuePtr converted_ret = nullptr;
    parse::ConvertData(obj, &converted_ret);
    node = NewValueNode(converted_ret);
    set_obj_node_map(curr_g_, obj_id, node);
  }

  MS_LOG(DEBUG) << "Now getinput " << py::str(obj) << " node " << node->ToString();
  return node;
}

void PynativeExecutor::Pushp() { graph_p_.push(curr_g_); }

void PynativeExecutor::Popp() {
  if (graph_p_.empty()) {
    MS_LOG(EXCEPTION) << "Stack graph_p_ is empty";
  }
  curr_g_ = graph_p_.top();
  graph_p_.pop();
}

void PynativeExecutor::EndGraph(const py::object &cell, const py::object &out, const py::args &args) {
  auto cell_id = GetId(cell);
  if (cell_graph_map_.count(cell_id) != 0) {
    MS_LOG(DEBUG) << "Endgraph already compiled";
    return;
  }
  cell_graph_map_[cell_id] = curr_g_;
  auto out_id = GetId(out);
  if (!graph_info_map_[curr_g_].obj_node_map.count(out_id) && !graph_info_map_[curr_g_].param_map.count(out_id)) {
    // cell construct return x, y
    if (py::isinstance<py::tuple>(out)) {
      std::vector<AnfNodePtr> args;
      args.push_back(NewValueNode(prim::kPrimMakeTuple));

      auto tuple = out.cast<py::tuple>();
      MS_LOG(DEBUG) << "End graph start tuple size" << tuple.size();
      auto tuple_size = static_cast<int>(tuple.size());
      auto cnode = curr_g_->NewCNode(args);
      for (int i = 0; i < tuple_size; i++) {
        args.push_back(GetInput(tuple[i], py::object()));
        set_obj_node_map(curr_g_, GetId(tuple[i]), cnode, i);
      }
      cnode->set_inputs(args);
      set_obj_node_map(curr_g_, out_id, cnode);
    } else {
      MS_LOG(ERROR) << "Graph has no this out: " << out_id;
      return;
    }
  }
  EndGraphByOutId(out_id, cell, out, args);
}

void PynativeExecutor::EndGraphByOutId(const std::string &out_id, const py::object &cell, const py::object &out,
                                       const py::args &args) {
  AnfNodePtr output_node;
  if (graph_info_map_[curr_g_].param_map.count(out_id)) {
    output_node = graph_info_map_[curr_g_].param_map[out_id];
  } else {
    output_node = GetObjNode(out);
  }
  curr_g_->set_output(output_node);
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(curr_g_));
  MS_LOG(DEBUG) << "Current graph" << curr_g_->output()->DebugString();
  resource_->manager()->AddFuncGraph(curr_g_);
  // custom bprop debug
  if (py::hasattr(cell, parse::CUSTOM_BPROP_NAME)) {
    MS_LOG(DEBUG) << "Use cell custom bprop function.";
    FuncGraphPtr bprop_graph = parse::ConvertToBpropCut(cell);
    if (bprop_graph != nullptr) {
      (void)curr_g_->transforms().insert(std::make_pair(parse::CUSTOM_BPROP_NAME, FuncGraphTransform(bprop_graph)));
      (void)bprop_graph->transforms().insert(std::make_pair("primal", FuncGraphTransform(curr_g_)));
    }
  }
  auto newfg = ad::Grad(curr_g_, resource_, curr_g_ == top_g_);
  if (curr_g_ != top_g_) {
    Popp();
    for (size_t i = 0; i < args.size(); i++) {
      auto input = GetInput(args[i], py::object());
      inputs.push_back(input);
    }
    auto out_cnode = curr_g_->NewCNode(inputs);
    set_pyobj(curr_g_, GetId(cell));
    if (py::isinstance<py::tuple>(out)) {
      auto out_list = py::cast<py::tuple>(out);
      auto out_size = static_cast<int>(out_list.size());
      for (int i = 0; i < out_size; i++) {
        set_obj_node_map(curr_g_, GetId(out_list[i]), out_cnode, i);
      }
    }
    set_obj_node_map(curr_g_, GetId(out), out_cnode);
  } else {
    parse::ResolveFuncGraph(newfg, resource_);
    resource_->set_func_graph(newfg);
  }
}

std::vector<AnfNodePtr> PynativeExecutor::GetWeightsArgs(const py::object &weights) {
  std::vector<AnfNodePtr> w_args;
  if (py::hasattr(weights, "__parameter_tuple__")) {
    auto tuple = weights.cast<py::tuple>();
    MS_LOG(DEBUG) << "GradNet start weights tuple size" << tuple.size();
    w_args.push_back(NewValueNode(prim::kPrimMakeTuple));
    for (size_t it = 0; it < tuple.size(); ++it) {
      auto param = tuple[it];
      auto param_id = GetId(param);
      AnfNodePtr para_node = nullptr;
      if (graph_info_map_[df_builder_].param_map.count(param_id)) {
        para_node = graph_info_map_[df_builder_].param_map[param_id];

        AnfNodePtr value = parse::GetMixedPrecisionCastHelp(df_builder_, para_node);
        AnfNodePtr make_ref = NewValueNode(prim::kPrimMakeRef);
        auto refkey = std::make_shared<RefKey>(para_node->cast<ParameterPtr>()->name());
        AnfNodePtr ref_key_node = NewValueNode(refkey);
        AnfNodePtr ref_node = df_builder_->NewCNode({make_ref, ref_key_node, value, para_node});

        w_args.push_back(ref_node);
      }
    }
  } else {
    MS_LOG(EXCEPTION) << "training not paramter_tuple";
  }
  return w_args;
}

abstract::AbstractBasePtrList PynativeExecutor::GetArgsSpec(const py::args &args) {
  abstract::AbstractBasePtrList args_spec;
  std::size_t size = args.size();
  for (std::size_t i = 0; i < size; i++) {
    ValuePtr converted = nullptr;
    bool succ = parse::ConvertData(args[i], &converted);
    if (!succ) {
      MS_LOG(EXCEPTION) << "Args convert error";
    }
    bool broaden = true;
    auto abs = abstract::FromValue(converted, broaden);
    args_spec.push_back(abs);
    auto param_node = std::static_pointer_cast<Parameter>(df_builder_->parameters()[i]);
    param_node->set_abstract(abs);
  }

  for (const auto &param : df_builder_->parameters()) {
    auto param_node = std::static_pointer_cast<Parameter>(param);
    if (param_node->has_default()) {
      auto param_value = std::dynamic_pointer_cast<ParamValuePy>(param_node->default_param());
      AbstractBasePtr ptr = abstract::FromValue(parse::data_converter::PyDataToValue(param_value->value()), true);
      if (ptr == nullptr) {
        MS_LOG(EXCEPTION) << "Args convert error";
      }
      args_spec.push_back(ptr);
      param_node->set_abstract(ptr);
    }
  }

  return args_spec;
}

void PynativeExecutor::GradNet(const GradOperationPtr &grad, const py::object &cell, const py::object &weights,
                               const py::args &args) {
  MS_LOG(INFO) << "GradNet start" << args.size();

  std::size_t size = args.size();
  auto cell_id = GetId(cell);
  if (graph_map_.count(cell_id) != 0) {
    MS_LOG(DEBUG) << "GradNet already compiled";
    return;
  }
  MS_LOG(DEBUG) << "GradNet first compiled";
  std::vector<AnfNodePtr> new_params;
  for (size_t i = 0; i < size; i++) {
    ParameterPtr p = std::make_shared<Parameter>(df_builder_);
    new_params.push_back(p);
  }
  MS_LOG(DEBUG) << "GradNet start weight size" << df_builder_->parameters().size();
  new_params.insert(new_params.end(), df_builder_->parameters().begin(), df_builder_->parameters().end());
  df_builder_->set_parameters(new_params);
  resource_->manager()->SetParameters(df_builder_, new_params);

  std::vector<AnfNodePtr> w_args = GetWeightsArgs(weights);
  MS_EXCEPTION_IF_NULL(resource_->func_graph());
  auto g = GradGraph(resource_->func_graph(), grad, w_args, size);
  resource_->set_func_graph(g);

  // get the parameters items and add the value to args_spec
  abstract::AbstractBasePtrList args_spec = GetArgsSpec(args);
  MS_LOG(DEBUG) << "Args_spec size" << args_spec.size();

  resource_->set_args_spec(args_spec);
  MS_LOG(DEBUG) << "Start opt";

  // Create backend and session
  resource_->results()[pipeline::kBackend] = compile::CreateBackend();

  graph_map_[cell_id] = g;
  PynativeOptimizeAction(resource_);
  TaskEmitAction(resource_);
  ExecuteAction(resource_);
  resource_->Clean();
  ad::CleanRes();
  pipeline::ReclaimOptimizer();
}

void PynativeExecutor::Clear(const std::string &flag) {
  if (flag == "resource") {
    MS_LOG(INFO) << "Clear res";
    Clean();
    // Maybe exit in the pynative runing op, so need reset pynative flag.
    auto ms_context = MsContext::GetInstance();
    if (ms_context != nullptr) {
      ms_context->set_enable_pynative_infer(false);
    }
    return;
  }
  MS_LOG(INFO) << "Clear";
  top_g_ = nullptr;
  curr_g_ = nullptr;
  graph_info_map_.clear();
  std::stack<FuncGraphPtr>().swap(graph_p_);
}

void PynativeExecutor::Clean() {
  MS_LOG(INFO) << "Clean all res";
  Clear();
  grad_flag_ = false;
  df_builder_ = nullptr;
  ad::CleanRes();
  pipeline::ReclaimOptimizer();
}

void PynativeExecutor::ClearRes() {
  Clean();
  resource_.reset();
}

py::object PynativeExecutor::Run(const py::tuple &args, const py::object &phase) {
  VectorRef arg_list;
  pipeline::ProcessVmArgInner(args, resource_, &arg_list);
  if (resource_->results().find(pipeline::kOutput) == resource_->results().end() ||
      !resource_->results()[pipeline::kOutput].is<compile::VmEvalFuncPtr>()) {
    MS_LOG(EXCEPTION) << "Can't find run graph func for ";
  }
  compile::VmEvalFuncPtr run = resource_->results()[pipeline::kOutput].cast<compile::VmEvalFuncPtr>();
  if (run == nullptr) {
    MS_LOG(EXCEPTION) << "Can't find run graph func for ";
  }

  std::string backend = MsContext::GetInstance()->backend_policy();

  MS_LOG(DEBUG) << "Eval run" << backend;
  BaseRef value = (*run)(arg_list);
  MS_LOG(DEBUG) << "Run end" << value.ToString();
  return BaseRefToPyData(value);
}

FuncGraphPtr PynativeExecutor::GradGraph(FuncGraphPtr g, const GradOperationPtr &grad_op,
                                         const std::vector<AnfNodePtr> &weights, size_t arg_size) {
  auto nparam = top_g_->parameters().size();
  std::ostringstream ss;
  ss << "grad{" << nparam << "}";
  df_builder_->set_flags(FUNC_GRAPH_FLAG_CORE, true);
  df_builder_->debug_info()->set_name(ss.str());

  auto df = grad_op->GetGrad(NewValueNode(g), nullptr, top_g_->parameters(), weights);
  std::vector<AnfNodePtr> inputs = {NewValueNode(df)};
  for (size_t i = 0; i < arg_size; ++i) {
    inputs.push_back(df_builder_->parameters()[i]);
  }
  auto out = df_builder_->NewCNode(inputs);
  df_builder_->set_output(out);
  resource_->manager()->AddFuncGraph(df);
  resource_->manager()->AddFuncGraph(df_builder_);
  return df_builder_;
}

REGISTER_PYBIND_DEFINE(PynativeExecutor_, ([](const py::module *m) {
                         (void)py::class_<PynativeExecutor, std::shared_ptr<PynativeExecutor>>(*m, "PynativeExecutor_")
                           .def_static("get_instance", &PynativeExecutor::GetInstance, "PynativeExecutor get_instance.")
                           .def("new_graph", &PynativeExecutor::NewGraph, "pynative new a graph.")
                           .def("end_graph", &PynativeExecutor::EndGraph, "pynative end a graph.")
                           .def("grad_net", &PynativeExecutor::GradNet, "pynative grad graph.")
                           .def("clear", &PynativeExecutor::Clear, "pynative clear status.")
                           .def("__call__", &PynativeExecutor::Run, py::arg("args"), py::arg("phase") = py::str(""),
                                "Executor run function.")
                           .def("set_grad_flag", &PynativeExecutor::set_grad_flag, py::arg("flag") = py::bool_(false),
                                "Executor set grad flag.");
                       }));
}  // namespace pynative
}  // namespace mindspore
