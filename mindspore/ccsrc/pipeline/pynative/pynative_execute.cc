/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "pipeline/pynative/pynative_execute.h"

#include <typeinfo>
#include <set>
#include <memory>
#include <sstream>
#include <unordered_set>
#include <algorithm>

#include "debug/trace.h"
#include "debug/anf_ir_dump.h"
#include "pybind_api/api_register.h"
#include "pybind_api/pybind_patch.h"
#include "pybind_api/ir/tensor_py.h"
#include "ir/param_info.h"
#include "ir/anf.h"
#include "ir/cell.h"
#include "ir/tensor.h"
#include "utils/any.h"
#include "utils/utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "utils/context/context_extends.h"
#include "utils/config_manager.h"
#include "utils/convert_utils_py.h"
#include "utils/scoped_long_running.h"
#include "frontend/optimizer/ad/grad.h"
#include "frontend/optimizer/ad/prim_bprop_optimizer.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/do_signature.h"
#include "frontend/parallel/context.h"
#include "pipeline/jit/action.h"
#include "pipeline/jit/pass.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/parse/parse_dynamic.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "pipeline/jit/static_analysis/auto_monad.h"
#include "pipeline/jit/pipeline.h"
#include "pipeline/jit/resource.h"
#include "pipeline/pynative/base.h"
#include "backend/session/session_factory.h"
#include "backend/optimizer/common/const_input_to_attr_registry.h"
#include "backend/optimizer/common/helper.h"
#include "runtime/hardware/device_context_manager.h"
#include "vm/transform.h"

#ifdef ENABLE_GE
#include "pipeline/pynative/pynative_execute_ge.h"
#endif

using mindspore::tensor::TensorPy;

namespace mindspore::pynative {
PynativeExecutorPtr PynativeExecutor::executor_ = nullptr;
ForwardExecutorPtr PynativeExecutor::forward_executor_ = nullptr;
GradExecutorPtr PynativeExecutor::grad_executor_ = nullptr;
std::mutex PynativeExecutor::instance_lock_;

namespace {
const size_t PTR_LEN = 15;
const size_t ARG_SIZE = 2;
const size_t MAX_TOP_CELL_COUNTS = 20;

// primitive unable to infer value for constant input in PyNative mode
const std::set<std::string> kVmOperators = {"make_ref", "HookBackward", "InsertGradientOf", "stop_gradient",
                                            "mixed_precision_cast"};
const char kOpsFunctionModelName[] = "mindspore.ops.functional";
std::shared_ptr<session::SessionBasic> kSession = nullptr;
std::shared_ptr<compile::MindRTBackend> mind_rt_backend = nullptr;
PyObjectIdCache g_pyobj_id_cache;

template <typename T, typename... Args>
void PynativeExecutorTry(const std::function<void(T *ret, const Args &...)> &method, T *ret, const Args &... args) {
  const auto inst = PynativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(inst);
  MS_EXCEPTION_IF_NULL(method);
  try {
    method(ret, args...);
  } catch (const py::error_already_set &ex) {
    // print function call stack info before release
    std::ostringstream oss;
    trace::TraceGraphEval();
    trace::GetEvalStackInfo(oss);
    // call py::print to output function call stack to STDOUT, in case of output the log to file, the user can see
    // these info from screen, no need to open log file to find these info
    py::print(oss.str());
    MS_LOG(ERROR) << oss.str();
    inst->ClearRes();
    // re-throw this exception to Python interpreter to handle it
    throw(py::error_already_set(ex));
  } catch (const py::type_error &ex) {
    inst->ClearRes();
    throw py::type_error(ex);
  } catch (const py::value_error &ex) {
    inst->ClearRes();
    throw py::value_error(ex);
  } catch (const py::index_error &ex) {
    inst->ClearRes();
    throw py::index_error(ex);
  } catch (const py::name_error &ex) {
    inst->ClearRes();
    throw py::name_error(ex);
  } catch (const std::exception &ex) {
    inst->ClearRes();
    // re-throw this exception to Python interpreter to handle it
    throw(std::runtime_error(ex.what()));
  } catch (...) {
    inst->ClearRes();
    auto exception_type = abi::__cxa_current_exception_type();
    MS_EXCEPTION_IF_NULL(exception_type);
    std::string ex_name(exception_type->name());
    MS_LOG(EXCEPTION) << "Error occurred when compile graph. Exception name: " << ex_name;
  }
}

inline ValuePtr PyObjToValue(const py::object &obj) {
  ValuePtr converted_ret = parse::data_converter::PyDataToValue(obj);
  if (!converted_ret) {
    MS_LOG(EXCEPTION) << "Attribute convert error with type: " << std::string(py::str(obj));
  }
  return converted_ret;
}

std::string GetPyObjId(const py::handle &obj) {
  py::object out = parse::python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_MOD_GET_OBJ_ID, obj);
  if (py::isinstance<py::none>(out)) {
    MS_LOG(EXCEPTION) << "Get pyobj failed";
  }
  return out.cast<std::string>();
}

std::string GetId(const py::handle &obj) {
  if (py::isinstance<tensor::Tensor>(obj)) {
    auto tensor_ptr = py::cast<tensor::TensorPtr>(obj);
    return tensor_ptr->id();
  } else if (py::isinstance<mindspore::Type>(obj)) {
    auto type_ptr = py::cast<mindspore::TypePtr>(obj);
    return "type" + type_ptr->ToString();
  } else if (py::isinstance<py::str>(obj) || py::isinstance<py::int_>(obj) || py::isinstance<py::float_>(obj)) {
    return std::string(py::str(obj));
  } else if (py::isinstance<py::none>(obj)) {
    return "none";
  } else if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    auto p_list = py::cast<py::tuple>(obj);
    string prefix = py::isinstance<py::tuple>(obj) ? "tuple:" : "list";
    if (p_list.empty()) {
      prefix = "empty";
    } else {
      std::string key;
      for (size_t i = 0; i < p_list.size(); ++i) {
        key += std::string(py::str(GetId(p_list[i]))) + ":";
      }
      prefix += key;
    }
    return prefix;
  }

  if (py::isinstance<Cell>(obj) || py::isinstance<py::function>(obj)) {
    const auto &it = g_pyobj_id_cache.find(obj);
    if (it == g_pyobj_id_cache.end()) {
      auto &&id = GetPyObjId(obj);
      g_pyobj_id_cache[obj] = id;
      return std::move(id);
    } else {
      return it->second;
    }
  } else {
    return GetPyObjId(obj);
  }
}

void GetTypeIndex(const std::vector<SignatureEnumDType> &dtypes,
                  std::unordered_map<SignatureEnumDType, std::vector<size_t>> *type_indexes) {
  MS_EXCEPTION_IF_NULL(type_indexes);
  for (size_t i = 0; i < dtypes.size(); ++i) {
    auto it = type_indexes->find(dtypes[i]);
    if (it == type_indexes->end()) {
      (void)type_indexes->emplace(std::make_pair(dtypes[i], std::vector<size_t>{i}));
    } else {
      it->second.emplace_back(i);
    }
  }
}

TypeId JudgeMaxType(TypeId max_type, bool has_scalar_float32, bool has_scalar_int64, bool has_tensor_int8) {
  if (max_type == TypeId::kNumberTypeBool) {
    if (has_scalar_int64) {
      max_type = TypeId::kNumberTypeInt64;
    }
    if (has_scalar_float32) {
      max_type = TypeId::kNumberTypeFloat32;
    }
  }
  if (max_type != TypeId::kNumberTypeFloat16 && max_type != TypeId::kNumberTypeFloat32 &&
      max_type != TypeId::kNumberTypeFloat64 && max_type != TypeId::kTypeUnknown && has_scalar_float32) {
    max_type = TypeId::kNumberTypeFloat32;
  }
  if (max_type == TypeId::kNumberTypeUInt8 && has_tensor_int8) {
    max_type = TypeId::kNumberTypeInt16;
  }
  return max_type;
}

void GetDstType(const py::tuple &py_args,
                const std::unordered_map<SignatureEnumDType, std::vector<size_t>> &type_indexes,
                std::unordered_map<SignatureEnumDType, TypeId> *dst_type) {
  for (auto it = type_indexes.begin(); it != type_indexes.end(); (void)++it) {
    const auto &type = it->first;
    const auto &indexes = it->second;
    if (type == SignatureEnumDType::kDTypeEmptyDefaultValue || indexes.size() < ARG_SIZE) {
      continue;
    }
    size_t priority = 0;
    TypeId max_type = TypeId::kTypeUnknown;
    bool has_scalar_float32 = false;
    bool has_scalar_int64 = false;
    bool has_tensor_int8 = false;
    // Find the maximum priority of the same dtype
    for (size_t index : indexes) {
      if (index >= py_args.size()) {
        MS_LOG(EXCEPTION) << "The index " << index << " exceeds the size of py_args " << py_args.size();
      }
      const auto &obj = py_args[index];
      if (py::isinstance<py::float_>(obj)) {
        has_scalar_float32 = true;
      }
      if (!py::isinstance<py::bool_>(obj) && py::isinstance<py::int_>(obj)) {
        has_scalar_int64 = true;
      }
      if (py::isinstance<tensor::Tensor>(obj)) {
        auto arg = py::cast<tensor::TensorPtr>(obj);
        TypeId arg_type_id = arg->data_type();
        auto type_priority = prim::type_map.find(arg_type_id);
        if (type_priority == prim::type_map.end()) {
          continue;
        }
        if (arg_type_id == kNumberTypeInt8) {
          has_tensor_int8 = true;
        }
        if (type_priority->second > priority) {
          max_type = type_priority->first;
          priority = type_priority->second;
        }
      }
    }
    max_type = JudgeMaxType(max_type, has_scalar_float32, has_scalar_int64, has_tensor_int8);
    MS_EXCEPTION_IF_NULL(dst_type);
    (void)dst_type->emplace(std::make_pair(type, max_type));
  }
}

const std::string &TypeIdToMsTypeStr(const TypeId &type_id) {
  const auto &type_name = type_name_map.find(type_id);
  if (type_name == type_name_map.end()) {
    MS_LOG(EXCEPTION) << "For implicit type conversion, not support convert to the type: " << TypeIdToType(type_id);
  }
  return type_name->second;
}

bool GetSignatureType(const PrimitivePyPtr &prim, std::vector<SignatureEnumDType> *dtypes) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(dtypes);
  const auto &signature = prim->signatures();
  bool has_sig_dtype = false;
  (void)std::transform(signature.begin(), signature.end(), std::back_inserter(*dtypes),
                       [&has_sig_dtype](const Signature &sig) {
                         auto dtype = sig.dtype;
                         if (dtype != SignatureEnumDType::kDTypeEmptyDefaultValue) {
                           has_sig_dtype = true;
                         }
                         return dtype;
                       });
  return has_sig_dtype;
}

void PynativeInfer(const PrimitivePyPtr &prim, OpExecInfo *const op_exec_info,
                   const abstract::AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_LOG(DEBUG) << "Prim " << prim->name() << " input infer " << mindspore::ToString(args_spec_list);
  prim->BeginRecordAddAttr();
  auto eval_ret = EvalOnePrim(prim, args_spec_list);
  MS_EXCEPTION_IF_NULL(eval_ret);
  AbstractBasePtr infer_res = eval_ret->abstract();
  MS_EXCEPTION_IF_NULL(infer_res);
  prim->EndRecordAddAttr();
  MS_EXCEPTION_IF_NULL(op_exec_info);
  op_exec_info->abstract = infer_res;
  MS_EXCEPTION_IF_NULL(op_exec_info->abstract);
  MS_LOG(DEBUG) << "Prim " << prim->name() << " infer result " << op_exec_info->abstract->ToString();
}

void GetSingleOpGraphInfo(const OpExecInfoPtr &op_exec_info, const std::vector<tensor::TensorPtr> &input_tensors,
                          const std::vector<int64_t> &tensors_mask, std::string *graph_info_key) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(graph_info_key);
  auto &graph_info = *graph_info_key;
  if (input_tensors.size() != tensors_mask.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size() << " should be equal to tensors mask size "
                      << tensors_mask.size();
  }
  std::ostringstream buf;
  buf << op_exec_info->op_name;
  bool has_const_input = false;
  for (size_t index = 0; index < input_tensors.size(); ++index) {
    MS_EXCEPTION_IF_NULL(input_tensors[index]);
    buf << input_tensors[index]->shape();
    buf << input_tensors[index]->data_type();
    buf << input_tensors[index]->padding_type();
    // In the case of the same shape, but dtype and format are inconsistent
    auto tensor_addr = input_tensors[index]->device_address();
    if (tensor_addr != nullptr) {
      auto p_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor_addr);
      MS_EXCEPTION_IF_NULL(p_address);
      buf << p_address->type_id();
      buf << p_address->format();
    }
    // For constant input
    if (tensors_mask[index] == kValueNodeTensorMask) {
      has_const_input = true;
      auto dtype = input_tensors[index]->Dtype();
      MS_EXCEPTION_IF_NULL(dtype);
      if (dtype->type_id() == kNumberTypeInt64) {
        buf << *reinterpret_cast<int *>(input_tensors[index]->data_c());
      } else if (dtype->type_id() == kNumberTypeFloat32 || dtype->type_id() == kNumberTypeFloat16) {
        buf << *reinterpret_cast<float *>(input_tensors[index]->data_c());
      } else {
        MS_LOG(EXCEPTION) << "The dtype of the constant input is not int64 or float32!";
      }
    }
    buf << "_";
  }
  // The value of the attribute affects the operator selection
  const auto &op_prim = op_exec_info->py_primitive;
  MS_EXCEPTION_IF_NULL(op_prim);
  const auto &attr_map = op_prim->attrs();
  (void)std::for_each(attr_map.begin(), attr_map.end(),
                      [&buf](const auto &element) { buf << element.second->ToString(); });

  // Constant input affects output, operators like DropoutGenMask whose output is related to values of input when input
  // shapes are the same but values are different
  if (has_const_input) {
    buf << "_";
    auto abstr = op_exec_info->abstract;
    MS_EXCEPTION_IF_NULL(abstr);
    auto build_shape = abstr->BuildShape();
    MS_EXCEPTION_IF_NULL(build_shape);
    buf << build_shape->ToString();
    auto build_type = abstr->BuildType();
    MS_EXCEPTION_IF_NULL(build_type);
    buf << build_type->type_id();
  }
  graph_info = buf.str();
}

py::list FilterTensorArgs(const py::args &args, bool has_sens = false) {
  size_t size = args.size();
  if (size == 0 && has_sens) {
    MS_LOG(EXCEPTION) << "The size of args is 0, when the flag of sens is set to True";
  }
  py::list only_tensors;
  size_t forward_args_size = has_sens ? size - 1 : size;
  for (size_t i = 0; i < forward_args_size; ++i) {
    if (py::isinstance<tensor::Tensor>(args[i])) {
      only_tensors.append(args[i]);
    }
  }
  if (has_sens) {
    only_tensors.append(args[forward_args_size]);
  }
  return only_tensors;
}

bool RunOpConvertConstInputToAttr(const py::object &input_object, size_t input_index, const PrimitivePtr &op_prim,
                                  const std::unordered_set<size_t> &input_attrs) {
  MS_EXCEPTION_IF_NULL(op_prim);
  const auto &input_names_value = op_prim->GetAttr(kAttrInputNames);
  if (input_names_value == nullptr) {
    return false;
  }
  const auto &input_names_vec = GetValue<std::vector<std::string>>(input_names_value);
  if (input_index >= input_names_vec.size()) {
    MS_LOG(EXCEPTION) << "The input index: " << input_index << " is large than the input names vector size!";
  }

  if (input_attrs.find(input_index) != input_attrs.end()) {
    const auto &value = PyObjToValue(input_object);
    auto input_name = input_names_vec[input_index];
    op_prim->AddAttr(input_name, value);
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
    input_tensors->emplace_back(tensor);
  }
  op_prim->set_attr(kAttrDynInputSizes, MakeValue(std::vector<int64_t>{SizeToLong(tuple_inputs.size())}));
}

void ConvertValueTupleToTensor(const py::object &input_object, std::vector<tensor::TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  const auto &input_value = PyObjToValue(input_object);
  MS_EXCEPTION_IF_NULL(input_value);
  if (!input_value->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "The input object is not a value tuple!";
  }
  auto value_tuple = input_value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple);
  tensor::TensorPtr tensor_ptr = opt::CreateTupleTensor(value_tuple);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  input_tensors->emplace_back(tensor_ptr);
}

void ConvertMultiPyObjectToTensor(const py::object &input_object, const PrimitivePtr &op_prim,
                                  std::vector<tensor::TensorPtr> *input_tensors, int64_t *const tensor_mask) {
  MS_EXCEPTION_IF_NULL(op_prim);
  MS_EXCEPTION_IF_NULL(input_tensors);
  MS_EXCEPTION_IF_NULL(tensor_mask);

  if (!py::isinstance<py::tuple>(input_object)) {
    MS_LOG(EXCEPTION) << "The input should be a tuple!";
  }
  auto tuple_inputs = py::cast<py::tuple>(input_object);
  if (tuple_inputs.empty()) {
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
                             std::vector<tensor::TensorPtr> *input_tensors, int64_t *const tensor_mask) {
  MS_EXCEPTION_IF_NULL(op_prim);
  MS_EXCEPTION_IF_NULL(input_tensors);
  MS_EXCEPTION_IF_NULL(tensor_mask);
  tensor::TensorPtr tensor_ptr = nullptr;
  if (py::isinstance<tensor::Tensor>(input_object)) {
    tensor_ptr = py::cast<tensor::TensorPtr>(input_object);
  } else if (py::isinstance<py::float_>(input_object)) {
    double input_value = py::cast<py::float_>(input_object);
    tensor_ptr = std::make_shared<tensor::Tensor>(input_value, kFloat32);
    *tensor_mask = kValueNodeTensorMask;
  } else if (py::isinstance<py::int_>(input_object)) {
    tensor_ptr = std::make_shared<tensor::Tensor>(py::cast<int64_t>(input_object), kInt64);
    *tensor_mask = kValueNodeTensorMask;
  } else if (py::isinstance<py::array>(input_object)) {
    tensor_ptr = TensorPy::MakeTensor(py::cast<py::array>(input_object), nullptr);
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
  input_tensors->emplace_back(tensor_ptr);
}

void ConstructInputTensor(const OpExecInfoPtr &op_run_info, std::vector<int64_t> *tensors_mask,
                          std::vector<tensor::TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(tensors_mask);
  MS_EXCEPTION_IF_NULL(input_tensors);
  PrimitivePtr op_prim = op_run_info->py_primitive;
  // Checking whether attr conversion is needed.
  opt::ConstInputToAttrInfoRegister reg;
  bool reg_exist = opt::ConstInputToAttrInfoRegistry::Instance().GetRegisterByOpName(op_run_info->op_name, &reg);
  if (op_run_info->is_dynamic_shape &&
      dynamic_shape_const_input_to_attr.find(op_run_info->op_name) == dynamic_shape_const_input_to_attr.end()) {
    MS_LOG(DEBUG) << "current node is dynamic shape: " << op_run_info->op_name;
    reg_exist = false;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target != kCPUDevice && op_run_info->op_name == prim::kPrimEmbeddingLookup->name()) {
    reg_exist = false;
  }
  // Gather op needs converting const input to attr on GPU device
  if (device_target != kGPUDevice && op_run_info->op_name == prim::kPrimGatherD->name()) {
    reg_exist = false;
  }
  // Get input tensors
  MS_EXCEPTION_IF_NULL(op_prim);
  op_prim->BeginRecordAddAttr();
  size_t input_num = op_run_info->op_inputs.size();
  if (input_num != op_run_info->inputs_mask.size()) {
    MS_LOG(EXCEPTION) << "The op input size " << input_num << ", but the size of input mask "
                      << op_run_info->inputs_mask.size();
  }
  for (size_t index = 0; index < input_num; ++index) {
    // convert const input to attr
    if (reg_exist &&
        RunOpConvertConstInputToAttr(op_run_info->op_inputs[index], index, op_prim, reg.GetConstInputAttrInfo())) {
      continue;
    }
    // convert const and tuple input to tensor
    int64_t tensor_mask = op_run_info->inputs_mask[index];
    ConvertPyObjectToTensor(op_run_info->op_inputs[index], op_prim, input_tensors, &tensor_mask);
    // Mark tensors, common tensor data : 0, weight param: 1, valuenode(float_, int_): 2
    op_run_info->inputs_mask[index] = tensor_mask;
    std::vector<int64_t> new_mask(input_tensors->size() - tensors_mask->size(), tensor_mask);
    tensors_mask->insert(tensors_mask->end(), new_mask.begin(), new_mask.end());
  }
  op_prim->EndRecordAddAttr();
}

void ConvertAttrToUnifyMindIR(const OpExecInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &op_prim = op_run_info->py_primitive;
  MS_EXCEPTION_IF_NULL(op_prim);

  const auto &op_name = op_run_info->op_name;
  auto attrs = op_prim->attrs();
  for (auto attr : attrs) {
    bool converted = CheckAndConvertUtils::ConvertAttrValueToString(op_name, attr.first, &attr.second);
    if (converted) {
      op_prim->set_attr(attr.first, attr.second);
    }
    bool converted_ir_attr = CheckAndConvertUtils::CheckIrAttrtoOpAttr(op_name, attr.first, &attr.second);
    if (converted_ir_attr) {
      op_prim->set_attr(attr.first, attr.second);
    }
  }
}

size_t GetTupleSize(const py::tuple &args) {
  size_t count = 0;
  for (size_t i = 0; i < args.size(); i++) {
    if (py::isinstance<py::tuple>(args[i])) {
      count += GetTupleSize(args[i]);
    } else {
      count += 1;
    }
  }
  return count;
}

void ConvertTupleArg(py::tuple *res, size_t *const index, const py::tuple &arg) {
  MS_EXCEPTION_IF_NULL(res);
  MS_EXCEPTION_IF_NULL(index);
  auto res_size = res->size();
  for (size_t i = 0; i < arg.size(); i++) {
    if (py::isinstance<py::tuple>(arg[i])) {
      ConvertTupleArg(res, index, arg[i]);
    } else {
      if (*index >= res_size) {
        MS_LOG(EXCEPTION) << "Convert tuple error, index is greater than tuple size, index " << (*index)
                          << ", tuple size " << res_size;
      }
      (*res)[(*index)++] = arg[i];
    }
  }
}

py::tuple ConvertArgs(const py::tuple &args) {
  size_t tuple_size = GetTupleSize(args);
  py::tuple res(tuple_size);
  size_t index = 0;
  for (size_t i = 0; i < args.size(); i++) {
    if (py::isinstance<py::tuple>(args[i])) {
      ConvertTupleArg(&res, &index, args[i]);
    } else {
      if (index >= tuple_size) {
        MS_LOG(EXCEPTION) << "Convert error, index is greater than tuple size, index " << index << ", tuple size "
                          << tuple_size;
      }
      res[index++] = args[i];
    }
  }
  return res;
}

void ResetTopCellInfo(const TopCellInfoPtr &top_cell, const py::args &args) {
  MS_EXCEPTION_IF_NULL(top_cell);
  top_cell->set_op_num(0);
  top_cell->all_op_info().clear();
  top_cell->set_forward_already_run(true);
  std::string input_args_id;
  for (size_t i = 0; i < args.size(); ++i) {
    input_args_id += GetId(args[i]) + "_";
  }
  top_cell->set_input_args_id(input_args_id);
}

void RunReplace(const CNodePtr &added_make_tuple, const std::vector<tensor::TensorPtr> &total_output_tensors,
                const FuncGraphPtr &grad_graph) {
  MS_EXCEPTION_IF_NULL(grad_graph);
  MS_EXCEPTION_IF_NULL(added_make_tuple);
  size_t index = 0;
  for (size_t i = 1; i < added_make_tuple->size(); ++i) {
    const auto &input_i = added_make_tuple->input(i);
    MS_EXCEPTION_IF_NULL(input_i);
    auto cnode = input_i->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    MS_LOG(DEBUG) << "Replace new output tensors for cnode: " << cnode->DebugString();
    auto output_vnode = cnode->forward().first;
    MS_EXCEPTION_IF_NULL(output_vnode);
    grad_graph->AddValueNode(output_vnode);
    MS_LOG(DEBUG) << "Original output value node: " << output_vnode << " info: " << output_vnode->ToString();
    size_t output_num = AnfAlgo::GetOutputTensorNum(cnode);
    if (index + output_num > total_output_tensors.size()) {
      MS_LOG(EXCEPTION) << "The size of total_output_tensors: " << total_output_tensors.size()
                        << ", but the current index: " << index << ", output num: " << output_num;
    }
    // Get new tensors.
    std::vector<ValuePtr> new_values;
    for (size_t j = index; j < index + output_num; ++j) {
      new_values.push_back(total_output_tensors[j]);
    }
    index = index + output_num;
    // Replace new tensors.
    if (output_num == 1) {
      output_vnode->set_value(new_values[0]);
    } else if (output_num > 1) {
      output_vnode->set_value(std::make_shared<ValueTuple>(new_values));
    } else {
      MS_LOG(EXCEPTION) << "The output value of forward cnode is empty, forward cnode info: " << cnode->ToString();
    }
    MS_LOG(DEBUG) << "New output value node: " << output_vnode << " info: " << output_vnode->ToString();
  }
  // Save op info with new tensors for current running ms_function func graph.
  if (index != total_output_tensors.size()) {
    MS_LOG(EXCEPTION) << "The index: " << index
                      << " should be equal to the size of total_output_tensors: " << total_output_tensors.size();
  }
}

void ReplaceNewTensorsInGradGraph(const TopCellInfoPtr &top_cell, const OpExecInfoPtr &op_exec_info,
                                  const ValuePtr &added_out, const FuncGraphPtr &ms_func_graph,
                                  const FuncGraphPtr &grad_graph) {
  MS_EXCEPTION_IF_NULL(top_cell);
  MS_EXCEPTION_IF_NULL(grad_graph);
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  // Get added forward nodes.
  auto merge_node = ms_func_graph->output();
  MS_EXCEPTION_IF_NULL(merge_node);
  auto merge_make_tuple = merge_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(merge_make_tuple);
  constexpr size_t merge_output_size = 3;
  if (merge_make_tuple->size() != merge_output_size) {
    MS_LOG(EXCEPTION) << "The input size of merge make tuple node should be 3, but it is: " << merge_make_tuple->size();
  }
  constexpr size_t added_output_index = 2;
  const auto &added_forward_node = merge_make_tuple->input(added_output_index);
  MS_EXCEPTION_IF_NULL(added_forward_node);
  if (added_forward_node->isa<ValueNode>()) {
    MS_LOG(DEBUG) << "The added forward output node is value node: " << added_forward_node->DebugString();
    std::vector<tensor::TensorPtr> total_output_tensors;
    TensorValueToTensor(added_out, &total_output_tensors);
    top_cell->set_op_info_with_ms_func_forward_tensors(op_exec_info->op_info, total_output_tensors);
    return;
  }
  // Replace new output tensors for forward nodes, it will also work in grad graph with same value node.
  auto added_make_tuple = added_forward_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(added_make_tuple);
  MS_LOG(DEBUG) << "The added forward make tuple node info: " << added_make_tuple->DebugString();
  std::vector<tensor::TensorPtr> total_output_tensors;
  TensorValueToTensor(added_out, &total_output_tensors);
  RunReplace(added_make_tuple, total_output_tensors, grad_graph);
  top_cell->set_op_info_with_ms_func_forward_tensors(op_exec_info->op_info, total_output_tensors);
}

void SaveOpInfo(const TopCellInfoPtr &top_cell, const std::string &op_info,
                const std::vector<tensor::TensorPtr> &op_out_tensors) {
  MS_EXCEPTION_IF_NULL(top_cell);
  auto &op_info_with_tensor_id = top_cell->op_info_with_tensor_id();
  if (op_info_with_tensor_id.find(op_info) != op_info_with_tensor_id.end()) {
    MS_LOG(EXCEPTION) << "Top cell: " << top_cell.get() << " records op info with tensor id, but get op info "
                      << op_info << " in op_info_with_tensor_id map";
  }
  // Record the relationship between the forward op and its output tensor id
  std::for_each(op_out_tensors.begin(), op_out_tensors.end(),
                [&op_info_with_tensor_id, &op_info](const tensor::TensorPtr &tensor) {
                  op_info_with_tensor_id[op_info].emplace_back(tensor->id());
                });
}

void UpdateTensorInfo(const tensor::TensorPtr &new_tensor, const std::vector<tensor::TensorPtr> &pre_tensors) {
  MS_EXCEPTION_IF_NULL(new_tensor);
  if (pre_tensors.empty()) {
    MS_LOG(EXCEPTION) << "The size of pre tensors is empty.";
  }

  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  for (auto &pre_tensor : pre_tensors) {
    MS_EXCEPTION_IF_NULL(pre_tensor);
    MS_LOG(DEBUG) << "Replace Old tensor " << pre_tensor.get() << " id " << pre_tensor->id()
                  << " device_address: " << pre_tensor->device_address() << " shape and type "
                  << pre_tensor->GetShapeAndDataTypeInfo() << " with New tensor " << new_tensor.get() << " id "
                  << new_tensor->id() << " device_address " << new_tensor->device_address() << " shape and dtype "
                  << new_tensor->GetShapeAndDataTypeInfo();
    pre_tensor->set_shape(new_tensor->shape());
    pre_tensor->set_data_type(new_tensor->data_type());
    if (device_target != kCPUDevice) {
      pre_tensor->set_device_address(new_tensor->device_address());
      continue;
    }
    // Replace data in device address when run in CPU device.
    if (pre_tensor->device_address() != nullptr) {
      auto old_device_address = std::dynamic_pointer_cast<device::DeviceAddress>(pre_tensor->device_address());
      auto new_device_address = std::dynamic_pointer_cast<device::DeviceAddress>(new_tensor->device_address());
      MS_EXCEPTION_IF_NULL(old_device_address);
      auto old_ptr = old_device_address->GetMutablePtr();
      MS_EXCEPTION_IF_NULL(old_ptr);
      MS_EXCEPTION_IF_NULL(new_device_address);
      auto new_ptr = new_device_address->GetPtr();
      MS_EXCEPTION_IF_NULL(new_ptr);
      auto ret = memcpy_s(old_ptr, old_device_address->GetSize(), new_ptr, new_device_address->GetSize());
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "Memory copy failed. ret: " << ret;
      }
    }
  }
}

void CheckPyNativeContext() {
  const auto &parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  const auto &ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &parallel_mode = parallel_context->parallel_mode();
  if (parallel_mode != parallel::STAND_ALONE && parallel_mode != parallel::DATA_PARALLEL &&
      ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    MS_LOG(EXCEPTION) << "PyNative Only support STAND_ALONE and DATA_PARALLEL, but got:" << parallel_mode;
  }
}

py::object GetDstType(const TypeId &type_id) {
  ValuePtr value = nullptr;
  if (type_id == kNumberTypeFloat16) {
    value = std::make_shared<Float>(16);
  } else if (type_id == kNumberTypeFloat32) {
    value = std::make_shared<Float>(32);
  } else if (type_id == kNumberTypeFloat64) {
    value = std::make_shared<Float>(64);
  } else if (type_id == kNumberTypeBool) {
    value = std::make_shared<Bool>();
  } else if (type_id == kNumberTypeInt8) {
    value = std::make_shared<Int>(8);
  } else if (type_id == kNumberTypeUInt8) {
    value = std::make_shared<UInt>(8);
  } else if (type_id == kNumberTypeInt16) {
    value = std::make_shared<Int>(16);
  } else if (type_id == kNumberTypeInt32) {
    value = std::make_shared<Int>(32);
  } else if (type_id == kNumberTypeInt64) {
    value = std::make_shared<Int>(64);
  } else {
    MS_LOG(EXCEPTION) << "Not support dst type";
  }
  MS_EXCEPTION_IF_NULL(value);
  return py::cast(value);
}
}  // namespace

py::object RealRunOp(const py::args &args) {
  CheckPyNativeContext();
  const auto &executor = PynativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor);
  OpExecInfoPtr op_exec_info = executor->forward_executor()->GenerateOpExecInfo(args);
  MS_EXCEPTION_IF_NULL(op_exec_info);
  py::object ret = py::none();
  PynativeExecutorTry(executor->forward_executor()->RunOpS, &ret, op_exec_info);
  return ret;
}

GradExecutorPtr ForwardExecutor::grad() const {
  auto grad_executor = grad_executor_.lock();
  MS_EXCEPTION_IF_NULL(grad_executor);
  return grad_executor;
}

bool TopCellInfo::IsSubCell(const std::string &cell_id) const {
  if (sub_cell_list_.empty()) {
    MS_LOG(DEBUG) << "The sub cell list is empty, there is no sub cell";
    return false;
  }
  if (sub_cell_list_.find(cell_id) != sub_cell_list_.end()) {
    return true;
  }
  return false;
}

void TopCellInfo::ClearDeviceMemory() {
  MS_LOG(DEBUG) << "Clear device memory in value nodes of bprop graph, top cell: " << cell_id_;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_target == kCPUDevice) {
    MS_LOG(DEBUG) << "No need to clear device address when run in CPU device.";
    return;
  }
  k_pynative_cell_ptr_ = nullptr;
  // Get all tensors obj in value node of running graph
  std::vector<tensor::TensorPtr> tensors_in_bprop_graph;
  MS_EXCEPTION_IF_NULL(resource_);
  const auto &bprop_graph = resource_->func_graph();
  MS_EXCEPTION_IF_NULL(bprop_graph);
  const auto &value_node_list = bprop_graph->value_nodes();
  for (const auto &elem : value_node_list) {
    auto &node = elem.first;
    MS_EXCEPTION_IF_NULL(node);
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    TensorValueToTensor(value_node->value(), &tensors_in_bprop_graph);
  }
  for (const auto &tensor : tensors_in_bprop_graph) {
    MS_EXCEPTION_IF_NULL(tensor);
    MS_LOG(DEBUG) << "Clear device address for tensor: " << tensor->ToString();
    tensor->set_device_address(nullptr);
  }
}

void TopCellInfo::Clear() {
  MS_LOG(DEBUG) << "Clear top cell info. Cell id " << cell_id_;
  op_num_ = 0;
  is_dynamic_ = false;
  vm_compiled_ = false;
  ms_function_flag_ = false;
  is_init_kpynative_ = false;
  need_compile_graph_ = false;
  forward_already_run_ = false;
  input_args_id_.clear();
  all_op_info_.clear();
  resource_ = nullptr;
  df_builder_ = nullptr;
  k_pynative_cell_ptr_ = nullptr;
  graph_info_map_.clear();
  sub_cell_list_.clear();
  op_info_with_tensor_id_.clear();
  tensor_id_with_tensor_object_.clear();
  op_info_with_ms_func_forward_tensors_.clear();
}

void ForwardExecutor::RunOpInner(py::object *ret, const OpExecInfoPtr &op_exec_info) {
  MS_EXCEPTION_IF_NULL(ret);
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_LOG(DEBUG) << "RunOp name: " << op_exec_info->op_name;
  if (op_exec_info->op_name == prim::kPrimMixedPrecisionCast->name()) {
    RunMixedPrecisionCastOp(op_exec_info, ret);
    return;
  }

  // 1.Set cast for inputs
  SetCastForInputs(op_exec_info);
  // 2.Construct graph, first step abs will update by node
  auto cnode = ConstructForwardGraph(op_exec_info);
  // 3.Get inputs abstract
  abstract::AbstractBasePtrList args_spec_list;
  GetInputsArgsSpec(op_exec_info, &args_spec_list);
  // 4.Get output abstract
  bool prim_cache_hit = false;
  GetOpOutputAbstract(op_exec_info, args_spec_list, &prim_cache_hit);
  // 5.Get output
  GetOpOutput(op_exec_info, args_spec_list, cnode, prim_cache_hit, ret);
}

OpExecInfoPtr ForwardExecutor::GenerateOpExecInfo(const py::args &args) {
  if (args.size() != PY_ARGS_NUM) {
    MS_LOG(EXCEPTION) << "Three args are needed by RunOp";
  }
  const auto &op_exec_info = std::make_shared<OpExecInfo>();
  const auto &op_name = py::cast<std::string>(args[PY_NAME]);
  op_exec_info->op_name = op_name;

  const auto &adapter = py::cast<PrimitivePyAdapterPtr>(args[PY_PRIM]);
  MS_EXCEPTION_IF_NULL(adapter);
  auto prim = adapter->attached_primitive();
  if (prim == nullptr) {
    prim = std::make_shared<PrimitivePy>(args[PY_PRIM], adapter);
    adapter->set_attached_primitive(prim);
  }

  if (!prim->HasPyObj()) {
    MS_LOG(EXCEPTION) << "Pyobj is empty";
  }
  op_exec_info->py_primitive = prim;
  op_exec_info->op_inputs = args[PY_INPUTS];
  op_exec_info->lazy_build = lazy_build_;
  return op_exec_info;
}

void ForwardExecutor::SetCastForInputs(const OpExecInfoPtr &op_exec_info) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  // No need cast self
  if (op_exec_info->op_name == prim::kPrimCast->name()) {
    return;
  }

  // Mixed precision conversion tensors which has cast dtype
  SetTensorMixPrecisionCast(op_exec_info);
  // Implicit transform
  SetImplicitCast(op_exec_info);
}

void ForwardExecutor::RunMixedPrecisionCastOp(const OpExecInfoPtr &op_exec_info, py::object *ret) {
  py::tuple res = RunOpWithInitBackendPolicy(op_exec_info);
  MS_EXCEPTION_IF_NULL(ret);
  if (res.size() == 1) {
    *ret = res[0];
    return;
  }
  *ret = std::move(res);
}

void ForwardExecutor::SetNonCostantValueAbs(const AbstractBasePtr &abs, size_t i, const std::string &id) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractTensor>()) {
    abs->set_value(kAnyValue);
  } else if (abs->isa<abstract::AbstractTuple>() || abs->isa<abstract::AbstractList>()) {
    const auto &abs_seq = abs->cast<abstract::AbstractSequeuePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    for (auto &item : abs_seq->elements()) {
      MS_EXCEPTION_IF_NULL(item);
      if (item->isa<abstract::AbstractTensor>()) {
        item->set_value(kAnyValue);
      }
    }
  }
  MS_LOG(DEBUG) << "Set " << i << "th abs " << abs->ToString();
  node_abs_map_[id] = abs;
}

void ForwardExecutor::GetInputsArgsSpec(const OpExecInfoPtr &op_exec_info,
                                        abstract::AbstractBasePtrList *args_spec_list) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(args_spec_list);
  auto prim = op_exec_info->py_primitive;
  MS_EXCEPTION_IF_NULL(prim);
  for (size_t i = 0; i < op_exec_info->op_inputs.size(); i++) {
    abstract::AbstractBasePtr abs = nullptr;
    const auto &obj = op_exec_info->op_inputs[i];
    const auto &id = GetId(obj);
    MS_LOG(DEBUG) << "Set input abs " << id;
    auto it = node_abs_map_.find(id);
    if (it != node_abs_map_.end()) {
      abs = it->second;
    }
    const auto const_input_index = prim->get_const_input_indexes();
    bool have_const_input = !const_input_index.empty();
    bool is_const_prim = prim->is_const_prim();
    MS_LOG(DEBUG) << prim->ToString() << " abs is nullptr " << (abs == nullptr) << " is_const_value "
                  << prim->is_const_prim();
    bool is_const_input =
      have_const_input && std::find(const_input_index.begin(), const_input_index.end(), i) != const_input_index.end();
    if (abs == nullptr || is_const_prim || is_const_input) {
      abs = PyObjToValue(obj)->ToAbstract();
      if (!is_const_prim && !is_const_input) {
        SetNonCostantValueAbs(abs, i, id);
      }
    }
    args_spec_list->emplace_back(abs);
  }
}

CNodePtr ForwardExecutor::ConstructForwardGraph(const OpExecInfoPtr &op_exec_info) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  auto prim = op_exec_info->py_primitive;
  std::vector<AnfNodePtr> inputs;
  std::vector<int64_t> op_masks;
  inputs.emplace_back(NewValueNode(prim));
  for (size_t i = 0; i < op_exec_info->op_inputs.size(); i++) {
    const auto &obj = op_exec_info->op_inputs[i];
    bool op_mask = false;
    tensor::MetaTensorPtr meta_tensor = nullptr;
    if (py::isinstance<tensor::MetaTensor>(obj)) {
      meta_tensor = obj.cast<tensor::MetaTensorPtr>();
      if (meta_tensor) {
        op_mask = meta_tensor->is_parameter();
      }
    }
    MS_LOG(DEBUG) << "Args i " << i << ", op mask " << op_mask;
    op_masks.emplace_back(static_cast<int64_t>(op_mask));

    // Construct grad graph
    if (grad()->need_construct_graph()) {
      const auto &id = GetId(obj);
      AnfNodePtr input_node = nullptr;
      input_node = grad()->GetInput(obj, op_mask);
      // update abstract
      if (input_node != nullptr) {
        if (input_node->abstract() != nullptr) {
          abstract::AbstractBasePtr abs = input_node->abstract();
          node_abs_map_[id] = abs;
        }
        inputs.emplace_back(input_node);
      }
    }
  }
  op_exec_info->inputs_mask = std::move(op_masks);
  CNodePtr cnode = nullptr;
  if (grad()->need_construct_graph()) {
    cnode = grad()->curr_g()->NewCNodeInOrder(inputs);
    MS_LOG(DEBUG) << "Make CNode for " << op_exec_info->op_name << ", new cnode is " << cnode->DebugString();
  }
  return cnode;
}

void ForwardExecutor::GetOpOutputAbstract(const OpExecInfoPtr &op_exec_info,
                                          const abstract::AbstractBasePtrList &args_spec_list, bool *prim_cache_hit) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(prim_cache_hit);
  auto op_name = op_exec_info->op_name;
  auto prim = op_exec_info->py_primitive;
  MS_EXCEPTION_IF_NULL(prim);

  AbsCacheKey key{prim->name(), prim->Hash(), prim->attrs()};
  auto temp = prim_abs_list_.find(key);
  if (temp != prim_abs_list_.end()) {
    MS_LOG(DEBUG) << "Match prim input args " << op_name << mindspore::ToString(args_spec_list);
    auto iter = temp->second.find(args_spec_list);
    if (iter != temp->second.end()) {
      MS_LOG(DEBUG) << "Match prim ok " << op_name;
      op_exec_info->abstract = iter->second.abs;
      prim->set_evaluate_added_attrs(iter->second.attrs);
      *prim_cache_hit = true;
    }
  }

  if (op_exec_info->abstract == nullptr || force_infer_prim.find(op_name) != force_infer_prim.end()) {
    // Use python infer method
    if (ignore_infer_prim.find(op_name) == ignore_infer_prim.end()) {
      PynativeInfer(prim, op_exec_info.get(), args_spec_list);
    }
  }
  // Get output dynamic shape info
  auto abstract = op_exec_info->abstract;
  MS_EXCEPTION_IF_NULL(abstract);
  auto shape = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape);

  if (shape->IsDynamic()) {
    op_exec_info->is_dynamic_shape = true;
    // Dynamic shape operator in the current top cell, disable backend cache
    grad()->EnableOpGraphCache(false);
  }
}

void ForwardExecutor::GetOpOutput(const OpExecInfoPtr &op_exec_info,
                                  const abstract::AbstractBasePtrList &args_spec_list, const CNodePtr &cnode,
                                  bool prim_cache_hit, py::object *ret) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  auto prim = op_exec_info->py_primitive;
  MS_EXCEPTION_IF_NULL(prim);
  // Infer output value by constant folding
  MS_EXCEPTION_IF_NULL(ret);
  py::dict output = abstract::ConvertAbstractToPython(op_exec_info->abstract);
  if (!output["value"].is_none()) {
    *ret = output["value"];
    grad()->RecordGradOpInfo(op_exec_info, PyObjToValue(*ret));
    return;
  }
  if (prim->is_const_prim()) {
    *ret = py::cast("");
    grad()->RecordGradOpInfo(op_exec_info, PyObjToValue(*ret));
    return;
  }

  // Add output abstract info into cache, the const value needs to infer evert step
  if (grad()->enable_op_cache() && !prim_cache_hit && !op_exec_info->is_dynamic_shape) {
    AbsCacheKey key{prim->name(), prim->Hash(), prim->attrs()};
    auto &out = prim_abs_list_[key];
    out[args_spec_list].abs = op_exec_info->abstract;
    out[args_spec_list].attrs = prim->evaluate_added_attrs();
  }
  // run op with selected backend
  auto result = RunOpWithInitBackendPolicy(op_exec_info);
  py::object out_real = result;
  if (result.size() == 1 && op_exec_info->abstract != nullptr &&
      !op_exec_info->abstract->isa<abstract::AbstractSequeue>()) {
    out_real = result[0];
  }
  // get output value
  ValuePtr out_real_value = nullptr;
  if (grad()->grad_flag()) {
    out_real_value = PyObjToValue(out_real);
  }
  // Save cnode info and build grad graph
  if (grad()->need_construct_graph() && !grad()->in_cell_with_custom_bprop_()) {
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &obj_id = GetId(out_real);
    cnode->set_abstract(op_exec_info->abstract);
    node_abs_map_[obj_id] = op_exec_info->abstract;
    grad()->SaveOutputNodeMap(obj_id, out_real, cnode);
    grad()->DoOpGrad(op_exec_info, cnode, out_real_value);
  } else {
    node_abs_map_.clear();
  }
  // Record op info for judge whether the construct of cell has been changed
  grad()->RecordGradOpInfo(op_exec_info, out_real_value);
  grad()->UpdateForwardTensorInfoInBpropGraph(op_exec_info, out_real_value);
  *ret = out_real;
}

py::object ForwardExecutor::DoAutoCast(const py::object &arg, const TypeId &type_id, const std::string &op_name,
                                       size_t index) {
  static py::object cast_prim = parse::python_adapter::GetPyFn(kOpsFunctionModelName, "cast");
  const auto &op_exec_info = std::make_shared<OpExecInfo>();
  op_exec_info->op_name = prim::kPrimCast->name();
  const auto &adapter = py::cast<PrimitivePyAdapterPtr>(cast_prim);
  MS_EXCEPTION_IF_NULL(adapter);
  auto prim = adapter->attached_primitive();
  if (prim == nullptr) {
    prim = std::make_shared<PrimitivePy>(cast_prim, adapter);
    adapter->set_attached_primitive(prim);
  }
  op_exec_info->py_primitive = prim;
  op_exec_info->is_mixed_precision_cast = true;
  op_exec_info->next_op_name = op_name;
  op_exec_info->next_input_index = index;
  py::object dst_type = GetDstType(type_id);
  py::tuple inputs(ARG_SIZE);
  inputs[0] = arg;
  inputs[1] = dst_type;
  op_exec_info->op_inputs = inputs;
  op_exec_info->lazy_build = lazy_build_;
  py::object ret = py::none();
  RunOpInner(&ret, op_exec_info);
  return ret;
}

py::object ForwardExecutor::DoAutoCastTuple(const py::tuple &tuple, const TypeId &type_id, const std::string &op_name,
                                            size_t index) {
  auto tuple_size = tuple.size();
  py::tuple result(tuple_size);
  for (size_t i = 0; i < tuple_size; i++) {
    if (py::isinstance<py::tuple>(tuple[i]) || py::isinstance<py::list>(tuple[i])) {
      result[i] = DoAutoCastTuple(tuple[i], type_id, op_name, index);
    } else {
      result[i] = DoAutoCast(tuple[i], type_id, op_name, index);
    }
  }
  return std::move(result);
}

py::object ForwardExecutor::DoParamMixPrecisionCast(bool *is_cast, const py::object &obj, const std::string &op_name,
                                                    size_t index) {
  MS_EXCEPTION_IF_NULL(is_cast);
  const auto &tensor = py::cast<tensor::TensorPtr>(obj);
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &cast_type = tensor->cast_dtype();
  if (cast_type != nullptr) {
    auto source_element = tensor->Dtype();
    if (source_element != nullptr && IsSubType(source_element, kFloat) && *source_element != *cast_type) {
      MS_LOG(DEBUG) << "Cast to " << cast_type->ToString();
      *is_cast = true;
      return DoAutoCast(obj, cast_type->type_id(), op_name, index);
    }
  }
  return obj;
}

py::object ForwardExecutor::DoParamMixPrecisionCastTuple(bool *is_cast, const py::tuple &tuple,
                                                         const std::string &op_name, size_t index) {
  MS_EXCEPTION_IF_NULL(is_cast);
  auto tuple_size = tuple.size();
  py::tuple result(tuple_size);
  for (size_t i = 0; i < tuple_size; i++) {
    if (py::isinstance<tensor::MetaTensor>(tuple[i])) {
      MS_LOG(DEBUG) << "Call cast for item " << i;
      result[i] = DoParamMixPrecisionCast(is_cast, tuple[i], op_name, index);
    } else if (py::isinstance<py::tuple>(tuple[i]) || py::isinstance<py::list>(tuple[i])) {
      result[i] = DoParamMixPrecisionCastTuple(is_cast, tuple[i], op_name, index);
    } else {
      result[i] = tuple[i];
    }
  }
  return std::move(result);
}

void ForwardExecutor::DoSignatrueCast(const PrimitivePyPtr &prim,
                                      const std::unordered_map<SignatureEnumDType, TypeId> &dst_type,
                                      const std::vector<SignatureEnumDType> &dtypes,
                                      const OpExecInfoPtr &op_exec_info) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(op_exec_info);
  const auto &signature = prim->signatures();
  auto &input_args = op_exec_info->op_inputs;
  size_t input_args_size = input_args.size();
  if (!dtypes.empty() && input_args_size > dtypes.size()) {
    MS_LOG(EXCEPTION) << "The input args size " << input_args_size << " exceeds the size of dtypes " << dtypes.size();
  }
  for (size_t i = 0; i < input_args_size; ++i) {
    // No need to implicit cast if no dtype.
    if (dtypes.empty() || dtypes[i] == SignatureEnumDType::kDTypeEmptyDefaultValue) {
      continue;
    }
    auto it = dst_type.find(dtypes[i]);
    if (it == dst_type.end() || it->second == kTypeUnknown) {
      continue;
    }
    MS_LOG(DEBUG) << "Check inputs " << i;
    const auto &obj = input_args[i];
    auto sig = SignatureEnumRW::kRWDefault;
    if (!signature.empty()) {
      if (i >= signature.size()) {
        MS_EXCEPTION(ValueError) << "Signature size is not equal to index, signature size " << signature.size()
                                 << ", index " << i;
      }
      sig = signature[i].rw;
    }
    TypeId arg_type_id = kTypeUnknown;
    if (py::isinstance<tensor::MetaTensor>(obj)) {
      const auto &arg = py::cast<tensor::MetaTensorPtr>(obj);
      arg_type_id = arg->data_type();
    }
    // Implicit cast
    bool is_same_type = false;
    if (arg_type_id != kTypeUnknown) {
      is_same_type = (prim::type_map.find(arg_type_id) == prim::type_map.end() || arg_type_id == it->second);
    }
    if (sig == SignatureEnumRW::kRWWrite && arg_type_id != kTypeUnknown && !is_same_type) {
      prim::RaiseExceptionForConvertRefDtype(prim->name(), TypeIdToMsTypeStr(arg_type_id),
                                             TypeIdToMsTypeStr(it->second));
    }
    if (is_same_type) {
      continue;
    }

    if (!py::isinstance<tensor::Tensor>(obj) && !py::isinstance<py::int_>(obj) && !py::isinstance<py::float_>(obj)) {
      MS_EXCEPTION(TypeError) << "For '" << prim->name() << "', the " << i
                              << "th input is a not support implicit conversion type: "
                              << py::cast<std::string>(obj.attr("__class__").attr("__name__")) << ", and the value is "
                              << py::cast<py::str>(obj) << ".";
    }
    py::object cast_output = DoAutoCast(input_args[i], it->second, op_exec_info->op_name, i);
    input_args[i] = cast_output;
  }
}

void ForwardExecutor::SetTensorMixPrecisionCast(const OpExecInfoPtr &op_exec_info) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  const auto &prim = op_exec_info->py_primitive;
  MS_EXCEPTION_IF_NULL(prim);
  const auto &signature = prim->signatures();
  for (size_t i = 0; i < op_exec_info->op_inputs.size(); i++) {
    const auto &obj = op_exec_info->op_inputs[i];
    auto sig = SignatureEnumRW::kRWDefault;
    if (!signature.empty()) {
      if (i >= signature.size()) {
        MS_EXCEPTION(ValueError) << "Signature size is not equal to index, signature size " << signature.size()
                                 << ", index " << i;
      }
      sig = signature[i].rw;
    }
    MS_LOG(DEBUG) << "Check mix precision " << op_exec_info->op_name << " input " << i;
    // mix precision for non param
    bool is_cast = false;
    py::object cast_output;
    if (py::isinstance<tensor::MetaTensor>(obj)) {
      auto meta_tensor = obj.cast<tensor::MetaTensorPtr>();
      if (meta_tensor && meta_tensor->is_parameter()) {
        // If parameter write(not kRWRead), no need cast
        if (sig != SignatureEnumRW::kRWRead) {
          continue;
        }
      }
      cast_output = DoParamMixPrecisionCast(&is_cast, obj, prim->name(), i);
    } else if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
      // mix precision for tuple inputs
      cast_output = DoParamMixPrecisionCastTuple(&is_cast, obj, prim->name(), i);
    }
    if (is_cast) {
      op_exec_info->op_inputs[i] = cast_output;
    }
  }
}

void ForwardExecutor::SetImplicitCast(const OpExecInfoPtr &op_exec_info) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  const auto &prim = op_exec_info->py_primitive;
  MS_EXCEPTION_IF_NULL(prim);
  const auto &it = implicit_cast_map_.find(prim->name());
  if (it == implicit_cast_map_.end()) {
    MS_LOG(DEBUG) << "Do signature for " << op_exec_info->op_name << " first";
    const auto &signature = prim->signatures();
    auto sig_size = signature.size();
    // Ignore monad signature
    for (const auto &sig : signature) {
      if (sig.default_value != nullptr && sig.default_value->isa<Monad>()) {
        --sig_size;
      }
    }
    auto size = op_exec_info->op_inputs.size();
    if (sig_size > 0 && sig_size != size) {
      MS_EXCEPTION(ValueError) << op_exec_info->op_name << " inputs size " << size << " does not match the requires "
                               << "signature size " << sig_size;
    }
    std::vector<SignatureEnumDType> dtypes;
    std::unordered_map<SignatureEnumDType, std::vector<size_t>> type_indexes;
    bool has_dtype_sig = GetSignatureType(op_exec_info->py_primitive, &dtypes);
    if (has_dtype_sig) {
      std::unordered_map<SignatureEnumDType, TypeId> dst_type;
      GetTypeIndex(dtypes, &type_indexes);
      GetDstType(op_exec_info->op_inputs, type_indexes, &dst_type);
      DoSignatrueCast(op_exec_info->py_primitive, dst_type, dtypes, op_exec_info);
    }
    PrimSignature sig_value{has_dtype_sig, dtypes, type_indexes};
    implicit_cast_map_[prim->name()] = sig_value;
  } else {
    if (!it->second.has_dtype_sig) {
      MS_LOG(DEBUG) << op_exec_info->op_name << " have no dtype sig";
      return;
    }
    MS_LOG(DEBUG) << "Do signature for " << op_exec_info->op_name << " with cache";
    std::unordered_map<SignatureEnumDType, TypeId> dst_type;
    GetDstType(op_exec_info->op_inputs, it->second.type_indexes, &dst_type);
    DoSignatrueCast(op_exec_info->py_primitive, dst_type, it->second.dtypes, op_exec_info);
  }
}

AnfNodePtr GradExecutor::GetInput(const py::object &obj, bool op_mask) {
  AnfNodePtr node = nullptr;
  const auto &obj_id = GetId(obj);

  if (op_mask) {
    MS_LOG(DEBUG) << "Cell parameters(weights)";
    // get the parameter name from parameter object
    auto name_attr = parse::python_adapter::GetPyObjAttr(obj, "name");
    if (py::isinstance<py::none>(name_attr)) {
      MS_LOG(EXCEPTION) << "Parameter object should have name attribute";
    }
    const auto &param_name = py::cast<std::string>(name_attr);
    auto df_builder = top_cell()->df_builder();
    MS_EXCEPTION_IF_NULL(df_builder);
    auto graph_info = top_cell()->graph_info_map().at(df_builder);
    MS_EXCEPTION_IF_NULL(graph_info);
    if (graph_info->params.find(obj_id) == graph_info->params.end()) {
      auto free_param = df_builder->add_parameter();
      free_param->set_name(param_name);
      free_param->debug_info()->set_name(param_name);
      auto value = py::cast<tensor::TensorPtr>(obj);
      free_param->set_default_param(value);
      MS_LOG(DEBUG) << "Top graph set free parameter " << obj_id;
      SetParamNodeMapInGraphInfoMap(df_builder, obj_id, free_param);
      SetParamNodeMapInGraphInfoMap(curr_g_, obj_id, free_param);
      SetNodeMapInGraphInfoMap(df_builder, obj_id, free_param);
      SetNodeMapInGraphInfoMap(curr_g_, obj_id, free_param);
      return free_param;
    }
    node = graph_info->params.at(obj_id);
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Get input param node " << node->ToString() << " " << obj_id;
    return node;
  }

  auto curr_graph_info = top_cell()->graph_info_map().at(curr_g_);
  MS_EXCEPTION_IF_NULL(curr_graph_info);
  if (curr_graph_info->node_map.find(obj_id) != curr_graph_info->node_map.end()) {
    // op(x, y)
    // out = op(op1(x, y))
    // out = op(cell1(x, y))
    // out = op(cell1(x, y)[0])
    node = GetObjNode(obj, obj_id);
  } else if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    // out = op((x, y))
    // out = cell((x, y))
    auto tuple = obj.cast<py::tuple>();
    // cell((1,2)): support not mix (scalar, tensor)
    if (!tuple.empty() && !py::isinstance<tensor::Tensor>(tuple[0])) {
      return MakeValueNode(obj, obj_id);
    }
    std::vector<AnfNodePtr> args;
    args.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    auto tuple_size = tuple.size();
    for (size_t i = 0; i < tuple_size; i++) {
      args.emplace_back(GetInput(tuple[i], false));
    }
    auto cnode = curr_g_->NewCNode(args);
    SetNodeMapInGraphInfoMap(curr_g_, GetId(obj), cnode);
    node = cnode;
  } else {
    node = MakeValueNode(obj, obj_id);
  }
  node == nullptr ? MS_LOG(DEBUG) << "Get node is nullptr"
                  : MS_LOG(DEBUG) << "Get input node " << node->ToString() << ", id " << obj_id;
  return node;
}

AnfNodePtr GradExecutor::GetObjNode(const py::object &obj, const std::string &obj_id) {
  auto graph_info = top_cell()->graph_info_map().at(curr_g_);
  MS_EXCEPTION_IF_NULL(graph_info);
  const auto &out = graph_info->node_map.at(obj_id);
  if (out.second.size() == 1 && out.second[0] == -1) {
    return out.first;
  }
  MS_LOG(DEBUG) << "Output size " << out.second.size();

  // Params node
  if (graph_info->params.find(obj_id) != graph_info->params.end()) {
    auto para_node = out.first;
    for (auto &v : out.second) {
      std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), para_node, NewValueNode(v)};
      MS_EXCEPTION_IF_NULL(curr_g_);
      para_node = curr_g_->NewCNode(tuple_get_item_inputs);
    }
    return para_node;
  }

  // Normal node
  auto node = out.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(node);
  auto abs = node->abstract();
  ValuePtr out_obj = nullptr;
  if (node->forward().first != nullptr) {
    out_obj = node->forward().first->value();
  } else {
    out_obj = PyObjToValue(obj);
  }
  for (const auto idx : out.second) {
    std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), node, NewValueNode(idx)};
    node = curr_g_->NewCNode(tuple_get_item_inputs);
    if (out_obj->isa<ValueTuple>()) {
      node->add_input_value(out_obj, "");
      node->add_input_value(MakeValue(idx), "");
      auto out_tuple = out_obj->cast<ValueTuplePtr>();
      MS_EXCEPTION_IF_NULL(out_tuple);
      if (static_cast<size_t>(idx) >= out_tuple->size()) {
        MS_LOG(EXCEPTION) << "Index exceeds the size of tuple. Index " << idx << ", tuple size " << out_tuple->size();
      }
      out_obj = (*out_tuple)[static_cast<size_t>(idx)];
      node->set_forward(NewValueNode(out_obj), "");
    }
    if (abs != nullptr && abs->isa<abstract::AbstractTuple>()) {
      auto abs_tuple = dyn_cast<abstract::AbstractTuple>(abs);
      MS_EXCEPTION_IF_NULL(abs_tuple);
      const auto &elements = abs_tuple->elements();
      if (static_cast<size_t>(idx) >= elements.size()) {
        MS_LOG(EXCEPTION) << "Index exceeds the size of elements. Index " << idx << ", elements size "
                          << elements.size();
      }
      auto prim_abs = elements[static_cast<size_t>(idx)];
      MS_EXCEPTION_IF_NULL(prim_abs);
      MS_LOG(DEBUG) << "Set tuple getitem abs " << prim_abs->ToString();
      node->set_abstract(prim_abs);
    }
  }
  if (node->abstract() != nullptr) {
    forward()->node_abs_map()[obj_id] = node->abstract();
  }
  MS_LOG(DEBUG) << "GetObjNode output " << node->DebugString();
  return node;
}

AnfNodePtr GradExecutor::MakeValueNode(const py::object &obj, const std::string &obj_id) {
  ValuePtr converted_ret = nullptr;
  parse::ConvertData(obj, &converted_ret);
  auto node = NewValueNode(converted_ret);
  SetNodeMapInGraphInfoMap(curr_g_, obj_id, node);
  return node;
}

TopCellInfoPtr GradExecutor::GetTopCell(const std::string &already_run_cell_id) {
  TopCellInfoPtr find_top_cell = nullptr;
  for (const auto &top_cell : top_cell_list_) {
    MS_EXCEPTION_IF_NULL(top_cell);
    // Complete match, means run grad operation first
    if (top_cell->already_run_cell_id() == already_run_cell_id) {
      return top_cell;
    }
    // Partial match, means run forward first
    if (already_run_cell_id.find(top_cell->already_run_cell_id()) != std::string::npos &&
        top_cell->already_run_cell_id().back() == '_') {
      find_top_cell = top_cell;
      break;
    }
  }
  // Same topcell info, but grad operation is not the same, construct backward graph again
  if (find_top_cell != nullptr) {
    if (!find_top_cell->grad_operation().empty() && find_top_cell->grad_operation() != grad_operation_) {
      MS_LOG(DEBUG) << "Already exist grad operation " << find_top_cell->grad_operation() << " is different with new "
                    << grad_operation_;
      EraseTopCellFromTopCellList(find_top_cell);
      (void)already_run_top_cell_.erase(find_top_cell->already_run_cell_id());
      return nullptr;
    } else {
      return find_top_cell;
    }
  }
  return nullptr;
}

void GradExecutor::EnableOpGraphCache(bool is_enable) {
  MS_LOG(DEBUG) << "Op cache is enable: " << is_enable;
  enable_op_cache_ = is_enable;
  const auto inst = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(inst);
  inst->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_OP_GRAPH_CACHE, is_enable);
}

void GradExecutor::RecordGradOpInfo(const OpExecInfoPtr &op_exec_info, const ValuePtr &op_out) {
  if (!grad_flag_) {
    MS_LOG(DEBUG) << "Grad flag is set to false, no need to record op info";
    return;
  }
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(op_out);
  std::string input_args_info;
  // Record input args info (weight or data)
  for (const auto mask : op_exec_info->inputs_mask) {
    if (mask) {
      input_args_info += "w";
      continue;
    }
    input_args_info += "d";
  }
  // Record op name and index
  op_exec_info->op_info.clear();
  const auto &curr_op_num = top_cell()->op_num();
  op_exec_info->op_info += op_exec_info->op_name + "-" + std::to_string(curr_op_num) + "-" + input_args_info;
  // The out shape is added to determine those ops that change the shape
  auto out_abs = op_out->ToAbstract();
  if (out_abs != nullptr) {
    auto out_shape = out_abs->BuildShape()->ToString();
    if (out_shape.find("()") == std::string::npos && out_shape.find("NoShape") == std::string::npos) {
      op_exec_info->op_info += "-" + out_shape;
    }
  }
  top_cell()->all_op_info() += "-" + op_exec_info->op_info;
  top_cell()->set_op_num(curr_op_num + 1);
}

void GradExecutor::SaveOutputNodeMap(const std::string &obj_id, const py::object &out_real, const CNodePtr &cnode) {
  if (cell_stack_.empty()) {
    MS_LOG(DEBUG) << "No need save output";
    return;
  }
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "Cnode is " << cnode->DebugString() << " id " << obj_id;
  if (py::isinstance<py::tuple>(out_real)) {
    auto value = py::cast<py::tuple>(out_real);
    auto size = static_cast<int64_t>(value.size());
    if (size > 1) {
      for (int64_t i = 0; i < size; ++i) {
        auto value_id = GetId(value[static_cast<size_t>(i)]);
        SetNodeMapInGraphInfoMap(curr_g_, value_id, cnode, i);
      }
    }
  }
  SetNodeMapInGraphInfoMap(curr_g_, obj_id, cnode);
}

// Run ad grad for curr op and connect grad graph with previous op
void GradExecutor::DoOpGrad(const OpExecInfoPtr &op_exec_info, const CNodePtr &cnode, const ValuePtr &op_out) {
  MS_EXCEPTION_IF_NULL(op_out);
  if (grad_is_running_ && !bprop_grad_stack_.top().second) {
    MS_LOG(DEBUG) << "Custom bprop, no need do op grad";
    return;
  }
  ValuePtrList input_args;
  for (size_t i = 0; i < op_exec_info->op_inputs.size(); ++i) {
    const auto &arg = PyObjToValue(op_exec_info->op_inputs[i]);
    input_args.emplace_back(arg);
  }

  if (!ad::GradPynativeOp(top_cell()->k_pynative_cell_ptr(), cnode, input_args, op_out)) {
    MS_LOG(EXCEPTION) << "Failed to run ad grad for op " << op_exec_info->op_name;
  }
}

void GradExecutor::UpdateMsFunctionForwardTensors(const OpExecInfoPtr &op_exec_info,
                                                  const ValuePtr &new_forward_value) {
  MS_LOG(DEBUG) << "Ms func graph has already ran before. The graph phase is: " << graph_phase();
  MS_EXCEPTION_IF_NULL(new_forward_value);
  MS_LOG(DEBUG) << "The output values of added forward nodes are: " << new_forward_value->ToString();
  std::vector<tensor::TensorPtr> new_tensors;
  TensorValueToTensor(new_forward_value, &new_tensors);
  if (new_tensors.empty()) {
    MS_LOG(DEBUG) << "The size of added forward tensors is zero, no need to update.";
    return;
  }

  MS_EXCEPTION_IF_NULL(op_exec_info);
  const auto &old_tensors = top_cell()->op_info_with_ms_func_forward_tensors().at(op_exec_info->op_info);
  if (old_tensors.size() != new_tensors.size()) {
    MS_LOG(EXCEPTION) << "The size of old tensors is: " << old_tensors.size()
                      << ", but the size of new tensors is: " << new_tensors.size()
                      << ", the current op info is: " << op_exec_info->op_info;
  }
  for (size_t i = 0; i < new_tensors.size(); ++i) {
    UpdateTensorInfo(new_tensors[i], {old_tensors[i]});
    old_tensors[i]->set_sync_status(kNeedSyncDeviceToHost);
  }
}

void GradExecutor::MakeCNodeForMsFunction(const FuncGraphPtr &ms_func_graph, const py::args &args,
                                          ValuePtrList *input_values, CNodePtr *ms_function_cnode) {
  // Get input node info of ms_function
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  std::vector<AnfNodePtr> input_nodes{NewValueNode(ms_func_graph)};
  MS_EXCEPTION_IF_NULL(input_values);
  for (size_t i = 0; i < args.size(); ++i) {
    auto input_i_node = GetInput(args[i], false);
    MS_EXCEPTION_IF_NULL(input_i_node);
    MS_LOG(DEBUG) << "The input " << i << " node of ms_function graph is: " << input_i_node->DebugString();
    input_nodes.emplace_back(input_i_node);
    const auto &inp_i_value = PyObjToValue(args[i]);
    MS_LOG(DEBUG) << "The input " << i << " value of ms_function graph is: " << inp_i_value->ToString();
    (*input_values).emplace_back(inp_i_value);
  }

  // Get dfbuilder and graph info map
  auto df_builder = top_cell()->df_builder();
  MS_EXCEPTION_IF_NULL(df_builder);
  const auto &graph_info = top_cell()->graph_info_map().at(df_builder);
  MS_EXCEPTION_IF_NULL(graph_info);
  // Get weights info of ms_function
  std::vector<AnfNodePtr> new_params;
  auto manage = Manage(ms_func_graph, false);
  for (const auto &anf_node : ms_func_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto param = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    if (!param->has_default()) {
      new_params.push_back(param);
      continue;
    }
    auto param_info = param->param_info();
    MS_EXCEPTION_IF_NULL(param_info);
    auto param_name = param_info->name();
    if (graph_info->params.count(param_name)) {
      // Share same weight parameter in different ms_function call.
      auto same_param = graph_info->params.at(param_name);
      manage->Replace(anf_node, same_param);
      param = same_param;
    } else {
      df_builder->add_parameter(param);
      param->debug_info()->set_name(param_name);
    }
    new_params.push_back(param);
    input_nodes.emplace_back(param);
    (*input_values).emplace_back(param->default_param());
    SetParamNodeMapInGraphInfoMap(df_builder, param_name, param);
    MS_LOG(DEBUG) << "Top graph set free parameter " << param->DebugString() << ". Its default value is "
                  << param->default_param()->ToString() << ". Its name is: " << param_name;
  }
  ms_func_graph->set_parameters(new_params);
  manage->Clear();

  // Make a CNode which includes ms_function fprop graph and inputs node
  MS_EXCEPTION_IF_NULL(ms_function_cnode);
  *ms_function_cnode = curr_g_->NewCNode(input_nodes);
  MS_LOG(DEBUG) << "Make ms function forward cnode: " << (*ms_function_cnode)->DebugString();
}

// Make adjoint for ms_function fprop graph and connect it with previous op
void GradExecutor::MakeAdjointForMsFunction(const FuncGraphPtr &ms_func_graph, const FuncGraphPtr &grad_graph,
                                            const py::object &actual_out, const py::args &args,
                                            const ValuePtr &actual_out_v) {
  ValuePtrList input_values;
  CNodePtr ms_function_cnode = nullptr;
  MakeCNodeForMsFunction(ms_func_graph, args, &input_values, &ms_function_cnode);
  MS_EXCEPTION_IF_NULL(ms_function_cnode);
  SetTupleArgsToGraphInfoMap(curr_g_, actual_out, ms_function_cnode);
  SetNodeMapInGraphInfoMap(curr_g_, GetId(actual_out), ms_function_cnode);

  // Connect grad graph of ms_function to context.
  auto k_pynative_cell_ptr = top_cell()->k_pynative_cell_ptr();
  MS_EXCEPTION_IF_NULL(k_pynative_cell_ptr);
  MS_EXCEPTION_IF_NULL(grad_graph);
  if (!k_pynative_cell_ptr->KPynativeWithFProp(ms_function_cnode, input_values, actual_out_v, grad_graph)) {
    MS_LOG(EXCEPTION) << "Failed to make adjoint for ms_function cnode, ms_function cnode info: "
                      << ms_function_cnode->DebugString();
  }
  top_cell()->set_ms_function_flag(true);
}

void GradExecutor::UpdateForwardTensorInfoInBpropGraph(const OpExecInfoPtr &op_exec_info, const ValuePtr &op_out) {
  if (!grad_flag_) {
    MS_LOG(DEBUG) << "The grad flag is false, no need to update forward op info in bprop graph";
    return;
  }
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(op_out);
  const auto &op_info = op_exec_info->op_info;
  MS_LOG(DEBUG) << "Current op info: " << op_info;

  std::vector<tensor::TensorPtr> all_op_tensors;
  // Get output tensors
  TensorValueToTensor(op_out, &all_op_tensors);
  // Save all tensors info of current op
  if (need_construct_graph()) {
    SaveOpInfo(top_cell_, op_info, all_op_tensors);
  }

  // First run top cell
  if (already_run_top_cell_.find(top_cell_->already_run_cell_id()) == already_run_top_cell_.end()) {
    MS_LOG(DEBUG) << "Top cell " << top_cell_->cell_id() << " run firstly";
    if (!need_construct_graph()) {
      MS_LOG(EXCEPTION) << "The cell stack is empty when running a new top cell " << top_cell_->cell_id();
    }
    return;
  }
  // Non-first run
  const auto &pre_top_cell = already_run_top_cell_.at(top_cell_->already_run_cell_id());
  MS_EXCEPTION_IF_NULL(pre_top_cell);
  if (pre_top_cell->op_info_with_tensor_id().find(op_info) == pre_top_cell->op_info_with_tensor_id().end()) {
    MS_LOG(DEBUG) << "Can not find op info " << op_info << " in op info with tensor id map. Top cell "
                  << top_cell_->cell_id();
    return;
  }

  // Update new output tensor info in bprop graph
  const auto &pre_op_tensor_id = pre_top_cell->op_info_with_tensor_id().at(op_info);
  if (pre_op_tensor_id.size() != all_op_tensors.size()) {
    MS_LOG(EXCEPTION) << "The size of pre op tensor id: " << pre_op_tensor_id.size()
                      << " is not equal to the size of all tensors of current op " << all_op_tensors.size();
  }
  const auto &pre_tensor_id_with_tensor_object = pre_top_cell->tensor_id_with_tensor_object();
  for (size_t i = 0; i < pre_op_tensor_id.size(); ++i) {
    auto pre_id = pre_op_tensor_id[i];
    if (pre_tensor_id_with_tensor_object.find(pre_id) == pre_tensor_id_with_tensor_object.end()) {
      continue;
    }
    const auto &new_tensor = all_op_tensors[i];
    const auto &pre_tensor_object = pre_tensor_id_with_tensor_object.at(pre_id);
    UpdateTensorInfo(new_tensor, pre_tensor_object);
  }
}

void GradExecutor::SaveForwardTensorInfoInBpropGraph(const pipeline::ResourcePtr &resource) const {
  MS_EXCEPTION_IF_NULL(resource);
  // Get all tensors id of forward op
  std::unordered_set<std::string> forward_op_tensor_id;
  const auto &op_info_with_tensor_id = top_cell()->op_info_with_tensor_id();
  for (const auto &record : op_info_with_tensor_id) {
    std::for_each(record.second.begin(), record.second.end(),
                  [&forward_op_tensor_id](const std::string &tensor_id) { forward_op_tensor_id.emplace(tensor_id); });
  }
  // Get all tensors obj in value node of bprop graph
  const auto &bprop_graph = resource->func_graph();
  MS_EXCEPTION_IF_NULL(bprop_graph);
  const auto &value_node_list = bprop_graph->value_nodes();
  std::vector<tensor::TensorPtr> tensors_in_bprop_graph;
  for (const auto &elem : value_node_list) {
    auto value_node = elem.first->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    TensorValueToTensor(value_node->value(), &tensors_in_bprop_graph);
  }

  auto &tensor_id_with_tensor_object = top_cell()->tensor_id_with_tensor_object();
  if (!tensor_id_with_tensor_object.empty()) {
    MS_LOG(EXCEPTION) << "When compile a top graph, the tensor_id_with_tensor_object map should be empty. Top cell: "
                      << top_cell()->cell_id();
  }
  // Save tensor in value node of bprop graph
  for (const auto &tensor : tensors_in_bprop_graph) {
    MS_EXCEPTION_IF_NULL(tensor);
    if (forward_op_tensor_id.find(tensor->id()) == forward_op_tensor_id.end() || tensor->device_address() == nullptr) {
      continue;
    }
    tensor_id_with_tensor_object[tensor->id()].emplace_back(tensor);
    MS_LOG(DEBUG) << "Save forward tensor " << tensor.get() << " id " << tensor->id()
                  << " device address: " << tensor->device_address() << " shape and dtype "
                  << tensor->GetShapeAndDataTypeInfo();
  }
}

py::tuple ForwardExecutor::RunOpWithInitBackendPolicy(const OpExecInfoPtr &op_exec_info) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  auto backend_policy = InitEnv(op_exec_info);
  PynativeStatusCode status = PYNATIVE_UNKNOWN_STATE;
  // returns a null py::tuple on error
  py::object result = RunOpWithBackendPolicy(backend_policy, op_exec_info, &status);
  if (status != PYNATIVE_SUCCESS) {
    MS_LOG(EXCEPTION) << "Failed to run " << op_exec_info->op_name;
  }
  MS_LOG(DEBUG) << "RunOp end";
  return result;
}

MsBackendPolicy ForwardExecutor::InitEnv(const OpExecInfoPtr &op_exec_info) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_LOG(DEBUG) << "RunOp start, op name is: " << op_exec_info->op_name;
  parse::python_adapter::set_python_env_flag(true);
  MsBackendPolicy backend_policy;
#if (!defined ENABLE_GE)
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!context::IsTsdOpened(ms_context)) {
    if (!context::OpenTsd(ms_context)) {
      MS_LOG(EXCEPTION) << "Open tsd failed";
    }
  }
  if (ms_context->backend_policy() == "ms") {
    backend_policy = kMsBackendMsPrior;
  } else {
    backend_policy = kMsBackendVmOnly;
  }
#else
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  context::PynativeInitGe(ms_context);
  backend_policy = kMsBackendGeOnly;
#endif
  if (kVmOperators.find(op_exec_info->op_name) != kVmOperators.end()) {
    backend_policy = kMsBackendVmOnly;
  }
  return backend_policy;
}

py::object ForwardExecutor::RunOpWithBackendPolicy(MsBackendPolicy backend_policy, const OpExecInfoPtr &op_exec_info,
                                                   PynativeStatusCode *status) {
  MS_EXCEPTION_IF_NULL(status);
  py::object result;
  switch (backend_policy) {
    case kMsBackendVmOnly: {
      // use vm only
      MS_LOG(DEBUG) << "RunOp use VM only backend";
      result = RunOpInVM(op_exec_info, status);
      break;
    }
    case kMsBackendGePrior: {
#ifdef ENABLE_GE
      // use GE first, use vm when GE fails
      MS_LOG(DEBUG) << "RunOp use GE first backend";
      result = RunOpInGE(op_exec_info, status);
      if (*status != PYNATIVE_SUCCESS) {
        result = RunOpInVM(op_exec_info, status);
      }
#endif
      break;
    }
    case kMsBackendMsPrior: {
      // use Ms first,use others when ms failed
      MS_LOG(DEBUG) << "RunOp use Ms first backend";
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

py::object ForwardExecutor::RunOpInVM(const OpExecInfoPtr &op_exec_info, PynativeStatusCode *status) {
  MS_LOG(DEBUG) << "RunOpInVM start";
  MS_EXCEPTION_IF_NULL(status);
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(op_exec_info->py_primitive);

  auto &op_inputs = op_exec_info->op_inputs;
  if (op_exec_info->op_name == "HookBackward" || op_exec_info->op_name == "InsertGradientOf" ||
      op_exec_info->op_name == "stop_gradient") {
    py::tuple result(op_inputs.size());
    for (size_t i = 0; i < op_inputs.size(); i++) {
      py::object input = op_inputs[i];
      auto tensor = py::cast<tensor::TensorPtr>(input);
      MS_EXCEPTION_IF_NULL(tensor);
      if (op_exec_info->op_name == "HookBackward") {
        // the input object is not a output of forward cnode, eg: parameter
        result[i] = tensor;
      } else {
        // the input object is a output of forward cnode
        auto new_tensor = std::make_shared<tensor::Tensor>(tensor->data_type(), tensor->shape(), tensor->data_ptr());
        new_tensor->set_device_address(tensor->device_address());
        new_tensor->set_sync_status(tensor->sync_status());
        result[i] = new_tensor;
      }
    }
    *status = PYNATIVE_SUCCESS;
    MS_LOG(DEBUG) << "RunOpInVM end";
    return std::move(result);
  }

  auto primitive = op_exec_info->py_primitive;
  MS_EXCEPTION_IF_NULL(primitive);
  auto result = primitive->RunPyComputeFunction(op_inputs);
  MS_LOG(DEBUG) << "RunOpInVM end";
  if (py::isinstance<py::none>(result)) {
    MS_LOG(ERROR) << "VM got the result none, please check whether it is failed to get func";
    *status = PYNATIVE_OP_NOT_IMPLEMENTED_ERR;
    py::tuple err_ret(0);
    return std::move(err_ret);
  }
  *status = PYNATIVE_SUCCESS;
  if (py::isinstance<py::tuple>(result)) {
    return result;
  }
  py::tuple tuple_result = py::make_tuple(result);
  return std::move(tuple_result);
}

py::object ForwardExecutor::RunOpInMs(const OpExecInfoPtr &op_exec_info, PynativeStatusCode *status) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(status);
  MS_LOG(DEBUG) << "Start run op [" << op_exec_info->op_name << "] with backend policy ms";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, true);
  compile::SetMindRTEnable();

  if (kSession == nullptr && !ms_context->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    const auto &device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    kSession = session::SessionFactory::Get().Create(device_target);
    MS_EXCEPTION_IF_NULL(kSession);
    kSession->Init(ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID));
  }

  std::vector<tensor::TensorPtr> input_tensors;
  std::vector<int64_t> tensors_mask;
  std::string graph_info;
  ConstructInputTensor(op_exec_info, &tensors_mask, &input_tensors);
  ConvertAttrToUnifyMindIR(op_exec_info);
  // get graph info for checking it whether existing in the cache
  GetSingleOpGraphInfo(op_exec_info, input_tensors, tensors_mask, &graph_info);
#if defined(__APPLE__)
  session::OpRunInfo op_run_info = {op_exec_info->op_name,
                                    op_exec_info->py_primitive,
                                    op_exec_info->abstract,
                                    op_exec_info->is_dynamic_shape,
                                    op_exec_info->is_mixed_precision_cast,
                                    op_exec_info->lazy_build,
                                    op_exec_info->next_op_name,
                                    static_cast<int>(op_exec_info->next_input_index)};
#else
  session::OpRunInfo op_run_info = {op_exec_info->op_name,
                                    op_exec_info->py_primitive,
                                    op_exec_info->abstract,
                                    op_exec_info->is_dynamic_shape,
                                    op_exec_info->is_mixed_precision_cast,
                                    op_exec_info->lazy_build,
                                    op_exec_info->next_op_name,
                                    op_exec_info->next_input_index};
#endif
  VectorRef outputs;
  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    kSession->RunOp(&op_run_info, graph_info, &input_tensors, &outputs, tensors_mask);
  } else {
    if (mind_rt_backend == nullptr) {
      const auto &device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
      uint32_t device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
      mind_rt_backend = std::make_shared<compile::MindRTBackend>("ms", device_target, device_id);
    }

    mindspore::ScopedLongRunning long_running;
    const compile::ActorInfo &actor_info =
      mind_rt_backend->CompileGraph(op_run_info, graph_info, &tensors_mask, &input_tensors);
    mind_rt_backend->RunGraph(actor_info, &op_run_info, &tensors_mask, &input_tensors, &outputs);
  }

  if (op_exec_info->is_dynamic_shape) {
    op_exec_info->abstract = op_run_info.abstract;
  }
  auto result = BaseRefToPyData(outputs);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, false);
  *status = PYNATIVE_SUCCESS;
  MS_LOG(DEBUG) << "End run op [" << op_exec_info->op_name << "] with backend policy ms";
  return result;
}

void ForwardExecutor::ClearRes() {
  MS_LOG(DEBUG) << "Clear forward res";
  lazy_build_ = false;
  implicit_cast_map_.clear();
  prim_abs_list_.clear();
  node_abs_map_.clear();
}

ForwardExecutorPtr GradExecutor::forward() const {
  auto forward_executor = forward_executor_.lock();
  MS_EXCEPTION_IF_NULL(forward_executor);
  return forward_executor;
}

TopCellInfoPtr GradExecutor::top_cell() const {
  MS_EXCEPTION_IF_NULL(top_cell_);
  return top_cell_;
}

FuncGraphPtr GradExecutor::curr_g() const {
  MS_EXCEPTION_IF_NULL(curr_g_);
  return curr_g_;
}

void GradExecutor::PushCellStack(const std::string &cell_id) { cell_stack_.push(cell_id); }

void GradExecutor::PopCellStack() {
  if (cell_stack_.empty()) {
    MS_LOG(EXCEPTION) << "Stack cell_statck_ is empty";
  }
  cell_stack_.pop();
}

void GradExecutor::PushHighOrderGraphStack(const TopCellInfoPtr &top_cell) {
  high_order_stack_.push(std::make_pair(curr_g_, top_cell));
}

TopCellInfoPtr GradExecutor::PopHighOrderGraphStack() {
  if (high_order_stack_.empty()) {
    MS_LOG(EXCEPTION) << "Stack high_order_stack_ is empty";
  }
  high_order_stack_.pop();
  TopCellInfoPtr top_cell = nullptr;
  if (!high_order_stack_.empty()) {
    auto t = high_order_stack_.top();
    curr_g_ = t.first;
    top_cell = t.second;
  }
  return top_cell;
}

std::string GradExecutor::GetCellId(const py::object &cell, const py::args &args) {
  auto cell_id = GetId(cell);
  for (size_t i = 0; i < args.size(); i++) {
    const auto &arg_id = GetId(args[i]);
    auto it = forward()->node_abs_map().find(arg_id);
    if (it != forward()->node_abs_map().end()) {
      auto &abs = it->second;
      MS_EXCEPTION_IF_NULL(abs);
      auto shape = abs->BuildShape();
      MS_EXCEPTION_IF_NULL(shape);
      auto type = abs->BuildType();
      MS_EXCEPTION_IF_NULL(type);
      cell_id += "_" + shape->ToString();
      cell_id += type->ToString();
    } else {
      auto value = PyObjToValue(args[i]);
      MS_EXCEPTION_IF_NULL(value);
      auto abs = value->ToAbstract();
      MS_EXCEPTION_IF_NULL(abs);
      if (abs->isa<abstract::AbstractTensor>()) {
        abs->set_value(kAnyValue);
      }
      forward()->node_abs_map()[arg_id] = abs;
      auto shape = abs->BuildShape();
      MS_EXCEPTION_IF_NULL(shape);
      auto type = abs->BuildType();
      MS_EXCEPTION_IF_NULL(type);
      cell_id += "_" + shape->ToString();
      cell_id += type->ToString();
    }
  }
  return cell_id;
}

void GradExecutor::DumpGraphIR(const std::string &filename, const FuncGraphPtr &graph) {
#ifdef ENABLE_DUMP_IR
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
    DumpIR(filename, graph);
  }
#endif
}

inline bool GradExecutor::IsNestedGrad() const {
  MS_LOG(DEBUG) << "Grad nested order is " << grad_order_;
  return grad_order_ > 1;
}

bool GradExecutor::IsCellObjIdEq(const std::string &l_cell_id, const std::string &r_cell_id) const {
  // just compare obj_id, ignore args id
  return l_cell_id.compare(0, PTR_LEN, r_cell_id, 0, PTR_LEN) == 0;
}

bool GradExecutor::IsBpropGraph(const std::string &cell_id) {
  if (top_cell_ == nullptr) {
    return false;
  }
  return std::any_of(bprop_cell_list_.begin(), bprop_cell_list_.end(),
                     [&cell_id](const std::string &value) { return cell_id.find(value) != std::string::npos; });
}

void GradExecutor::UpdateTopCellInfo(bool forward_already_run, bool need_compile_graph, bool vm_compiled) {
  top_cell()->set_vm_compiled(vm_compiled);
  top_cell()->set_need_compile_graph(need_compile_graph);
  top_cell()->set_forward_already_run(forward_already_run);
}

void GradExecutor::ClearCellRes(const std::string &cell_id) {
  static bool clear_all_cell_res = false;
  // Grad clean
  if (cell_id.empty()) {
    MS_LOG(DEBUG) << "Clear all cell resources";
    clear_all_cell_res = true;
    for (const auto &iter : top_cell_list_) {
      MS_EXCEPTION_IF_NULL(iter);
      iter->Clear();
    }
    top_cell_list_.clear();
    already_run_top_cell_.clear();
    clear_all_cell_res = false;
    return;
  }
  if (clear_all_cell_res) {
    MS_LOG(DEBUG) << "In process of clearing all cell resources, so no need to clear single cell resource again";
    return;
  }
  // clear when cell destruction
  for (auto it = top_cell_list_.begin(); it != top_cell_list_.end();) {
    MS_EXCEPTION_IF_NULL(*it);
    const auto &top_cell_id = (*it)->cell_id();
    const auto &already_run_cell_id = (*it)->already_run_cell_id();
    if (IsCellObjIdEq(cell_id, top_cell_id)) {
      MS_LOG(DEBUG) << "Clear top cell resource. Top cell id " << top_cell_id;
      (*it)->Clear();
      it = top_cell_list_.erase(it);
      (void)already_run_top_cell_.erase(already_run_cell_id);
      continue;
    }
    ++it;
  }
}

void GradExecutor::HandleInputArgsForTopCell(const py::args &args, bool is_bprop_top) {
  if (is_bprop_top) {
    // Convert input args to parameters for top cell graph in bprop.
    for (size_t i = 0; i < args.size(); ++i) {
      auto param = args[i];
      auto new_param = curr_g_->add_parameter();
      const auto &param_id = GetId(param);
      SetTupleArgsToGraphInfoMap(curr_g_, param, new_param, true);
      SetNodeMapInGraphInfoMap(curr_g_, param_id, new_param);
      SetParamNodeMapInGraphInfoMap(curr_g_, param_id, new_param);
    }
    return;
  }
  // Convert input args to parameters for top cell graph in construct.
  std::vector<ValuePtr> input_param_values;
  const auto &only_tensors = FilterTensorArgs(args);
  for (size_t i = 0; i < only_tensors.size(); ++i) {
    auto new_param = curr_g_->add_parameter();
    auto param_i = only_tensors[i];
    const auto &param_i_value = PyObjToValue(param_i);
    input_param_values.emplace_back(param_i_value);
    auto param_i_abs = param_i_value->ToAbstract();
    MS_EXCEPTION_IF_NULL(param_i_abs);
    new_param->set_abstract(param_i_abs->Broaden());
    const auto &param_i_id = GetId(param_i);
    SetTupleArgsToGraphInfoMap(curr_g_, param_i, new_param, true);
    SetNodeMapInGraphInfoMap(curr_g_, param_i_id, new_param);
    SetParamNodeMapInGraphInfoMap(curr_g_, param_i_id, new_param);
    SetParamNodeMapInGraphInfoMap(top_cell_->df_builder(), param_i_id, new_param);
  }
  top_cell()->set_k_pynative_cell_ptr(ad::GradPynativeCellBegin(curr_g_->parameters(), input_param_values));
}

void GradExecutor::InitResourceAndDfBuilder(const std::string &cell_id, const py::args &args) {
  if (cell_stack_.empty() || IsNestedGrad()) {
    if (cell_stack_.empty() && !grad_is_running_) {
      MS_LOG(DEBUG) << "Make new topest graph";
      MakeNewTopGraph(cell_id, args, true);
    } else if (grad_is_running_ && IsBpropGraph(cell_id)) {
      MS_LOG(DEBUG) << "Run bprop cell";
      curr_g_ = std::make_shared<FuncGraph>();
      auto graph_info_cg = std::make_shared<GraphInfo>(cell_id);
      top_cell()->graph_info_map()[curr_g_] = graph_info_cg;
      HandleInputArgsForTopCell(args, true);
      bprop_grad_stack_.push(std::make_pair(cell_id, false));
    } else if (grad_is_running_ && top_cell()->grad_order() != grad_order_) {
      MS_LOG(DEBUG) << "Nested grad graph existed in bprop";
      MakeNewTopGraph(cell_id, args, false);
      bprop_grad_stack_.push(std::make_pair(cell_id, true));
    } else if (!cell_stack_.empty() && IsNestedGrad() && top_cell()->grad_order() != grad_order_) {
      MS_LOG(DEBUG) << "Nested grad graph existed in construct";
      auto cur_top_is_dynamic = top_cell()->is_dynamic();
      MakeNewTopGraph(cell_id, args, false);
      top_cell()->set_is_dynamic(cur_top_is_dynamic);
    }
  }

  PushCellStack(cell_id);
  // Init kPynativeCellPtr with input parameters of top cell
  if (!top_cell()->is_init_kpynative()) {
    auto graph_info_cg = std::make_shared<GraphInfo>(cell_id);
    top_cell()->graph_info_map()[curr_g_] = graph_info_cg;
    auto graph_info_df = std::make_shared<GraphInfo>(cell_id);
    top_cell()->graph_info_map()[top_cell_->df_builder()] = graph_info_df;
    HandleInputArgsForTopCell(args, false);
    top_cell()->set_need_compile_graph(true);
    top_cell()->set_init_kpynative(true);
  } else {
    // Non-top cell
    top_cell()->sub_cell_list().emplace(cell_id);
  }
}

void GradExecutor::NewGraphInner(py::object *ret, const py::object &cell, const py::args &args) {
  MS_EXCEPTION_IF_NULL(ret);
  const auto &cell_id = GetCellId(cell, args);
  MS_LOG(DEBUG) << "NewGraphInner start " << args.size() << " " << cell_id;
  if (top_cell_ != nullptr && cell_stack_.empty()) {
    // Already run top cell need distinguish high order; high order add "0" otherwise "1"
    const auto &already_run_cell_id = GetAlreadyRunCellId(cell_id);
    auto top_it = already_run_top_cell_.find(already_run_cell_id);
    if (top_it != already_run_top_cell_.end()) {
      // Top cell forward run.
      const auto &pre_top_cell = top_it->second;
      MS_EXCEPTION_IF_NULL(pre_top_cell);
      if (!pre_top_cell->is_dynamic()) {
        MS_LOG(DEBUG) << "Top cell " << cell_id << " is not dynamic, no need to run NewGraphInner again";
        ResetTopCellInfo(pre_top_cell, args);
        PushHighOrderGraphStack(pre_top_cell);
        set_top_cell(pre_top_cell);
        grad_order_ = pre_top_cell->grad_order();
        return;
      }
    } else if ((top_cell()->IsSubCell(cell_id) || GetHighOrderStackSize() >= 1) &&
               !IsCellObjIdEq(cell_id, check_graph_cell_id_)) {
      // Sub cell ( or may be a temporary cell, but must be non top) forward run in cache process.
      MS_LOG(DEBUG) << "Sub cell no need to run NewGraphInner again";
      return;
    }
  }
  // When the cell has custom bprop, in_custom_bprop_cell is lager than 0
  if (py::hasattr(cell, parse::CUSTOM_BPROP_NAME)) {
    custom_bprop_cell_count_ += 1;
  }
  // Make top graph and init resource for resource and df_builder
  InitResourceAndDfBuilder(cell_id, args);
  // Check whether cell has dynamic construct
  if (!top_cell()->is_dynamic()) {
    bool is_dynamic = parse::DynamicParser::IsDynamicCell(cell);
    MS_LOG(DEBUG) << "Current cell dynamic " << is_dynamic;
    if (is_dynamic) {
      top_cell()->set_is_dynamic(is_dynamic);
    }
  }
}

void GradExecutor::MakeNewTopGraph(const string &cell_id, const py::args &args, bool is_topest) {
  pipeline::CheckArgsValid(args);
  // Record input args info
  std::string input_args_id;
  for (size_t i = 0; i < args.size(); ++i) {
    input_args_id += GetId(args[i]) + "_";
  }
  // Run forward first need plus 1
  if (grad_order_ == 0) {
    ++grad_order_;
  }
  // The number of top cell exceeds MAX_TOP_CELL_COUNTS, delete the last one to keep the maximum length of the list,
  // disable backend cache
  if (top_cell_list_.size() >= MAX_TOP_CELL_COUNTS) {
    EnableOpGraphCache(false);
    const auto last_top_cell = top_cell_list_.back();
    top_cell_list_.pop_back();
    MS_EXCEPTION_IF_NULL(last_top_cell);
    last_top_cell->Clear();
    (void)already_run_top_cell_.erase(last_top_cell->already_run_cell_id());
  }
  // Create top cell
  curr_g_ = std::make_shared<FuncGraph>();
  auto df_builder = std::make_shared<FuncGraph>();
  auto resource = std::make_shared<pipeline::Resource>();
  const auto &already_run_cell_id = GetAlreadyRunCellId(cell_id);
  auto top_cell =
    std::make_shared<TopCellInfo>(is_topest, grad_order_, resource, df_builder, cell_id, already_run_cell_id);
  top_cell->set_forward_already_run(true);
  top_cell->set_input_args_id(input_args_id);
  top_cell_list_.emplace_back(top_cell);
  PushHighOrderGraphStack(top_cell);
  set_top_cell(top_cell);
  MS_LOG(DEBUG) << "New top graph, curr_g ptr " << curr_g_.get() << " resource ptr " << resource.get();
}

void GradExecutor::SetTupleArgsToGraphInfoMap(const FuncGraphPtr &g, const py::object &args, const AnfNodePtr &node,
                                              bool is_param) {
  if (!py::isinstance<py::tuple>(args) && !py::isinstance<py::list>(args)) {
    return;
  }
  auto tuple = args.cast<py::tuple>();
  auto tuple_size = static_cast<int64_t>(tuple.size());
  for (int64_t i = 0; i < tuple_size; ++i) {
    // tuple slice used size_t
    auto id = GetId(tuple[static_cast<size_t>(i)]);
    if (is_param && node->isa<Parameter>()) {
      auto param = node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(param);
      SetParamNodeMapInGraphInfoMap(g, id, param);
    }
    SetNodeMapInGraphInfoMap(g, id, node, i);
    SetTupleItemArgsToGraphInfoMap(g, tuple[i], node, std::vector<int64_t>{i}, is_param);
  }
}

void GradExecutor::SetTupleItemArgsToGraphInfoMap(const FuncGraphPtr &g, const py::object &args, const AnfNodePtr &node,
                                                  const std::vector<int64_t> &index_sequence, bool is_param) {
  if (!py::isinstance<py::tuple>(args) && !py::isinstance<py::list>(args)) {
    return;
  }
  MS_EXCEPTION_IF_NULL(node);
  auto tuple = args.cast<py::tuple>();
  auto tuple_size = static_cast<int64_t>(tuple.size());
  for (int64_t i = 0; i < tuple_size; ++i) {
    std::vector<int64_t> tmp = index_sequence;
    tmp.emplace_back(i);
    // tuple slice used size_t
    auto id = GetId(tuple[static_cast<size_t>(i)]);
    if (is_param && node->isa<Parameter>()) {
      auto param = node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(param);
      SetParamNodeMapInGraphInfoMap(g, id, param);
    }
    SetNodeMapInGraphInfoMap(g, id, node, tmp);
    SetTupleItemArgsToGraphInfoMap(g, tuple[i], node, tmp, is_param);
  }
}

void GradExecutor::CreateMakeTupleNodeForMultiOut(const FuncGraphPtr &curr_g, const py::object &out,
                                                  const std::string &out_id) {
  MS_EXCEPTION_IF_NULL(curr_g);
  const auto &out_tuple = out.cast<py::tuple>();
  // get input node and value
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimMakeTuple)};
  ValuePtrList input_args;
  std::vector<size_t> value_index;
  for (size_t i = 0; i < out_tuple.size(); i++) {
    const auto &v = PyObjToValue(out_tuple[i]);
    // Graph have no define for grad
    if (v->isa<FuncGraph>()) {
      continue;
    }
    value_index.emplace_back(i);
    input_args.emplace_back(v);
    inputs.emplace_back(GetInput(out_tuple[i], false));
  }
  py::tuple value_outs(value_index.size());
  for (size_t i = 0; i < value_index.size(); ++i) {
    value_outs[i] = out_tuple[value_index[i]];
  }
  auto cnode = curr_g_->NewCNode(inputs);
  MS_LOG(DEBUG) << "Tuple output node info " << cnode->DebugString();
  // record node info in graph map
  SetTupleArgsToGraphInfoMap(curr_g_, out, cnode);
  SetNodeMapInGraphInfoMap(curr_g_, out_id, cnode);
  if (grad_is_running_ && !bprop_grad_stack_.top().second) {
    MS_LOG(DEBUG) << "Custom bprop, no need GradPynativeOp";
    return;
  }
  // run ad for maketuple node
  const auto &out_value = PyObjToValue(value_outs);
  ad::GradPynativeOp(top_cell()->k_pynative_cell_ptr(), cnode, input_args, out_value);
}

void GradExecutor::EndGraphInner(py::object *ret, const py::object &cell, const py::object &out, const py::args &args) {
  MS_EXCEPTION_IF_NULL(ret);
  const auto &cell_id = GetCellId(cell, args);
  MS_LOG(DEBUG) << "EndGraphInner start " << args.size() << " " << cell_id;
  if (cell_stack_.empty()) {
    MS_LOG(DEBUG) << "Current cell " << cell_id << " no need to run EndGraphInner again";
    if (cell_id == top_cell()->cell_id()) {
      if (top_cell()->is_topest()) {
        set_grad_flag(false);
      }
      if (GetHighOrderStackSize() < ARG_SIZE) {
        auto outer_top_cell = PopHighOrderGraphStack();
        if (outer_top_cell != nullptr) {
          set_top_cell(outer_top_cell);
        }
      }
    }
    return;
  }
  // Make output node in this case: x = op1, y = op2, return (x, y)
  const auto &out_id = GetId(out);
  const auto &graph_info = top_cell()->graph_info_map().at(curr_g_);
  MS_EXCEPTION_IF_NULL(graph_info);
  if (graph_info->node_map.find(out_id) == graph_info->node_map.end()) {
    if (py::isinstance<py::tuple>(out) || py::isinstance<py::list>(out)) {
      CreateMakeTupleNodeForMultiOut(curr_g_, out, out_id);
    } else {
      MS_LOG(DEBUG) << "Set ValueNode as output for graph, out id: " << out_id;
      MakeValueNode(out, out_id);
    }
  }
  DoGradForCustomBprop(cell, out, args);
  // Set output node for forward graph when need.
  PopCellStack();
  if (grad_is_running_ && !bprop_grad_stack_.empty()) {
    if (!bprop_grad_stack_.top().second) {
      bprop_grad_stack_.pop();
      MS_EXCEPTION_IF_NULL(curr_g_);
      curr_g_->set_output(GetObjNode(out, out_id));
      return;
    } else if (bprop_grad_stack_.top().first == cell_id) {
      bprop_grad_stack_.pop();
    }
  }

  bool is_top_cell_end = cell_id == top_cell()->cell_id();
  // Just only dump the last forward graph
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG) && is_top_cell_end) {
    curr_g_->set_output(GetObjNode(out, out_id));
#ifdef ENABLE_DUMP_IR
    DumpIR("fg.ir", curr_g_);
#endif
  }

  // Reset grad flag and update output node of top cell
  if (cell_stack_.empty() && is_top_cell_end) {
    MS_LOG(DEBUG) << "Cur top last cell " << cell_id;
    set_grad_flag(false);
    PopHighOrderGraphStack();
    // Update real output node of top cell for generating bprop graph
    AnfNodePtr output_node = GetObjNode(out, out_id);
    MS_EXCEPTION_IF_NULL(output_node);
    auto k_pynative_cell_ptr = top_cell()->k_pynative_cell_ptr();
    MS_EXCEPTION_IF_NULL(k_pynative_cell_ptr);
    k_pynative_cell_ptr->UpdateOutputNodeOfTopCell(output_node);
  }

  // Checkout whether need to compile graph when top cell has ran finished
  if (is_top_cell_end) {
    CheckNeedCompileGraph();
  }
}

void GradExecutor::DoGradForCustomBprop(const py::object &cell, const py::object &out, const py::args &args) {
  if (!py::hasattr(cell, parse::CUSTOM_BPROP_NAME)) {
    return;
  }
  custom_bprop_cell_count_ -= 1;
  if (custom_bprop_cell_count_ != 0) {
    return;
  }
  MS_LOG(DEBUG) << "Do grad for custom bprop";
  size_t par_number = py::tuple(parse::python_adapter::CallPyObjMethod(cell, "get_parameters")).size();
  if (par_number > 0) {
    MS_LOG(EXCEPTION) << "When user defines the net bprop, the 'Parameter' data type is not supported in the net.";
  }
  py::function bprop_func = py::getattr(cell, parse::CUSTOM_BPROP_NAME);
  auto bprop_func_cellid = GetId(bprop_func);
  bprop_cell_list_.emplace_back(bprop_func_cellid);
  auto fake_prim = std::make_shared<PrimitivePy>(prim::kPrimHookBackward->name());
  fake_prim->set_hook(bprop_func);
  const auto &cell_id = GetCellId(cell, args);
  (void)fake_prim->AddAttr("cell_id", MakeValue(cell_id));
  (void)fake_prim->AddAttr(parse::CUSTOM_BPROP_NAME, MakeValue(true));

  py::object code_obj = py::getattr(bprop_func, "__code__");
  py::object co_name = py::getattr(code_obj, "co_name");
  if (std::string(py::str(co_name)) == "staging_specialize") {
    MS_LOG(EXCEPTION) << "Decorating bprop with '@ms_function' is not supported.";
  }
  // Three parameters self, out and dout need to be excluded
  const size_t inputs_num = py::cast<int64_t>(py::getattr(code_obj, "co_argcount")) - 3;
  if (inputs_num > args.size()) {
    MS_EXCEPTION(TypeError) << "Size of bprop func inputs[" << inputs_num << "] is larger than size of cell inputs["
                            << args.size() << "]";
  }

  py::list cell_inputs;
  for (size_t i = 0; i < inputs_num; i += 1) {
    cell_inputs.append(args[i]);
  }
  OpExecInfoPtr op_exec_info = std::make_shared<OpExecInfo>();
  op_exec_info->op_name = fake_prim->name();
  op_exec_info->py_primitive = fake_prim;
  op_exec_info->op_inputs = cell_inputs;
  auto cnode = forward()->ConstructForwardGraph(op_exec_info);
  const auto &v_out = PyObjToValue(out);
  DoOpGrad(op_exec_info, cnode, v_out);
  const auto &out_obj_id = GetId(out);
  SaveOutputNodeMap(out_obj_id, out, cnode);
}

std::string GradExecutor::GetAlreadyRunCellId(const std::string &cell_id) {
  std::string already_run_cell_id;
  if (IsNestedGrad()) {
    already_run_cell_id = cell_id + "0";
  } else {
    already_run_cell_id = cell_id + "1";
  }
  already_run_cell_id += "_" + grad_operation_;
  MS_LOG(DEBUG) << "Get already run top cell id " << already_run_cell_id;
  return already_run_cell_id;
}

std::string GradExecutor::GetGradCellId(bool has_sens, const py::object &cell, const py::args &args) {
  size_t forward_args_size = args.size();
  py::args tmp = args;
  if (has_sens) {
    forward_args_size--;
    py::tuple f_args(forward_args_size);
    for (size_t i = 0; i < forward_args_size; ++i) {
      f_args[i] = args[i];
    }
    tmp = f_args;
  }
  const auto &cell_id = GetCellId(cell, tmp);
  return cell_id;
}

void GradExecutor::GradNetInner(py::object *ret, const prim::GradOperationPtr &grad, const py::object &cell,
                                const py::object &weights, const py::args &args) {
  MS_EXCEPTION_IF_NULL(ret);
  MS_EXCEPTION_IF_NULL(grad);
  auto size = args.size();
  const auto &cell_id = GetGradCellId(grad->sens_param(), cell, args);
  MS_LOG(DEBUG) << "GradNet start " << size << " " << cell_id;
  if (!top_cell()->need_compile_graph()) {
    MS_LOG(DEBUG) << "No need compile graph";
    UpdateTopCellInfo(false, false, false);
    return;
  }
  top_cell()->set_grad_operation(grad_operation_);
  auto resource = top_cell()->resource();
  MS_EXCEPTION_IF_NULL(resource);
  auto df_builder = top_cell()->df_builder();
  MS_EXCEPTION_IF_NULL(df_builder);
  MS_LOG(DEBUG) << "curr_g ptr " << curr_g_.get() << " resource ptr " << resource.get();

  // Get params(weights) require derivative
  auto w_args = GetWeightsArgs(weights, df_builder);
  if (w_args.empty() && !df_builder->parameters().empty()) {
    MS_LOG(DEBUG) << "Add weights params to w_args";
    w_args.insert(w_args.end(), df_builder->parameters().begin(), df_builder->parameters().end());
  }
  // Get bprop graph of top cell
  auto bprop_graph = GetBpropGraph(grad, cell, w_args, size, args);
  resource->set_func_graph(bprop_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_graph, true);
  DumpGraphIR("launch_bprop_graph.ir", bprop_graph);
  // Launch bprop graph to backend
  SaveForwardTensorInfoInBpropGraph(resource);
  compile::SetMindRTEnable();
  resource->results()[pipeline::kBackend] = compile::CreateBackend();
  MS_LOG(DEBUG) << "Start task emit action";
  TaskEmitAction(resource);
  MS_LOG(DEBUG) << "Start execute action";
  ExecuteAction(resource);
  MS_LOG(DEBUG) << "Start update top cell info when run finish";
  UpdateTopCellInfo(false, false, true);
  resource->Clean();
  abstract::AnalysisContext::ClearContext();
}

std::vector<AnfNodePtr> GradExecutor::GetWeightsArgs(const py::object &weights, const FuncGraphPtr &df_builder) {
  MS_EXCEPTION_IF_NULL(df_builder);
  if (!py::hasattr(weights, "__parameter_tuple__")) {
    MS_LOG(DEBUG) << "No parameter tuple get";
    return {};
  }

  const auto &tuple = weights.cast<py::tuple>();
  MS_LOG(DEBUG) << "Get weights tuple size " << tuple.size();
  std::vector<AnfNodePtr> w_args;
  for (size_t it = 0; it < tuple.size(); ++it) {
    auto param = tuple[it];
    auto param_id = GetId(param);
    auto &graph_info_map = top_cell()->graph_info_map();
    if (graph_info_map.find(df_builder) == graph_info_map.end()) {
      MS_LOG(EXCEPTION) << "Can not find df_builder " << df_builder.get() << " Top cell " << top_cell().get()
                        << " cell id " << top_cell()->cell_id();
    }
    auto graph_info = graph_info_map.at(df_builder);
    MS_EXCEPTION_IF_NULL(graph_info);
    AnfNodePtr para_node = nullptr;
    if (graph_info->params.find(param_id) != graph_info->params.end()) {
      para_node = graph_info->params.at(param_id);
      w_args.emplace_back(para_node);
      continue;
    }
    const auto &name_attr = parse::python_adapter::GetPyObjAttr(param, "name");
    if (py::isinstance<py::none>(name_attr)) {
      MS_LOG(EXCEPTION) << "Parameter object should have name attribute";
    }
    const auto &param_name = py::cast<std::string>(name_attr);
    MS_LOG(DEBUG) << "The input " << it << " parameter weight name " << param_name;
    if (graph_info->params.find(param_name) != graph_info->params.end()) {
      para_node = graph_info->params.at(param_name);
    } else {
      MS_LOG(DEBUG) << "Can not find input param in graph info map, make a new parameter";
      auto free_param = df_builder->add_parameter();
      free_param->set_name(param_name);
      auto value = py::cast<tensor::TensorPtr>(param);
      free_param->set_default_param(value);
      free_param->debug_info()->set_name(param_name);
      para_node = free_param;
    }
    w_args.emplace_back(para_node);
  }
  return w_args;
}

abstract::AbstractBasePtrList GradExecutor::GetArgsSpec(const py::list &args, const FuncGraphPtr &bprop_graph) {
  MS_EXCEPTION_IF_NULL(bprop_graph);
  std::size_t size = args.size();
  abstract::AbstractBasePtrList args_spec;
  const auto &bprop_params = bprop_graph->parameters();
  // bprop_params include inputs, parameters, more than size(inputs)
  if (bprop_params.size() < size) {
    MS_LOG(EXCEPTION) << "Df parameters size " << bprop_params.size() << " less than " << size;
  }
  // Update abstract info for parameters in bprop graph
  size_t index = 0;
  for (const auto &param : bprop_params) {
    auto param_node = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_node);
    if (param_node->has_default()) {
      // update abstract info for weights
      ValuePtr value = param_node->default_param();
      auto ptr = value->ToAbstract();
      MS_EXCEPTION_IF_NULL(ptr);
      args_spec.emplace_back(ptr);
      param_node->set_abstract(ptr->Broaden());
    } else {
      // update abstract info for input params
      const auto &input_value = PyObjToValue(args[index]);
      auto input_abs = abstract::FromValue(input_value, true);
      if (param_node->abstract() != nullptr) {
        auto input_shape = input_abs->BuildShape()->ToString();
        auto param_tensor_abs = param_node->abstract();
        if (param_tensor_abs->isa<abstract::AbstractRef>()) {
          param_tensor_abs = param_tensor_abs->cast<abstract::AbstractRefPtr>()->CloneAsTensor();
        }
        auto ir_shape = param_tensor_abs->BuildShape()->ToString();
        // Exclude const input
        if (input_shape != "()" && ir_shape != "()") {
          if (input_shape != ir_shape) {
            MS_EXCEPTION(ValueError) << "The shape should be " << ir_shape << ", but got " << input_shape << ", "
                                     << param->DebugString();
          }
          auto ir_dtype = param_tensor_abs->BuildType()->ToString();
          auto input_dtype = input_abs->BuildType()->ToString();
          if (input_dtype != ir_dtype) {
            MS_EXCEPTION(TypeError) << "The dtype should be " << ir_dtype << ", but got " << input_dtype << ", "
                                    << param->DebugString();
          }
        }
      }
      args_spec.emplace_back(input_abs);
      param_node->set_abstract(input_abs->Broaden());
      index++;
    }
  }
  MS_LOG(DEBUG) << "Args_spec size " << args_spec.size();
  return args_spec;
}

FuncGraphPtr GradExecutor::GetBpropGraph(const prim::GradOperationPtr &grad, const py::object &cell,
                                         const std::vector<AnfNodePtr> &weights, size_t arg_size,
                                         const py::args &args) {
  bool build_formal_param = false;
  if (!py::hasattr(cell, parse::CUSTOM_BPROP_NAME) && !cell_stack_.empty() && IsNestedGrad()) {
    build_formal_param = true;
    need_renormalize_ = true;
  }
  if (top_cell()->ms_function_flag()) {
    need_renormalize_ = true;
  }

  auto k_pynative_cell_ptr = top_cell()->k_pynative_cell_ptr();
  MS_EXCEPTION_IF_NULL(k_pynative_cell_ptr);
  MS_EXCEPTION_IF_NULL(grad);
  FuncGraphPtr bprop_graph = ad::GradPynativeCellEnd(k_pynative_cell_ptr, weights, grad->get_all_, grad->get_by_list_,
                                                     grad->sens_param_, build_formal_param);
  MS_EXCEPTION_IF_NULL(bprop_graph);

  MS_LOG(DEBUG) << "Top graph input params size " << arg_size;
  std::ostringstream ss;
  ss << "grad{" << arg_size << "}";
  bprop_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  bprop_graph->debug_info()->set_name(ss.str());
  // Get the parameters items and add the value to args_spec
  (void)GetArgsSpec(FilterTensorArgs(args, grad->sens_param_), bprop_graph);

  // Do opt for final bprop graph
  pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(bprop_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_graph);
  auto optimized_bg = ad::PrimBpropOptimizer::GetPrimBpropOptimizerInst().BpropGraphFinalOpt(resource);

  if (cell_stack_.empty()) {
    need_renormalize_ = false;
  }
  DumpGraphIR("after_final_opt.ir", optimized_bg);
  return optimized_bg;
}

py::object GradExecutor::CheckGraph(const py::object &cell, const py::args &args) {
  BaseRef ret = false;
  check_graph_cell_id_ = GetCellId(cell, args);
  if (!(top_cell_ != nullptr && check_graph_cell_id_.find(top_cell_->cell_id()) != std::string::npos &&
        grad_order_ >= 1)) {
    ++grad_order_;
  }
  if (!grad_is_running_) {
    MS_LOG(DEBUG) << "Grad not running yet";
    return BaseRefToPyData(ret);
  }
  MS_LOG(DEBUG) << "Key is " << check_graph_cell_id_;
  if (top_cell_ != nullptr) {
    for (auto it = top_cell_->sub_cell_list().begin(); it != top_cell_->sub_cell_list().end(); ++it) {
      MS_LOG(DEBUG) << "Cur cell id " << *it;
      if (!IsCellObjIdEq(*it, check_graph_cell_id_)) {
        continue;
      }
      MS_LOG(DEBUG) << "Delete cellid from cell graph list, top cell is " << top_cell_;
      top_cell_->sub_cell_list().erase(it);
      ret = true;
      break;
    }
  }
  return BaseRefToPyData(ret);
}

py::object GradExecutor::CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &cell,
                                         const py::args &args) {
  bool forward_run = false;
  // Get cell id and input args info
  const auto &cell_id = GetCellId(cell, args);
  grad_operation_ = std::to_string(grad->get_all_) + std::to_string(grad->get_by_list_);

  std::string input_args_id;
  for (size_t i = 0; i < args.size(); ++i) {
    input_args_id += GetId(args[i]) + "_";
  }
  // Check whether need to run forward process
  const auto &check_already_run_cell_id = GetAlreadyRunCellId(cell_id);
  auto find_top_cell = GetTopCell(check_already_run_cell_id);
  if (find_top_cell != nullptr) {
    forward_run = find_top_cell->forward_already_run();
    auto curr_top_cell = top_cell();
    set_top_cell(find_top_cell);
    bool input_args_changed =
      !find_top_cell->input_args_id().empty() && find_top_cell->input_args_id() != input_args_id;
    if (forward_run && input_args_changed && find_top_cell->is_dynamic()) {
      MS_LOG(WARNING) << "The construct of running cell is dynamic and the input info of this cell has changed, "
                         "forward process will run again";
      forward_run = false;
    }
    if (forward_run && GetHighOrderStackSize() >= 1) {
      PushHighOrderGraphStack(curr_top_cell);
    }
  }
  MS_LOG(DEBUG) << "Graph have already ran " << forward_run << " top cell id " << cell_id;
  return BaseRefToPyData(forward_run);
}

void GradExecutor::CheckNeedCompileGraph() {
  auto new_top_cell = top_cell();
  const auto &already_top_cell_id = new_top_cell->already_run_cell_id();
  // Update top cell by current cell op info
  if (already_run_top_cell_.find(already_top_cell_id) == already_run_top_cell_.end()) {
    MS_LOG(DEBUG) << "Top cell " << new_top_cell->cell_id() << " has never been ran, need compile graph";
    already_run_top_cell_[already_top_cell_id] = new_top_cell;
    return;
  }

  MS_LOG(DEBUG) << "Top cell " << new_top_cell->cell_id() << " has been ran";
  auto pre_top_cell = already_run_top_cell_.at(already_top_cell_id);
  MS_EXCEPTION_IF_NULL(pre_top_cell);
  const auto &pre_all_op_info = pre_top_cell->all_op_info();
  const auto &new_all_op_info = new_top_cell->all_op_info();
  MS_LOG(DEBUG) << "Pre all op info : " << pre_all_op_info;
  MS_LOG(DEBUG) << "New all op info : " << new_all_op_info;
  if (pre_all_op_info != new_all_op_info) {
    MS_LOG(DEBUG) << "The op info has been changed, need to compile graph again";
    // The top cell switches exceeds MAX_TOP_CELL_COUNTS under the control flow, disable backend cache
    if (top_cell_switch_counts_ >= MAX_TOP_CELL_COUNTS) {
      EnableOpGraphCache(false);
    } else {
      // Increase top cell switches counts
      ++top_cell_switch_counts_;
    }
    EraseTopCellFromTopCellList(pre_top_cell);
    pre_top_cell->Clear();
    already_run_top_cell_[already_top_cell_id] = new_top_cell;
    g_pyobj_id_cache.clear();
  } else {
    MS_LOG(DEBUG) << "The op info has not been changed, no need to compile graph again";
    pre_top_cell->set_input_args_id(new_top_cell->input_args_id());
    EraseTopCellFromTopCellList(new_top_cell);
    new_top_cell->Clear();
    pre_top_cell->set_forward_already_run(true);
    set_top_cell(pre_top_cell);
  }
}

void GradExecutor::RunGradGraph(py::object *ret, const py::object &cell, const py::tuple &args) {
  MS_EXCEPTION_IF_NULL(ret);
  const auto &cell_id = GetCellId(cell, args);
  MS_LOG(DEBUG) << "Run start cell id " << cell_id;
  auto has_sens = std::any_of(top_cell_list_.begin(), top_cell_list_.end(), [&cell_id](const TopCellInfoPtr &value) {
    return cell_id.find(value->cell_id()) != std::string::npos && cell_id != value->cell_id();
  });
  MS_LOG(DEBUG) << "Run has sens " << has_sens << " cell id " << cell_id;
  auto resource = top_cell()->resource();
  MS_EXCEPTION_IF_NULL(resource);
  MS_LOG(DEBUG) << "Run resource ptr " << resource.get();

  VectorRef arg_list;
  py::tuple converted_args = ConvertArgs(FilterTensorArgs(args, has_sens));
  pipeline::ProcessVmArgInner(converted_args, resource, &arg_list);
  if (resource->results().find(pipeline::kOutput) == resource->results().end()) {
    MS_LOG(EXCEPTION) << "Can't find run graph output";
  }
  if (!resource->results()[pipeline::kOutput].is<compile::VmEvalFuncPtr>()) {
    MS_LOG(EXCEPTION) << "Run graph is not VmEvalFuncPtr";
  }
  compile::VmEvalFuncPtr run = resource->results()[pipeline::kOutput].cast<compile::VmEvalFuncPtr>();
  MS_EXCEPTION_IF_NULL(run);

  const auto &backend = MsContext::GetInstance()->backend_policy();
  MS_LOG(DEBUG) << "Eval run " << backend;
  grad_is_running_ = true;
  BaseRef value = (*run)(arg_list);
  grad_is_running_ = false;
  MS_LOG(DEBUG) << "Eval run end " << value.ToString();
  *ret = BaseRefToPyData(value);
  // Clear device memory resource of top cell when it has been ran.
  auto has_higher_order = std::any_of(top_cell_list_.begin(), top_cell_list_.end(),
                                      [](const TopCellInfoPtr &value) { return !value->is_topest(); });
  if (top_cell()->is_topest() && !has_higher_order) {
    top_cell()->ClearDeviceMemory();
  }
  // High order
  if (top_cell()->vm_compiled()) {
    MakeNestedCnode(cell, converted_args, resource, *ret);
  } else if (GetHighOrderStackSize() >= ARG_SIZE) {
    SwitchTopcell();
  }
}

void GradExecutor::SwitchTopcell() {
  const auto &inner_top_cell_all_op_info = top_cell()->all_op_info();
  bool inner_top_cell_is_dynamic = top_cell()->is_dynamic();
  top_cell()->set_grad_order(1);

  // Get outer top cell
  auto outer_top_cell = PopHighOrderGraphStack();
  MS_EXCEPTION_IF_NULL(outer_top_cell);
  outer_top_cell->all_op_info() += inner_top_cell_all_op_info;
  // If inner is dynamic, outer set dynamic too
  if (inner_top_cell_is_dynamic) {
    outer_top_cell->set_is_dynamic(inner_top_cell_is_dynamic);
  }
  set_top_cell(outer_top_cell);
}

void GradExecutor::DoParameterReplace(const FuncGraphPtr &first_grad_fg, const py::tuple &forward_args,
                                      std::vector<AnfNodePtr> *inputs, ValuePtrList *weights_args) {
  MS_EXCEPTION_IF_NULL(inputs);
  MS_EXCEPTION_IF_NULL(weights_args);
  auto first_df_builder = top_cell()->df_builder();
  MS_EXCEPTION_IF_NULL(first_df_builder);
  auto first_graph_info = top_cell()->graph_info_map().at(first_df_builder);
  MS_EXCEPTION_IF_NULL(first_graph_info);
  SwitchTopcell();
  auto second_df_builder = top_cell()->df_builder();
  MS_EXCEPTION_IF_NULL(second_df_builder);
  auto second_graph_info = top_cell()->graph_info_map().at(second_df_builder);
  MS_EXCEPTION_IF_NULL(second_graph_info);

  std::unordered_set<std::string> params_weights_set;
  std::unordered_set<std::string> params_inputs_set;
  for (const auto &sec : second_graph_info->params) {
    if (sec.second->has_default()) {
      params_weights_set.emplace(sec.first);
    } else {
      params_inputs_set.insert(sec.first);
    }
  }
  auto manager = Manage({first_grad_fg}, false);
  // Replace inputs param
  for (size_t i = 0; i < forward_args.size(); ++i) {
    const auto &id = GetId(forward_args[i]);
    if (params_inputs_set.count(id)) {
      // Can find in second graph
      const auto &input_param_second = second_graph_info->params.at(id);
      manager->Replace(first_graph_info->params.at(id), input_param_second);
      inputs->emplace_back(input_param_second);
    } else {
      inputs->emplace_back(GetInput(forward_args[i], false));
    }
  }

  // Replace weights param
  for (const auto &fir : first_graph_info->params) {
    if (!fir.second->has_default()) {
      continue;
    }
    // Second graph no this weight param, need add to second graph
    if (!params_weights_set.count(fir.first)) {
      SetParamNodeMapInGraphInfoMap(second_df_builder, fir.first, fir.second);
      inputs->emplace_back(fir.second);
      weights_args->emplace_back(fir.second->default_param());
    } else {
      // Need replace
      for (const auto &sec : second_graph_info->params) {
        MS_LOG(DEBUG) << "Param name " << fir.first << " ptr " << fir.second.get();
        if (sec.second->has_default() && fir.second->name() == sec.second->name()) {
          manager->Replace(fir.second, sec.second);
          inputs->emplace_back(sec.second);
          weights_args->emplace_back(sec.second->default_param());
          break;
        }
      }
    }
  }
}

void GradExecutor::MakeNestedCnode(const py::object &cell, const py::tuple &forward_args,
                                   const pipeline::ResourcePtr &resource, const py::object &out) {
  if (cell_stack_.empty()) {
    MS_LOG(DEBUG) << "No nested grad find";
    return;
  }
  FuncGraphPtr first_grad_fg = nullptr;
  if (py::hasattr(cell, parse::CUSTOM_BPROP_NAME)) {
    first_grad_fg = curr_g_;
    MS_LOG(DEBUG) << "Bprop nested";
  } else {
    first_grad_fg = resource->func_graph();
  }
  MS_EXCEPTION_IF_NULL(first_grad_fg);
  DumpGraphIR("first_grad_fg.ir", first_grad_fg);

  std::vector<AnfNodePtr> inputs{NewValueNode(first_grad_fg)};
  ValuePtrList weights_args;
  DoParameterReplace(first_grad_fg, forward_args, &inputs, &weights_args);

  pipeline::ResourcePtr r = std::make_shared<pipeline::Resource>();
  r->manager()->AddFuncGraph(first_grad_fg);
  set_eliminate_forward(false);
  FuncGraphPtr second_grad_fg = ad::Grad(first_grad_fg, r);
  set_eliminate_forward(true);
  DumpGraphIR("second_grad_fg.ir", second_grad_fg);
  r->Clean();

  MS_LOG(DEBUG) << "Get pre graph ptr " << curr_g().get();
  auto cnode = curr_g()->NewCNode(inputs);
  auto out_id = GetId(out);
  SetTupleArgsToGraphInfoMap(curr_g(), out, cnode);
  SetNodeMapInGraphInfoMap(curr_g(), out_id, cnode);
  MS_LOG(DEBUG) << "Nested make cnode is " << cnode->DebugString();

  // Get input values
  ValuePtrList input_args;
  for (size_t i = 0; i < forward_args.size(); ++i) {
    const auto &arg = PyObjToValue(forward_args[i]);
    input_args.emplace_back(arg);
  }
  input_args.insert(input_args.end(), weights_args.begin(), weights_args.end());
  // Get output values
  py::object new_out;
  if (py::hasattr(cell, parse::CUSTOM_BPROP_NAME) && !py::isinstance<py::tuple>(out)) {
    new_out = py::make_tuple(out);
  } else {
    new_out = out;
  }
  const auto &out_value = PyObjToValue(new_out);
  if (!top_cell()->k_pynative_cell_ptr()->KPynativeWithFProp(cnode, input_args, out_value, second_grad_fg)) {
    MS_LOG(EXCEPTION) << "Failed to run ad grad for second grad graph " << cnode->ToString();
  }
  need_renormalize_ = true;
}

void GradExecutor::EraseTopCellFromTopCellList(const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(top_cell);
  auto iter = std::find_if(top_cell_list_.begin(), top_cell_list_.end(),
                           [&](const TopCellInfoPtr &elem) { return elem.get() == top_cell.get(); });
  if (iter == top_cell_list_.end()) {
    MS_LOG(WARNING) << "Can not find top cell " << top_cell.get() << " cell id " << top_cell->cell_id()
                    << " from top cell list";
  } else {
    (void)top_cell_list_.erase(iter);
  }
}

void GradExecutor::GradMsFunctionInner(const std::string &phase, const py::object &out, const py::args &args,
                                       const FuncGraphPtr &ms_func_graph, const FuncGraphPtr &grad_graph) {
  // Get actual output value and added output value.
  if (!py::isinstance<py::tuple>(out)) {
    MS_LOG(EXCEPTION) << "The output value of ms_function func graph should be a tuple.";
  }
  auto tuple_out = py::cast<py::tuple>(out);
  constexpr size_t tuple_out_size = 2;
  if (tuple_out.size() != tuple_out_size) {
    MS_LOG(EXCEPTION) << "The tuple size of output value of ms_function func graph should be 2.";
  }
  py::object actual_out = tuple_out[0];
  auto actual_out_v = PyObjToValue(actual_out);
  auto added_out = PyObjToValue(tuple_out[1]);
  MS_LOG(DEBUG) << "Added output value is: " << added_out->ToString();

  // Identity op info for current running ms_func graph.
  OpExecInfoPtr op_exec_info = std::make_shared<OpExecInfo>();
  op_exec_info->op_name = phase;
  RecordGradOpInfo(op_exec_info, actual_out_v);
  MS_LOG(DEBUG) << "ms_function cnode op info: " << op_exec_info->op_info;

  // Step 1: Update actual output tensors used in grad graph.
  MS_LOG(DEBUG) << "ms_function actual output value: " << actual_out_v->ToString();
  UpdateForwardTensorInfoInBpropGraph(op_exec_info, actual_out_v);

  // Step 2: Update output tensors of added forward nodes, which are added to return node of ms_function func graph.
  if (top_cell()->op_info_with_ms_func_forward_tensors().count(op_exec_info->op_info)) {
    UpdateMsFunctionForwardTensors(op_exec_info, added_out);
    return;
  }
  MS_LOG(DEBUG) << "Ms func graph run firstly. The graph phase is: " << graph_phase();
  if (!need_construct_graph()) {
    MS_LOG(EXCEPTION) << "The flag of need construct graph is False.";
  }
  ReplaceNewTensorsInGradGraph(top_cell(), op_exec_info, added_out, ms_func_graph, grad_graph);

  // Clone new ms_function func graph and grad graph.
  auto new_ms_func_graph = BasicClone(ms_func_graph);
  auto new_grad_graph = BasicClone(grad_graph, true);
  auto new_make_tuple = new_ms_func_graph->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(new_make_tuple);
  new_ms_func_graph->set_output(new_make_tuple->input(1));

  // Make Adjoint for grad graph
  MakeAdjointForMsFunction(new_ms_func_graph, new_grad_graph, actual_out, args, actual_out_v);
}

py::object GradExecutor::GradMsFunction(const py::object &out, const py::args &args) {
  // Get actual forward output object.
  if (graph_phase().empty()) {
    MS_LOG(EXCEPTION) << "The graph phase is empty, can not obtain ms_function func graph.";
  }
  const auto &phase = graph_phase();
  MS_LOG(DEBUG) << "ms_function func graph phase: " << phase;
  auto executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(executor);
  FuncGraphPtr ms_func_graph = executor->GetFuncGraph(phase);
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  py::object ret = out;
  if (ms_func_graph->modify_output()) {
    auto tuple_out = py::cast<py::tuple>(out);
    ret = tuple_out[0];
  }

  // Make Adjoint for grad graph of ms_function.
  if (!grad_flag_) {
    MS_LOG(DEBUG) << "Only run forward infer computation, no need to construct grad graph.";
    set_graph_phase("");
    return ret;
  }
  FuncGraphPtr grad_graph = executor->GetGradGraph(phase);
  MS_EXCEPTION_IF_NULL(grad_graph);
  GradMsFunctionInner(phase, out, args, ms_func_graph, grad_graph);
  set_graph_phase("");
  return ret;
}

void GradExecutor::ClearGrad(const py::object &cell, const py::args &args) {
  MS_LOG(DEBUG) << "Clear top cell grad resource " << GetCellId(cell, args);
  if (grad_order_ > 0) {
    --grad_order_;
  }
  check_graph_cell_id_.clear();
  grad_operation_.clear();
  forward()->node_abs_map().clear();
  ad::CleanRes();
  pipeline::ReclaimOptimizer();
}

void GradExecutor::ClearRes() {
  MS_LOG(DEBUG) << "Clear grad res";
  grad_flag_ = false;
  enable_op_cache_ = true;
  grad_is_running_ = false;
  need_renormalize_ = false;
  eliminate_forward_ = true;
  custom_bprop_cell_count_ = 0;
  grad_order_ = 0;
  top_cell_switch_counts_ = 0;

  check_graph_cell_id_.clear();
  grad_operation_.clear();
  top_cell_ = nullptr;
  curr_g_ = nullptr;
  bprop_cell_list_.clear();
  already_run_top_cell_.clear();
  ClearCellRes();
  std::stack<std::pair<std::string, bool>>().swap(bprop_grad_stack_);
  std::stack<std::string>().swap(cell_stack_);
  std::stack<std::pair<FuncGraphPtr, TopCellInfoPtr>>().swap(high_order_stack_);
}

GradExecutorPtr PynativeExecutor::grad_executor() const {
  MS_EXCEPTION_IF_NULL(grad_executor_);
  return grad_executor_;
}
ForwardExecutorPtr PynativeExecutor::forward_executor() const {
  MS_EXCEPTION_IF_NULL(forward_executor_);
  return forward_executor_;
}

bool PynativeExecutor::grad_flag() const { return grad_executor()->grad_flag(); }

void PynativeExecutor::set_grad_flag(bool flag) { grad_executor()->set_grad_flag(flag); }

void PynativeExecutor::set_graph_phase(const std::string &graph_phase) {
  grad_executor()->set_graph_phase(graph_phase);
}

void PynativeExecutor::set_py_exe_path(const py::object &py_exe_path) {
  if (!py::isinstance<py::str>(py_exe_path)) {
    MS_LOG(EXCEPTION) << "Failed, py_exe_path input is not a str";
  }
  auto py_exe_path_s = py::cast<std::string>(py_exe_path);
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_PYTHON_EXE_PATH, py_exe_path_s);
}

void PynativeExecutor::set_kernel_build_server_dir(const py::object &kernel_build_server_dir) {
  if (!py::isinstance<py::str>(kernel_build_server_dir)) {
    MS_LOG(EXCEPTION) << "Failed, kernel_build_server_dir input is not a str";
  }
  auto kernel_build_server_dir_s = py::cast<std::string>(kernel_build_server_dir);
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_KERNEL_BUILD_SERVER_DIR, kernel_build_server_dir_s);
}

py::object PynativeExecutor::CheckGraph(const py::object &cell, const py::args &args) {
  return grad_executor()->CheckGraph(cell, args);
}

py::object PynativeExecutor::CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &cell,
                                             const py::args &args) {
  return grad_executor()->CheckAlreadyRun(grad, cell, args);
}

py::object PynativeExecutor::Run(const py::object &cell, const py::tuple &args) {
  py::object ret;
  PynativeExecutorTry(grad_executor()->RunGraph, &ret, cell, args);
  return ret;
}

void PynativeExecutor::ClearCell(const std::string &cell_id) {
  MS_LOG(DEBUG) << "Clear cell res, cell id " << cell_id;
  grad_executor()->ClearCellRes(cell_id);
}

void PynativeExecutor::ClearGrad(const py::object &cell, const py::args &args) {
  MS_LOG(DEBUG) << "Clear grad";
  return grad_executor()->ClearGrad(cell, args);
}

void PynativeExecutor::ClearRes() {
  MS_LOG(DEBUG) << "Clear all res";
  session::PynativeTaskManager::GetInstance().Reset();
  SetLazyBuild(false);
  cell_depth_ = 0;

  // Maybe exit in runop step
  auto ms_context = MsContext::GetInstance();
  if (ms_context != nullptr) {
    ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, false);
  }
  ConfigManager::GetInstance().ResetIterNum();
  if (forward_executor_ != nullptr) {
    forward_executor_->ClearRes();
  }
  if (grad_executor_ != nullptr) {
    grad_executor_->ClearRes();
  }
  ad::CleanRes();
  pipeline::ReclaimOptimizer();
  kSession = nullptr;
  mind_rt_backend = nullptr;
  g_pyobj_id_cache.clear();
}

void PynativeExecutor::NewGraph(const py::object &cell, const py::args &args) {
  // Make a flag for new cell
  if (!grad_executor()->grad_flag()) {
    MS_LOG(DEBUG) << "Grad flag is false";
    return;
  }
  py::object ret;
  PynativeExecutorTry(grad_executor()->InitGraph, &ret, cell, args);
}

void PynativeExecutor::EndGraph(const py::object &cell, const py::object &out, const py::args &args) {
  if (!grad_executor()->grad_flag()) {
    MS_LOG(DEBUG) << "Grad flag is false";
    return;
  }
  MS_LOG(DEBUG) << "Enter end graph process.";
  py::object ret;
  PynativeExecutorTry(grad_executor()->LinkGraph, &ret, cell, out, args);
  MS_LOG(DEBUG) << "Leave end graph process.";
}

py::object PynativeExecutor::GradMsFunction(const py::object &out, const py::args &args) {
  return grad_executor()->GradMsFunction(out, args);
}

void PynativeExecutor::GradNet(const prim::GradOperationPtr &grad, const py::object &cell, const py::object &weights,
                               const py::args &args) {
  py::object ret;
  PynativeExecutorTry(grad_executor()->GradGraph, &ret, grad, cell, weights, args);
}

void PynativeExecutor::Sync() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    if (kSession == nullptr) {
      MS_EXCEPTION(NotExistsError) << "No session has been created!";
    }
    kSession->SyncStream();
  } else {
    std::string device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    uint32_t device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    const auto &device_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_target, device_id});
    MS_EXCEPTION_IF_NULL(device_context);
    (void)device_context->SyncStream();
  }
}

void PynativeExecutor::SetLazyBuild(bool enable) { forward_executor()->set_lazy_build(enable); }

void PynativeExecutor::EnterCell() {
  if (cell_depth_ < UINT32_MAX) {
    ++cell_depth_;
  } else {
    MS_LOG(ERROR) << "Cell call stack too deep";
  }
}

void PynativeExecutor::ExitCell() {
  if (cell_depth_ > 0) {
    --cell_depth_;
  }
}

bool PynativeExecutor::IsTopCell() const { return cell_depth_ == 0; }

void PynativeExecutor::ExecuteAllTask() { session::PynativeTaskManager::GetInstance().ExecuteRemainingTasks(); }

REGISTER_PYBIND_DEFINE(PynativeExecutor_, ([](const py::module *m) {
                         (void)py::class_<PynativeExecutor, std::shared_ptr<PynativeExecutor>>(*m, "PynativeExecutor_")
                           .def_static("get_instance", &PynativeExecutor::GetInstance, "PynativeExecutor get_instance.")
                           .def("enter_cell", &PynativeExecutor::EnterCell, "enter cell.")
                           .def("exit_cell", &PynativeExecutor::ExitCell, "exit cell.")
                           .def("is_top_cell", &PynativeExecutor::IsTopCell, "check top cell.")
                           .def("new_graph", &PynativeExecutor::NewGraph, "pynative new a graph.")
                           .def("end_graph", &PynativeExecutor::EndGraph, "pynative end a graph.")
                           .def("check_graph", &PynativeExecutor::CheckGraph, "pynative check a grad graph.")
                           .def("check_run", &PynativeExecutor::CheckAlreadyRun, "pynative check graph run before.")
                           .def("grad_ms_function", &PynativeExecutor::GradMsFunction, "pynative grad for ms_function.")
                           .def("grad_net", &PynativeExecutor::GradNet, "pynative grad graph.")
                           .def("clear_cell", &PynativeExecutor::ClearCell, "pynative clear status.")
                           .def("clear_res", &PynativeExecutor::ClearRes, "pynative clear exception res.")
                           .def("clear_grad", &PynativeExecutor::ClearGrad, "pynative clear grad status.")
                           .def("sync", &PynativeExecutor::Sync, "pynative sync stream.")
                           .def("set_lazy_build", &PynativeExecutor::SetLazyBuild, "pynative build kernel async")
                           .def("execute_all_task", &PynativeExecutor::ExecuteAllTask, "clear all task")
                           .def("__call__", &PynativeExecutor::Run, "pynative executor run grad graph.")
                           .def("set_graph_phase", &PynativeExecutor::set_graph_phase, "pynative set graph phase")
                           .def("grad_flag", &PynativeExecutor::grad_flag, "pynative grad flag")
                           .def("set_grad_flag", &PynativeExecutor::set_grad_flag, py::arg("flag") = py::bool_(false),
                                "Executor set grad flag.")
                           .def("set_py_exe_path", &PynativeExecutor::set_py_exe_path,
                                py::arg("py_exe_path") = py::str(""), "set python executable path.")
                           .def("set_kernel_build_server_dir", &PynativeExecutor::set_kernel_build_server_dir,
                                py::arg("kernel_build_server_dir") = py::str(""),
                                "set kernel build server directory path.");
                       }));
}  // namespace mindspore::pynative
