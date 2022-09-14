/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <algorithm>

#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "pipeline/jit/debug/trace.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/pybind_api/api_register.h"
#include "pybind_api/pybind_patch.h"
#include "pybind_api/ir/tensor_py.h"
#include "ir/param_info.h"
#include "ir/anf.h"
#include "ir/cell.h"
#include "ir/tensor.h"
#include "ir/func_graph_cloner.h"
#include "utils/any.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "runtime/device/context_extends.h"
#include "include/common/utils/config_manager.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/scoped_long_running.h"
#include "frontend/optimizer/ad/grad.h"
#include "frontend/optimizer/ad/prim_bprop_optimizer.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/do_signature.h"
#include "include/common/utils/parallel_context.h"
#include "pipeline/jit/action.h"
#include "pipeline/jit/pass.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/parse/parse_dynamic.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "pipeline/jit/static_analysis/auto_monad.h"
#include "pipeline/jit/pipeline.h"
#include "pipeline/jit/resource.h"
#include "pipeline/pynative/base.h"
#include "backend/common/optimizer/const_input_to_attr_factory.h"
#include "backend/common/optimizer/helper.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/hardware/device_context_manager.h"
#include "backend/graph_compiler/transform.h"

using mindspore::tensor::TensorPy;

namespace mindspore::pynative {
PynativeExecutorPtr PynativeExecutor::executor_ = nullptr;
ForwardExecutorPtr PynativeExecutor::forward_executor_ = nullptr;
GradExecutorPtr PynativeExecutor::grad_executor_ = nullptr;
std::mutex PynativeExecutor::instance_lock_;

namespace {
const size_t ARG_SIZE = 2;
const size_t MAX_TOP_CELL_COUNTS = 20;

// primitive unable to infer value for constant input in PyNative mode
const std::set<std::string> kVmOperators = {"InsertGradientOf", "stop_gradient", "mixed_precision_cast", "HookBackward",
                                            "CellBackwardHook"};
const std::set<std::string> kAxisNone = {"ReduceSum"};
const std::set<std::string> kIgnoreInferPrim = {"mixed_precision_cast"};
const std::set<std::string> kForceInferPrim = {"TopK", "DropoutGenMask"};
const std::set<std::string> kSummaryOperators = {"ScalarSummary", "ImageSummary", "TensorSummary", "HistogramSummary"};
const char kOpsFunctionModelName[] = "mindspore.ops.functional";
const char kGrad[] = "grad";
const char kSensInfo[] = "SensInfo";
std::map<std::string, std::shared_ptr<session::SessionBasic>> kSessionBackends;
std::map<std::string, std::shared_ptr<compile::MindRTBackend>> kMindRtBackends;
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
  } catch (const py::index_error &ex) {
    inst->ClearRes();
    throw py::index_error(ex);
  } catch (const py::value_error &ex) {
    inst->ClearRes();
    throw py::value_error(ex);
  } catch (const py::type_error &ex) {
    inst->ClearRes();
    throw py::type_error(ex);
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
  py::object out = python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_MOD_GET_OBJ_ID, obj);
  if (py::isinstance<py::none>(out)) {
    MS_LOG(EXCEPTION) << "Get pyobj failed";
  }
  return out.cast<std::string>();
}

std::string GetId(const py::handle &obj) {
  if (py::isinstance<tensor::Tensor>(obj)) {
    auto tensor_ptr = py::cast<tensor::TensorPtr>(obj);
    return tensor_ptr->id();
  } else if (py::isinstance<Cell>(obj)) {
    return obj.cast<CellPtr>()->id();
  } else if (py::isinstance<mindspore::Type>(obj)) {
    auto type_ptr = py::cast<mindspore::TypePtr>(obj);
    return "type" + type_ptr->ToString();
  } else if (py::isinstance<py::str>(obj)) {
    return "S" + obj.cast<std::string>();
  } else if (py::isinstance<py::int_>(obj)) {
    return "I" + py::str(obj).cast<std::string>();
  } else if (py::isinstance<py::float_>(obj)) {
    return "F" + py::str(obj).cast<std::string>();
  } else if (py::isinstance<py::none>(obj)) {
    return "none";
  } else if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    auto p_list = py::cast<py::tuple>(obj);
    string prefix = py::isinstance<py::tuple>(obj) ? "tuple:" : "list";
    if (p_list.empty()) {
      prefix = "empty";
    } else {
      std::string key;
      prefix += std::accumulate(p_list.begin(), p_list.end(), key,
                                [](const std::string &str, const py::handle &b) { return str + ":" + GetId(b); });
    }
    return prefix;
  }

  if (py::isinstance<py::function>(obj)) {
    auto it = g_pyobj_id_cache.find(obj);
    if (it == g_pyobj_id_cache.end()) {
      auto id = GetPyObjId(obj);
      (void)g_pyobj_id_cache.emplace(obj, id);
      return id;
    } else {
      return it->second;
    }
  } else {
    return GetPyObjId(obj);
  }
}

bool IsFunctionType(const py::object &cell) { return !py::isinstance<Cell>(cell); }

inline bool IsDynamicShape(const OpExecInfoPtr &op_exec_info) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  return op_exec_info->has_dynamic_input || op_exec_info->has_dynamic_output;
}

bool IsParameter(const py::object &inp_obj) {
  bool ret = false;
  if (py::isinstance<tensor::MetaTensor>(inp_obj)) {
    auto meta_tensor = inp_obj.cast<tensor::MetaTensorPtr>();
    MS_EXCEPTION_IF_NULL(meta_tensor);
    ret = meta_tensor->is_parameter();
  }
  return ret;
}

void GetTypeIndex(const std::vector<SignatureEnumDType> &dtypes,
                  mindspore::HashMap<SignatureEnumDType, std::vector<size_t>> *type_indexes) {
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

std::string GetCurrentDeviceTarget(const std::string &device_target, const PrimitivePyPtr &op_prim) {
  MS_EXCEPTION_IF_NULL(op_prim);
  const auto &attr_map = op_prim->attrs();
  auto iter = attr_map.find("primitive_target");
  if (iter != attr_map.end()) {
    return GetValue<std::string>(iter->second);
  }
  return device_target;
}

session::SessionPtr GetCurrentSession(const std::string &device_target, uint32_t device_id) {
  auto iter = kSessionBackends.find(device_target);
  if (iter == kSessionBackends.end()) {
    auto session = session::SessionFactory::Get().Create(device_target);
    MS_EXCEPTION_IF_NULL(session);
    session->Init(device_id);
    kSessionBackends[device_target] = session;
    return session;
  } else {
    return iter->second;
  }
}

compile::MindRTBackendPtr GetMindRtBackend(const std::string &device_target, uint32_t device_id) {
  auto iter = kMindRtBackends.find(device_target);
  if (iter == kMindRtBackends.end()) {
    auto backend = std::make_shared<compile::MindRTBackend>("ms", device_target, device_id);
    MS_EXCEPTION_IF_NULL(backend);
    kMindRtBackends[device_target] = backend;
    return backend;
  } else {
    return iter->second;
  }
}

void GetDstType(const py::tuple &py_args,
                const mindspore::HashMap<SignatureEnumDType, std::vector<size_t>> &type_indexes,
                mindspore::HashMap<SignatureEnumDType, TypeId> *dst_type) {
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
  const auto &type_name = type_name_map().find(type_id);
  if (type_name == type_name_map().cend()) {
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
  MS_LOG(DEBUG) << "Prim " << prim->name() << " infer input: " << mindspore::ToString(args_spec_list);
  prim->BeginRecordAddAttr();
  auto eval_ret = EvalOnePrim(prim, args_spec_list);
  MS_EXCEPTION_IF_NULL(eval_ret);
  AbstractBasePtr infer_res = eval_ret->abstract();
  MS_EXCEPTION_IF_NULL(infer_res);
  prim->EndRecordAddAttr();
  MS_EXCEPTION_IF_NULL(op_exec_info);
  op_exec_info->abstract = infer_res;
  MS_EXCEPTION_IF_NULL(op_exec_info->abstract);
  MS_LOG(DEBUG) << "Prim " << prim->name() << " infer result: " << op_exec_info->abstract->ToString();
}

bool ValueHasDynamicShape(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    return value->cast<tensor::TensorPtr>()->base_shape_ptr() != nullptr;
  } else if (value->isa<ValueSequence>()) {
    auto value_seq = value->cast<ValueSequencePtr>();
    return std::any_of(value_seq->value().begin(), value_seq->value().end(),
                       [](const ValuePtr &elem) { return ValueHasDynamicShape(elem); });
  }
  return false;
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
  const auto &op_prim = op_exec_info->py_primitive;
  MS_EXCEPTION_IF_NULL(op_prim);
  bool has_hidden_side_effect = op_prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_HIDDEN);
  for (size_t index = 0; index < input_tensors.size(); ++index) {
    const auto &input_tensor = input_tensors[index];
    MS_EXCEPTION_IF_NULL(input_tensor);
    if (input_tensor->base_shape_ptr() != nullptr) {
      buf << input_tensor->base_shape_ptr()->ToString();
    } else {
      buf << input_tensor->shape();
    }
    buf << input_tensor->data_type();
    buf << input_tensor->padding_type();
    // In the case of the same shape, but dtype and format are inconsistent
    auto tensor_addr = input_tensor->device_address();
    if (tensor_addr != nullptr && !has_hidden_side_effect) {
      auto p_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor_addr);
      MS_EXCEPTION_IF_NULL(p_address);
      buf << p_address->type_id();
      buf << p_address->format();
    }
    // For constant input
    if (tensors_mask[index] == kValueNodeTensorMask) {
      has_const_input = true;
      buf << common::AnfAlgo::GetTensorValueString(input_tensor);
    }
    buf << "_";
  }
  // The value of the attribute affects the operator selection
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

  // Operator with hidden side effect.
  if (has_hidden_side_effect) {
    buf << "_" << std::to_string(op_prim->id());
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
    if (py::isinstance<tensor::Tensor>(args[i]) || py::isinstance<tensor::CSRTensor>(args[i]) ||
        py::isinstance<tensor::COOTensor>(args[i])) {
      only_tensors.append(args[i]);
    }
  }
  if (has_sens) {
    only_tensors.append(args[forward_args_size]);
  }
  return only_tensors;
}

bool RunOpConvertConstInputToAttr(const OpExecInfoPtr &op_run_info, size_t input_index, const PrimitivePtr &op_prim,
                                  const mindspore::HashSet<size_t> &input_attrs) {
  MS_EXCEPTION_IF_NULL(op_prim);
  const py::object &input_object = op_run_info->op_inputs[input_index];
  if (input_attrs.find(input_index) != input_attrs.end()) {
    const auto &input_names_value = op_prim->GetAttr(kAttrInputNames);
    if (input_names_value == nullptr) {
      return false;
    }
    const auto &input_names_vec = GetValue<std::vector<std::string>>(input_names_value);
    if (input_index >= input_names_vec.size()) {
      MS_LOG(EXCEPTION) << "The input index: " << input_index << " is larger than the input names vector size!";
    }
    const auto &value = PyObjToValue(input_object);
    auto input_name = input_names_vec[input_index];
    if (value->isa<tensor::Tensor>()) {
      auto tensor = value->cast<tensor::TensorPtr>();
      if (tensor->data().const_data() == nullptr) {
        return false;
      }
    }
    op_prim->AddAttr(input_name, value);
    (void)op_run_info->index_with_value.emplace_back(std::make_pair(input_index, value));
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
    (void)input_tensors->emplace_back(tensor);
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
  (void)input_tensors->emplace_back(tensor_ptr);
}

void ConvertCSRTensorToTensorList(const py::object &input_object, const PrimitivePtr &op_prim,
                                  std::vector<tensor::TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(op_prim);
  MS_EXCEPTION_IF_NULL(input_tensors);
  if (!py::isinstance<tensor::CSRTensor>(input_object)) {
    MS_LOG(EXCEPTION) << "The input should be a csr_tensor! ";
  }
  auto input_names = op_prim->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(DEBUG) << "input_names are nullptr";
    return;
  }
  auto csr_inputs = py::cast<tensor::CSRTensor>(input_object);
  (void)input_tensors->emplace_back(csr_inputs.GetIndptr());
  (void)input_tensors->emplace_back(csr_inputs.GetIndices());
  (void)input_tensors->emplace_back(csr_inputs.GetValues());
  op_prim->set_attr("is_csr", MakeValue(true));
  op_prim->set_attr("dense_shape", MakeValue(csr_inputs.shape()));
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
    if (kAxisNone.find(op_prim->name()) != kAxisNone.end()) {
      (void)input_tensors->emplace_back(std::make_shared<tensor::Tensor>(static_cast<int64_t>(0), kInt64));
      return;
    } else {
      MS_LOG(EXCEPTION) << "The size of input list or tuple is 0!";
    }
  }
  if (py::isinstance<tensor::Tensor>(tuple_inputs[0])) {
    PlantTensorTupleToVector(tuple_inputs, op_prim, input_tensors);
  } else {
    ConvertValueTupleToTensor(input_object, input_tensors);
    *tensor_mask = kValueNodeTensorMask;
  }
}

void ConvertPyObjectToTensor(const OpExecInfoPtr &op_run_info, size_t index, const PrimitivePtr &op_prim,
                             std::vector<tensor::TensorPtr> *input_tensors, int64_t *const tensor_mask) {
  MS_EXCEPTION_IF_NULL(op_prim);
  MS_EXCEPTION_IF_NULL(input_tensors);
  MS_EXCEPTION_IF_NULL(tensor_mask);
  const py::object &input_object = op_run_info->op_inputs[index];
  tensor::TensorPtr tensor_ptr = nullptr;
  if (py::isinstance<tensor::Tensor>(input_object)) {
    tensor_ptr = py::cast<tensor::TensorPtr>(input_object);
  } else if (py::isinstance<py::float_>(input_object)) {
    double input_value = py::cast<py::float_>(input_object);
    tensor_ptr = std::make_shared<tensor::Tensor>(input_value, kFloat32);
    *tensor_mask = kValueNodeTensorMask;
  } else if (py::isinstance<py::bool_>(input_object)) {
    tensor_ptr = std::make_shared<tensor::Tensor>(py::cast<bool>(input_object), kBool);
    *tensor_mask = kValueNodeTensorMask;
  } else if (py::isinstance<py::int_>(input_object)) {
    if (op_prim->name() == prim::kPrimCSRReduceSum->name()) {
      op_prim->set_attr("axis", MakeValue(py::cast<int64_t>(input_object)));
      return;
    }
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
  } else if (py::isinstance<tensor::CSRTensor>(input_object)) {
    ConvertCSRTensorToTensorList(input_object, op_prim, input_tensors);
    return;
  } else if (py::isinstance<py::none>(input_object)) {
    (void)op_run_info->index_with_value.emplace_back(std::make_pair(index, kNone));
    return;
  } else {
    MS_LOG(EXCEPTION) << "Run op inputs type is invalid!";
  }
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  (void)input_tensors->emplace_back(tensor_ptr);
}

bool NeedConvertConstInputToAttr(const OpExecInfoPtr &op_run_info, mindspore::HashSet<size_t> *input_to_attr_ptr) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(input_to_attr_ptr);
  if (op_run_info->op_name == prim::kPrimCustom->name()) {
    // Custom op needs to set reg dynamically
    const PrimitivePtr &op_prim = op_run_info->py_primitive;
    MS_EXCEPTION_IF_NULL(op_prim);
    opt::GetCustomOpAttrIndex(op_prim, input_to_attr_ptr);
    return !input_to_attr_ptr->empty();
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto cur_target = GetCurrentDeviceTarget(device_target, op_run_info->py_primitive);
  if (device_target != cur_target) {
    MS_LOG(DEBUG) << "primitive target does not match backend: " << device_target
                  << ", primitive_target: " << cur_target;
    device_target = cur_target;
  }
  *input_to_attr_ptr = opt::ConstInputToAttrRegister::GetInstance().GetConstToAttr(op_run_info->op_name, device_target,
                                                                                   IsDynamicShape(op_run_info));
  return !input_to_attr_ptr->empty();
}

void ConstructInputTensor(const OpExecInfoPtr &op_run_info, std::vector<int64_t> *tensors_mask,
                          std::vector<tensor::TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(tensors_mask);
  MS_EXCEPTION_IF_NULL(input_tensors);

  mindspore::HashSet<size_t> input_to_attr = {};
  bool need_convert_input_to_attr = NeedConvertConstInputToAttr(op_run_info, &input_to_attr);
  MS_LOG(DEBUG) << "Need convert input to addr " << need_convert_input_to_attr;
  if (need_convert_input_to_attr) {
    // Clone a new prim
    op_run_info->py_primitive = std::make_shared<PrimitivePy>(*op_run_info->py_primitive);
  }
  const auto &op_prim = op_run_info->py_primitive;

  // Get input tensors.
  op_prim->BeginRecordAddAttr();
  size_t input_num = op_run_info->op_inputs.size();
  if (input_num != op_run_info->inputs_mask.size()) {
    MS_LOG(EXCEPTION) << "The op input size " << input_num << ", but the size of input mask "
                      << op_run_info->inputs_mask.size();
  }
  for (size_t index = 0; index < input_num; ++index) {
    // convert const input to attr
    if (need_convert_input_to_attr && RunOpConvertConstInputToAttr(op_run_info, index, op_prim, input_to_attr)) {
      continue;
    }
    // convert const and tuple input to tensor
    int64_t tensor_mask = op_run_info->inputs_mask[index];
    ConvertPyObjectToTensor(op_run_info, index, op_prim, input_tensors, &tensor_mask);
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
  top_cell->set_all_op_info("");
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
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(cnode);
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
  (void)std::for_each(total_output_tensors.begin(), total_output_tensors.end(),
                      [](const tensor::TensorPtr &tensor) { tensor->set_is_forward_output(true); });
  top_cell->set_op_info_with_ms_func_forward_tensors(op_exec_info->op_info, total_output_tensors);
}

void SaveOpInfo(const TopCellInfoPtr &top_cell, const std::string &op_info,
                const std::vector<tensor::TensorPtr> &op_out_tensors) {
  MS_EXCEPTION_IF_NULL(top_cell);
  const auto &op_info_with_tensor_id = top_cell->op_info_with_tensor_id();
  if (op_info_with_tensor_id.find(op_info) != op_info_with_tensor_id.end()) {
    MS_LOG(EXCEPTION) << "Top cell: " << top_cell.get() << " records op info with tensor id, but get op info "
                      << op_info << " in op_info_with_tensor_id map";
  }
  // Record the relationship between the forward op and its output tensor id
  (void)std::for_each(
    op_out_tensors.begin(), op_out_tensors.end(),
    [&top_cell, &op_info](const tensor::TensorPtr &tensor) { top_cell->SetOpInfoWithTensorId(op_info, tensor->id()); });
}

void UpdateTensorInfo(const tensor::TensorPtr &new_tensor, const std::vector<tensor::TensorPtr> &pre_tensors) {
  MS_EXCEPTION_IF_NULL(new_tensor);
  if (pre_tensors.empty() || new_tensor->device_address() == nullptr) {
    MS_LOG(DEBUG) << "The number of pre tensors is zero or the device address of new tensor is nullptr.";
    return;
  }
  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  for (auto &pre_tensor : pre_tensors) {
    MS_EXCEPTION_IF_NULL(pre_tensor);
    MS_LOG(DEBUG) << "Replace Old tensor id " << pre_tensor->id() << " device_address: " << pre_tensor->device_address()
                  << " shape and type " << pre_tensor->GetShapeAndDataTypeInfo() << " with New tensor id "
                  << new_tensor->id() << " device_address " << new_tensor->device_address() << " shape and dtype "
                  << new_tensor->GetShapeAndDataTypeInfo();
    pre_tensor->set_shape(new_tensor->shape());
    pre_tensor->set_data_type(new_tensor->data_type());
    auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(new_tensor->device_address());
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_target != kCPUDevice && device_address->GetDeviceType() != device::DeviceType::kCPU) {
      pre_tensor->set_device_address(new_tensor->device_address());
      continue;
    }
    for (auto &item : kMindRtBackends) {
      MS_EXCEPTION_IF_NULL(item.second);
      item.second->WaitTaskFinish();
    }
    // Replace data in device address when run in CPU device.
    if (pre_tensor->device_address() != nullptr) {
      // If tensor is dynamic shape, Just replace device address.
      if (ValueHasDynamicShape(pre_tensor)) {
        pre_tensor->set_device_address(new_tensor->device_address());
        continue;
      }
      auto old_device_address = std::dynamic_pointer_cast<device::DeviceAddress>(pre_tensor->device_address());
      MS_EXCEPTION_IF_NULL(old_device_address);
      auto new_device_address = std::dynamic_pointer_cast<device::DeviceAddress>(new_tensor->device_address());
      MS_EXCEPTION_IF_NULL(new_device_address);

      // CPU host tensor data_c is different from device address if the address is from mem_pool.
      if (new_device_address->from_mem_pool()) {
        pre_tensor->set_device_address(new_device_address);
        continue;
      }

      auto old_ptr = old_device_address->GetMutablePtr();
      MS_EXCEPTION_IF_NULL(old_ptr);
      auto new_ptr = new_device_address->GetPtr();
      MS_EXCEPTION_IF_NULL(new_ptr);
      MS_EXCEPTION_IF_CHECK_FAIL(old_device_address->GetSize() == new_device_address->GetSize(), "Size not equal");
      if (old_device_address->GetSize() < SECUREC_MEM_MAX_LEN) {
        auto ret_code = memcpy_s(old_ptr, old_device_address->GetSize(), new_ptr, new_device_address->GetSize());
        MS_EXCEPTION_IF_CHECK_FAIL(ret_code == EOK, "Memory copy failed, ret code: " + std::to_string(ret_code));
      } else {
        auto ret_code = std::memcpy(old_ptr, new_ptr, old_device_address->GetSize());
        MS_EXCEPTION_IF_CHECK_FAIL(ret_code == old_ptr, "Memory copy failed");
      }
    } else {
      pre_tensor->set_device_address(device_address);
      pre_tensor->data_sync();
      pre_tensor->set_device_address(nullptr);
      pre_tensor->set_sync_status(kNeedSyncHostToDevice);
    }
  }
}

void CheckPyNativeContext() {
  const auto &parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  const auto &ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &parallel_mode = parallel_context->parallel_mode();
  const auto &search_mode = parallel_context->strategy_search_mode();
  if (parallel_mode == parallel::kAutoParallel && search_mode != parallel::kShardingPropagation) {
    MS_LOG(EXCEPTION)
      << "PyNative only supports Auto_Parallel under search mode of sharding_propagation using shard function, but got "
      << search_mode;
  }
}

py::object GetDstType(const TypeId &type_id) {
  constexpr int k8Bits = 8;
  constexpr int k16Bits = 16;
  constexpr int k32Bits = 32;
  constexpr int k64Bits = 64;
  ValuePtr value = nullptr;
  if (type_id == kNumberTypeFloat16) {
    value = std::make_shared<Float>(k16Bits);
  } else if (type_id == kNumberTypeFloat32) {
    value = std::make_shared<Float>(k32Bits);
  } else if (type_id == kNumberTypeFloat64) {
    value = std::make_shared<Float>(k64Bits);
  } else if (type_id == kNumberTypeBool) {
    value = std::make_shared<Bool>();
  } else if (type_id == kNumberTypeInt8) {
    value = std::make_shared<Int>(k8Bits);
  } else if (type_id == kNumberTypeUInt8) {
    value = std::make_shared<UInt>(k8Bits);
  } else if (type_id == kNumberTypeInt16) {
    value = std::make_shared<Int>(k16Bits);
  } else if (type_id == kNumberTypeInt32) {
    value = std::make_shared<Int>(k32Bits);
  } else if (type_id == kNumberTypeInt64) {
    value = std::make_shared<Int>(k64Bits);
  } else {
    MS_LOG(EXCEPTION) << "Not support dst type";
  }
  MS_EXCEPTION_IF_NULL(value);
  return py::cast(value);
}

bool IsPyObjTypeInvalid(const py::object &obj) {
  return !py::isinstance<tensor::Tensor>(obj) && !py::isinstance<tensor::CSRTensor>(obj) &&
         !py::isinstance<py::int_>(obj) && !py::isinstance<py::float_>(obj);
}

bool IsConstPrimOrConstInput(const OpExecInfoPtr &op_exec_info, size_t index) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  auto prim = op_exec_info->py_primitive;
  MS_EXCEPTION_IF_NULL(prim);
  bool is_const_prim = prim->is_const_prim();
  const auto &const_input_index = prim->get_const_input_indexes();
  bool have_const_input = !const_input_index.empty();
  bool is_const_input =
    have_const_input && std::find(const_input_index.begin(), const_input_index.end(), index) != const_input_index.end();
  MS_LOG(DEBUG) << prim->ToString() << " is const prim " << prim->is_const_prim() << ", is_const_input "
                << is_const_input;
  return is_const_prim || is_const_input;
}

// Shallow Copy Value and change shape
ValuePtr ShallowCopyValue(const OpExecInfoPtr &op_exec_info, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(value);
  auto tensor_abs = op_exec_info->abstract;
  if (tensor_abs->isa<abstract::AbstractRefTensor>()) {
    tensor_abs = tensor_abs->cast<abstract::AbstractRefPtr>()->CloneAsTensor();
  }
  auto new_shape = tensor_abs->BuildShape()->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(new_shape);
  if (value->isa<mindspore::tensor::Tensor>()) {
    auto tensor_value = value->cast<mindspore::tensor::TensorPtr>();
    return std::make_shared<mindspore::tensor::Tensor>(tensor_value->data_type(), new_shape->shape(),
                                                       tensor_value->data_c(), tensor_value->Size());
  } else if (value->isa<ValueTuple>()) {
    std::vector<ValuePtr> values;
    auto value_tuple = value->cast<ValueTuplePtr>();
    (void)std::transform(value_tuple->value().begin(), value_tuple->value().end(), std::back_inserter(values),
                         [op_exec_info](const ValuePtr &elem) { return ShallowCopyValue(op_exec_info, elem); });
    return std::make_shared<ValueTuple>(values);
  } else {
    return value;
  }
}

void SaveIdWithDynamicAbstract(const py::object &obj, const AbstractBasePtr &abs,
                               OrderedMap<std::string, abstract::AbstractBasePtr> *obj_id_with_dynamic_abs) {
  MS_EXCEPTION_IF_NULL(abs);
  MS_EXCEPTION_IF_NULL(obj_id_with_dynamic_abs);
  if (py::isinstance<py::tuple>(obj) && abs->isa<abstract::AbstractTuple>()) {
    const auto &obj_tuple = py::cast<py::tuple>(obj);
    const auto &abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
    if (obj_tuple.size() != abs_tuple->size()) {
      MS_LOG(EXCEPTION) << "Obj tuple size " << obj_tuple.size() << ", but abstract tuple size " << abs_tuple->size();
    }
    for (size_t i = 0; i < obj_tuple.size(); ++i) {
      SaveIdWithDynamicAbstract(obj_tuple[i], abs_tuple->elements()[i], obj_id_with_dynamic_abs);
    }
  } else if (py::isinstance<py::tuple>(obj) && !abs->isa<abstract::AbstractTuple>()) {
    const auto &obj_tuple = py::cast<py::tuple>(obj);
    if (obj_tuple.size() != 1) {
      MS_LOG(EXCEPTION) << "Not match: obj " << py::str(obj) << " and abs " << abs->ToString();
    }
    // Like Unique, has two outputs, but one output is static shape, and should not be stored
    if (abs->BuildShape()->IsDynamic()) {
      (void)obj_id_with_dynamic_abs->emplace(std::make_pair(GetId(obj_tuple[0]), abs));
    }
  } else if (!py::isinstance<py::tuple>(obj) && !abs->isa<abstract::AbstractTuple>()) {
    if (abs->BuildShape()->IsDynamic()) {
      (void)obj_id_with_dynamic_abs->emplace(std::make_pair(GetId(obj), abs));
    }
  } else {
    MS_LOG(EXCEPTION) << "Not match: obj " << py::str(obj) << " and abs " << abs->ToString();
  }
}

ShapeVector GetTensorShape(const py::object &obj) {
  if (py::isinstance<tensor::Tensor>(obj)) {
    return obj.cast<tensor::TensorPtr>()->shape();
  }
  return {};
}

TypePtr GetTypeFromAbstract(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractSequence>()) {
    MS_LOG(EXCEPTION) << "Get tuple or list abs";
  }
  const auto &type = abs->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  return type;
}

abstract::ShapePtr GetShapeFromAbstract(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractSequence>()) {
    MS_LOG(EXCEPTION) << "Get tuple or list abs";
  }
  const auto &base_shape = abs->BuildShape();
  MS_EXCEPTION_IF_NULL(base_shape);
  const auto &shape = base_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  return shape;
}

void SaveIdWithDynamicShape(const OpExecInfoPtr &op_exec_info, const std::string &id, const py::object &real_obj,
                            const abstract::AbstractBasePtr &dynamic_abs) {
  if (py::isinstance<py::list>(real_obj) || py::isinstance<py::tuple>(real_obj)) {
    const auto &obj_tuple = real_obj.cast<py::tuple>();
    const auto &obj_abs_seq = dynamic_abs->cast<mindspore::abstract::AbstractSequencePtr>();
    if (obj_tuple.size() != obj_abs_seq->size()) {
      MS_LOG(EXCEPTION) << "Input tuple/list obj size " << obj_tuple.size() << " not equal to AbstractSequence size "
                        << obj_abs_seq->size();
    }
    for (size_t i = 0; i < obj_tuple.size(); ++i) {
      SaveIdWithDynamicShape(op_exec_info, GetId(obj_tuple[i]), obj_tuple[i], obj_abs_seq->elements()[i]);
    }
  } else {
    const auto &dynamic_shape_vec = GetShapeFromAbstract(dynamic_abs);
    MS_LOG(DEBUG) << "Save tensor " << id << ", real shape " << GetTensorShape(real_obj) << ", dynamic shape "
                  << dynamic_shape_vec->ToString();
    (void)op_exec_info->id_with_dynamic_shape.emplace(std::make_pair(id, dynamic_shape_vec));
  }
}

void UpdateInputTensorToDynamicShape(const OpExecInfoPtr &op_exec_info, std::vector<tensor::TensorPtr> *input_tensors) {
  if (!op_exec_info->has_dynamic_input) {
    return;
  }
  MS_EXCEPTION_IF_NULL(input_tensors);
  // Set tensor dynamic base shape
  for (auto &input_tensor : *input_tensors) {
    auto it = op_exec_info->id_with_dynamic_shape.find(input_tensor->id());
    if (it != op_exec_info->id_with_dynamic_shape.end()) {
      input_tensor->set_base_shape(it->second);
    }
  }
}

void UpdateValueToDynamicShape(const ValuePtr &value,
                               const OrderedMap<std::string, abstract::AbstractBasePtr> &obj_id_with_dynamic_abs) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<mindspore::tensor::Tensor>()) {
    auto tensor_value = value->cast<tensor::TensorPtr>();
    auto it = obj_id_with_dynamic_abs.find(tensor_value->id());
    if (it != obj_id_with_dynamic_abs.end()) {
      tensor_value->set_base_shape(GetShapeFromAbstract(it->second));
    }
  } else if (value->isa<ValueTuple>()) {
    auto value_tuple = value->cast<ValueTuplePtr>();
    for (const auto &v : value_tuple->value()) {
      UpdateValueToDynamicShape(v, obj_id_with_dynamic_abs);
    }
  } else {
    MS_LOG(DEBUG) << "Out put is not a tensor";
  }
}

ValuePtr SetSensValue(const ValuePtr &value, const TopCellInfoPtr &top_cell) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(top_cell);
  if (value->isa<ValueTuple>()) {
    ValuePtrList values;
    auto value_tuple = value->cast<ValueTuplePtr>();
    (void)std::transform(value_tuple->value().begin(), value_tuple->value().end(), std::back_inserter(values),
                         [&top_cell](const ValuePtr &elem) { return SetSensValue(elem, top_cell); });
    return std::make_shared<ValueTuple>(values);
  } else if (value->isa<ValueList>()) {
    ValuePtrList values;
    auto value_list = value->cast<ValueTuplePtr>();
    (void)std::transform(value_list->value().begin(), value_list->value().end(), std::back_inserter(values),
                         [&top_cell](const ValuePtr &elem) { return SetSensValue(elem, top_cell); });
    return std::make_shared<ValueList>(values);
  } else if (value->isa<tensor::Tensor>()) {
    auto tensor_value = value->cast<tensor::TensorPtr>();
    // Sens tensor has the same shape and dtype with output tensor
    auto sens_tensor = std::make_shared<tensor::Tensor>(tensor_value->data_type(), tensor_value->shape());
    sens_tensor->set_base_shape(tensor_value->base_shape_ptr());
    MS_LOG(DEBUG) << "Make new tensor for sens id " << sens_tensor->id() << ", abstract "
                  << sens_tensor->ToAbstract()->ToString();
    top_cell->SetTensorIdWithTensorObject(sens_tensor->id(), sens_tensor);
    return sens_tensor;
  } else {
    return value;
  }
}

void FindMatchTopCell(const TopCellInfoPtr &top_cell, const py::args &args, std::vector<ShapeVector> *new_args_shape) {
  MS_EXCEPTION_IF_NULL(top_cell);
  MS_EXCEPTION_IF_NULL(new_args_shape);
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &cur_value_abs = PyObjToValue(args[i])->ToAbstract();
    MS_EXCEPTION_IF_NULL(cur_value_abs);
    const auto &cur_type = GetTypeFromAbstract(cur_value_abs);
    const auto &elem_type = top_cell->cell_self_info()->args_type[i];
    // Type is not the same
    if (cur_type->hash() != elem_type->hash()) {
      MS_LOG(DEBUG) << "The " << i << "th args type is not the same, cur is " << cur_type->ToString()
                    << " and the elem is " << elem_type->ToString();
      return;
    }
    // Check shape
    const auto &cur_shape = GetShapeFromAbstract(cur_value_abs)->shape();
    auto elem_shape = top_cell->cell_self_info()->args_shape[i]->shape();
    if (cur_shape.size() != elem_shape.size()) {
      MS_LOG(DEBUG) << "The " << i << "th args shape size is not the same, cur is " << cur_shape.size()
                    << " and the elem is " << elem_shape.size();
      return;
    }
    ShapeVector new_shape;
    for (size_t j = 0; j < cur_shape.size(); ++j) {
      if (cur_shape[j] == elem_shape[j]) {
        (void)new_shape.emplace_back(cur_shape[j]);
      } else {
        (void)new_shape.emplace_back(-1);
      }
    }
    // All shape can not be -1, and all shape can not be actual.
    bool is_any_unknown = std::any_of(new_shape.begin(), new_shape.end(), [](int64_t s) { return s == -1; });
    bool is_any_actual = std::any_of(new_shape.begin(), new_shape.end(), [](int64_t s) { return s != -1; });
    if (is_any_unknown && is_any_actual) {
      (void)new_args_shape->emplace_back(new_shape);
    } else {
      MS_LOG(DEBUG) << "Not support all shape unknown or actual.Cur shape " << cur_shape << ", elem shape "
                    << elem_shape << ", and new shape is " << new_shape;
    }
  }
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

DynamicShapeInfoPtr ForwardExecutor::dynamic_shape_info_ptr() {
  if (dynamic_shape_info_ptr_ == nullptr) {
    dynamic_shape_info_ptr_ = std::make_shared<DynamicShapeInfo>();
  }
  MS_EXCEPTION_IF_NULL(dynamic_shape_info_ptr_);
  return dynamic_shape_info_ptr_;
}

GradExecutorPtr ForwardExecutor::grad() const {
  auto grad_executor = grad_executor_.lock();
  MS_EXCEPTION_IF_NULL(grad_executor);
  return grad_executor;
}

void TopCellInfo::SetCellSelfInfoForTopCell(const py::object &cell, const py::args &args) {
  std::vector<std::string> args_id;
  std::vector<abstract::ShapePtr> args_shape;
  std::vector<TypePtr> args_type;
  for (size_t i = 0; i < args.size(); ++i) {
    auto value = PyObjToValue(args[i]);
    MS_EXCEPTION_IF_NULL(value);
    auto abs = value->ToAbstract();
    auto shape_ptr = abs->BuildShape()->cast<abstract::ShapePtr>();
    if (shape_ptr == nullptr) {
      return;
    }
    (void)args_id.emplace_back(GetId(args[i]));
    (void)args_shape.emplace_back(shape_ptr);
    (void)args_type.emplace_back(abs->BuildType());
  }
  set_cell_self_info(std::make_shared<CellSelfInfo>(GetId(cell), args_id, args_shape, args_type));
}

bool TopCellInfo::IsSubCell(const std::string &cell_id) const {
  if (sub_cell_list_.empty()) {
    MS_LOG(DEBUG) << "The sub cell list is empty, there is no sub cell";
    return false;
  }
  return sub_cell_list_.find(cell_id) != sub_cell_list_.end();
}

void TopCellInfo::RecordCellBackwardHookOp(const std::string &cell_order, const AnfNodePtr &hook_op) {
  MS_EXCEPTION_IF_NULL(hook_op);
  (void)cell_backward_hook_op_[cell_order].emplace_back(hook_op);
  constexpr size_t cell_backward_hook_max_num = 2;
  if (cell_backward_hook_op_[cell_order].size() > cell_backward_hook_max_num) {
    MS_LOG(EXCEPTION) << "Cell order: " << cell_order << " only has two backward hook op.";
  }
}

void TopCellInfo::CheckSubCellHookChanged() {
  if (!hook_changed_) {
    for (const auto &sub_cell : sub_cell_list_) {
      const auto sub_cell_id = sub_cell.substr(0, sub_cell.find('_'));
      if (sub_cell_hook_changed_.find(sub_cell_id) != sub_cell_hook_changed_.end()) {
        hook_changed_ = true;
        break;
      }
    }
  }
  sub_cell_hook_changed_.clear();
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
    auto device_sync = tensor->device_address();
    auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
    if (device_address == nullptr) {
      continue;
    }
    if (!device_address->from_persistent_mem()) {
      tensor->set_device_address(nullptr);
    }
  }
}

void TopCellInfo::Clear() {
  MS_LOG(DEBUG) << "Clear top cell info. Cell id " << cell_id_;
  op_num_ = 0;
  dynamic_graph_structure_ = false;
  vm_compiled_ = false;
  ms_function_flag_ = false;
  is_init_kpynative_ = false;
  need_compile_graph_ = false;
  forward_already_run_ = false;
  input_args_id_.clear();
  all_op_info_.clear();
  resource_ = nullptr;
  df_builder_ = nullptr;
  fg_ = nullptr;
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
  if (kSummaryOperators.count(op_exec_info->op_name) != 0) {
    MS_LOG(DEBUG) << "PyNative not support Operator " << op_exec_info->op_name;
    return;
  }
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

OpExecInfoPtr ForwardExecutor::GenerateOpExecInfo(const py::args &args) const {
  if (args.size() != PY_ARGS_NUM) {
    MS_LOG(EXCEPTION) << "Three args are needed by RunOp";
  }
  python_adapter::set_python_env_flag(true);
  const auto &op_exec_info = std::make_shared<OpExecInfo>();
  const auto &op_name = py::cast<std::string>(args[PY_NAME]);
  op_exec_info->op_name = op_name;
  op_exec_info->is_nop_prim = false;

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
  if (op_exec_info->op_name == prim::kPrimCast->name() || op_exec_info->is_nop_prim) {
    return;
  }

  // Mixed precision conversion tensors which has cast dtype
  SetTensorMixPrecisionCast(op_exec_info);
  // Implicit transform
  SetImplicitCast(op_exec_info);
}

void ForwardExecutor::RunMixedPrecisionCastOp(const OpExecInfoPtr &op_exec_info, py::object *ret) {
  MS_EXCEPTION_IF_NULL(ret);
  MS_EXCEPTION_IF_NULL(op_exec_info);
  py::tuple res = RunOpWithInitBackendPolicy(op_exec_info);
  if (res.size() == 1) {
    *ret = res[0];
    return;
  }
  *ret = std::move(res);
}

void ForwardExecutor::SetNonCostantValueAbs(const AbstractBasePtr &abs, const std::string &id) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractTensor>()) {
    abs->set_value(kAnyValue);
  } else if (abs->isa<abstract::AbstractTuple>() || abs->isa<abstract::AbstractList>()) {
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    for (auto &item : abs_seq->elements()) {
      MS_EXCEPTION_IF_NULL(item);
      if (item->isa<abstract::AbstractTensor>()) {
        item->set_value(kAnyValue);
      }
    }
  }
  node_abs_map_[id] = abs;
}

AbstractBasePtr ForwardExecutor::GetInputObjAbstract(const OpExecInfoPtr &op_exec_info, size_t i,
                                                     const py::object &obj) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  const auto &id = GetId(obj);
  AbstractBasePtr abs = nullptr;

  auto it = node_abs_map_.find(id);
  if (it != node_abs_map_.end()) {
    abs = it->second;
  }
  MS_LOG(DEBUG) << "Abstract cache hit " << (abs != nullptr);
  bool is_const_prim_or_input = IsConstPrimOrConstInput(op_exec_info, i);
  if (abs == nullptr || is_const_prim_or_input) {
    abs = PyObjToValue(obj)->ToAbstract();
    if (!is_const_prim_or_input) {
      SetNonCostantValueAbs(abs, id);
    }
  }
  return abs;
}

AbstractBasePtr ForwardExecutor::GetTupleInputAbstract(const OpExecInfoPtr &op_exec_info, const py::object &obj,
                                                       const std::string &id, size_t input_index) {
  abstract::AbstractBasePtrList abs_list;
  if (!IsConstPrimOrConstInput(op_exec_info, input_index)) {
    auto it = node_abs_map_.find(id);
    if (it != node_abs_map_.end()) {
      return it->second;
    }
  }
  MS_LOG(DEBUG) << "Abstract cache not hit";
  auto tuple = obj.cast<py::tuple>();
  auto tuple_size = tuple.size();
  for (size_t i = 0; i < tuple_size; ++i) {
    const auto &item_id = GetId(tuple[i]);
    const auto item_it = dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs.find(item_id);
    if (item_it != dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs.end()) {
      (void)abs_list.emplace_back(item_it->second);
    } else {
      auto abs = GetInputObjAbstract(op_exec_info, input_index, tuple[i]);
      (void)abs_list.emplace_back(abs);
    }
  }
  abstract::AbstractBasePtr node_abs;
  if (py::isinstance<py::tuple>(obj)) {
    node_abs = std::make_shared<abstract::AbstractTuple>(abs_list);
  } else {
    node_abs = std::make_shared<abstract::AbstractList>(abs_list);
  }
  node_abs_map_[id] = node_abs;
  return node_abs;
}

void ForwardExecutor::GetInputsArgsSpec(const OpExecInfoPtr &op_exec_info,
                                        abstract::AbstractBasePtrList *args_spec_list) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(args_spec_list);

  for (size_t i = 0; i < op_exec_info->op_inputs.size(); i++) {
    const auto &obj = op_exec_info->op_inputs[i];
    const auto &id = GetId(obj);
    // Get tuple or list abs
    MS_LOG(DEBUG) << "Set input abs for arg id " << id;
    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
      auto abs = GetTupleInputAbstract(op_exec_info, obj, id, i);
      auto shape = abs->BuildShape();
      MS_EXCEPTION_IF_NULL(shape);
      if (shape->IsDynamic()) {
        MS_LOG(DEBUG) << "Input " << i << " get input of prev op dynamic output";
        op_exec_info->has_dynamic_input = true;
        SaveIdWithDynamicShape(op_exec_info, id, obj, abs);
      }
      (void)args_spec_list->emplace_back(abs);
      MS_LOG(DEBUG) << "Set " << i << "th abs " << args_spec_list->back()->ToString();
      continue;
    }
    auto out_it = dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs.find(id);
    if (out_it != dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs.end()) {
      MS_LOG(DEBUG) << "Input " << i << " get input of prev op dynamic output";
      op_exec_info->has_dynamic_input = true;
      (void)args_spec_list->emplace_back(out_it->second);
      SaveIdWithDynamicShape(op_exec_info, id, obj, out_it->second);
    } else {
      const auto &input_abs = GetInputObjAbstract(op_exec_info, i, obj);
      const auto &shape = input_abs->BuildShape();
      MS_EXCEPTION_IF_NULL(shape);
      // For ms function
      if (shape->IsDynamic()) {
        MS_LOG(DEBUG) << "Input " << i << " get dynamic shape";
        op_exec_info->has_dynamic_input = true;
      }
      (void)args_spec_list->emplace_back(input_abs);
    }
    MS_LOG(DEBUG) << "Set " << i << "th abs " << args_spec_list->back()->ToString();
  }
}

AnfNodePtr ForwardExecutor::GetRealInputNodeBySkipHook(const AnfNodePtr &input_node) const {
  if (input_node == nullptr) {
    MS_LOG(DEBUG) << "The input node is nullptr.";
    return input_node;
  }
  const auto &cell_backward_hook_op = grad()->top_cell()->cell_backward_hook_op();
  for (const auto &elem : cell_backward_hook_op) {
    constexpr size_t cell_backward_hook_num = 2;
    if (elem.second.size() < cell_backward_hook_num) {  // In cell own scope, no need to skip backward hook op.
      continue;
    }
    // The input node is the first backward hook op of another cell, skip the backward hook op.
    if (IsPrimitiveCNode(input_node, prim::kPrimCellBackwardHook) && input_node == elem.second[0]) {
      // Single input.
      auto backward_hook_op = input_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(backward_hook_op);
      return backward_hook_op->input(1);
    } else if (IsPrimitiveCNode(input_node, prim::kPrimTupleGetItem)) {
      // Multi inputs.
      auto tuple_get_item = input_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(tuple_get_item);
      auto inp_in_tuple = tuple_get_item->input(1);
      MS_EXCEPTION_IF_NULL(inp_in_tuple);
      if (IsPrimitiveCNode(inp_in_tuple, prim::kPrimCellBackwardHook) && inp_in_tuple == elem.second[0]) {
        constexpr size_t idx = 2;
        auto idx_node = tuple_get_item->input(idx);
        MS_EXCEPTION_IF_NULL(idx_node);
        auto value_node = idx_node->cast<ValueNodePtr>();
        MS_EXCEPTION_IF_NULL(value_node);
        auto out_idx = GetValue<int64_t>(value_node->value());
        auto backward_hook_op = inp_in_tuple->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(backward_hook_op);
        return backward_hook_op->input(1 + LongToSize(out_idx));
      }
    }
  }
  return input_node;
}

CNodePtr ForwardExecutor::ConstructForwardGraph(const OpExecInfoPtr &op_exec_info) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  auto prim = op_exec_info->py_primitive;
  std::vector<AnfNodePtr> inputs;
  std::vector<int64_t> op_masks;
  inputs.emplace_back(NewValueNode(prim));
  for (size_t i = 0; i < op_exec_info->op_inputs.size(); i++) {
    const auto &obj = op_exec_info->op_inputs[i];
    bool op_mask = IsParameter(obj);
    MS_LOG(DEBUG) << "Args i " << i << ", op mask " << op_mask;
    op_masks.emplace_back(static_cast<int64_t>(op_mask));

    // Construct grad graph
    if (grad()->need_construct_graph()) {
      const auto &id = GetId(obj);
      AnfNodePtr input_node = nullptr;
      input_node = GetRealInputNodeBySkipHook(grad()->GetInput(obj, op_mask));
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
    if (IsPrimitiveCNode(cnode, prim::kPrimCellBackwardHook)) {
      grad()->top_cell()->RecordCellBackwardHookOp(grad()->GetCurCellOrder(), cnode);
    }
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

  if (op_exec_info->abstract == nullptr || kForceInferPrim.find(op_name) != kForceInferPrim.end()) {
    // Use python infer method
    if (kIgnoreInferPrim.find(op_name) == kIgnoreInferPrim.end()) {
      PynativeInfer(prim, op_exec_info.get(), args_spec_list);
    }
  }
  // Get output dynamic shape info from infer steprr
  auto abstract = op_exec_info->abstract;
  MS_EXCEPTION_IF_NULL(abstract);
  auto shape = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape);
  op_exec_info->has_dynamic_output = shape->IsDynamic();
  if (IsDynamicShape(op_exec_info)) {
    MS_LOG(DEBUG) << "Set dynamic op " << op_name;
  }
}

void ForwardExecutor::DoNopOutput(const OpExecInfoPtr &op_exec_info, ValuePtr *out_real_value) const {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  // Get First input
  if (op_exec_info->op_inputs.empty()) {
    MS_LOG(EXCEPTION) << "Inputs of " << op_exec_info->op_name << " is empty";
  }
  const auto &obj = op_exec_info->op_inputs[0];
  if (!py::isinstance<tensor::Tensor>(obj)) {
    MS_LOG(EXCEPTION) << "First input of " << op_exec_info->op_name << " must be a tensor";
  }
  const auto &tensor_ptr = py::cast<tensor::TensorPtr>(obj);
  *out_real_value = ShallowCopyValue(op_exec_info, tensor_ptr);
  MS_LOG(DEBUG) << "New copy value is " << (*out_real_value)->ToString();
}

void ForwardExecutor::GetOpOutput(const OpExecInfoPtr &op_exec_info,
                                  const abstract::AbstractBasePtrList &args_spec_list, const CNodePtr &cnode,
                                  bool prim_cache_hit, py::object *ret) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  const auto &prim = op_exec_info->py_primitive;
  MS_EXCEPTION_IF_NULL(prim);
  // Infer output value by constant folding
  MS_EXCEPTION_IF_NULL(ret);
  py::dict output = abstract::ConvertAbstractToPython(op_exec_info->abstract, true);
  if (!output[ATTR_VALUE].is_none()) {
    *ret = output[ATTR_VALUE];
    grad()->RecordGradOpInfo(op_exec_info);
    MS_LOG(DEBUG) << "Get output by constant folding, output is " << py::str(*ret);
    return;
  } else if (prim->is_const_prim()) {
    *ret = py::cast("");
    grad()->RecordGradOpInfo(op_exec_info);
    MS_LOG(DEBUG) << "Get const prim";
    return;
  }

  // Add output abstract info into cache, the const value needs to infer evert step
  if (grad()->enable_op_cache() && !prim_cache_hit && !IsDynamicShape(op_exec_info)) {
    AbsCacheKey key{prim->name(), prim->Hash(), prim->attrs()};
    auto &out = prim_abs_list_[key];
    out[args_spec_list].abs = op_exec_info->abstract;
    out[args_spec_list].attrs = prim->evaluate_added_attrs();
  }

  // Run op with selected backend, nop is no need run backend
  ValuePtr out_real_value = nullptr;
  if (op_exec_info->is_nop_prim) {
    DoNopOutput(op_exec_info, &out_real_value);
    *ret = BaseRefToPyData(out_real_value);
  } else {
    auto result = RunOpWithInitBackendPolicy(op_exec_info);
    py::object out_real = result;
    if (result.size() == 1 && op_exec_info->abstract != nullptr &&
        !op_exec_info->abstract->isa<abstract::AbstractSequence>()) {
      out_real = result[0];
    }
    // Get output value
    if (grad()->grad_flag()) {
      out_real_value = PyObjToValue(out_real);
    }
    *ret = out_real;
  }

  if (grad()->need_construct_graph() && !grad()->in_cell_with_custom_bprop_()) {
    MS_EXCEPTION_IF_NULL(cnode);
    const auto &obj_id = GetId(*ret);
    cnode->set_abstract(op_exec_info->abstract);
    node_abs_map_[obj_id] = op_exec_info->abstract;
    grad()->SaveOutputNodeMap(obj_id, *ret, cnode);
    grad()->DoOpGrad(op_exec_info, cnode, out_real_value);
    // Dynamic shape should update to top cell
    if (IsDynamicShape(op_exec_info)) {
      grad()->top_cell()->set_dynamic_shape(true);
    }
  } else {
    node_abs_map_.clear();
  }
  // Record op info for judge whether the construct of cell has been changed
  grad()->RecordGradOpInfo(op_exec_info);
  grad()->UpdateForwardTensorInfoInBpropGraph(op_exec_info->op_info, out_real_value);
}

py::object ForwardExecutor::DoAutoCast(const py::object &arg, const TypeId &type_id, const std::string &op_name,
                                       size_t index) {
  static py::object cast_prim = python_adapter::GetPyFn(kOpsFunctionModelName, "cast");
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
  return result;
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
  return result;
}

void ForwardExecutor::DoSignatureCast(const PrimitivePyPtr &prim,
                                      const mindspore::HashMap<SignatureEnumDType, TypeId> &dst_type,
                                      const std::vector<SignatureEnumDType> &dtypes,
                                      const OpExecInfoPtr &op_exec_info) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(op_exec_info);
  const auto &signature = prim->signatures();
  auto &input_args = op_exec_info->op_inputs;
  size_t input_args_size = input_args.size();
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
      prim::RaiseExceptionForConvertRefDtype(prim, TypeIdToMsTypeStr(arg_type_id), TypeIdToMsTypeStr(it->second), i);
    }
    if (is_same_type) {
      continue;
    }

    if (IsPyObjTypeInvalid(obj)) {
      MS_EXCEPTION(TypeError) << "For '" << prim->name() << "', the " << (i + 1) << "th input " << signature[i].name
                              << " can not be implicitly converted. "
                              << "Its type is " << py::cast<std::string>(obj.attr("__class__").attr("__name__"))
                              << ", and the value is " << py::cast<py::str>(obj) << ". Only support Tensor or Scalar.";
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
    mindspore::HashMap<SignatureEnumDType, std::vector<size_t>> type_indexes;
    bool has_dtype_sig = GetSignatureType(op_exec_info->py_primitive, &dtypes);
    if (has_dtype_sig) {
      mindspore::HashMap<SignatureEnumDType, TypeId> dst_type;
      GetTypeIndex(dtypes, &type_indexes);
      GetDstType(op_exec_info->op_inputs, type_indexes, &dst_type);
      DoSignatureCast(op_exec_info->py_primitive, dst_type, dtypes, op_exec_info);
    }
    PrimSignature sig_value{has_dtype_sig, dtypes, type_indexes};
    implicit_cast_map_[prim->name()] = sig_value;
  } else {
    if (!it->second.has_dtype_sig) {
      MS_LOG(DEBUG) << op_exec_info->op_name << " have no dtype sig";
      return;
    }
    MS_LOG(DEBUG) << "Do signature for " << op_exec_info->op_name << " with cache";
    mindspore::HashMap<SignatureEnumDType, TypeId> dst_type;
    GetDstType(op_exec_info->op_inputs, it->second.type_indexes, &dst_type);
    DoSignatureCast(op_exec_info->py_primitive, dst_type, it->second.dtypes, op_exec_info);
  }
}

AnfNodePtr GradExecutor::GetInput(const py::object &obj, bool op_mask) const {
  AnfNodePtr node = nullptr;
  const auto &obj_id = GetId(obj);

  if (op_mask) {
    MS_LOG(DEBUG) << "Cell parameters(weights)";
    // get the parameter name from parameter object
    auto name_attr = python_adapter::GetPyObjAttr(obj, "name");
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
      SetParamNodeMapInGraphInfoMap(curr_g(), obj_id, free_param);
      SetNodeMapInGraphInfoMap(df_builder, obj_id, free_param);
      SetNodeMapInGraphInfoMap(curr_g(), obj_id, free_param);
      return free_param;
    }
    node = graph_info->params.at(obj_id);
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Get input param node " << node->ToString() << ", obj id " << obj_id;
    return node;
  }

  auto curr_graph_info = top_cell()->graph_info_map().at(curr_g());
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
      bool is_parameter = IsParameter(tuple[i]);
      args.emplace_back(GetInput(tuple[i], is_parameter));
    }
    auto cnode = curr_g()->NewCNode(args);
    SetNodeMapInGraphInfoMap(curr_g(), obj_id, cnode);
    node = cnode;
  } else {
    node = MakeValueNode(obj, obj_id);
  }
  node == nullptr ? MS_LOG(DEBUG) << "Get node is nullptr"
                  : MS_LOG(DEBUG) << "Get input node " << node->ToString() << ", id " << obj_id;
  return node;
}

AnfNodePtr GradExecutor::GetObjNode(const py::object &obj, const std::string &obj_id) const {
  auto graph_info = top_cell()->graph_info_map().at(curr_g());
  MS_EXCEPTION_IF_NULL(graph_info);
  if (graph_info->node_map.find(obj_id) == graph_info->node_map.end()) {
    // A tuple returns in this case: x = op1, y = op2, return (x, y)
    // or a constant returns in this case
    auto make_tuple = CreateMakeTupleNode(obj, obj_id);
    if (make_tuple == nullptr) {
      MS_LOG(DEBUG) << "Create value node for obj id: " << obj_id;
      return MakeValueNode(obj, obj_id);
    }
    return make_tuple;
  }
  // single output CNode
  const auto &out = graph_info->node_map.at(obj_id);
  if (out.second.size() == 1 && out.second[0] == -1) {
    return out.first;
  }
  // Params node
  if (graph_info->params.find(obj_id) != graph_info->params.end()) {
    auto para_node = out.first;
    for (auto &v : out.second) {
      std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), para_node, NewValueNode(v)};
      para_node = curr_g()->NewCNode(tuple_get_item_inputs);
    }
    return para_node;
  }
  // Create tuple get item node for multiple output CNode
  return CreateTupleGetItemNode(obj_id);
}

AnfNodePtr GradExecutor::MakeValueNode(const py::object &obj, const std::string &obj_id) const {
  ValuePtr converted_ret = nullptr;
  if (!parse::ConvertData(obj, &converted_ret)) {
    MS_LOG(EXCEPTION) << "Failed to convert obj to value node.";
  }
  auto node = NewValueNode(converted_ret);
  SetNodeMapInGraphInfoMap(curr_g(), obj_id, node);
  return node;
}

AnfNodePtr GradExecutor::CreateMakeTupleNode(const py::object &obj, const std::string &obj_id) const {
  if (!py::isinstance<py::tuple>(obj) && !py::isinstance<py::list>(obj)) {
    MS_LOG(DEBUG) << "The input obj is not a tuple or list.";
    return nullptr;
  }
  // get input node and value
  const auto &obj_tuple = obj.cast<py::tuple>();
  ValuePtrList input_args;
  std::vector<size_t> value_index;
  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < obj_tuple.size(); ++i) {
    const auto &v = PyObjToValue(obj_tuple[i]);
    // Graph have no define for grad
    if (v->isa<FuncGraph>()) {
      continue;
    }
    value_index.emplace_back(i);
    input_args.emplace_back(v);
    (void)CreateMakeTupleNode(obj_tuple[i], GetId(obj_tuple[i]));
    inputs.emplace_back(GetInput(obj_tuple[i], false));
  }
  py::tuple value_outs(value_index.size());
  for (size_t i = 0; i < value_index.size(); ++i) {
    value_outs[i] = obj_tuple[value_index[i]];
  }
  // create make tuple node and record in graph info map
  auto cnode = curr_g()->NewCNode(inputs);
  MS_LOG(DEBUG) << "Create make tuple node: " << cnode->DebugString();
  SetTupleArgsToGraphInfoMap(curr_g(), obj, cnode);
  SetNodeMapInGraphInfoMap(curr_g(), obj_id, cnode);
  // run ad for make tuple node
  if (grad_flag_) {
    if (grad_is_running_ && !bprop_grad_stack_.empty() && !bprop_grad_stack_.top().second) {
      MS_LOG(DEBUG) << "Running custom bprop, no need to do GradPynativeOp.";
    } else {
      (void)ad::GradPynativeOp(top_cell()->k_pynative_cell_ptr(), cnode, input_args, PyObjToValue(value_outs));
    }
  }
  return cnode;
}

AnfNodePtr GradExecutor::CreateTupleGetItemNode(const std::string &obj_id) const {
  // obj_id is obtained by calling the 'GetId()'
  auto graph_info = top_cell()->graph_info_map().at(curr_g());
  MS_EXCEPTION_IF_NULL(graph_info);
  if (graph_info->node_map.find(obj_id) == graph_info->node_map.end()) {
    MS_LOG(DEBUG) << "Can not find CNode for obj id: " << obj_id;
    return nullptr;
  }
  const auto &out = graph_info->node_map.at(obj_id);
  MS_LOG(DEBUG) << "Output size: " << out.second.size();
  auto c_node = out.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(c_node);
  auto abs = c_node->abstract();
  // Create tuple get item node
  for (const auto &idx : out.second) {
    std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), c_node, NewValueNode(idx)};
    c_node = curr_g()->NewCNode(tuple_get_item_inputs);
    if (abs != nullptr && abs->isa<abstract::AbstractTuple>()) {
      auto abs_tuple = dyn_cast<abstract::AbstractTuple>(abs);
      MS_EXCEPTION_IF_NULL(abs_tuple);
      const auto &elements = abs_tuple->elements();
      if (static_cast<size_t>(idx) >= elements.size()) {
        MS_LOG(EXCEPTION) << "Index exceeds the size of elements. Index " << idx << ", element size "
                          << elements.size();
      }
      auto prim_abs = elements[static_cast<size_t>(idx)];
      MS_EXCEPTION_IF_NULL(prim_abs);
      MS_LOG(DEBUG) << "Set tuple getitem abs " << prim_abs->ToString();
      c_node->set_abstract(prim_abs);
    }
  }
  if (c_node->abstract() != nullptr) {
    forward()->SetNodeAbsMap(obj_id, c_node->abstract());
  }
  MS_LOG(DEBUG) << "Create tuple get item node: " << c_node->DebugString();
  return c_node;
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

void GradExecutor::SetHookChanged(const py::object &cell) {
  auto cell_id = GetId(cell);
  for (const auto &top_cell : top_cell_list_) {
    MS_EXCEPTION_IF_NULL(top_cell);
    if (top_cell->cell_id().find(cell_id) != std::string::npos) {
      top_cell->set_hook_changed(true);
    }
    const auto &sub_cells = top_cell->sub_cell_list();
    for (const auto &sub_cell_id : sub_cells) {
      if (sub_cell_id.find(cell_id) != std::string::npos) {
        top_cell->set_hook_changed(true);
      }
    }
  }
  if (need_construct_graph() && top_cell_ != nullptr) {
    top_cell_->set_sub_cell_hook_changed(cell_id);
  }
}

void GradExecutor::RecordGradOpInfo(const OpExecInfoPtr &op_exec_info) const {
  if (!grad_flag_) {
    MS_LOG(DEBUG) << "Grad flag is set to false, no need to record op info";
    return;
  }
  MS_EXCEPTION_IF_NULL(op_exec_info);
  std::string input_args_info;
  // Record input args info (weight or data)
  for (const auto mask : op_exec_info->inputs_mask) {
    if (mask != 0) {
      input_args_info += "w";
      continue;
    }
    input_args_info += "d";
  }
  // Record op name and index
  op_exec_info->op_info.clear();
  const auto &curr_op_num = top_cell()->op_num();
  op_exec_info->op_info += op_exec_info->op_name + "-" + std::to_string(curr_op_num) + "-" + input_args_info;
  // The out shape(not dynamic shape) is added to determine those ops that change the shape
  bool is_dynamic_shape_out = !top_cell()->dynamic_shape() || op_exec_info->has_dynamic_output;
  const auto &out_abs = op_exec_info->abstract;
  if (is_dynamic_shape_out && out_abs != nullptr) {
    auto shape = out_abs->BuildShape();
    MS_EXCEPTION_IF_NULL(shape);
    if (!shape->isa<abstract::NoShape>() && !shape->IsDimZero()) {
      op_exec_info->op_info += "-" + shape->ToString();
    }
  }
  const auto &all_op_info = top_cell()->all_op_info();
  top_cell()->set_all_op_info(all_op_info + "-" + op_exec_info->op_info);
  top_cell()->set_op_num(curr_op_num + 1);
}

void GradExecutor::SaveOutputNodeMap(const std::string &obj_id, const py::object &out_real, const CNodePtr &cnode) {
  if (cell_stack_.empty()) {
    MS_LOG(DEBUG) << "No need save output";
    return;
  }
  MS_EXCEPTION_IF_NULL(cnode);
  MS_LOG(DEBUG) << "Cnode is " << cnode->DebugString() << ", out value id " << obj_id;
  if (py::isinstance<py::tuple>(out_real)) {
    auto value = py::cast<py::tuple>(out_real);
    auto size = static_cast<int64_t>(value.size());
    if (size > 1) {
      for (int64_t i = 0; i < size; ++i) {
        auto value_id = GetId(value[static_cast<size_t>(i)]);
        SetNodeMapInGraphInfoMap(curr_g(), value_id, cnode, i);
      }
    }
  }
  SetNodeMapInGraphInfoMap(curr_g(), obj_id, cnode);
}

// Run ad grad for curr op and connect grad graph with previous op
void GradExecutor::DoOpGrad(const OpExecInfoPtr &op_exec_info, const CNodePtr &cnode, const ValuePtr &op_out) {
  MS_EXCEPTION_IF_NULL(op_out);
  if (grad_is_running_ && !bprop_grad_stack_.top().second) {
    MS_LOG(DEBUG) << "Custom bprop, no need do op grad";
    return;
  }

  ValuePtrList input_args;
  input_args.resize(op_exec_info->op_inputs.size(), nullptr);
  // Run in Vm, inputs not convert to tensor object, so need do transform it
  if (op_exec_info->input_tensors.empty()) {
    for (size_t i = 0; i < op_exec_info->op_inputs.size(); ++i) {
      input_args[i] = PyObjToValue(op_exec_info->op_inputs[i]);
    }
  } else {
    // Run in Ms, some op input tensor convert into attributes, so add them back
    for (auto &it : op_exec_info->index_with_value) {
      input_args[it.first] = it.second;
    }
    // Add other tensor
    for (size_t i = 0, j = 0; i < op_exec_info->op_inputs.size() && j < op_exec_info->input_tensors.size(); ++i) {
      if (input_args[i] == nullptr) {
        if (py::isinstance<tensor::Tensor>(op_exec_info->op_inputs[i])) {
          input_args[i] = op_exec_info->input_tensors[j];
        } else {
          // Like axis, can not be a tensor, just a value
          input_args[i] = PyObjToValue(op_exec_info->op_inputs[i]);
        }
        ++j;
      }
    }
  }
  if (op_exec_info->has_dynamic_output) {
    UpdateValueToDynamicShape(op_out, forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs);
  }

  if (!ad::GradPynativeOp(top_cell()->k_pynative_cell_ptr(), cnode, input_args, op_out)) {
    MS_LOG(EXCEPTION) << "Failed to run ad grad for op " << op_exec_info->op_name;
  }
}

void GradExecutor::SaveDynShapeAbsForMsFunction(const py::args &args, const py::object &out,
                                                const FuncGraphPtr &ms_func_graph) const {
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  auto output_node = ms_func_graph->output();
  MS_EXCEPTION_IF_NULL(output_node);

  // Update input to dynamic
  for (size_t i = 0; i < args.size(); ++i) {
    if (py::isinstance<tensor::Tensor>(args[i])) {
      const auto &input_i_tensor = args[i].cast<tensor::TensorPtr>();
      UpdateValueToDynamicShape(input_i_tensor, forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs);
    }
  }

  // Update output to dynamic
  SaveIdWithDynamicAbstract(out, output_node->abstract(),
                            &(forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs));
  const auto &output_value = PyObjToValue(out);
  UpdateValueToDynamicShape(output_value, forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs);

  // Save output by one id for abs get performance
  if (output_node->abstract()->BuildShape()->IsDynamic()) {
    forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs[GetId(out)] = output_node->abstract();
  }
}

void GradExecutor::UpdateMsFunctionForwardTensors(const OpExecInfoPtr &op_exec_info,
                                                  const ValuePtr &new_forward_value) const {
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
                                          ValuePtrList *input_values, CNodePtr *ms_function_cnode) const {
  // Get input node info of ms_function
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  std::vector<AnfNodePtr> input_nodes{NewValueNode(ms_func_graph)};
  MS_EXCEPTION_IF_NULL(input_values);
  for (size_t i = 0; i < args.size(); ++i) {
    const auto &input_i_node = GetInput(args[i], false);
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
    if (graph_info->params.count(param_name) != 0) {
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
  *ms_function_cnode = curr_g()->NewCNode(input_nodes);
  MS_LOG(DEBUG) << "Make ms function forward cnode: " << (*ms_function_cnode)->DebugString();
}

// Make adjoint for ms_function fprop graph and connect it with previous op
CNodePtr GradExecutor::MakeAdjointForMsFunction(const FuncGraphPtr &ms_func_graph, const FuncGraphPtr &grad_graph,
                                                const py::object &actual_out, const py::args &args,
                                                const ValuePtr &actual_out_v) const {
  ValuePtrList input_values;
  CNodePtr ms_function_cnode = nullptr;
  MakeCNodeForMsFunction(ms_func_graph, args, &input_values, &ms_function_cnode);
  MS_EXCEPTION_IF_NULL(ms_function_cnode);
  SetTupleArgsToGraphInfoMap(curr_g(), actual_out, ms_function_cnode);
  SetNodeMapInGraphInfoMap(curr_g(), GetId(actual_out), ms_function_cnode);

  // Connect grad graph of ms_function to context.
  auto k_pynative_cell_ptr = top_cell()->k_pynative_cell_ptr();
  MS_EXCEPTION_IF_NULL(k_pynative_cell_ptr);
  MS_EXCEPTION_IF_NULL(grad_graph);
  if (!k_pynative_cell_ptr->KPynativeWithFProp(ms_function_cnode, input_values, actual_out_v, grad_graph)) {
    MS_LOG(EXCEPTION) << "Failed to make adjoint for ms_function cnode, ms_function cnode info: "
                      << ms_function_cnode->DebugString();
  }
  top_cell()->set_ms_function_flag(true);
  return ms_function_cnode;
}

void GradExecutor::UpdateForwardTensorInfoInBpropGraph(const string &op_info, const ValuePtr &op_out) {
  if (!grad_flag_) {
    MS_LOG(DEBUG) << "The grad flag is false, no need to update forward op info in bprop graph";
    return;
  }
  MS_EXCEPTION_IF_NULL(op_out);
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
  mindspore::HashSet<std::string> forward_op_tensor_id;
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
  // Check exception case.
  const auto &tensor_id_with_tensor_object = top_cell()->tensor_id_with_tensor_object();
  MS_LOG(DEBUG) << "Current tensor_id_with_tensor_object size " << tensor_id_with_tensor_object.size();
  // Save tensor in value node of bprop graph
  for (const auto &tensor : tensors_in_bprop_graph) {
    MS_EXCEPTION_IF_NULL(tensor);
    if (forward_op_tensor_id.find(tensor->id()) == forward_op_tensor_id.end() || tensor->device_address() == nullptr) {
      continue;
    }
    tensor->set_is_forward_output(true);
    top_cell()->SetTensorIdWithTensorObject(tensor->id(), tensor);
    MS_LOG(DEBUG) << "Save forward tensor " << tensor.get() << " id " << tensor->id()
                  << " device address: " << tensor->device_address() << " shape and dtype "
                  << tensor->GetShapeAndDataTypeInfo();
  }
}

py::tuple ForwardExecutor::RunOpWithInitBackendPolicy(const OpExecInfoPtr &op_exec_info) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  auto backend_policy = GetBackendPolicy(op_exec_info);
  // returns a null py::tuple on error
  py::object result = RunOpWithBackendPolicy(backend_policy, op_exec_info);
  MS_LOG(DEBUG) << "RunOp end";
  return result;
}

MsBackendPolicy ForwardExecutor::GetBackendPolicy(const OpExecInfoPtr &op_exec_info) const {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_LOG(DEBUG) << "RunOp start, op name is: " << op_exec_info->op_name;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);

  MsBackendPolicy backend_policy = kMsBackendVmOnly;
#ifdef ENABLE_D
  if (ms_context->backend_policy() == "ge") {
    MS_LOG(EXCEPTION) << "In PyNative mode, not support ge backend!";
  }
  if (!context::IsTsdOpened(ms_context)) {
    if (!context::OpenTsd(ms_context)) {
      MS_LOG(EXCEPTION) << "Open tsd failed";
    }
  }
#endif
  return backend_policy;
}

py::object ForwardExecutor::RunOpWithBackendPolicy(MsBackendPolicy backend_policy, const OpExecInfoPtr &op_exec_info) {
  py::object result;
  if (backend_policy == kMsBackendVmOnly) {
#ifndef ENABLE_TEST
    if (kVmOperators.find(op_exec_info->op_name) != kVmOperators.end()) {
      result = RunOpInVM(op_exec_info);
    } else {
      result = RunOpInMs(op_exec_info);
    }
#else
    result = RunOpInVM(op_exec_info);
#endif
  }

  return result;
}

py::object ForwardExecutor::RunOpInVM(const OpExecInfoPtr &op_exec_info) {
  MS_LOG(DEBUG) << "RunOpInVM start";
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(op_exec_info->py_primitive);

  auto &op_inputs = op_exec_info->op_inputs;
  if (op_exec_info->op_name == prim::kPrimInsertGradientOf->name() ||
      op_exec_info->op_name == prim::kPrimStopGradient->name() ||
      op_exec_info->op_name == prim::kPrimHookBackward->name() ||
      op_exec_info->op_name == prim::kPrimCellBackwardHook->name()) {
    py::tuple result(op_inputs.size());
    for (size_t i = 0; i < op_inputs.size(); i++) {
      py::object input = op_inputs[i];
      auto tensor = py::cast<tensor::TensorPtr>(input);
      MS_EXCEPTION_IF_NULL(tensor);
      if (op_exec_info->op_name == prim::kPrimHookBackward->name() ||
          op_exec_info->op_name == prim::kPrimCellBackwardHook->name()) {
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
    SaveOutputDynamicShape(op_exec_info, op_exec_info->abstract, result);
    MS_LOG(DEBUG) << "RunOpInVM end";
    return result;
  }

  auto primitive = op_exec_info->py_primitive;
  MS_EXCEPTION_IF_NULL(primitive);
  auto result = primitive->RunPyComputeFunction(op_inputs);
  SaveOutputDynamicShape(op_exec_info, op_exec_info->abstract, result);
  MS_LOG(DEBUG) << "RunOpInVM end";
  if (py::isinstance<py::none>(result)) {
    MS_LOG(EXCEPTION) << "VM op " << op_exec_info->op_name << " run failed!";
  }
  if (py::isinstance<py::tuple>(result)) {
    return result;
  }
  py::tuple tuple_result = py::make_tuple(result);
  return tuple_result;
}

void ForwardExecutor::CheckIfNeedSyncForHeterogeneous(const std::string &cur_target) {
  if (last_target_ != "Unknown" && last_target_ != cur_target) {
    auto executor = PynativeExecutor::GetInstance();
    executor->Sync();
  }
  last_target_ = cur_target;
}

void ForwardExecutor::SetDynamicInput(const py::object &cell, const py::args &args) {
  auto &dynamic_index = dynamic_shape_info_ptr()->feed_dynamic_input[GetId(cell)];
  dynamic_index.resize(args.size());
  for (size_t i = 0; i < args.size(); i++) {
    auto value = PyObjToValue(args[i]);
    auto abstract = value->ToAbstract()->Broaden();
    MS_EXCEPTION_IF_NULL(abstract);
    dynamic_index[i] = abstract;
  }
}

void ForwardExecutor::ResetDynamicAbsMap() {
  if (IsFirstCell()) {
    // Clean up some resources for dynamic shape
    dynamic_shape_info_ptr()->reset();
  }
}

void ForwardExecutor::SetFeedDynamicInputAbs(const py::object &cell, const py::args &args) {
  if (!dynamic_shape_info_ptr()->HasFeedDynamicInput()) {
    return;
  }
  const auto &feed_dynamic_input = dynamic_shape_info_ptr()->feed_dynamic_input;
  const auto &cell_id = GetId(cell);
  auto it = feed_dynamic_input.find(cell_id);
  if (it != feed_dynamic_input.end()) {
    if (it->second.size() != args.size()) {
      MS_LOG(DEBUG) << "Dynamic input size " << it->second.size() << " is not equal to real input size " << args.size();
      return;
    }
    bool id_changed = false;
    for (size_t i = 0; i < args.size(); i++) {
      auto abs = it->second.at(i);
      MS_EXCEPTION_IF_NULL(abs);
      auto shape = abs->BuildShape();
      MS_EXCEPTION_IF_NULL(shape);
      if (shape->IsDynamic()) {
        const auto &arg_id = GetId(args[i]);
        MS_LOG(DEBUG) << "Set arg " << i << ", id " << arg_id << " to be dynamic shape; Arg self abs: "
                      << PyObjToValue(args[i])->ToAbstract()->Broaden()->ToString()
                      << ", dynamic abs: " << abs->ToString();
        dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs[arg_id] = abs;
        (void)node_abs_map_.erase(arg_id);
        id_changed = true;
      }
    }
    if (id_changed) {
      grad()->CheckPreviousTopCellCanBeDynamicShape(cell, args);
    }
  }
}

py::object ForwardExecutor::GetDynamicInput(const py::object &actual_input) {
  if (py::isinstance<py::tuple>(actual_input)) {
    py::tuple tuple_actual_args = py::cast<py::tuple>(actual_input);
    size_t args_size = tuple_actual_args.size();
    py::tuple dyn_shape_args = py::tuple(args_size);
    for (size_t i = 0; i < args_size; ++i) {
      dyn_shape_args[i] = GetDynamicInput(tuple_actual_args[i]);
    }
    return dyn_shape_args;
  } else if (py::isinstance<py::list>(actual_input)) {
    py::list list_actual_args = py::cast<py::list>(actual_input);
    size_t args_size = list_actual_args.size();
    py::list dyn_shape_args;
    for (size_t i = 0; i < args_size; ++i) {
      dyn_shape_args.append(GetDynamicInput(list_actual_args[i]));
    }
    return dyn_shape_args;
  } else if (py::isinstance<tensor::Tensor>(actual_input)) {
    const auto &obj_id_with_dynamic_output_abs = dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs;
    auto iter = obj_id_with_dynamic_output_abs.find(GetId(actual_input));
    if (iter != obj_id_with_dynamic_output_abs.end()) {
      auto tensor_ptr = py::cast<tensor::TensorPtr>(actual_input);
      MS_EXCEPTION_IF_NULL(tensor_ptr);
      auto dyn_tensor = std::make_shared<tensor::Tensor>(tensor_ptr->data_type(), tensor_ptr->shape_c());
      dyn_tensor->set_base_shape(GetShapeFromAbstract(iter->second));
      auto py_dyn_tensor = ValueToPyData(dyn_tensor);
      return py_dyn_tensor;
    }
  }
  return actual_input;
}

void ForwardExecutor::SaveOutputDynamicShape(const OpExecInfoPtr &op_exec_info, const AbstractBasePtr &real_abs,
                                             const py::object &obj) {
  // Save dynamic abs
  if (op_exec_info->has_dynamic_output) {
    SaveIdWithDynamicAbstract(obj, op_exec_info->abstract, &dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs);
  } else {
    // Update real abs
    op_exec_info->abstract = real_abs;
  }
}

py::object ForwardExecutor::RunOpInMs(const OpExecInfoPtr &op_exec_info) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  compile::SetMindRTEnable();
  MS_LOG(DEBUG) << "Start run op [" << op_exec_info->op_name << "] with backend policy ms";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, true);
  const std::string &device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  uint32_t device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto enable_mind_rt = ms_context->get_param<bool>(MS_CTX_ENABLE_MINDRT);

  std::string cur_target = GetCurrentDeviceTarget(device_target, op_exec_info->py_primitive);
  CheckIfNeedSyncForHeterogeneous(cur_target);

  std::vector<tensor::TensorPtr> input_tensors;
  std::vector<int64_t> tensors_mask;
  std::string graph_info;
  ConstructInputTensor(op_exec_info, &tensors_mask, &input_tensors);
  UpdateInputTensorToDynamicShape(op_exec_info, &input_tensors);
  op_exec_info->input_tensors = input_tensors;
  ConvertAttrToUnifyMindIR(op_exec_info);
  // get graph info for checking it whether existing in the cache
  GetSingleOpGraphInfo(op_exec_info, input_tensors, tensors_mask, &graph_info);
#if defined(__APPLE__)
  session::OpRunInfo op_run_info = {true,
                                    false,
                                    op_exec_info->op_name,
                                    op_exec_info->py_primitive.get(),
                                    op_exec_info->abstract,
                                    op_exec_info->has_dynamic_input,
                                    op_exec_info->has_dynamic_output,
                                    op_exec_info->is_mixed_precision_cast,
                                    false,
                                    op_exec_info->next_op_name,
                                    static_cast<int>(op_exec_info->next_input_index),
                                    graph_info,
                                    tensors_mask,
                                    input_tensors,
                                    cur_target};
#else
  session::OpRunInfo op_run_info = {true,
                                    false,
                                    op_exec_info->op_name,
                                    op_exec_info->py_primitive.get(),
                                    op_exec_info->abstract,
                                    op_exec_info->has_dynamic_input,
                                    op_exec_info->has_dynamic_output,
                                    op_exec_info->is_mixed_precision_cast,
                                    op_exec_info->lazy_build,
                                    op_exec_info->next_op_name,
                                    op_exec_info->next_input_index,
                                    graph_info,
                                    tensors_mask,
                                    input_tensors,
                                    cur_target};
#endif

  VectorRef outputs;
  if (!enable_mind_rt) {
    auto cur_session = GetCurrentSession(cur_target, device_id);
    MS_EXCEPTION_IF_NULL(cur_session);
    cur_session->RunOp(&op_run_info, &outputs);
  } else {
    auto cur_mind_rt_backend = GetMindRtBackend(cur_target, device_id);
    MS_EXCEPTION_IF_NULL(cur_mind_rt_backend);
    mindspore::ScopedLongRunning long_running;
    cur_mind_rt_backend->RunOp(&op_run_info, &outputs);
  }

  auto result = BaseRefToPyData(outputs);
  // Save dynamic shape for next op run
  SaveOutputDynamicShape(op_exec_info, op_run_info.abstract, result);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, false);
  MS_LOG(DEBUG) << "End run op [" << op_exec_info->op_name << "] with backend policy ms";
  return result;
}

void ForwardExecutor::ClearRes() {
  MS_LOG(DEBUG) << "Clear forward res";
  lazy_build_ = false;
  cell_depth_ = 0;
  implicit_cast_map_.clear();
  prim_abs_list_.clear();
  node_abs_map_.clear();
  dynamic_shape_info_ptr()->reset();
}

ForwardExecutorPtr GradExecutor::forward() const {
  auto forward_executor = forward_executor_.lock();
  MS_EXCEPTION_IF_NULL(forward_executor);
  return forward_executor;
}

const TopCellInfoPtr &GradExecutor::top_cell() const {
  MS_EXCEPTION_IF_NULL(top_cell_);
  return top_cell_;
}

FuncGraphPtr GradExecutor::curr_g() const {
  auto fg = top_cell()->fg();
  MS_EXCEPTION_IF_NULL(fg);
  return fg;
}

void GradExecutor::PushCellStack(const std::string &cell_id) {
  cell_stack_.push(cell_id);
  ++cell_order_;
}

void GradExecutor::PopCellStack() {
  if (cell_stack_.empty()) {
    MS_LOG(EXCEPTION) << "Stack cell_stack_ is empty";
  }
  cell_stack_.pop();
}

std::string GradExecutor::GetCurCellOrder() const {
  if (cell_stack_.empty()) {
    MS_LOG(EXCEPTION) << "The cell_stack_ is empty!";
  }
  return cell_stack_.top() + "_" + std::to_string(cell_order_);
}

void GradExecutor::PushHighOrderGraphStack(const TopCellInfoPtr &top_cell) { high_order_stack_.push(top_cell); }

TopCellInfoPtr GradExecutor::PopHighOrderGraphStack() {
  if (high_order_stack_.empty()) {
    MS_LOG(EXCEPTION) << "Stack high_order_stack_ is empty";
  }
  high_order_stack_.pop();
  TopCellInfoPtr top_cell = nullptr;
  if (!high_order_stack_.empty()) {
    top_cell = high_order_stack_.top();
  }
  return top_cell;
}

std::string GradExecutor::GetCellId(const py::object &cell, const py::args &args) const {
  auto cell_id = GetId(cell);
  auto fn = [&cell_id](const abstract::AbstractBasePtr &abs) {
    MS_EXCEPTION_IF_NULL(abs);
    auto shape = abs->BuildShape();
    auto type = abs->BuildType();
    cell_id += "_" + shape->ToString();
    cell_id += type->ToString();
  };

  for (size_t i = 0; i < args.size(); i++) {
    const auto &arg_id = GetId(args[i]);
    // Get dynamic input, like data sink
    const auto item = forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs.find(arg_id);
    if (item != forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs.end()) {
      MS_LOG(DEBUG) << "Input " << i << " get dynamic input";
      fn(item->second);
      continue;
    }

    // Find in step process
    auto it = forward()->node_abs_map().find(arg_id);
    if (it != forward()->node_abs_map().end()) {
      fn(it->second);
    } else {
      auto value = PyObjToValue(args[i]);
      MS_EXCEPTION_IF_NULL(value);
      auto abs = value->ToAbstract();
      MS_EXCEPTION_IF_NULL(abs);
      if (abs->isa<abstract::AbstractTensor>()) {
        abs->set_value(kAnyValue);
      }
      forward()->SetNodeAbsMap(arg_id, abs);
      fn(abs);
    }
  }
  return cell_id;
}

void GradExecutor::DumpGraphIR(const std::string &filename, const FuncGraphPtr &graph) const {
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
  auto l_index = l_cell_id.find('_');
  auto r_index = r_cell_id.find('_');
  return l_cell_id.substr(0, l_index) == r_cell_id.substr(0, r_index);
}

bool GradExecutor::IsBpropGraph(const std::string &cell_id) {
  if (top_cell_ == nullptr) {
    return false;
  }
  return std::any_of(bprop_cell_list_.begin(), bprop_cell_list_.end(),
                     [&cell_id](const std::string &value) { return cell_id.find(value) != std::string::npos; });
}

void GradExecutor::UpdateTopCellInfo(bool forward_already_run, bool need_compile_graph, bool vm_compiled) const {
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
      (void)already_run_top_cell_.erase(already_run_cell_id);
      it = top_cell_list_.erase(it);
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
      auto new_param = curr_g()->add_parameter();
      const auto &param_id = GetId(param);
      SetTupleArgsToGraphInfoMap(curr_g(), param, new_param, true);
      SetNodeMapInGraphInfoMap(curr_g(), param_id, new_param);
      SetParamNodeMapInGraphInfoMap(curr_g(), param_id, new_param);
    }
    return;
  }
  // Convert input args to parameters for top cell graph in construct.
  std::vector<ValuePtr> input_param_values;
  const auto &only_tensors = FilterTensorArgs(args);
  for (size_t i = 0; i < only_tensors.size(); ++i) {
    auto new_param = curr_g()->add_parameter();
    auto param_i = only_tensors[i];
    const auto &param_i_value = PyObjToValue(param_i);
    input_param_values.emplace_back(param_i_value);
    const auto &param_i_id = GetId(param_i);
    abstract::AbstractBasePtr param_i_abs = nullptr;
    auto item = forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs.find(param_i_id);
    if (item != forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs.end()) {
      MS_LOG(DEBUG) << "Param " << i << " is dynamic input";
      param_i_abs = item->second;
    } else {
      param_i_abs = param_i_value->ToAbstract();
      MS_EXCEPTION_IF_NULL(param_i_abs);
      param_i_abs = param_i_abs->Broaden();
    }
    MS_EXCEPTION_IF_NULL(param_i_abs);
    new_param->set_abstract(param_i_abs);
    SetTupleArgsToGraphInfoMap(curr_g(), param_i, new_param, true);
    SetNodeMapInGraphInfoMap(curr_g(), param_i_id, new_param);
    SetParamNodeMapInGraphInfoMap(curr_g(), param_i_id, new_param);
    SetParamNodeMapInGraphInfoMap(top_cell_->df_builder(), param_i_id, new_param);
  }
  top_cell()->set_k_pynative_cell_ptr(ad::GradPynativeCellBegin(curr_g()->parameters(), input_param_values));
}

void GradExecutor::InitResourceAndDfBuilder(const std::string &cell_id, const py::object &cell, const py::args &args) {
  if (cell_stack_.empty() || IsNestedGrad()) {
    if (cell_stack_.empty() && !grad_is_running_) {
      MS_LOG(DEBUG) << "Make new topest graph";
      MakeNewTopGraph(cell_id, cell, args, true);
    } else if (grad_is_running_ && IsBpropGraph(cell_id)) {
      MS_LOG(DEBUG) << "Run bprop cell";
      auto fg = std::make_shared<FuncGraph>();
      top_cell()->set_fg(fg);
      auto graph_info_cg = std::make_shared<GraphInfo>(cell_id);
      top_cell()->SetGraphInfoMap(fg, graph_info_cg);
      HandleInputArgsForTopCell(args, true);
      bprop_grad_stack_.push(std::make_pair(cell_id, false));
    } else if (grad_is_running_ && top_cell()->grad_order() != grad_order_) {
      MS_LOG(DEBUG) << "Nested grad graph existed in bprop";
      MakeNewTopGraph(cell_id, cell, args, false);
      bprop_grad_stack_.push(std::make_pair(cell_id, true));
    } else if (!cell_stack_.empty() && IsNestedGrad() && top_cell()->grad_order() != grad_order_) {
      MS_LOG(DEBUG) << "Nested grad graph existed in construct";
      auto cur_top_is_dynamic = top_cell()->dynamic_graph_structure();
      MakeNewTopGraph(cell_id, cell, args, false);
      top_cell()->set_dynamic_graph_structure(cur_top_is_dynamic);
    }
  }

  PushCellStack(cell_id);
  // Init kPynativeCellPtr with input parameters of top cell
  if (!top_cell()->is_init_kpynative()) {
    auto graph_info_cg = std::make_shared<GraphInfo>(cell_id);
    top_cell()->SetGraphInfoMap(curr_g(), graph_info_cg);
    auto graph_info_df = std::make_shared<GraphInfo>(cell_id);
    top_cell()->SetGraphInfoMap(top_cell_->df_builder(), graph_info_df);
    HandleInputArgsForTopCell(args, false);
    top_cell()->set_need_compile_graph(true);
    top_cell()->set_init_kpynative(true);
  } else {
    // Non-top cell
    top_cell()->SetSubCellList(cell_id);
  }
}

void GradExecutor::NewGraphInner(const py::object *ret, const py::object &cell, const py::args &args) {
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
      MS_LOG(DEBUG) << "Pre top cell, hook_changed " << pre_top_cell->hook_changed() << ", dynamic_graph_structure "
                    << pre_top_cell->dynamic_graph_structure();
      if (pre_top_cell->hook_changed()) {
        (void)already_run_top_cell_.erase(top_it);
        EraseTopCellFromTopCellList(pre_top_cell);
      } else if (!pre_top_cell->dynamic_graph_structure()) {
        MS_LOG(DEBUG) << "Top cell " << cell_id << " is not dynamic structure, no need to run NewGraphInner again";
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
  InitResourceAndDfBuilder(cell_id, cell, args);
  // Check whether cell has dynamic construct
  if (!top_cell()->dynamic_graph_structure()) {
    bool is_dynamic_structure = parse::DynamicParser::IsDynamicCell(cell);
    MS_LOG(DEBUG) << "Current cell dynamic " << is_dynamic_structure;
    if (is_dynamic_structure) {
      top_cell()->set_dynamic_graph_structure(is_dynamic_structure);
    }
  }
}

void GradExecutor::ChangeTopCellInfo(const TopCellInfoPtr &top_cell, size_t args_size) {
  MS_EXCEPTION_IF_NULL(top_cell);
  std::string new_cell_id = top_cell->cell_self_info()->cell_self_id;
  for (size_t i = 0; i < args_size; ++i) {
    new_cell_id += "_" + top_cell->cell_self_info()->args_shape[i]->ToString();
    new_cell_id += top_cell->cell_self_info()->args_type[i]->ToString();
  }
  MS_LOG(DEBUG) << "Change top cell " << top_cell->cell_id() << " to be dynamic " << new_cell_id;
  top_cell->set_cell_id(new_cell_id);
  top_cell->set_already_run_cell_id(GetAlreadyRunCellId(new_cell_id));
}

TopCellInfoPtr GradExecutor::ChangeTopCellToDynamicShapeByAuto(const TopCellInfoPtr &top_cell,
                                                               const std::vector<ShapeVector> &new_args_shape,
                                                               const py::object &cell, const py::args &args) {
  MS_EXCEPTION_IF_NULL(top_cell);
  // Change args shape
  for (size_t i = 0; i < args.size(); ++i) {
    top_cell->cell_self_info()->args_shape[i] = std::make_shared<abstract::Shape>(new_args_shape[i]);
    if (py::isinstance<tensor::Tensor>(args[i])) {
      auto tensor = args[i].cast<tensor::TensorPtr>();
      tensor->set_base_shape(top_cell->cell_self_info()->args_shape[i]);
    }
    forward()->EraseFromNodeAbsMap(GetId(args[i]));
  }
  // Set to feed dynamic map, later shapes can match it
  MS_LOG(DEBUG) << "Set dynamic input for auto dynamic shape";
  forward()->SetDynamicInput(cell, args);
  forward()->SetFeedDynamicInputAbs(cell, args);
  ChangeTopCellInfo(top_cell, new_args_shape.size());
  return top_cell;
}

TopCellInfoPtr GradExecutor::ChangeTopCellToDynamicShapeBySetInputs(const TopCellInfoPtr &top_cell,
                                                                    const std::vector<ShapeVector> &new_args_shape,
                                                                    const py::object &cell) {
  MS_EXCEPTION_IF_NULL(top_cell);
  // Change args shape
  for (size_t i = 0; i < new_args_shape.size(); ++i) {
    top_cell->cell_self_info()->args_shape[i] = std::make_shared<abstract::Shape>(new_args_shape[i]);
  }
  const auto &feed_dynamic_input = forward()->dynamic_shape_info_ptr()->feed_dynamic_input;
  auto it = feed_dynamic_input.find(GetId(cell));
  if (it != feed_dynamic_input.end()) {
    for (size_t i = 0; i < new_args_shape.size(); i++) {
      auto abs = it->second.at(i);
      MS_EXCEPTION_IF_NULL(abs);
      auto shape = abs->BuildShape();
      MS_EXCEPTION_IF_NULL(shape);
      if (shape->IsDynamic()) {
        const auto &arg_id = top_cell->cell_self_info()->args_id[i];
        MS_LOG(DEBUG) << "Set arg " << i << ", id " << arg_id << ", dynamic abs: " << abs->ToString();
        forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs[arg_id] = abs;
        forward()->ClearNodeAbsMap();
      }
    }
  }
  ChangeTopCellInfo(top_cell, new_args_shape.size());
  return top_cell;
}

void GradExecutor::UpdateTopCellId(const py::args &args) {
  if (top_cell_ == nullptr || top_cell_->cell_self_info() == nullptr) {
    return;
  }
  mindspore::HashMap<std::string, ShapeVector> id_with_shape;
  ShapeVector empty_shape;
  bool has_dynamic_id = false;
  for (size_t i = 0; i < args.size(); i++) {
    const auto &arg_id = GetId(args[i]);
    const auto item = forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs.find(arg_id);
    if (item != forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs.end()) {
      id_with_shape[arg_id] = (GetShapeFromAbstract(item->second))->shape();
      has_dynamic_id = true;
    } else {
      id_with_shape[arg_id] = empty_shape;
    }
  }
  if (!has_dynamic_id) {
    return;
  }
  // Check current top cell need change id to dynamic id
  const auto &args_id = top_cell()->cell_self_info()->args_id;
  bool need_change = std::any_of(args_id.begin(), args_id.end(), [&id_with_shape](const std::string &id) {
    return id_with_shape.find(id) != id_with_shape.end();
  });
  if (need_change) {
    // Change args shape
    for (size_t i = 0; i < id_with_shape.size(); ++i) {
      const auto it = std::next(id_with_shape.begin(), static_cast<int64_t>(i));
      if (!it->second.empty() && top_cell()->cell_self_info()->args_id[i] == it->first) {
        top_cell()->cell_self_info()->args_shape[i] = std::make_shared<abstract::Shape>(it->second);
      }
    }
    ChangeTopCellInfo(top_cell(), top_cell()->cell_self_info()->args_id.size());
  }
}

TopCellInfoPtr GradExecutor::GetTopCellWithDynamicShape(const py::object &cell, const py::args &args, bool is_auto) {
  // Current return nullptr for disable auto dynamic shape feature; Later after a complete test will enable this
  if (is_auto && !py::isinstance<py::none>(cell)) {
    return nullptr;
  }
  const auto &cell_self_id = GetId(cell);
  const auto it =
    std::find_if(top_cell_list_.begin(), top_cell_list_.end(), [&cell_self_id](const TopCellInfoPtr &elem) {
      return elem->cell_self_info() != nullptr && elem->cell_self_info()->cell_self_id == cell_self_id;
    });
  if (it != top_cell_list_.end()) {
    const auto &elem = *it;
    if (elem->dynamic_shape()) {
      MS_LOG(DEBUG) << "Elem has already dynamic shape";
      return nullptr;
    }
    std::vector<ShapeVector> new_args_shape;
    FindMatchTopCell(elem, args, &new_args_shape);
    // Change top cell to be dynamic
    if (new_args_shape.size() == args.size()) {
      if (is_auto) {
        return ChangeTopCellToDynamicShapeByAuto(elem, new_args_shape, cell, args);
      } else {
        return ChangeTopCellToDynamicShapeBySetInputs(elem, new_args_shape, cell);
      }
    }
  }
  UpdateTopCellId(args);
  return nullptr;
}

void GradExecutor::CheckPreviousTopCellCanBeDynamicShape(const py::object &cell, const py::args &args) {
  if (!grad_flag()) {
    return;
  }
  // In ms_function, new graph run before construct, so top cell create first; After that, set_dynamic_input call
  // in construct, here change top cell to dynamic.
  if (GetTopCellWithDynamicShape(cell, args, false) != nullptr) {
    MS_LOG(DEBUG) << "Convert ms_function top cell to dynamic shape.";
  }
}

void GradExecutor::MakeNewTopGraph(const string &cell_id, const py::object &cell, const py::args &args,
                                   bool is_topest) {
  pipeline::CheckArgsValid(cell, args);
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
    // Delete top cell from begin
    auto delete_first_top_cell = top_cell_list_.front();
    MS_EXCEPTION_IF_NULL(delete_first_top_cell);
    delete_first_top_cell->Clear();
    (void)already_run_top_cell_.erase(delete_first_top_cell->already_run_cell_id());
    (void)top_cell_list_.erase(top_cell_list_.begin());
    MS_LOG(WARNING) << "Too many top cell has been built, please check if the cell " << cell.cast<CellPtr>()->ToString()
                    << " is repeatedly defined in each epoch";
  }
  // Create top cell
  auto fg = std::make_shared<FuncGraph>();
  auto df_builder = std::make_shared<FuncGraph>();
  auto resource = std::make_shared<pipeline::Resource>();
  const auto &already_run_cell_id = GetAlreadyRunCellId(cell_id);
  auto top_cell =
    std::make_shared<TopCellInfo>(is_topest, grad_order_, resource, fg, df_builder, cell_id, already_run_cell_id);
  top_cell->set_forward_already_run(true);
  top_cell->set_input_args_id(input_args_id);
  TopCellInfoPtr top_cell_with_dynamic_shape = GetTopCellWithDynamicShape(cell, args, true);
  if (top_cell_with_dynamic_shape != nullptr) {
    top_cell->set_cell_id(top_cell_with_dynamic_shape->cell_id());
    top_cell->set_already_run_cell_id(top_cell_with_dynamic_shape->already_run_cell_id());
    top_cell->set_cell_self_info(top_cell_with_dynamic_shape->cell_self_info());
    EraseTopCellFromTopCellList(top_cell_with_dynamic_shape);
    MS_LOG(DEBUG) << "Pre top cell and current top cell merged to one top cell with dynamic shape";
  } else {
    top_cell->SetCellSelfInfoForTopCell(cell, args);
  }
  (void)top_cell_list_.emplace_back(top_cell);
  PushHighOrderGraphStack(top_cell);
  set_top_cell(top_cell);
  MS_LOG(DEBUG) << "New top graph, fg ptr " << fg.get() << " resource ptr " << resource.get();
}

void GradExecutor::SetTupleArgsToGraphInfoMap(const FuncGraphPtr &g, const py::object &args, const AnfNodePtr &node,
                                              bool is_param) const {
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
                                                  const std::vector<int64_t> &index_sequence, bool is_param) const {
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

ValuePtr GradExecutor::GetSensValueForDynamicShapeOutput(const py::object &out, const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  auto out_value = PyObjToValue(out);
  if (!ValueHasDynamicShape(out_value)) {
    return out_value;
  }
  MS_LOG(DEBUG) << "Set sens value with op info: " << kSensInfo;
  // Create sens value
  auto sens_value = SetSensValue(out_value, top_cell());
  // Ready for replace
  std::vector<tensor::TensorPtr> all_op_tensors;
  // Get output tensors
  TensorValueToTensor(sens_value, &all_op_tensors);
  // Save all tensors info of current op
  SaveOpInfo(top_cell_, kSensInfo, all_op_tensors);
  return sens_value;
}

void GradExecutor::UpdateSensValueForDynamicShapeOutput(const py::object &out) const {
  if (top_cell()->op_info_with_tensor_id().count(kSensInfo) == 0) {
    return;
  }
  MS_LOG(DEBUG) << "Update sens value with op info: " << kSensInfo;
  std::vector<tensor::TensorPtr> new_tensors;
  auto out_value = PyObjToValue(out);
  TensorValueToTensor(out_value, &new_tensors);
  if (new_tensors.empty()) {
    MS_LOG(DEBUG) << "The size of added forward tensors is zero, no need to update.";
    return;
  }
  // Update new output tensor info in bprop graph
  const auto &pre_op_tensor_id = top_cell()->op_info_with_tensor_id().at(kSensInfo);
  if (pre_op_tensor_id.size() != new_tensors.size()) {
    MS_LOG(EXCEPTION) << "The size of pre op tensor id: " << pre_op_tensor_id.size()
                      << " is not equal to the size of all tensors of current op " << new_tensors.size();
  }
  const auto &pre_tensor_id_with_tensor_object = top_cell()->tensor_id_with_tensor_object();
  for (size_t i = 0; i < pre_op_tensor_id.size(); ++i) {
    auto pre_id = pre_op_tensor_id[i];
    if (pre_tensor_id_with_tensor_object.find(pre_id) == pre_tensor_id_with_tensor_object.end()) {
      continue;
    }
    const auto &old_tensor_list = pre_tensor_id_with_tensor_object.at(pre_id);
    if (old_tensor_list.empty()) {
      MS_LOG(EXCEPTION) << "Get empty old tensor list";
    }
    const auto &old_tensor = old_tensor_list.front();
    const auto &new_tensor = new_tensors[i];
    MS_EXCEPTION_IF_NULL(old_tensor);
    MS_EXCEPTION_IF_NULL(new_tensor);
    MS_LOG(DEBUG) << "Replace Old tensor id " << old_tensor->id() << ", shape and type "
                  << old_tensor->GetShapeAndDataTypeInfo() << "; With new tensor id " << new_tensor->id()
                  << ", shape and dtype " << new_tensor->GetShapeAndDataTypeInfo();
    (void)old_tensor->set_shape(new_tensor->shape());
    (void)old_tensor->set_data_type(new_tensor->data_type());
    // New tensor have no device address, let old tensor device address nullptr for realloc in later stage
    old_tensor->set_device_address(nullptr);
  }
}

void GradExecutor::SetForwardLastNodeInfo(const py::object &out) const {
  auto output_node = GetObjNode(out, GetId(out));
  MS_EXCEPTION_IF_NULL(output_node);
  if (top_cell()->dynamic_shape()) {
    abstract::AbstractBasePtr last_node_abs = nullptr;
    if (output_node->abstract() == nullptr) {
      last_node_abs = PyObjToValue(out)->ToAbstract()->Broaden();
    } else {
      last_node_abs = output_node->abstract();
    }
    MS_EXCEPTION_IF_NULL(last_node_abs);
    // Set last output abstract and will be used for sens
    top_cell()->set_last_output_abs(last_node_abs);
  }
  // Set last node and sens for build adjoint
  const auto &sens_value = GetSensValueForDynamicShapeOutput(out, output_node);
  auto k_pynative_cell_ptr = top_cell()->k_pynative_cell_ptr();
  MS_EXCEPTION_IF_NULL(k_pynative_cell_ptr);
  k_pynative_cell_ptr->UpdateOutputNodeOfTopCell(output_node, sens_value);
}

void GradExecutor::EndGraphInner(const py::object *ret, const py::object &cell, const py::object &out,
                                 const py::args &args) {
  MS_EXCEPTION_IF_NULL(ret);
  const auto &cell_id = GetCellId(cell, args);
  MS_LOG(DEBUG) << "EndGraphInner start " << args.size() << " " << cell_id;
  if (cell_stack_.empty()) {
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
      // Top cell update sens
      UpdateSensValueForDynamicShapeOutput(out);
    }
    MS_LOG(DEBUG) << "Current cell " << cell_id << " no need to run EndGraphInner again";
    return;
  }
  DoGradForCustomBprop(cell, out, args);
  PopCellStack();
  if (grad_is_running_ && !bprop_grad_stack_.empty()) {
    if (!bprop_grad_stack_.top().second) {
      curr_g()->set_output(GetObjNode(out, GetId(out)));
      bprop_grad_stack_.pop();
      return;
    } else if (bprop_grad_stack_.top().first == cell_id) {
      bprop_grad_stack_.pop();
    }
  }
  // Just only dump the last forward graph
  bool is_top_cell_end = cell_id == top_cell()->cell_id();
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG) && is_top_cell_end) {
    curr_g()->set_output(GetObjNode(out, GetId(out)));
#ifdef ENABLE_DUMP_IR
    DumpIR("fg.ir", curr_g());
#endif
  }
  // Reset grad flag and update output node of the outermost cell
  if (cell_stack_.empty() && is_top_cell_end) {
    MS_LOG(DEBUG) << "Cur top last cell " << cell_id;
    PopHighOrderGraphStack();
    SetForwardLastNodeInfo(out);
    top_cell()->ClearCellHookOp();
    cell_order_ = 0;
    set_grad_flag(false);
  }
  // Checkout whether need to compile graph when each top cell has ran finished
  if (is_top_cell_end) {
    // In high grad cases, the output of the internal graph may be a tuple, and node needs to be created in the getobj
    if (!cell_stack_.empty()) {
      SetForwardLastNodeInfo(out);
    }
    top_cell()->CheckSubCellHookChanged();
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
  py::function bprop_func = py::getattr(cell, parse::CUSTOM_BPROP_NAME);
  py::object code_obj = py::getattr(bprop_func, "__code__");
  // When the co_names is empty, we will still get a tuple which is empty.
  auto co_names = py::getattr(code_obj, "co_names").cast<py::tuple>();
  for (auto name : co_names) {
    if (!py::hasattr(cell, name)) {
      continue;
    }
    auto var = py::getattr(cell, name);
    if (py::hasattr(var, "__parameter__") && py::isinstance<tensor::MetaTensor>(var)) {
      MS_LOG(EXCEPTION) << "The user defined 'bprop' function does not support using Parameter.";
    }
  }

  auto bprop_func_cellid = GetId(bprop_func);
  bprop_cell_list_.emplace_back(bprop_func_cellid);
  auto fake_prim = std::make_shared<PrimitivePy>(prim::kPrimHookBackward->name());
  if (py::isinstance<Cell>(cell)) {
    auto cell_ptr = py::cast<CellPtr>(cell);
    fake_prim->set_bprop_cls_name(cell_ptr->name());
  }
  fake_prim->AddBackwardHookFn(0, bprop_func);

  const auto &cell_id = GetCellId(cell, args);
  (void)fake_prim->AddAttr("cell_id", MakeValue(cell_id));
  (void)fake_prim->AddAttr(parse::CUSTOM_BPROP_NAME, MakeValue(true));

  py::object co_name = py::getattr(code_obj, "co_name");
  if (std::string(py::str(co_name)) == "staging_specialize") {
    MS_LOG(EXCEPTION) << "Decorating bprop with '@ms_function' is not supported.";
  }
  // Three parameters self, out and dout need to be excluded
  const size_t inputs_num = py::cast<int64_t>(py::getattr(code_obj, "co_argcount")) - 3;
  if (inputs_num != args.size()) {
    MS_EXCEPTION(TypeError) << "Size of bprop func inputs[" << inputs_num
                            << "] is not equal to the size of cell inputs[" << args.size() << "]";
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
  std::string already_run_cell_id(cell_id);
  already_run_cell_id += std::to_string(grad_order_ == 0 ? 1 : grad_order_);
  already_run_cell_id += "_" + grad_operation_;
  MS_LOG(DEBUG) << "Get already run top cell id " << already_run_cell_id;
  return already_run_cell_id;
}

std::string GradExecutor::GetGradCellId(bool has_sens, const py::object &cell, const py::args &args) const {
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

void GradExecutor::MarkMsFunctionNodes(const pipeline::ResourcePtr &resource) {
  auto func_graph = resource->func_graph();
  std::vector<size_t> in_ms_function;
  const auto &parameters = func_graph->parameters();
  for (const auto &parameter : parameters) {
    auto param = parameter->cast<ParameterPtr>();
    if (!param->has_default()) {
      continue;
    }
    auto iter = std::find(ms_function_params_.begin(), ms_function_params_.end(), param->name());
    if (iter != ms_function_params_.end()) {
      in_ms_function.push_back(1);
    } else {
      in_ms_function.push_back(0);
    }
  }

  auto ret = func_graph->get_return();
  auto ret_cnode = ret->cast<CNodePtr>();
  auto grads = ret_cnode->input(1)->cast<CNodePtr>();
  for (size_t i = 1; i < grads->inputs().size(); i++) {
    if (in_ms_function[i - 1] != 0) {
      auto node = grads->input(i);
      if (!node->isa<CNode>()) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      cnode->set_parallel(true);
    }
  }
}

void GradExecutor::GradNetInner(const py::object *ret, const prim::GradOperationPtr &grad, const py::object &cell,
                                const py::object &weights, const py::object &grad_position, const py::args &args) {
  MS_EXCEPTION_IF_NULL(ret);
  MS_EXCEPTION_IF_NULL(grad);
  auto size = args.size();
  const auto &cell_id = GetGradCellId(grad->sens_param(), cell, args);
  MS_LOG(DEBUG) << "GradNet start " << size << " " << cell_id;
  if (!top_cell()->need_compile_graph()) {
    MS_LOG(DEBUG) << "No need compile graph";
    UpdateTopCellInfo(false, false, !cell_stack_.empty());
    return;
  }
  top_cell()->set_grad_operation(grad_operation_);
  auto resource = top_cell()->resource();
  MS_EXCEPTION_IF_NULL(resource);
  auto df_builder = top_cell()->df_builder();
  MS_EXCEPTION_IF_NULL(df_builder);
  MS_LOG(DEBUG) << "fg ptr " << curr_g().get() << " resource ptr " << resource.get();

  // Get params(weights) require derivative
  auto w_args = GetWeightsArgs(weights, df_builder);
  auto p_args = GetGradPositionArgs(grad_position, grad->get_by_position_);
  if (w_args.empty() && !df_builder->parameters().empty()) {
    MS_LOG(DEBUG) << "Add weights params to w_args";
    (void)w_args.insert(w_args.end(), df_builder->parameters().cbegin(), df_builder->parameters().cend());
  }
  // Get bprop graph of top cell
  auto bprop_graph = GetBpropGraph(grad, cell, w_args, p_args, size, args);
  MS_EXCEPTION_IF_NULL(bprop_graph);
  bprop_graph->set_flag(kFlagIsPynativeBpropGraph, true);
  resource->set_func_graph(bprop_graph);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(bprop_graph, true);
  DumpGraphIR("launch_bprop_graph.ir", bprop_graph);
  // Launch bprop graph to backend
  SaveForwardTensorInfoInBpropGraph(resource);
  compile::SetMindRTEnable();
  resource->SetResult(pipeline::kBackend, compile::CreateBackend());
  MS_LOG(DEBUG) << "Start task emit action";
  auto parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode == parallel::kSemiAutoParallel || parallel_mode == parallel::kAutoParallel) {
    MarkMsFunctionNodes(resource);
  }
  TaskEmitAction(resource);
  MS_LOG(DEBUG) << "Start execute action";
  ExecuteAction(resource);
  MS_LOG(DEBUG) << "Start update top cell info when run finish";
  UpdateTopCellInfo(false, false, true);
  resource->Clean();
  abstract::AnalysisContext::ClearContext();
  // Clean cache used for parse. As static variable is released after
  // Python threads is released.
  parse::data_converter::ClearObjectCache();
  parse::Parser::CleanParserResource();
  trace::ClearTraceStack();
}

std::vector<AnfNodePtr> GradExecutor::GetWeightsArgs(const py::object &weights, const FuncGraphPtr &df_builder) const {
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
    const auto &name_attr = python_adapter::GetPyObjAttr(param, "name");
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

std::vector<size_t> GradExecutor::GetGradPositionArgs(const py::object &grad_position, bool get_by_position) const {
  std::vector<size_t> pos_args;
  if (!get_by_position) {
    return pos_args;
  }

  if (py::isinstance<py::tuple>(grad_position)) {
    const auto &tuple = grad_position.cast<py::tuple>();
    (void)std::transform(tuple.begin(), tuple.end(), std::back_inserter(pos_args),
                         [](const py::handle &elem) { return py::cast<int64_t>(elem); });
    return pos_args;
  }
  MS_LOG(EXCEPTION) << "Grad position only support tuple when grad by position.";
}

void GradExecutor::ShallowCopySensValue(const py::tuple &input_args, bool has_sens, VectorRef *run_args) const {
  if (!has_sens) {
    return;
  }
  // Get index and number of sens args.
  size_t sens_index = input_args.size() - 1;
  size_t sens_num = 1;
  if (py::isinstance<py::tuple>(input_args[sens_index])) {
    py::tuple tuple_sens = py::cast<py::tuple>(input_args[sens_index]);
    sens_num = ConvertArgs(tuple_sens).size();
  }
  // Shallow copy sens args to new sens args.
  MS_EXCEPTION_IF_NULL(run_args);
  for (size_t i = sens_index; i < sens_index + sens_num; ++i) {
    const auto &original_sens = (*run_args)[i];
    if (utils::isa<ValuePtr>(original_sens)) {
      auto sens_value = utils::cast<ValuePtr>(original_sens);
      MS_EXCEPTION_IF_NULL(sens_value);
      auto new_sens_value = ShallowCopyTensorValue(sens_value);
      MS_EXCEPTION_IF_NULL(new_sens_value);
      MS_LOG(DEBUG) << "sens args [" << sens_value->ToString() << "] has been shallow copied to ["
                    << new_sens_value->ToString() << "].";
      (*run_args)[i] = new_sens_value;
    }
  }
}

void GradExecutor::CheckParamShapeAndType(const AnfNodePtr &param, const ParameterPtr &param_node,
                                          const abstract::AbstractBasePtr &input_abs,
                                          const abstract::AbstractBasePtr &param_tensor_abs,
                                          const std::string &input_shape) {
  MS_EXCEPTION_IF_NULL(param);
  MS_EXCEPTION_IF_NULL(param_node);
  MS_EXCEPTION_IF_NULL(param_tensor_abs);
  auto ir_base_shape = param_tensor_abs->BuildShape();
  MS_EXCEPTION_IF_NULL(ir_base_shape);
  auto ir_shape = ir_base_shape->ToString();
  if (input_shape != "()" && ir_shape != "()") {
    if (input_shape != ir_shape) {
      // Sens shape in ir graph is determined by graph output, so it can be dynamic shape; But input shape is
      // determined by user input, which could not be dynamic shape.
      if (param_node->debug_info()->name() != "sens" || !ir_base_shape->IsDynamic()) {
        MS_EXCEPTION(ValueError) << "The shape should be " << ir_shape << ", but got " << input_shape << ", "
                                 << param->DebugString();
      }
    }
    auto ir_dtype = param_tensor_abs->BuildType()->ToString();
    MS_EXCEPTION_IF_NULL(input_abs);
    auto input_dtype = input_abs->BuildType()->ToString();
    if (input_dtype != ir_dtype) {
      MS_EXCEPTION(TypeError) << "The dtype should be " << ir_dtype << ", but got " << input_dtype << ", "
                              << param->DebugString();
    }
  }
  if (param_node->debug_info()->name() == "sens" && ir_shape != input_shape) {
    need_renormalize_ = true;
  }
}

void GradExecutor::UpdateParamAbsByArgs(const py::list &args, const FuncGraphPtr &bprop_graph) {
  MS_EXCEPTION_IF_NULL(bprop_graph);
  const auto &bprop_params = bprop_graph->parameters();
  // bprop_params include inputs, parameters, more than size(inputs)
  if (bprop_params.size() < args.size()) {
    MS_LOG(EXCEPTION) << "Df parameters size " << bprop_params.size() << " less than " << args.size();
  }
  size_t index = 0;
  for (const auto &param : bprop_params) {
    auto param_node = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_node);
    if (param_node->has_default()) {
      // update abstract info for weights
      ValuePtr value = param_node->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto ptr = value->ToAbstract();
      MS_EXCEPTION_IF_NULL(ptr);
      param_node->set_abstract(ptr->Broaden());
    } else {
      // update abstract info for input params
      abstract::AbstractBasePtr input_abs;
      auto it = forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs.find(GetId(args[index]));
      if (it != forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs.end()) {
        input_abs = it->second;
      } else {
        input_abs = abstract::FromValue(PyObjToValue(args[index]), true);
      }
      MS_EXCEPTION_IF_NULL(input_abs);
      if (param_node->abstract() != nullptr) {
        auto input_shape = input_abs->BuildShape()->ToString();
        auto param_tensor_abs = param_node->abstract();
        if (param_tensor_abs->isa<abstract::AbstractRefTensor>()) {
          param_tensor_abs = param_tensor_abs->cast<abstract::AbstractRefPtr>()->CloneAsTensor();
        }
        CheckParamShapeAndType(param, param_node, input_abs, param_tensor_abs, input_shape);
      }
      param_node->set_abstract(input_abs->Broaden());
      index++;
    }
  }
}

FuncGraphPtr GradExecutor::GetBpropGraph(const prim::GradOperationPtr &grad, const py::object &cell,
                                         const std::vector<AnfNodePtr> &weights,
                                         const std::vector<size_t> &grad_position, size_t arg_size,
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
  ad::GradAttr grad_attr(grad->get_all_, grad->get_by_list_, grad->sens_param_, grad->get_by_position_);
  FuncGraphPtr bprop_graph =
    ad::GradPynativeCellEnd(k_pynative_cell_ptr, weights, grad_position, grad_attr, build_formal_param);
  MS_EXCEPTION_IF_NULL(bprop_graph);

  MS_LOG(DEBUG) << "Top graph input params size " << arg_size;
  std::ostringstream ss;
  ss << "grad{" << arg_size << "}";
  bprop_graph->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  bprop_graph->debug_info()->set_name(ss.str());
  // Get the parameters items and add the value to args_spec
  if (top_cell()->dynamic_shape() && grad->sens_param()) {
    MS_EXCEPTION_IF_NULL(top_cell()->last_output_abs());
    auto shape = top_cell()->last_output_abs()->BuildShape();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->IsDynamic()) {
      const auto &sens_id = GetId(args[arg_size - 1]);
      forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs[sens_id] = top_cell()->last_output_abs();
    }
  }
  UpdateParamAbsByArgs(FilterTensorArgs(args, grad->sens_param_), bprop_graph);
  // Dynamic shape graph need add some other pass
  if (top_cell()->dynamic_shape()) {
    bprop_graph->set_flag(FUNC_GRAPH_FLAG_DYNAMIC_SHAPE, true);
  }
  if (top_cell()->is_dynamic_structure()) {
    bprop_graph->set_flag(kFlagIsDynamicStructure, true);
  }
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
      top_cell_->EraseFromSubCellList(*it);
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
  grad_operation_ = std::to_string(static_cast<int>(grad->get_all_)) +
                    std::to_string(static_cast<int>(grad->get_by_list_)) + grad->grad_position_ + grad->weights_id_;

  std::string input_args_id;
  for (size_t i = 0; i < args.size(); ++i) {
    input_args_id += GetId(args[i]) + "_";
  }
  // Under the condition that the stack is empty (forward process completed or no forward process),
  // check whether need to run forward process
  if (cell_stack_.empty() && top_cell_ != nullptr) {
    const auto &check_already_run_cell_id = GetAlreadyRunCellId(cell_id);
    auto find_top_cell = GetTopCell(check_already_run_cell_id);
    if (find_top_cell != nullptr) {
      MS_LOG(DEBUG) << "Find already run top cell";
      forward_run = find_top_cell->forward_already_run();
      const auto &curr_top_cell = top_cell();
      set_top_cell(find_top_cell);
      bool input_args_changed =
        !find_top_cell->input_args_id().empty() && find_top_cell->input_args_id() != input_args_id;
      if (forward_run && input_args_changed && find_top_cell->dynamic_graph_structure()) {
        MS_LOG(WARNING) << "The construct of running cell is dynamic and the input info of this cell has changed, "
                           "forward process will run again";
        forward_run = false;
      }
      if (forward_run && GetHighOrderStackSize() >= 1) {
        PushHighOrderGraphStack(curr_top_cell);
      }
    }
  }
  MS_LOG(DEBUG) << "Graph have already ran " << forward_run << " top cell id " << cell_id;
  return BaseRefToPyData(forward_run);
}

void GradExecutor::CheckNeedCompileGraph() {
  const auto &new_top_cell = top_cell();
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
    top_cell()->set_is_dynamic_structure(true);
  } else {
    MS_LOG(DEBUG) << "The op info has not been changed, no need to compile graph again";
    pre_top_cell->set_input_args_id(new_top_cell->input_args_id());
    // In high order situations, the internal top cell remains unchanged, but the external top cell has changed. Then
    // the graph info of the internal top cell needs to be updated so that the external top cell can perceive it.
    if (!cell_stack_.empty()) {
      pre_top_cell->SetGraphInfoMap(pre_top_cell->df_builder(),
                                    new_top_cell->graph_info_map().at(new_top_cell->df_builder()));
    }
    EraseTopCellFromTopCellList(new_top_cell);
    new_top_cell->Clear();
    pre_top_cell->set_forward_already_run(true);
    set_top_cell(pre_top_cell);
  }
}

void GradExecutor::RunGradGraph(py::object *ret, const py::object &cell, const py::object &sens_param,
                                const py::tuple &args) {
  MS_EXCEPTION_IF_NULL(ret);
  bool has_sens = sens_param.cast<bool>();
  const auto &cell_id = GetGradCellId(has_sens, cell, args);
  MS_LOG(DEBUG) << "Run has sens " << has_sens << " cell id " << cell_id;
  auto resource = top_cell()->resource();
  MS_EXCEPTION_IF_NULL(resource);
  MS_LOG(DEBUG) << "Run resource ptr " << resource.get();

  VectorRef arg_list;
  auto filter_args = FilterTensorArgs(args, has_sens);
  py::tuple converted_args = ConvertArgs(filter_args);
  pipeline::ProcessVmArgInner(converted_args, resource, &arg_list);
  ShallowCopySensValue(filter_args, has_sens, &arg_list);
  MS_LOG(DEBUG) << "Convert args size " << converted_args.size() << ", graph param size " << arg_list.size();
  compile::VmEvalFuncPtr run = resource->GetResult(pipeline::kOutput).cast<compile::VmEvalFuncPtr>();
  MS_EXCEPTION_IF_NULL(run);

  const auto &backend = MsContext::GetInstance()->backend_policy();
  MS_LOG(DEBUG) << "Eval run " << backend;
  grad_is_running_ = true;
  top_cell()->set_k_pynative_cell_ptr(nullptr);
  BaseRef value = (*run)(arg_list);
  grad_is_running_ = false;
  FuncGraphPtr fg = resource->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto output_abs = fg->output()->abstract();
  MS_LOG(DEBUG) << "Eval run end " << value.ToString();
  *ret = BaseRefToPyData(value, output_abs);
  // Clear device memory resource of top cell when it has been ran.
  auto has_higher_order = std::any_of(top_cell_list_.begin(), top_cell_list_.end(),
                                      [](const TopCellInfoPtr &value) { return !value->is_topest(); });
  if (top_cell()->is_topest() && !has_higher_order) {
    top_cell()->ClearDeviceMemory();
    if (IsFunctionType(cell)) {
      ClearCellRes(cell_id);
    }
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
  bool inner_top_cell_is_dynamic = top_cell()->dynamic_graph_structure();

  // Get outer top cell
  auto outer_top_cell = PopHighOrderGraphStack();
  MS_EXCEPTION_IF_NULL(outer_top_cell);
  const auto &outer_top_cell_all_op_info = outer_top_cell->all_op_info();
  outer_top_cell->set_all_op_info(outer_top_cell_all_op_info + inner_top_cell_all_op_info);
  // If inner is dynamic, outer set dynamic too
  if (inner_top_cell_is_dynamic) {
    outer_top_cell->set_dynamic_graph_structure(inner_top_cell_is_dynamic);
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

  mindspore::HashSet<std::string> params_weights_set;
  mindspore::HashSet<std::string> params_inputs_set;
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
    if (params_inputs_set.count(id) != 0) {
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
    if (params_weights_set.count(fir.first) == 0) {
      MS_LOG(DEBUG) << "Can't find " << fir.first << " in outer graph, add it";
      second_df_builder->add_parameter(fir.second);
      SetParamNodeMapInGraphInfoMap(second_df_builder, fir.first, fir.second);
      inputs->emplace_back(fir.second);
      weights_args->emplace_back(fir.second->default_param());
    } else {
      // Need replace
      MS_LOG(DEBUG) << "Param name " << fir.first << " ptr " << fir.second.get();
      auto it = std::find_if(second_graph_info->params.begin(), second_graph_info->params.end(),
                             [&fir](const std::pair<std::string, ParameterPtr> &sec) {
                               return sec.second->has_default() && fir.second->name() == sec.second->name();
                             });
      if (it != second_graph_info->params.end()) {
        manager->Replace(fir.second, it->second);
        inputs->emplace_back(it->second);
        weights_args->emplace_back(it->second->default_param());
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
    first_grad_fg = curr_g();
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
  first_grad_fg->transforms().erase(kGrad);
  FuncGraphPtr second_grad_fg = ad::Grad(first_grad_fg, opt::Optimizer::MakeEmptyOptimizer(r));
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
  (void)input_args.insert(input_args.end(), weights_args.cbegin(), weights_args.cend());
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
  auto it = forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs.find(GetId(actual_out));
  if (it != forward()->dynamic_shape_info_ptr()->obj_id_with_dynamic_output_abs.end()) {
    op_exec_info->abstract = it->second;
  } else {
    op_exec_info->abstract = actual_out_v->ToAbstract();
  }
  RecordGradOpInfo(op_exec_info);
  MS_LOG(DEBUG) << "ms_function cnode op info: " << op_exec_info->op_info;

  // Step 1: Update actual output tensors used in grad graph.
  MS_LOG(DEBUG) << "ms_function actual output value: " << actual_out_v->ToString();
  UpdateForwardTensorInfoInBpropGraph(op_exec_info->op_info, actual_out_v);

  // Step 2: Update output tensors of added forward nodes, which are added to return node of ms_function func graph.
  if (top_cell()->op_info_with_ms_func_forward_tensors().count(op_exec_info->op_info) != 0) {
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
  const auto &ms_function_cnode =
    MakeAdjointForMsFunction(new_ms_func_graph, new_grad_graph, actual_out, args, actual_out_v);
  ms_function_cnode->set_abstract(new_ms_func_graph->output()->abstract()->Broaden());
}

py::object GradExecutor::GradMsFunction(const py::object &out, const py::args &args) {
  if (graph_phase().empty()) {
    MS_LOG(EXCEPTION) << "The graph phase is empty, can not obtain ms_function func graph.";
  }
  // Get forward graph
  const auto &phase = graph_phase();
  MS_LOG(DEBUG) << "ms_function func graph phase: " << phase;
  auto executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(executor);
  FuncGraphPtr ms_func_graph = executor->GetFuncGraph(phase);
  MS_EXCEPTION_IF_NULL(ms_func_graph);
  // Get actual forward output object.
  py::object ret = out;
  if (ms_func_graph->modify_output()) {
    auto tuple_out = py::cast<py::tuple>(out);
    ret = tuple_out[0];
  }
  // Save dynamic shape info if output tensors of forward graph have dynamic shapes
  SaveDynShapeAbsForMsFunction(args, out, ms_func_graph);
  // Make Adjoint for grad graph of ms_function.
  if (!grad_flag_) {
    MS_LOG(DEBUG) << "Only run forward infer computation, no need to construct grad graph.";
    set_graph_phase("");
    return ret;
  }
  FuncGraphPtr grad_graph = executor->GetGradGraph(phase);
  MS_EXCEPTION_IF_NULL(grad_graph);
  GradMsFunctionInner(phase, out, args, ms_func_graph, grad_graph);
  auto parallel_mode = parallel::ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode == parallel::kSemiAutoParallel || parallel_mode == parallel::kAutoParallel) {
    for (auto &parameter : ms_func_graph->parameters()) {
      auto param = parameter->cast<ParameterPtr>();
      if (param->has_default()) {
        ms_function_params_.push_back(param->name());
      }
    }
  }
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
  forward()->ClearNodeAbsMap();
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
  bprop_cell_list_.clear();
  already_run_top_cell_.clear();
  ClearCellRes();
  std::stack<std::pair<std::string, bool>>().swap(bprop_grad_stack_);
  std::stack<std::string>().swap(cell_stack_);
  std::stack<TopCellInfoPtr>().swap(high_order_stack_);
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

void PynativeExecutor::set_grad_flag(bool flag) const { grad_executor()->set_grad_flag(flag); }

void PynativeExecutor::SetHookChanged(const py::object &cell) const {
  if (!py::isinstance<Cell>(cell)) {
    MS_LOG(EXCEPTION) << "The 'set_hook_changed' function is only supported on Cell object!";
  }
  grad_executor()->SetHookChanged(cell);
}

void PynativeExecutor::set_graph_phase(const std::string &graph_phase) const {
  grad_executor()->set_graph_phase(graph_phase);
}

void PynativeExecutor::SetDynamicInput(const py::object &cell, const py::args &args) const {
  MS_LOG(DEBUG) << "Set dynamic input for feed mode from cell id " << GetId(cell);
  forward_executor()->SetDynamicInput(cell, args);
  // After set input, check previous top cell can be make to dynamic shape
  grad_executor()->CheckPreviousTopCellCanBeDynamicShape(cell, args);
}

py::object PynativeExecutor::GetDynamicInput(const py::object &actual_input) const {
  return forward_executor()->GetDynamicInput(actual_input);
}

void PynativeExecutor::set_py_exe_path(const py::object &py_exe_path) const {
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

void PynativeExecutor::set_grad_position(const prim::GradOperationPtr &grad, const py::object &grad_position) const {
  grad->set_grad_position(std::string(py::str(grad_position)));
}

void PynativeExecutor::set_weights_id(const prim::GradOperationPtr &grad, const py::object &weights_id) {
  if (!py::isinstance<py::str>(weights_id)) {
    MS_LOG(EXCEPTION) << "Failed, weights_id is not a str";
  }
  std::string weights_id_s = py::cast<std::string>(weights_id);
  grad->set_weights_id(weights_id_s);
}

py::object PynativeExecutor::CheckAlreadyRun(const prim::GradOperationPtr &grad, const py::object &cell,
                                             const py::args &args) const {
  return grad_executor()->CheckAlreadyRun(grad, cell, args);
}

py::object PynativeExecutor::Run(const py::object &cell, const py::object &sens_param, const py::tuple &args) const {
  py::object ret;
  PynativeExecutorTry(grad_executor()->RunGraph, &ret, cell, sens_param, args);
  return ret;
}

void PynativeExecutor::ClearCell(const py::object &cell) const {
  const auto &cell_id = GetId(cell);
  MS_LOG(DEBUG) << "Clear cell res, cell id " << cell_id;
  grad_executor()->ClearCellRes(cell_id);
}

void PynativeExecutor::ClearGrad(const py::object &cell, const py::args &args) const {
  MS_LOG(DEBUG) << "Clear grad";
  return grad_executor()->ClearGrad(cell, args);
}

void PynativeExecutor::ClearRes() {
  MS_LOG(DEBUG) << "Clear all res";
  runtime::OpExecutor::GetInstance().Reset();
  for (auto &item : kMindRtBackends) {
    MS_EXCEPTION_IF_NULL(item.second);
    item.second->ClearOpExecutorResource();
  }
  SetLazyBuild(false);

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
  kSessionBackends.clear();
  kMindRtBackends.clear();
  g_pyobj_id_cache.clear();
}

void PynativeExecutor::NewGraph(const py::object &cell, const py::args &args) const {
  if (py::isinstance<Cell>(cell)) {
    forward_executor()->IncreaseCellDepth();
  }

  // Do some initing work before new graph
  forward_executor()->SetFeedDynamicInputAbs(cell, args);

  if (!grad_flag()) {
    MS_LOG(DEBUG) << "Grad flag is false";
    return;
  }
  const py::object ret;
  PynativeExecutorTry(grad_executor()->InitGraph, &ret, cell, args);
}

void PynativeExecutor::EndGraph(const py::object &cell, const py::object &out, const py::args &args) {
  if (py::isinstance<Cell>(cell)) {
    forward_executor()->DecreaseCellDepth();
  }

  // Do some finishing work before end graph
  if (forward_executor()->IsFirstCell()) {
    // Reset lazy build
    SetLazyBuild(false);
    // Finish lazy task
    ExecuteLazyTask();
  }

  if (!grad_flag()) {
    forward_executor()->ResetDynamicAbsMap();
    MS_LOG(DEBUG) << "Grad flag is false";
    return;
  }
  const py::object ret;
  PynativeExecutorTry(grad_executor()->LinkGraph, &ret, cell, out, args);
  forward_executor()->ResetDynamicAbsMap();
}

py::object PynativeExecutor::GradMsFunction(const py::object &out, const py::args &args) const {
  return grad_executor()->GradMsFunction(out, args);
}

void PynativeExecutor::GradNet(const prim::GradOperationPtr &grad, const py::object &cell, const py::object &weights,
                               const py::object &grad_position, const py::args &args) {
  const py::object ret;
  PynativeExecutorTry(grad_executor()->GradGraph, &ret, grad, cell, weights, grad_position, args);
}

void PynativeExecutor::Sync() const {
  ExecuteLazyTask();

  mindspore::ScopedLongRunning long_running;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
    for (auto &item : kSessionBackends) {
      MS_EXCEPTION_IF_NULL(item.second);
      item.second->SyncStream();
    }
  } else {
    for (auto &item : kMindRtBackends) {
      MS_EXCEPTION_IF_NULL(item.second);
      item.second->SyncStream();
    }
    for (auto &item : kSessionBackends) {
      MS_EXCEPTION_IF_NULL(item.second);
      item.second->SyncStream();
    }
  }
}

void PynativeExecutor::SetLazyBuild(bool enable) const { forward_executor()->set_lazy_build(enable); }

bool PynativeExecutor::IsFirstCell() const { return forward_executor()->IsFirstCell(); }

void PynativeExecutor::ExecuteLazyTask() const {
  mindspore::ScopedLongRunning long_running;
  for (auto &item : kMindRtBackends) {
    MS_EXCEPTION_IF_NULL(item.second);
    item.second->WaitTaskFinish();
  }
}

REGISTER_PYBIND_DEFINE(PynativeExecutor_, ([](const py::module *m) {
                         (void)py::class_<PynativeExecutor, std::shared_ptr<PynativeExecutor>>(*m, "PynativeExecutor_")
                           .def_static("get_instance", &PynativeExecutor::GetInstance, "PynativeExecutor get_instance.")
                           .def("is_first_cell", &PynativeExecutor::IsFirstCell, "check if the first cell.")
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
                           .def("__call__", &PynativeExecutor::Run, "pynative executor run grad graph.")
                           .def("set_graph_phase", &PynativeExecutor::set_graph_phase, "pynative set graph phase")
                           .def("grad_flag", &PynativeExecutor::grad_flag, "pynative grad flag")
                           .def("set_hook_changed", &PynativeExecutor::SetHookChanged, "set pynative hook changed")
                           .def("set_grad_position", &PynativeExecutor::set_grad_position, "set pynative grad position")
                           .def("set_weights_id", &PynativeExecutor::set_weights_id, "set pynative grad weights id")
                           .def("set_grad_flag", &PynativeExecutor::set_grad_flag, py::arg("flag") = py::bool_(false),
                                "Executor set grad flag.")
                           .def("set_dynamic_input", &PynativeExecutor::SetDynamicInput, "set dynamic input")
                           .def("get_dynamic_input", &PynativeExecutor::GetDynamicInput, "get dynamic input")
                           .def("set_py_exe_path", &PynativeExecutor::set_py_exe_path,
                                py::arg("py_exe_path") = py::str(""), "set python executable path.")
                           .def("set_kernel_build_server_dir", &PynativeExecutor::set_kernel_build_server_dir,
                                py::arg("kernel_build_server_dir") = py::str(""),
                                "set kernel build server directory path.");
                       }));
}  // namespace mindspore::pynative
