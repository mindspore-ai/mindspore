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
#include <map>
#include <set>
#include <memory>
#include <sstream>
#include <unordered_set>
#include <algorithm>

#include "debug/trace.h"
#include "pybind_api/ir/tensor_py.h"
#include "ir/param_info.h"
#include "ir/anf.h"
#include "ir/cell.h"
#include "ir/tensor.h"
#include "utils/any.h"
#include "utils/utils.h"
#include "utils/ms_context.h"
#include "utils/context/context_extends.h"
#include "utils/config_manager.h"
#include "utils/convert_utils_py.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/do_signature.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/parse/resolve.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "backend/session/session_factory.h"
#include "backend/optimizer/pass/const_input_to_attr_registry.h"
#include "backend/optimizer/common/helper.h"
#include "pipeline/jit/action.h"

#include "pipeline/pynative/base.h"
#include "pybind_api/api_register.h"
#include "vm/transform.h"

#include "frontend/optimizer/ad/grad.h"
#include "pipeline/jit/resource.h"
#include "pipeline/jit/pipeline.h"
#include "pipeline/jit/pass.h"

#ifdef ENABLE_GE
#include "pipeline/pynative/pynative_execute_ge.h"
#endif

#include "debug/anf_ir_dump.h"

using mindspore::tensor::TensorPy;

const size_t PTR_LEN = 15;
// primitive unable to infer value for constant input in PyNative mode
const std::set<std::string> vm_operators = {"make_ref", "HookBackward", "InsertGradientOf", "stop_gradient",
                                            "mixed_precision_cast"};

namespace mindspore::pynative {
static std::shared_ptr<session::SessionBasic> session = nullptr;
PynativeExecutorPtr PynativeExecutor::executor_ = nullptr;
std::mutex PynativeExecutor::instance_lock_;
int64_t PynativeExecutor::graph_id_ = 0;

template <typename... Args>
void PynativeExecutorTry(PynativeExecutor *const executor, void (PynativeExecutor::*method)(Args...), Args &&... args) {
  MS_EXCEPTION_IF_NULL(executor);
  try {
    (executor->*method)(args...);
  } catch (const py::error_already_set &ex) {
    // print function call stack info before release
    std::ostringstream oss;
    trace::TraceGraphEval();
    trace::GetEvalStackInfo(oss);
    // call py::print to output function call stack to STDOUT, in case of output the log to file, the user can see
    // these info from screen, no need to open log file to find these info
    py::print(oss.str());
    MS_LOG(ERROR) << oss.str();
    PynativeExecutor::GetInstance()->Clean();
    // re-throw this exception to Python interpreter to handle it
    throw(py::error_already_set(ex));
  } catch (const py::type_error &ex) {
    PynativeExecutor::GetInstance()->Clean();
    throw py::type_error(ex);
  } catch (const py::value_error &ex) {
    PynativeExecutor::GetInstance()->Clean();
    throw py::value_error(ex);
  } catch (const py::index_error &ex) {
    PynativeExecutor::GetInstance()->Clean();
    throw py::index_error(ex);
  } catch (const std::exception &ex) {
    PynativeExecutor::GetInstance()->Clean();
    // re-throw this exception to Python interpreter to handle it
    throw(std::runtime_error(ex.what()));
  } catch (...) {
    PynativeExecutor::GetInstance()->Clean();
    std::string exName(abi::__cxa_current_exception_type()->name());
    MS_LOG(EXCEPTION) << "Error occurred when compile graph. Exception name: " << exName;
  }
}

inline ValuePtr PyAttrValue(const py::object &obj) {
  ValuePtr converted_ret = parse::data_converter::PyDataToValue(obj);
  if (!converted_ret) {
    MS_LOG(EXCEPTION) << "Attribute convert error with type: " << std::string(py::str(obj));
  }
  return converted_ret;
}

static std::string GetId(const py::object &obj) {
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

  py::object ret = parse::python_adapter::CallPyFn(parse::PYTHON_MOD_PARSE_MODULE, parse::PYTHON_MOD_GET_OBJ_ID, obj);
  return py::cast<std::string>(ret);
}

std::map<SignatureEnumDType, std::vector<size_t>> GetTypeIndex(const std::vector<SignatureEnumDType> &dtypes) {
  std::map<SignatureEnumDType, std::vector<size_t>> type_indexes;
  for (size_t i = 0; i < dtypes.size(); ++i) {
    auto it = type_indexes.find(dtypes[i]);
    if (it == type_indexes.end()) {
      (void)type_indexes.emplace(std::make_pair(dtypes[i], std::vector<size_t>{i}));
    } else {
      it->second.emplace_back(i);
    }
  }
  return type_indexes;
}

std::map<SignatureEnumDType, TypeId> GetDstType(const py::tuple &py_args,
                                                const std::map<SignatureEnumDType, std::vector<size_t>> &type_indexes) {
  std::map<SignatureEnumDType, TypeId> dst_type;
  for (auto it = type_indexes.begin(); it != type_indexes.end(); (void)++it) {
    auto type = it->first;
    auto indexes = it->second;
    if (type == SignatureEnumDType::kDTypeEmptyDefaultValue || indexes.size() < 2) {
      continue;
    }
    size_t priority = 0;
    TypeId max_type = TypeId::kTypeUnknown;
    bool has_scalar_float32 = false;
    bool has_scalar_int64 = false;
    bool has_tensor_int8 = false;
    for (size_t index : indexes) {
      if (!has_scalar_float32 && py::isinstance<py::float_>(py_args[index])) {
        has_scalar_float32 = true;
      }
      if (!has_scalar_int64 && !py::isinstance<py::bool_>(py_args[index]) && py::isinstance<py::int_>(py_args[index])) {
        has_scalar_int64 = true;
      }

      auto obj = py_args[index];
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
    (void)dst_type.emplace(std::make_pair(type, max_type));
  }
  return dst_type;
}

std::string TypeIdToMsTypeStr(const TypeId &type_id) {
  auto type_name = type_name_map.find(type_id);
  if (type_name == type_name_map.end()) {
    MS_LOG(EXCEPTION) << "For implicit type conversion, not support convert to the type: " << TypeIdToType(type_id);
  }
  return type_name->second;
}

bool GetSignatureType(const PrimitivePyPtr &prim, std::vector<SignatureEnumDType> *dtypes) {
  MS_EXCEPTION_IF_NULL(dtypes);
  auto signature = prim->signatures();
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

void PynativeInfer(const PrimitivePyPtr &prim, const py::list &py_args, OpExecInfo *const op_exec_info,
                   const abstract::AbstractBasePtrList &args_spec_list) {
  MS_LOG(DEBUG) << "Prim " << prim->name() << " input infer " << mindspore::ToString(args_spec_list);
  prim->BeginRecordAddAttr();
  AbstractBasePtr infer_res = EvalOnePrim(prim, args_spec_list)->abstract();
  prim->EndRecordAddAttr();
  op_exec_info->abstract = infer_res;
  MS_LOG(DEBUG) << "Prim " << prim->name() << " infer result " << op_exec_info->abstract->ToString();
}

std::string GetSingleOpGraphInfo(const OpExecInfoPtr &op_exec_info,
                                 const std::vector<tensor::TensorPtr> &input_tensors) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  std::string graph_info;
  // get input tensor info
  for (const auto &tensor : input_tensors) {
    MS_EXCEPTION_IF_NULL(tensor);
    auto tensor_shape = tensor->shape();
    (void)std::for_each(tensor_shape.begin(), tensor_shape.end(),
                        [&](const auto &dim) { (void)graph_info.append(std::to_string(dim) + "_"); });
    (void)graph_info.append(std::to_string(tensor->data_type()) + "_");
    if (tensor->device_address() != nullptr) {
      (void)graph_info.append(
        std::to_string(std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address())->type_id()) + "_");
      (void)graph_info.append(std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address())->format() +
                              "_");
    }
  }
  // get prim and abstract info
  (void)graph_info.append(op_exec_info->prim_id + "_");
  // get attr info
  const auto &op_prim = op_exec_info->py_primitive;
  MS_EXCEPTION_IF_NULL(op_prim);
  const auto &attr_map = op_prim->evaluate_added_attrs();
  (void)std::for_each(attr_map.begin(), attr_map.end(),
                      [&](const auto &element) { (void)graph_info.append(element.second->ToString() + "_"); });
  return graph_info;
}

py::object RunOpInVM(const OpExecInfoPtr &op_exec_info, PynativeStatusCode *status) {
  MS_LOG(INFO) << "RunOpInVM start";

  MS_EXCEPTION_IF_NULL(status);
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(op_exec_info->py_primitive);

  auto &op_inputs = op_exec_info->op_inputs;
  if (op_exec_info->op_name == "HookBackward" || op_exec_info->op_name == "InsertGradientOf") {
    py::tuple result(op_inputs.size());
    for (size_t i = 0; i < op_inputs.size(); i++) {
      py::object input = op_inputs[i];
      auto tensor = py::cast<tensor::TensorPtr>(input);
      auto new_tensor = std::make_shared<tensor::Tensor>(tensor->data_type(), tensor->shape(), tensor->data_ptr());
      new_tensor->set_device_address(tensor->device_address());
      new_tensor->set_sync_status(tensor->sync_status());
      result[i] = new_tensor;
    }
    *status = PYNATIVE_SUCCESS;
    MS_LOG(INFO) << "RunOpInVM end";
    return std::move(result);
  }
  auto primitive = op_exec_info->py_primitive;
  MS_EXCEPTION_IF_NULL(primitive);
  auto result = primitive->RunPyComputeFunction(op_inputs);
  if (py::isinstance<py::none>(result)) {
    MS_LOG(ERROR) << "VM got the result none, please check whether it is failed to get func";
    *status = PYNATIVE_OP_NOT_IMPLEMENTED_ERR;
    py::tuple err_ret(0);
    return std::move(err_ret);
  }

  // execute op
  py::tuple tuple_result = py::make_tuple(result);
  *status = PYNATIVE_SUCCESS;
  MS_LOG(INFO) << "RunOpInVM end";
  return std::move(tuple_result);
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
  ValuePtr input_value = parse::data_converter::PyDataToValue(input_object);
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
                                  std::vector<tensor::TensorPtr> *input_tensors, int64_t *tensor_mask) {
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
  auto inputs = py::cast<py::tuple>(input_object);
  if (py::isinstance<tensor::Tensor>(inputs[0])) {
    PlantTensorTupleToVector(inputs, op_prim, input_tensors);
  } else {
    ConvertValueTupleToTensor(input_object, input_tensors);
    *tensor_mask = kValueNodeTensorMask;
  }
}

void ConvertPyObjectToTensor(const py::object &input_object, const PrimitivePtr &op_prim,
                             std::vector<tensor::TensorPtr> *input_tensors, int64_t *tensor_mask) {
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
    tensor_ptr = std::make_shared<tensor::Tensor>(py::cast<py::int_>(input_object), kInt64);
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
  MS_EXCEPTION_IF_NULL(op_prim);

  opt::ConstInputToAttrInfoRegister reg;
  bool reg_exist = opt::ConstInputToAttrInfoRegistry::Instance().GetRegisterByOpName(op_run_info->op_name, &reg);
  if (op_run_info->op_name == prim::kPrimEmbeddingLookup->name()) {
    reg_exist = false;
  }
  if (op_run_info->op_name == prim::kPrimGatherD->name()) {
    auto ms_context = MsContext::GetInstance();
    // Gather op needs converting const input to attr on GPU device
    if (ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kGPUDevice) {
      reg_exist = false;
    }
  }

  op_prim->BeginRecordAddAttr();
  size_t input_num = op_run_info->op_inputs.size();
  for (size_t index = 0; index < input_num; ++index) {
    // convert const input to attr
    if (reg_exist &&
        RunOpConvertConstInputToAttr(op_run_info->op_inputs[index], index, op_prim, reg.GetConstInputAttrInfo())) {
      continue;
    }
    // convert const and tuple input to tensor
    int64_t tensor_mask = static_cast<int64_t>(op_run_info->inputs_mask[index]);
    ConvertPyObjectToTensor(op_run_info->op_inputs[index], op_prim, input_tensors, &tensor_mask);
    // mark tensors, data : 0, weight : 1, valuenode: 2
    std::vector<int64_t> new_mask(input_tensors->size() - tensors_mask->size(), tensor_mask);
    tensors_mask->insert(tensors_mask->end(), new_mask.begin(), new_mask.end());
  }
  op_prim->EndRecordAddAttr();
}

BaseRef TransformBaseRefListToTuple(const BaseRef &base_ref) {
  if (utils::isa<VectorRef>(base_ref)) {
    auto ref_list = utils::cast<VectorRef>(base_ref);
    py::tuple output_tensors(ref_list.size());
    for (size_t i = 0; i < ref_list.size(); ++i) {
      auto output = TransformBaseRefListToTuple(ref_list[i]);
      if (utils::isa<tensor::TensorPtr>(output)) {
        auto tensor_ptr = utils::cast<tensor::TensorPtr>(output);
        MS_EXCEPTION_IF_NULL(tensor_ptr);
        output_tensors[i] = tensor_ptr;
      } else if (utils::isa<PyObjectRef>(output)) {
        py::object obj = utils::cast<PyObjectRef>(output).object_;
        py::tuple tensor_tuple = py::cast<py::tuple>(obj);
        output_tensors[i] = tensor_tuple;
      } else {
        MS_LOG(EXCEPTION) << "The output is not a base ref list or a tensor!";
      }
    }
    return std::make_shared<PyObjectRef>(output_tensors);
  } else if (utils::isa<tensor::TensorPtr>(base_ref)) {
    return base_ref;
  } else {
    MS_LOG(EXCEPTION) << "The output is not a base ref list or a tensor!";
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

void ConvertTupleArg(py::tuple *res, size_t *index, const py::tuple &arg) {
  for (size_t i = 0; i < arg.size(); i++) {
    if (py::isinstance<py::tuple>(arg[i])) {
      ConvertTupleArg(res, index, arg[i]);
    } else {
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
      res[index++] = args[i];
    }
  }
  return res;
}

void ClearPyNativeSession() { session = nullptr; }

PynativeExecutor::~PynativeExecutor() { ClearRes(); }

py::tuple RunOp(const py::args &args) {
  auto executor = PynativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor);
  MS_LOG(DEBUG) << "RunOp start " << args.size();
  OpExecInfoPtr op_exec_info = executor->GenerateOpExecInfo(args);
  try {
    return executor->RunOpInner(op_exec_info);
  } catch (const py::error_already_set &ex) {
    executor->Clean();
    // re-throw this exception to Python interpreter to handle it
    throw(py::error_already_set(ex));
  } catch (const py::type_error &ex) {
    executor->Clean();
    throw py::type_error(ex);
  } catch (const py::value_error &ex) {
    executor->Clean();
    throw py::value_error(ex);
  } catch (const py::index_error &ex) {
    executor->Clean();
    throw py::index_error(ex);
  } catch (const std::exception &ex) {
    executor->Clean();
    // re-throw this exception to Python interpreter to handle it
    throw(std::runtime_error(ex.what()));
  } catch (...) {
    executor->Clean();
    std::string exName(abi::__cxa_current_exception_type()->name());
    MS_LOG(EXCEPTION) << "Error occurred when compile graph. Exception name: " << exName;
  }
}

py::tuple PynativeExecutor::RunOpInner(const OpExecInfoPtr &op_exec_info) {
  if (op_exec_info->op_name == prim::kPrimMixedPrecisionCast->name()) {
    return RunOpWithInitBackendPolicy(op_exec_info);
  }
  // make cnode for building grad graph if grad flag is set.
  abstract::AbstractBasePtrList args_spec_list;
  std::vector<bool> op_masks;
  auto cnode = MakeCNode(op_exec_info, &op_masks, &args_spec_list);
  op_exec_info->inputs_mask = op_masks;
  // get output abstract info
  bool is_find = false;
  auto prim = op_exec_info->py_primitive;
  if (prim_abs_list_.find(prim->id()) != prim_abs_list_.end()) {
    auto abs_list = prim_abs_list_[prim->id()];
    MS_LOG(DEBUG) << "Match prim input args " << op_exec_info->op_name << mindspore::ToString(args_spec_list);
    if (abs_list.find(args_spec_list) != abs_list.end()) {
      MS_LOG(DEBUG) << "Match prim ok " << op_exec_info->op_name;
      op_exec_info->abstract = abs_list[args_spec_list].abs;
      op_exec_info->is_dynamic_shape = abs_list[args_spec_list].is_dynamic_shape;
      prim->set_evaluate_added_attrs(abs_list[args_spec_list].attrs);
      is_find = true;
    }
  }
  if (op_exec_info->abstract == nullptr || force_infer_prim.find(op_exec_info->op_name) != force_infer_prim.end()) {
    // use python infer method
    if (ignore_infer_prim.find(op_exec_info->op_name) == ignore_infer_prim.end()) {
      PynativeInfer(prim, op_exec_info->op_inputs, op_exec_info.get(), args_spec_list);
    }
    // get output dynamic shape info
    auto abstract = op_exec_info->abstract;
    MS_EXCEPTION_IF_NULL(abstract);
    auto shape = abstract->BuildShape();
    MS_EXCEPTION_IF_NULL(shape);
    auto shape_info = shape->ToString();
    if (shape_info.find("-1") != string::npos) {
      op_exec_info->is_dynamic_shape = true;
    }
  }
  if (cnode != nullptr) {
    cnode->set_abstract(op_exec_info->abstract);
  }
  // infer output value for const prim
  MS_EXCEPTION_IF_NULL(op_exec_info);
  if (op_exec_info->abstract != nullptr) {
    MS_LOG(DEBUG) << "Run op infer " << op_exec_info->op_name << " " << op_exec_info->abstract->ToString();
    py::dict output = abstract::ConvertAbstractToPython(op_exec_info->abstract);
    if (!output["value"].is_none()) {
      py::tuple value_ret(1);
      value_ret[0] = output["value"];
      return value_ret;
    }
    if (op_exec_info->py_primitive->is_const_prim()) {
      py::tuple value_ret(1);
      value_ret[0] = "";
      return value_ret;
    }
  }
  // add output abstract info into cache
  if (!is_find) {
    // const_value need infer every step
    auto &out = prim_abs_list_[prim->id()];
    out[args_spec_list].abs = op_exec_info->abstract;
    out[args_spec_list].is_dynamic_shape = op_exec_info->is_dynamic_shape;
    out[args_spec_list].attrs = prim->evaluate_added_attrs();
    MS_LOG(DEBUG) << "Set prim " << op_exec_info->op_name << mindspore::ToString(args_spec_list);
  }
  // run op with selected backend
  auto result = RunOpWithInitBackendPolicy(op_exec_info);
  py::object out_real = result;
  if (result.size() == 1) {
    MS_LOG(DEBUG) << "Output size is 1";
    out_real = result[0];
  }
  std::string obj_id = GetId(out_real);
  node_abs_map_[obj_id] = op_exec_info->abstract;
  SaveOutputNodeMap(obj_id, out_real, cnode);
  SaveAllResult(op_exec_info, cnode, out_real);
  // Update the abstract and device address of value node with tensor in grad graph
  UpdateAbstractAndDeviceAddress(op_exec_info, out_real);
  return result;
}

OpExecInfoPtr PynativeExecutor::GenerateOpExecInfo(const py::args &args) {
  if (args.size() != PY_ARGS_NUM) {
    MS_LOG(ERROR) << "Three args are needed by RunOp";
    return nullptr;
  }
  auto op_exec_info = std::make_shared<OpExecInfo>();
  auto op_name = py::cast<std::string>(args[PY_NAME]);
  op_exec_info->op_name = op_name;
  if (grad_flag_) {
    MS_EXCEPTION_IF_NULL(resource_);
    int64_t graph_id = resource_->results()[pipeline::kPynativeGraphId].cast<int64_t>();
    op_exec_info->op_index = std::to_string(graph_id) + op_name + std::to_string(op_index_map_[op_name]);
    op_index_map_[op_name]++;
  }
  auto prim = py::cast<PrimitivePyPtr>(args[PY_PRIM]);
  MS_EXCEPTION_IF_NULL(prim);
  if (!prim->HasPyObj()) {
    MS_LOG(EXCEPTION) << "Pyobj is empty";
  }
  op_exec_info->prim_id = GetId(prim->GetPyObj());
  op_exec_info->py_primitive = prim;
  op_exec_info->op_attrs = py::getattr(args[PY_PRIM], "attrs");
  op_exec_info->op_inputs = args[PY_INPUTS];
  return op_exec_info;
}

AnfNodePtr PynativeExecutor::MakeCNode(const OpExecInfoPtr &op_exec_info, std::vector<bool> *op_masks,
                                       abstract::AbstractBasePtrList *args_spec_list) {
  MS_EXCEPTION_IF_NULL(op_masks);
  MS_EXCEPTION_IF_NULL(args_spec_list);
  MS_EXCEPTION_IF_NULL(op_exec_info);
  CNodePtr cnode = nullptr;

  auto prim = op_exec_info->py_primitive;
  std::vector<AnfNodePtr> inputs;
  inputs.emplace_back(NewValueNode(prim));

  const auto &signature = prim->signatures();
  auto sig_size = signature.size();
  auto size = op_exec_info->op_inputs.size();
  // ignore signature for cast op
  if (sig_size > 0 && sig_size != size) {
    MS_EXCEPTION(ValueError) << op_exec_info->op_name << " inputs size " << size << " does not match the requires "
                             << "inputs size " << sig_size;
  }
  if (op_exec_info->op_name != prim::kPrimCast->name()) {
    RunParameterAutoMixPrecisionCast(op_exec_info);
  }
  MS_LOG(DEBUG) << "Make cnode for " << op_exec_info->op_name;
  for (size_t i = 0; i < op_exec_info->op_inputs.size(); i++) {
    const auto &obj = op_exec_info->op_inputs[i];
    bool op_mask = false;
    if (py::isinstance<tensor::MetaTensor>(obj)) {
      auto meta_tensor = obj.cast<tensor::MetaTensorPtr>();
      if (meta_tensor) {
        op_mask = meta_tensor->is_parameter();
      }
    }
    (*op_masks).emplace_back(op_mask);
    MS_LOG(DEBUG) << "Gen args i " << i << " " << op_exec_info->op_name << " op mask " << op_mask << " grad_flag_ "
                  << grad_flag_;

    AnfNodePtr input_node = nullptr;
    abstract::AbstractBasePtr abs = nullptr;
    auto id = GetId(obj);
    auto it = node_abs_map_.find(id);
    if (it != node_abs_map_.end()) {
      abs = it->second;
    }
    if (!graph_info_map_.empty()) {
      input_node = GetInput(obj, op_mask);
    }
    // update abstract
    if (input_node != nullptr && input_node->abstract() != nullptr) {
      abs = input_node->abstract();
    }

    auto const_input_index = prim->get_const_input_indexes();
    bool have_const_input = !const_input_index.empty();
    bool is_const_prim = prim->is_const_prim();
    MS_LOG(DEBUG) << prim->ToString() << " abs is nullptr " << (abs == nullptr) << " is_const_value "
                  << prim->is_const_prim();
    bool is_const_input =
      have_const_input && std::find(const_input_index.begin(), const_input_index.end(), i) != const_input_index.end();
    if (abs == nullptr || is_const_prim || is_const_input) {
      MS_LOG(DEBUG) << "MakeCnode get node no in map " << id;
      ValuePtr input_value = PyAttrValue(obj);
      abs = input_value->ToAbstract();
      if (!is_const_prim && !is_const_input) {
        auto config = abstract::AbstractBase::kBroadenTensorOnly;
        abs = abs->Broaden(config);
        MS_LOG(DEBUG) << "Broaden for " << prim->ToString() << " " << config;
      }
      node_abs_map_[id] = abs;
    }
    (*args_spec_list).emplace_back(abs);
    if (input_node != nullptr) {
      inputs.emplace_back(input_node);
    }
  }

  MS_LOG(DEBUG) << "MakeCnode args end";
  if (grad_flag_ && curr_g_ != nullptr) {
    cnode = curr_g_->NewCNode(inputs);
    MS_LOG(DEBUG) << "Runop MakeCnode, new node is " << cnode->DebugString(4);
  }
  return cnode;
}

py::object PynativeExecutor::DoAutoCast(const py::object &arg, const TypeId &type_id, const std::string &op_name,
                                        size_t index) {
  py::tuple cast_args(3);
  cast_args[PY_PRIM] = parse::python_adapter::GetPyFn(kOpsFunctionModelName, "cast");
  cast_args[PY_NAME] = prim::kPrimCast->name();
  std::string dst_type_str = TypeIdToMsTypeStr(type_id);
  py::object dst_type = parse::python_adapter::GetPyFn(kMSDtypeModelName, dst_type_str);
  py::tuple inputs(2);
  inputs[0] = arg;
  inputs[1] = dst_type;
  cast_args[PY_INPUTS] = inputs;
  auto op_exec = GenerateOpExecInfo(cast_args);
  op_exec->is_mixed_precision_cast = true;
  op_exec->next_op_name = op_name;
  op_exec->next_input_index = index;
  return RunOpInner(op_exec)[0];
}

py::object PynativeExecutor::DoParamMixPrecisionCast(bool *is_cast, const py::object obj, const std::string &op_name,
                                                     size_t index) {
  MS_EXCEPTION_IF_NULL(is_cast);
  auto tensor = py::cast<tensor::TensorPtr>(obj);
  auto cast_type = tensor->cast_dtype();
  py::object cast_output = obj;
  if (cast_type != nullptr) {
    auto source_element = tensor->Dtype();
    if (source_element != nullptr && IsSubType(source_element, kFloat) && *source_element != *cast_type) {
      MS_LOG(DEBUG) << "Cast to " << cast_type->ToString();
      *is_cast = true;
      return DoAutoCast(obj, cast_type->type_id(), op_name, index);
    }
  }
  return cast_output;
}

py::object PynativeExecutor::DoParamMixPrecisionCastTuple(bool *is_cast, const py::tuple tuple,
                                                          const std::string &op_name, size_t index) {
  MS_EXCEPTION_IF_NULL(is_cast);
  auto tuple_size = static_cast<int64_t>(tuple.size());
  py::tuple result(tuple_size);

  for (int64_t i = 0; i < tuple_size; i++) {
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

void PynativeExecutor::DoSignatrueCast(const PrimitivePyPtr &prim, const std::map<SignatureEnumDType, TypeId> &dst_type,
                                       const std::vector<SignatureEnumDType> &dtypes,
                                       const OpExecInfoPtr &op_exec_info) {
  const auto &signature = prim->signatures();
  auto &out_args = op_exec_info->op_inputs;
  for (size_t i = 0; i < out_args.size(); ++i) {
    // No need to implicit cast if no dtype.
    if (dtypes.empty() || dtypes[i] == SignatureEnumDType::kDTypeEmptyDefaultValue) {
      continue;
    }
    auto it = dst_type.find(dtypes[i]);
    if (it == dst_type.end() || it->second == kTypeUnknown) {
      continue;
    }
    MS_LOG(DEBUG) << "Check inputs " << i;
    auto obj = out_args[i];
    auto sig = SignatureEnumRW::kRWDefault;
    if (!signature.empty()) {
      sig = signature[i].rw;
    }
    bool is_parameter = false;
    TypeId arg_type_id = kTypeUnknown;
    if (py::isinstance<tensor::MetaTensor>(obj)) {
      auto arg = py::cast<tensor::MetaTensorPtr>(obj);
      if (arg->is_parameter()) {
        is_parameter = true;
        MS_LOG(DEBUG) << "Parameter is read " << i;
      }
      arg_type_id = arg->data_type();
    }
    // implicit cast
    bool is_same_type = false;
    if (arg_type_id != kTypeUnknown) {
      is_same_type = (prim::type_map.find(arg_type_id) == prim::type_map.end() || arg_type_id == it->second);
    }
    if (sig == SignatureEnumRW::kRWWrite) {
      if (!is_parameter) {
        prim::RaiseExceptionForCheckParameter(prim->name(), i, "not");
      }
      if (arg_type_id != kTypeUnknown) {
        if (!is_same_type) {
          prim::RaiseExceptionForConvertRefDtype(prim->name(), TypeIdToMsTypeStr(arg_type_id),
                                                 TypeIdToMsTypeStr(it->second));
        }
      }
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
    py::object cast_output = DoAutoCast(out_args[i], it->second, op_exec_info->op_name, i);
    out_args[i] = cast_output;
  }
}

void PynativeExecutor::RunParameterAutoMixPrecisionCast(const OpExecInfoPtr &op_exec_info) {
  size_t size = op_exec_info->op_inputs.size();
  auto prim = op_exec_info->py_primitive;
  MS_EXCEPTION_IF_NULL(prim);
  const auto &signature = prim->signatures();
  for (size_t i = 0; i < size; i++) {
    auto obj = op_exec_info->op_inputs[i];
    auto sig = SignatureEnumRW::kRWDefault;
    if (!signature.empty()) {
      sig = signature[i].rw;
    }
    MS_LOG(DEBUG) << "Check mix precision " << op_exec_info->op_name << " input " << i << " "
                  << std::string(py::repr(obj));
    // mix precision for non param
    bool is_cast = false;
    py::object cast_output;
    if (py::isinstance<tensor::MetaTensor>(obj)) {
      auto meta_tensor = obj.cast<tensor::MetaTensorPtr>();
      if (meta_tensor && meta_tensor->is_parameter()) {
        if (sig != SignatureEnumRW::kRWRead) {
          continue;
        }
      }
      // redundant cast call if the tensor is a const Tensor.
      cast_output = DoParamMixPrecisionCast(&is_cast, obj, prim->name(), i);
    } else if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
      // mix precision for tuple inputs
      cast_output = DoParamMixPrecisionCastTuple(&is_cast, obj, prim->name(), i);
    }
    if (is_cast) {
      op_exec_info->op_inputs[i] = cast_output;
    }
  }
  std::vector<SignatureEnumDType> dtypes;

  bool has_dtype_sig = GetSignatureType(prim, &dtypes);
  std::map<SignatureEnumDType, TypeId> dst_types;
  if (has_dtype_sig) {
    // fetch info for implicit cast
    auto type_indexes = GetTypeIndex(dtypes);
    dst_types = GetDstType(op_exec_info->op_inputs, type_indexes);
  }
  MS_LOG(DEBUG) << "Do signature for " << op_exec_info->op_name;
  DoSignatrueCast(prim, dst_types, dtypes, op_exec_info);
}

AnfNodePtr PynativeExecutor::GetInput(const py::object &obj, bool op_mask) {
  AnfNodePtr node = nullptr;
  std::string obj_id = GetId(obj);

  if (op_mask) {
    MS_LOG(DEBUG) << "Cell paramsters(weights)";
    // get the parameter name from parameter object
    auto name_attr = parse::python_adapter::GetPyObjAttr(obj, "name");
    if (py::isinstance<py::none>(name_attr)) {
      MS_LOG(EXCEPTION) << "Parameter object should have name attribute";
    }
    auto param_name = py::cast<std::string>(name_attr);
    if (graph_info_map_[df_builder_].params.find(obj_id) == graph_info_map_[df_builder_].params.end()) {
      auto free_param = df_builder_->add_parameter();
      free_param->set_name(param_name);
      free_param->debug_info()->set_name(param_name);
      auto value = py::cast<tensor::TensorPtr>(obj);
      free_param->set_default_param(value);
      MS_LOG(DEBUG) << "Top graph set free parameter " << obj_id;
      graph_info_map_[df_builder_].params.emplace(obj_id);
      SetNodeMapInGraphInfoMap(df_builder_, obj_id, free_param);
      return free_param;
    }
    return graph_info_map_[df_builder_].node_map[obj_id].first;
  }

  if (graph_info_map_[curr_g_].node_map.find(obj_id) != graph_info_map_[curr_g_].node_map.end()) {
    // op(x, y)
    // out = op(op1(x, y))
    // out = op(cell1(x, y))
    // out = op(cell1(x, y)[0])
    return GetObjNode(obj, obj_id);
  } else if (py::isinstance<py::tuple>(obj)) {
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
                  : MS_LOG(DEBUG) << "Now getinput node " << node->ToString() << obj_id;
  return node;
}

void PynativeExecutor::UpdateAbstractAndDeviceAddress(const OpExecInfoPtr &op_exec_info, const py::object &out_real) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  if (!grad_flag_) {
    return;
  }
  auto op_index = op_exec_info->op_index;
  auto output_value = PyAttrValue(out_real);
  MS_EXCEPTION_IF_NULL(output_value);
  std::vector<tensor::TensorPtr> output_tensors;
  TensorValueToTensor(output_value, &output_tensors);
  if (op_index_with_tensor_id_.find(op_index) == op_index_with_tensor_id_.end()) {
    // first step
    std::for_each(output_tensors.begin(), output_tensors.end(), [&](const tensor::TensorPtr &tensor) {
      op_index_with_tensor_id_[op_index].emplace_back(tensor->id());
    });
    return;
  }
  auto ms_context = MsContext::GetInstance();
  auto target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  const auto &tensor_id_list = op_index_with_tensor_id_[op_index];
  for (size_t i = 0; i < tensor_id_list.size(); ++i) {
    auto tensor_id = tensor_id_list[i];
    if (tensor_id_with_tensor_.find(tensor_id) != tensor_id_with_tensor_.end()) {
      auto &new_tensor = output_tensors[i];
      auto &tensors_in_value_node = tensor_id_with_tensor_[tensor_id];
      std::for_each(tensors_in_value_node.begin(), tensors_in_value_node.end(), [&](tensor::TensorPtr &tensor) {
        tensor->set_shape(new_tensor->shape());
        tensor->set_data_type(new_tensor->data_type());
        if (target != kCPUDevice) {
          tensor->set_device_address(new_tensor->device_address());
        } else {
          auto old_device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
          auto new_device_address = std::dynamic_pointer_cast<device::DeviceAddress>(new_tensor->device_address());
          auto old_ptr = old_device_address->GetMutablePtr();
          auto new_ptr = new_device_address->GetPtr();
          MS_EXCEPTION_IF_NULL(old_ptr);
          MS_EXCEPTION_IF_NULL(new_ptr);
          auto ret = memcpy_s(old_ptr, old_device_address->GetSize(), new_ptr, new_device_address->GetSize());
          if (ret != EOK) {
            MS_LOG(EXCEPTION) << "Memory copy failed. ret: " << ret;
          }
        }
      });
    }
  }
}

void PynativeExecutor::SaveTensorsInValueNode(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  tensor_id_with_tensor_.clear();
  const auto &func_graph = resource->func_graph();
  const auto &value_node_list = func_graph->value_nodes();
  for (const auto &elem : value_node_list) {
    auto value_node = elem.first->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    std::vector<tensor::TensorPtr> tensors;
    TensorValueToTensor(value_node->value(), &tensors);
    for (const auto &tensor : tensors) {
      if (tensor->device_address() != nullptr) {
        tensor_id_with_tensor_[tensor->id()].emplace_back(tensor);
      }
    }
  }
}

AnfNodePtr PynativeExecutor::GetObjNode(const py::object &obj, const std::string &obj_id) {
  auto &out = graph_info_map_[curr_g_].node_map[obj_id];
  if (out.second.size() == 1 && out.second[0] == -1) {
    return out.first;
  }
  MS_LOG(DEBUG) << "Output size " << out.second.size();

  // Params node
  if (graph_info_map_[curr_g_].params.find(obj_id) != graph_info_map_[curr_g_].params.end()) {
    auto para_node = out.first;
    for (auto &idx : out.second) {
      std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), para_node,
                                                    NewValueNode(idx)};
      para_node = curr_g_->NewCNode(tuple_get_item_inputs);
    }
    return para_node;
  }

  // Normal node
  CNodePtr node = out.first->cast<CNodePtr>();
  auto abs = node->abstract();
  ValuePtr out_obj = nullptr;
  if (node->forward().first != nullptr) {
    out_obj = node->forward().first;
  } else {
    out_obj = PyAttrValue(obj);
  }
  for (auto &idx : out.second) {
    std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), node, NewValueNode(idx)};
    node = curr_g_->NewCNode(tuple_get_item_inputs);
    if (out_obj->isa<ValueTuple>()) {
      node->add_input_value(out_obj, "");
      node->add_input_value(MakeValue(idx), "");
      out_obj = (*out_obj->cast<ValueTuplePtr>())[idx];
      node->set_forward(out_obj, "");
    }
    if (abs != nullptr && abs->isa<abstract::AbstractTuple>()) {
      auto prim_abs = dyn_cast<abstract::AbstractTuple>(abs)->elements()[idx];
      MS_LOG(DEBUG) << "Set tuple getitem abs " << prim_abs->ToString();
      node->set_abstract(prim_abs);
    }
  }
  if (node->abstract() != nullptr) {
    node_abs_map_[obj_id] = node->abstract();
  }
  MS_LOG(DEBUG) << "GetObjNode output " << node->DebugString(6);
  return node;
}

AnfNodePtr PynativeExecutor::MakeValueNode(const py::object &obj, const std::string &obj_id) {
  ValuePtr converted_ret = nullptr;
  parse::ConvertData(obj, &converted_ret);
  auto node = NewValueNode(converted_ret);
  SetNodeMapInGraphInfoMap(curr_g_, obj_id, node);
  return node;
}

void PynativeExecutor::SaveOutputNodeMap(const std::string &obj_id, const py::object &out_real,
                                         const AnfNodePtr &cnode) {
  if (!grad_flag_ || graph_info_map_.empty()) {
    MS_LOG(DEBUG) << "No need save output";
    return;
  }
  MS_LOG(DEBUG) << "Cnode is " << cnode->DebugString(4) << " id " << obj_id;

  if (py::isinstance<py::tuple>(out_real)) {
    auto value = py::cast<py::tuple>(out_real);
    auto size = static_cast<int64_t>(value.size());
    if (size > 1) {
      for (int64_t i = 0; i < size; ++i) {
        auto value_id = GetId(value[i]);
        SetNodeMapInGraphInfoMap(curr_g_, value_id, cnode, i);
      }
    }
  }
  SetNodeMapInGraphInfoMap(curr_g_, obj_id, cnode);
  SetPyObjInGraphInfoMap(curr_g_, obj_id);
}

void PynativeExecutor::SaveAllResult(const OpExecInfoPtr &op_exec_info, const AnfNodePtr &node,
                                     const py::object &out_real) {
  if (!grad_flag_ || node == nullptr) {
    return;
  }

  MS_EXCEPTION_IF_NULL(op_exec_info);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // save input object
  size_t size = op_exec_info->op_inputs.size();
  for (size_t i = 0; i < size; i++) {
    auto obj = op_exec_info->op_inputs[i];
    auto obj_id = GetId(obj);
    if (obj_to_forward_id_.find(obj_id) != obj_to_forward_id_.end()) {
      cnode->add_input_value(PyAttrValue(obj), obj_to_forward_id_[obj_id]);
    } else {
      cnode->add_input_value(nullptr, "");
    }
  }
  // save output object
  auto output_value = PyAttrValue(out_real);
  MS_EXCEPTION_IF_NULL(output_value);
  cnode->set_forward(output_value, op_exec_info->op_index);
  auto out_id = GetId(out_real);
  if (py::isinstance<py::tuple>(out_real)) {
    auto tuple_item = py::cast<py::tuple>(out_real);
    for (size_t i = 0; i < tuple_item.size(); i++) {
      auto tuple_item_id = GetId(tuple_item[i]);
      obj_to_forward_id_[tuple_item_id] = op_exec_info->op_index;
    }
  }
  obj_to_forward_id_[out_id] = op_exec_info->op_index;
}

void PynativeExecutor::GenTupleMap(const ValueTuplePtr &tuple, std::map<std::string, tensor::TensorPtr> *t_map) {
  if (t_map == nullptr) {
    return;
  }
  for (size_t i = 0; i < tuple->size(); i++) {
    ValuePtr tuple_i = (*tuple)[i];
    if (tuple_i->isa<tensor::Tensor>()) {
      auto t = tuple_i->cast<tensor::TensorPtr>();
      (*t_map)[t->id()] = t;
    } else if (tuple_i->isa<ValueTuple>()) {
      GenTupleMap(tuple_i->cast<ValueTuplePtr>(), t_map);
    }
  }
  MS_LOG(DEBUG) << "End GenTupleMap " << tuple->ToString();
}

ValuePtr PynativeExecutor::CleanTupleAddr(const ValueTuplePtr &tuple) {
  std::vector<ValuePtr> value_list;
  for (size_t i = 0; i < tuple->size(); i++) {
    ValuePtr tuple_i = (*tuple)[i];
    if (tuple_i->isa<tensor::Tensor>()) {
      auto t = tuple_i->cast<tensor::TensorPtr>();
      auto new_tensor = std::make_shared<tensor::Tensor>(*t);
      new_tensor->set_device_address(nullptr);
      value_list.emplace_back(new_tensor);
    } else if (tuple_i->isa<ValueTuple>()) {
      value_list.emplace_back(CleanTupleAddr(tuple_i->cast<ValueTuplePtr>()));
    } else {
      MS_LOG(DEBUG) << "Tuple[i] value " << tuple_i->ToString();
      value_list.emplace_back(tuple_i);
    }
  }
  MS_LOG(DEBUG) << "End CleanTupleAddr";
  return std::make_shared<ValueTuple>(value_list);
}

py::tuple PynativeExecutor::RunOpWithInitBackendPolicy(const OpExecInfoPtr &op_exec_info) {
  auto backend_policy = InitEnv(op_exec_info);
  PynativeStatusCode status = PYNATIVE_UNKNOWN_STATE;
  // returns a null py::tuple on error
  py::tuple err_ret(0);
  py::object result = RunOpWithBackendPolicy(backend_policy, op_exec_info, &status);
  if (status != PYNATIVE_SUCCESS) {
    MS_LOG(ERROR) << "Failed to run " << op_exec_info->op_name;
    return err_ret;
  }

  MS_LOG(DEBUG) << "RunOp end";
  return result;
}

MsBackendPolicy PynativeExecutor::InitEnv(const OpExecInfoPtr &op_exec_info) {
  MS_LOG(INFO) << "RunOp start, op name is: " << op_exec_info->op_name;
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
  if (vm_operators.find(op_exec_info->op_name) != vm_operators.end()) {
    backend_policy = kMsBackendVmOnly;
  }
  return backend_policy;
}

py::object PynativeExecutor::RunOpWithBackendPolicy(MsBackendPolicy backend_policy, const OpExecInfoPtr &op_exec_info,
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

py::object PynativeExecutor::RunOpInMs(const OpExecInfoPtr &op_exec_info, PynativeStatusCode *status) {
  MS_EXCEPTION_IF_NULL(op_exec_info);
  MS_EXCEPTION_IF_NULL(status);
  MS_LOG(INFO) << "Start run op [" << op_exec_info->op_name << "] with backend policy ms";
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, true);

  if (session == nullptr) {
    std::string device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    session = session::SessionFactory::Get().Create(device_target);
    MS_EXCEPTION_IF_NULL(session);
    session->Init(ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID));
  }

  std::vector<tensor::TensorPtr> input_tensors;
  std::vector<int64_t> tensors_mask;
  ConstructInputTensor(op_exec_info, &tensors_mask, &input_tensors);
  // get graph info for checking it whether existing in the cache
  std::string graph_info = GetSingleOpGraphInfo(op_exec_info, input_tensors);
  session::OpRunInfo op_run_info = {op_exec_info->op_name,
                                    op_exec_info->py_primitive,
                                    op_exec_info->abstract,
                                    op_exec_info->is_dynamic_shape,
                                    op_exec_info->is_mixed_precision_cast,
                                    op_exec_info->next_op_name,
                                    op_exec_info->next_input_index};
  VectorRef outputs;
  session->RunOp(&op_run_info, graph_info, &input_tensors, &outputs, tensors_mask);
  auto result = BaseRefToPyData(outputs);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, false);
  *status = PYNATIVE_SUCCESS;
  MS_LOG(INFO) << "End run op [" << op_exec_info->op_name << "] with backend policy ms";
  return result;
}

void PynativeExecutor::PushCurrentGraphToStack() { graph_stack_.push(curr_g_); }

void PynativeExecutor::PopGraphStack() {
  if (graph_stack_.empty()) {
    MS_LOG(EXCEPTION) << "Stack graph_stack_ is empty";
  }
  graph_stack_.pop();
  if (!graph_stack_.empty()) {
    curr_g_ = graph_stack_.top();
  }
}

std::string PynativeExecutor::GetCellId(const py::object &cell, const py::args &args) {
  auto cell_id = GetId(cell);
  for (size_t i = 0; i < args.size(); i++) {
    std::string arg_id = GetId(args[i]);
    auto it = node_abs_map_.find(arg_id);
    if (it != node_abs_map_.end()) {
      cell_id += it->second->ToString();
    } else {
      auto abs = PyAttrValue(args[i])->ToAbstract();
      auto config = abstract::AbstractBase::kBroadenTensorOnly;
      abs = abs->Broaden(config);
      cell_id += abs->ToString();
      node_abs_map_[arg_id] = abs;
    }
  }
  return cell_id;
}

std::string PynativeExecutor::GetCellInfo(const py::object &cell) {
  if (py::isinstance<Cell>(cell)) {
    auto c_cell = py::cast<CellPtr>(cell);
    MS_EXCEPTION_IF_NULL(c_cell);
    auto cell_info = c_cell->ToString();
    return cell_info;
  }
  return "";
}

std::string PynativeExecutor::ParseNodeName(const std::shared_ptr<parse::ParseAst> &ast, const py::object &node,
                                            parse::AstMainType type) {
  MS_EXCEPTION_IF_NULL(ast);
  if (py::isinstance<py::none>(node)) {
    MS_LOG(DEBUG) << "Get none type node!";
    return "";
  }
  auto node_type = ast->GetNodeType(node);
  MS_EXCEPTION_IF_NULL(node_type);
  // check node type
  parse::AstMainType node_main_type = node_type->main_type();
  if (node_main_type != type) {
    MS_LOG(ERROR) << "Node type is wrong: " << node_main_type << ", it should be " << type;
    return "";
  }
  std::string node_name = node_type->node_name();
  MS_LOG(DEBUG) << "Ast node is " << node_name;
  return node_name;
}

void PynativeExecutor::ParseInputArgs(const std::shared_ptr<parse::ParseAst> &ast, const py::object &fn_node) {
  MS_EXCEPTION_IF_NULL(ast);
  py::list args = ast->GetArgs(fn_node);
  for (size_t i = 1; i < args.size(); i++) {
    std::string arg_name = py::cast<std::string>(args[i].attr("arg"));
    MS_LOG(DEBUG) << "Input arg name: " << arg_name;
    cell_input_args_.emplace(arg_name);
  }
}

bool PynativeExecutor::ParseIfWhileExprNode(const std::shared_ptr<parse::ParseAst> &ast, const py::object &node) {
  MS_LOG(DEBUG) << "Parse if/while expr";
  py::object test_node = parse::python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_TEST);
  const auto &node_name = ParseNodeName(ast, test_node, parse::AST_MAIN_TYPE_EXPR);
  if (node_name == parse::NAMED_PRIMITIVE_COMPARE) {
    py::object left_node = parse::python_adapter::GetPyObjAttr(test_node, parse::NAMED_PRIMITIVE_LEFT);
    py::list comparators_node = parse::python_adapter::GetPyObjAttr(test_node, parse::NAMED_PRIMITIVE_COMPARATORS);
    if (comparators_node.empty()) {
      MS_LOG(DEBUG) << "Get comparators node falied!";
      return false;
    }
    auto left = ParseNodeName(ast, left_node, parse::AST_MAIN_TYPE_EXPR);
    auto right = ParseNodeName(ast, comparators_node[0], parse::AST_MAIN_TYPE_EXPR);
    if (left == parse::NAMED_PRIMITIVE_SUBSCRIPT) {
      py::object value_in_subscript = parse::python_adapter::GetPyObjAttr(left_node, parse::NAMED_PRIMITIVE_VALUE);
      left = ParseNodeName(ast, value_in_subscript, parse::AST_MAIN_TYPE_EXPR);
    }
    MS_LOG(DEBUG) << "Left is " << left << " Right is " << right;
    if (unchanged_named_primitive.find(left) == unchanged_named_primitive.end() ||
        unchanged_named_primitive.find(right) == unchanged_named_primitive.end()) {
      return true;
    }
  }
  // if flag:
  if (node_name == parse::NAMED_PRIMITIVE_NAME) {
    std::string id = py::cast<std::string>(test_node.attr("id"));
    if (cell_input_args_.find(id) != cell_input_args_.end()) {
      return true;
    }
  }
  return false;
}

bool PynativeExecutor::ParseAssignExprNode(const std::shared_ptr<parse::ParseAst> &ast, const py::object &node) {
  MS_LOG(DEBUG) << "Parse assign expr";
  py::object value_node = parse::python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_VALUE);
  const auto &node_name = ParseNodeName(ast, value_node, parse::AST_MAIN_TYPE_EXPR);
  if (node_name == parse::NAMED_PRIMITIVE_CALL) {
    py::object func_node = parse::python_adapter::GetPyObjAttr(value_node, parse::NAMED_PRIMITIVE_FUNC);
    const auto &func_name = ParseNodeName(ast, func_node, parse::AST_MAIN_TYPE_EXPR);
    if (func_name == parse::NAMED_PRIMITIVE_SUBSCRIPT) {
      py::object slice_node = parse::python_adapter::GetPyObjAttr(func_node, parse::NAMED_PRIMITIVE_SLICE);
      py::object value_in_slice_node = parse::python_adapter::GetPyObjAttr(slice_node, parse::NAMED_PRIMITIVE_VALUE);
      const auto &node_name_in_slice_node = ParseNodeName(ast, value_in_slice_node, parse::AST_MAIN_TYPE_EXPR);
      if (cell_input_args_.find(node_name_in_slice_node) != cell_input_args_.end()) {
        return true;
      }
    }
  }
  return false;
}

bool PynativeExecutor::ParseForExprNode(const std::shared_ptr<parse::ParseAst> &ast, const py::object &node) {
  MS_LOG(DEBUG) << "Parse for expr";
  py::object body_node = parse::python_adapter::GetPyObjAttr(node, parse::NAMED_PRIMITIVE_BODY);
  if (py::isinstance<py::none>(body_node)) {
    MS_LOG(DEBUG) << "Parse body of for expression is none!";
    return false;
  }
  py::int_ pcount = parse::python_adapter::CallPyObjMethod(body_node, parse::PYTHON_GET_METHOD_LEN);
  size_t count = LongToSize(pcount);
  MS_LOG(DEBUG) << "The for nodes count in body is " << count;
  for (size_t i = 0; i < count; ++i) {
    auto it = py::cast<py::list>(body_node)[i];
    const auto &node_name = ParseNodeName(ast, it, parse::AST_MAIN_TYPE_STMT);
    if (node_name == parse::NAMED_PRIMITIVE_ASSIGN && ParseAssignExprNode(ast, it)) {
      return true;
    }
  }
  return false;
}

bool PynativeExecutor::ParseBodyContext(const std::shared_ptr<parse::ParseAst> &ast, const py::object &fn_node) {
  MS_EXCEPTION_IF_NULL(ast);
  py::object func_obj = parse::python_adapter::GetPyObjAttr(fn_node, parse::NAMED_PRIMITIVE_BODY);
  if (py::isinstance<py::none>(func_obj)) {
    MS_LOG(DEBUG) << "Parse body of cell is none!";
    return false;
  }
  py::int_ pcount = parse::python_adapter::CallPyObjMethod(func_obj, parse::PYTHON_GET_METHOD_LEN);
  size_t count = IntToSize(pcount);
  MS_LOG(DEBUG) << "The nodes count in body is " << count;
  bool ret = false;
  for (size_t i = 0; i < count; ++i) {
    auto node = py::cast<py::list>(func_obj)[i];
    const auto &node_name = ParseNodeName(ast, node, parse::AST_MAIN_TYPE_STMT);
    if (node_name == parse::NAMED_PRIMITIVE_ASSIGN) {
      ret = ParseAssignExprNode(ast, node);
    } else if (node_name == parse::NAMED_PRIMITIVE_FOR) {
      ret = ParseForExprNode(ast, node);
    } else if (node_name == parse::NAMED_PRIMITIVE_IF || node_name == parse::NAMED_PRIMITIVE_WHILE) {
      ret = ParseIfWhileExprNode(ast, node);
    }
    if (ret) {
      MS_LOG(INFO) << "Current cell is dynamic!";
      break;
    }
  }
  return ret;
}

bool PynativeExecutor::IsDynamicCell(const py::object &cell) {
  std::string cell_info = GetCellInfo(cell);
  if (ignore_judge_dynamic_cell.find(cell_info) != ignore_judge_dynamic_cell.end()) {
    return false;
  }
  // using ast parse to check whether the construct of cell will be changed
  auto ast = std::make_shared<parse::ParseAst>(cell);
  bool success = ast->InitParseAstInfo(parse::PYTHON_MOD_GET_PARSE_METHOD);
  if (!success) {
    MS_LOG(ERROR) << "Parse code to ast tree failed";
    return false;
  }
  py::object fn_node = ast->GetAstNode();
  // get the name of input args as the initialize of dynamic_variables
  ParseInputArgs(ast, fn_node);
  // parse body context
  bool ret = false;
  ret = ParseBodyContext(ast, fn_node);
  cell_input_args_.clear();
  return ret;
}

void PynativeExecutor::NewGraphInner(const py::object &cell, const py::args &args) {
  auto cell_id = GetCellId(cell, args);
  MS_LOG(DEBUG) << "NewGraphInner start, args size: " << args.size() << ", cell id: " << cell_id;
  // check whether cell needed to construct grad graph
  if (graph_stack_.empty() && cell_graph_map_.find(cell_id) != cell_graph_map_.end() && !dynamic_cell_) {
    auto it = cell_resource_map_.find(cell_id);
    if (it != cell_resource_map_.end()) {
      resource_ = it->second;
      MS_EXCEPTION_IF_NULL(resource_);
      op_index_map_.clear();
    }
    MS_LOG(DEBUG) << "Graph already compiled";
    return;
  }
  // init resource for constructing forward graph and grad graph
  auto g = std::make_shared<FuncGraph>();
  if (graph_stack_.empty()) {
    MakeNewTopGraph(cell_id, args, g);
  }
  MS_EXCEPTION_IF_NULL(df_builder_);
  curr_g_ = g;
  PushCurrentGraphToStack();
  if (graph_info_map_.find(curr_g_) == graph_info_map_.end()) {
    graph_info_map_.emplace(curr_g_, GraphInfo());
  }
  for (size_t i = 0; i < args.size(); ++i) {
    auto param = args[i];
    auto new_param = g->add_parameter();
    std::string param_id = GetId(param);
    SetTupleArgsToGraphInfoMap(curr_g_, param, new_param, true);
    SetNodeMapInGraphInfoMap(curr_g_, param_id, new_param);
  }
  // check whether the constrcut of cell will be changed
  if (!dynamic_cell_) {
    dynamic_cell_ = IsDynamicCell(cell);
    MS_LOG(DEBUG) << "cell id: " << cell_id << ", is dynamic cell: " << dynamic_cell_;
  }
}

void PynativeExecutor::MakeNewTopGraph(const string &cell_id, const py::args &args, const FuncGraphPtr &g) {
  for (auto arg : args) {
    if (py::isinstance<tensor::Tensor>(arg)) {
      auto tensor = arg.cast<tensor::TensorPtr>();
      if (tensor && tensor->is_parameter()) {
        MS_EXCEPTION(TypeError) << "The inputs could not be Parameter.";
      }
    }
  }
  top_g_ = curr_g_ = g;
  dynamic_cell_ = false;
  // a df builder is built for every top function graph
  df_builder_ = std::make_shared<FuncGraph>();
  df_builder_map_[cell_id] = std::make_pair(df_builder_, nullptr);
  resource_ = std::make_shared<pipeline::Resource>();
  resource_->results()[pipeline::kPynativeGraphId] = graph_id_++;
  cell_resource_map_[cell_id] = resource_;
  MS_LOG(DEBUG) << "New top graph for " << cell_id;
  op_index_map_.clear();
  op_index_with_tensor_id_.clear();
  top_graph_cells_.emplace(cell_id);
}

void PynativeExecutor::SetTupleArgsToGraphInfoMap(const FuncGraphPtr &g, const py::object &args, const AnfNodePtr &node,
                                                  bool is_param) {
  if (!py::isinstance<py::tuple>(args) && !py::isinstance<py::list>(args)) {
    return;
  }
  auto tuple = args.cast<py::tuple>();
  auto tuple_size = static_cast<int64_t>(tuple.size());
  for (int64_t i = 0; i < tuple_size; ++i) {
    auto id = GetId(tuple[i]);
    if (is_param) {
      graph_info_map_[g].params.emplace(id);
    }
    SetNodeMapInGraphInfoMap(g, id, node, i);
    SetTupleItemArgsToGraphInfoMap(g, tuple[i], node, std::vector<int64_t>{i}, is_param);
  }
}

void PynativeExecutor::SetTupleItemArgsToGraphInfoMap(const FuncGraphPtr &g, const py::object &args,
                                                      const AnfNodePtr &node,
                                                      const std::vector<int64_t> &index_sequence, bool is_param) {
  if (!py::isinstance<py::tuple>(args) && !py::isinstance<py::list>(args)) {
    return;
  }
  auto tuple = args.cast<py::tuple>();
  auto tuple_size = static_cast<int64_t>(tuple.size());
  for (int64_t i = 0; i < tuple_size; ++i) {
    std::vector<int64_t> tmp = index_sequence;
    tmp.emplace_back(i);
    auto id = GetId(tuple[i]);
    if (is_param) {
      graph_info_map_[g].params.emplace(id);
    }
    SetNodeMapInGraphInfoMap(g, id, node, tmp);
    SetTupleItemArgsToGraphInfoMap(g, tuple[i], node, tmp, is_param);
  }
}

void PynativeExecutor::EndGraphInner(const py::object &cell, const py::object &out, const py::args &args) {
  auto cell_id = GetCellId(cell, args);
  MS_LOG(DEBUG) << "EndGraphInner start " << args.size() << " " << cell_id;
  if (graph_stack_.empty() && cell_graph_map_.find(cell_id) != cell_graph_map_.end() && !dynamic_cell_) {
    MS_LOG(DEBUG) << "Endgraph already compiled";
    return;
  }
  cell_graph_map_[cell_id] = std::make_pair(curr_g_, false);
  auto out_id = GetId(out);
  // x =op1, y =op2, return (x, y)
  if (graph_info_map_[curr_g_].node_map.find(out_id) == graph_info_map_[curr_g_].node_map.end()) {
    if (py::isinstance<py::tuple>(out) || py::isinstance<py::list>(out)) {
      auto tuple = out.cast<py::tuple>();
      auto tuple_size = static_cast<int64_t>(tuple.size());

      std::vector<AnfNodePtr> inputs;
      inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
      for (int64_t i = 0; i < tuple_size; i++) {
        inputs.emplace_back(GetInput(tuple[i], false));
      }
      auto cnode = curr_g_->NewCNode(inputs);
      SetTupleArgsToGraphInfoMap(curr_g_, out, cnode);
      SetNodeMapInGraphInfoMap(curr_g_, out_id, cnode);
    } else {
      MS_LOG(DEBUG) << "Set ValueNode as output for graph, out id: " << out_id;
      MakeValueNode(out, out_id);
    }
  }
  EndGraphByOutId(out_id, cell, out, args);
}

void PynativeExecutor::EndGraphByOutId(const std::string &out_id, const py::object &cell, const py::object &out,
                                       const py::args &args) {
  AnfNodePtr output_node = GetObjNode(out, out_id);
  curr_g_->set_output(output_node);
  MS_LOG(DEBUG) << "Current graph " << curr_g_->output()->DebugString();
  resource_->manager()->AddFuncGraph(curr_g_);

  auto newfg = MakeGradGraph(cell, args);

  if (graph_stack_.size() > 1) {
    std::vector<AnfNodePtr> inputs;
    inputs.emplace_back(NewValueNode(curr_g_));

    PopGraphStack();
    // connect the previous graph to the inside graph
    auto graph_prev = graph_stack_.top();
    for (size_t i = 0; i < args.size(); i++) {
      auto input = GetInput(args[i], false);
      inputs.emplace_back(input);
    }
    auto out_cnode = graph_prev->NewCNode(inputs);
    SetPyObjInGraphInfoMap(graph_prev, GetCellId(cell, args));
    SetTupleArgsToGraphInfoMap(graph_prev, out, out_cnode);
    SetNodeMapInGraphInfoMap(graph_prev, GetId(out), out_cnode);
  } else {
    if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
      DumpIR("before_resolve.ir", newfg);
    }
    parse::ResolveFuncGraph(newfg, resource_);
    if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
      DumpIR("after_resolve.ir", newfg);
    }
    resource_->set_func_graph(newfg);
    PopGraphStack();
  }
}

FuncGraphPtr PynativeExecutor::MakeGradGraph(const py::object &cell, const py::args &args) {
  // custom bprop debug
  bool need_replace_param = false;
  if (py::hasattr(cell, parse::CUSTOM_BPROP_NAME)) {
    need_replace_param = true;
    size_t par_number = py::tuple(parse::python_adapter::CallPyObjMethod(cell, "get_parameters")).size();
    if (par_number > 0) {
      ClearRes();
      MS_LOG(EXCEPTION) << "When user defines the net bprop, there are " << par_number
                        << " parameters that is not supported in the net.";
    }
    MS_LOG(INFO) << "Use cell custom bprop function.";
    FuncGraphPtr bprop_graph = parse::ConvertToBpropCut(cell);
    if (bprop_graph != nullptr) {
      (void)curr_g_->transforms().emplace(std::make_pair(parse::CUSTOM_BPROP_NAME, FuncGraphTransform(bprop_graph)));
      (void)bprop_graph->transforms().emplace(std::make_pair("primal", FuncGraphTransform(curr_g_)));
    }
  }
  // Obtain grad graph
  auto newfg = ad::Grad(curr_g_, resource_, graph_stack_.size() == 1);
  graph_info_map_.erase(curr_g_);

  if (need_replace_param) {
    auto params = newfg->parameters();
    auto manager = Manage({newfg}, false);
    if (args.size() > params.size()) {
      MS_EXCEPTION(TypeError) << "The number of arguments " << args.size()
                              << " is more than the number of parameters required, which is " << params.size();
    }
    for (size_t i = 0; i < args.size(); i++) {
      ValuePtr value = PyAttrValue(args[i]);
      auto v_node = NewValueNode(value);
      manager->Replace(params[i], v_node);
    }
  }
  return newfg;
}

void PynativeExecutor::GradNetInner(const GradOperationPtr &grad, const py::object &cell, const py::object &weights,
                                    const py::args &args) {
  MS_LOG(INFO) << "GradNet start " << args.size();
  auto size = args.size();
  std::pair<bool, bool> sens_weights_changed(false, false);
  std::string cell_id = CheckCellChanged(grad, cell, weights, args, &sens_weights_changed);
  MS_LOG(DEBUG) << "GradNetInner cell_id " << cell_id;

  if (!sens_weights_changed.first && !sens_weights_changed.second &&
      cell_graph_map_.find(cell_id) != cell_graph_map_.end() && cell_graph_map_[cell_id].second && !dynamic_cell_) {
    if (cell_resource_map_.find(cell_id) == cell_resource_map_.end()) {
      MS_LOG(EXCEPTION) << "Can not find resource";
    }
    resource_ = cell_resource_map_[cell_id];
    MS_EXCEPTION_IF_NULL(resource_);
    MS_LOG(INFO) << "GradNetInner already compiled";
    return;
  }

  // set all params(input+weights)
  SetGradGraphParams(size, cell_id, sens_weights_changed);

  // get params(weights) require derivative
  auto w_args = GetWeightsArgs(weights);

  // get the parameters items and add the value to args_spec
  auto args_spec = GetArgsSpec(args);
  resource_->set_args_spec(args_spec);
  MS_LOG(DEBUG) << "Args_spec size " << args_spec.size();

  // Only need to set it at first time
  if (df_builder_map_[cell_id].second == nullptr) {
    auto cloned_df_builder = BasicClone(df_builder_);
    auto cloned_df_newfg = BasicClone(resource_->func_graph());
    df_builder_map_[cell_id] = std::make_pair(cloned_df_builder, cloned_df_newfg);
  } else {
    resource_->set_func_graph(df_builder_map_[cell_id].second);
  }

  // get real grad graph
  GradGraph(resource_->func_graph(), grad, w_args, size);

  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
    DumpIR("befor_grad.ir", resource_->func_graph());
    DumpIR("after_grad.ir", df_builder_);
  }

  resource_->set_func_graph(df_builder_);
  resource_->manager()->KeepRoots({df_builder_});
  resource_->results()[pipeline::kBackend] = compile::CreateBackend();

  MS_LOG(DEBUG) << "Start opt";
  PynativeOptimizeAction(resource_);
  SaveTensorsInValueNode(resource_);
  TaskEmitAction(resource_);
  ExecuteAction(resource_);
  cell_graph_map_[cell_id].second = true;

  resource_->Clean();
  ad::CleanRes();
  pipeline::ReclaimOptimizer();
}

std::string PynativeExecutor::CheckCellChanged(const GradOperationPtr &grad, const py::object &cell,
                                               const py::object &weights, const py::args &args,
                                               std::pair<bool, bool> *sens_weights_changed) {
  MS_EXCEPTION_IF_NULL(sens_weights_changed);
  auto fn = [](const py::object &arg) {
    std::string arg_id;
    if (py::isinstance<tensor::Tensor>(arg)) {
      auto tensor_ptr = py::cast<tensor::TensorPtr>(arg);
      auto dtype = tensor_ptr->data_type();
      auto shape = tensor_ptr->shape();
      std::stringstream ss;
      std::for_each(shape.begin(), shape.end(), [&ss](int i) { ss << i; });
      arg_id = ss.str() + std::to_string(dtype);
    } else {
      arg_id = std::string(py::str(arg));
    }
    return arg_id;
  };

  std::string sens_id = "sens";
  std::string cell_id;
  if (grad->sens_param()) {
    size_t size = args.size();
    size_t forward_args_count = size;
    if (size >= 1) {
      forward_args_count = size - 1;
      const py::object &sens = args[forward_args_count];
      sens_id = fn(sens);
    }
    py::tuple forward_args(forward_args_count);
    for (size_t i = 0; i < forward_args_count; ++i) {
      forward_args[i] = args[i];
    }
    cell_id = GetCellId(cell, forward_args);
  } else {
    cell_id = GetCellId(cell, args);
  }

  std::string wigths_id = fn(weights);

  // Check whether sens or weights changed
  auto it = cell_sw_map_.find(cell_id);
  if (it != cell_sw_map_.end() && it->second.first != sens_id) {
    MS_LOG(DEBUG) << "Sens_id, cur is " << it->second.first << " new is " << sens_id;
    (*sens_weights_changed).first = true;
  }
  if (it != cell_sw_map_.end() && it->second.second != wigths_id) {
    MS_LOG(DEBUG) << "Wigths_id, cur is " << it->second.first << " new is " << wigths_id;
    (*sens_weights_changed).second = true;
  }
  cell_sw_map_[cell_id] = std::make_pair(sens_id, wigths_id);
  return cell_id;
}

void PynativeExecutor::SetGradGraphParams(size_t size, const std::string &cell_id,
                                          const std::pair<bool, bool> &sens_weights_changed) {
  auto ic = cell_resource_map_.find(cell_id);
  if (ic == cell_resource_map_.end()) {
    MS_LOG(EXCEPTION) << "Can not find resource";
  }
  MS_EXCEPTION_IF_NULL(ic->second);
  resource_ = ic->second;

  auto it = df_builder_map_.find(cell_id);
  if (it == df_builder_map_.end()) {
    MS_LOG(EXCEPTION) << "Can not find df_builder";
  }
  MS_EXCEPTION_IF_NULL(it->second.first);
  df_builder_ = it->second.first;

  top_g_ = cell_graph_map_[cell_id].first;
  if (sens_weights_changed.first) {
    MS_LOG(INFO) << "Sens changed, no need reset df_builder params";
    return;
  }

  std::vector<AnfNodePtr> new_params;
  for (size_t i = 0; i < size; i++) {
    ParameterPtr p = std::make_shared<Parameter>(df_builder_);
    new_params.emplace_back(p);
  }
  MS_LOG(DEBUG) << "GradNet weight param size " << df_builder_->parameters().size();
  // df_builder_->parameters() set in GetInput, which are weights params
  new_params.insert(new_params.end(), df_builder_->parameters().begin(), df_builder_->parameters().end());
  df_builder_->set_parameters(new_params);
  resource_->manager()->SetParameters(df_builder_, new_params);
}

std::vector<AnfNodePtr> PynativeExecutor::GetWeightsArgs(const py::object &weights) {
  std::vector<AnfNodePtr> w_args;
  if (!py::hasattr(weights, "__parameter_tuple__")) {
    MS_LOG(DEBUG) << "No paramter_tuple get";
    return {};
  }
  auto tuple = weights.cast<py::tuple>();
  MS_LOG(DEBUG) << "GradNet start weights tuple size " << tuple.size();
  w_args.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  for (size_t it = 0; it < tuple.size(); ++it) {
    auto param = tuple[it];
    auto param_id = GetId(param);
    AnfNodePtr para_node = nullptr;
    if (graph_info_map_[df_builder_].params.find(param_id) != graph_info_map_[df_builder_].params.end() &&
        graph_info_map_[df_builder_].node_map.find(param_id) != graph_info_map_[df_builder_].node_map.end()) {
      para_node = graph_info_map_[df_builder_].node_map[param_id].first;
    } else {
      auto name_attr = parse::python_adapter::GetPyObjAttr(param, "name");
      if (py::isinstance<py::none>(name_attr)) {
        MS_LOG(EXCEPTION) << "Parameter object should have name attribute";
      }
      auto param_name = py::cast<std::string>(name_attr);
      auto free_param = df_builder_->add_parameter();
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

abstract::AbstractBasePtrList PynativeExecutor::GetArgsSpec(const py::args &args) {
  abstract::AbstractBasePtrList args_spec;
  std::size_t size = args.size();
  // input params
  for (std::size_t i = 0; i < size; i++) {
    ValuePtr converted = nullptr;
    bool succ = parse::ConvertData(args[i], &converted);
    if (!succ) {
      MS_LOG(EXCEPTION) << "Args convert error";
    }
    bool broaden = true;
    auto abs = abstract::FromValue(converted, broaden);
    args_spec.emplace_back(abs);
    auto param_node = std::static_pointer_cast<Parameter>(df_builder_->parameters()[i]);
    param_node->set_abstract(abs);
  }
  // weights params
  for (const auto &param : df_builder_->parameters()) {
    auto param_node = std::static_pointer_cast<Parameter>(param);
    if (param_node->has_default()) {
      ValuePtr value = param_node->default_param();
      auto ptr = value->ToAbstract();
      MS_EXCEPTION_IF_NULL(ptr);
      args_spec.emplace_back(ptr);
      param_node->set_abstract(ptr);
    }
  }
  return args_spec;
}

void PynativeExecutor::GradGraph(FuncGraphPtr g, const GradOperationPtr &grad_op,
                                 const std::vector<AnfNodePtr> &weights, size_t arg_size) {
  auto nparam = top_g_->parameters().size();
  std::ostringstream ss;
  ss << "grad{" << nparam << "}";
  df_builder_->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  df_builder_->debug_info()->set_name(ss.str());

  auto df = grad_op->GetGrad(NewValueNode(g), nullptr, top_g_->parameters(), weights);
  std::vector<AnfNodePtr> inputs = {NewValueNode(df)};
  for (size_t i = 0; i < arg_size; ++i) {
    inputs.emplace_back(df_builder_->parameters()[i]);
  }
  auto out = df_builder_->NewCNode(inputs);
  df_builder_->set_output(out);
  resource_->manager()->AddFuncGraph(df);
  resource_->manager()->AddFuncGraph(df_builder_);
}

py::object PynativeExecutor::CheckGraph(const py::object &cell, const py::args &args) {
  BaseRef ret = false;
  if (!grad_is_running) {
    MS_LOG(DEBUG) << "Grad not running yet";
    return BaseRefToPyData(ret);
  }
  auto cell_id = GetCellId(cell, args);
  string key = cell_id.substr(0, std::min(PTR_LEN, cell_id.size()));
  MS_LOG(DEBUG) << "Key is " << key;
  for (auto it = cell_graph_map_.begin(); it != cell_graph_map_.end(); ++it) {
    MS_LOG(DEBUG) << "Cur cell id " << it->first;
    if (key != it->first.substr(0, std::min(PTR_LEN, it->first.size()))) {
      continue;
    }
    MS_LOG(DEBUG) << "Delete cellid from cell_graph_map_";
    cell_graph_map_.erase(it->first);
    ret = true;
    break;
  }
  return BaseRefToPyData(ret);
}

py::object PynativeExecutor::Run(const py::tuple &args, const py::object &phase) {
  VectorRef arg_list;
  py::tuple converted_args = ConvertArgs(args);
  pipeline::ProcessVmArgInner(converted_args, resource_, &arg_list);
  if (resource_->results().find(pipeline::kOutput) == resource_->results().end()) {
    MS_LOG(EXCEPTION) << "Can't find run graph output";
  }
  if (!resource_->results()[pipeline::kOutput].is<compile::VmEvalFuncPtr>()) {
    MS_LOG(EXCEPTION) << "Run graph is not VmEvalFuncPtr";
  }
  compile::VmEvalFuncPtr run = resource_->results()[pipeline::kOutput].cast<compile::VmEvalFuncPtr>();
  MS_EXCEPTION_IF_NULL(run);

  std::string backend = MsContext::GetInstance()->backend_policy();
  MS_LOG(DEBUG) << "Eval run " << backend;
  grad_is_running = true;
  BaseRef value = (*run)(arg_list);
  grad_is_running = false;
  MS_LOG(DEBUG) << "Run end " << value.ToString();
  return BaseRefToPyData(value);
}

template <typename T>
void MapClear(T *map, const std::string &flag) {
  for (auto it = map->begin(); it != map->end();) {
    if (it->first.find(flag) != std::string::npos) {
      it = map->erase(it);
    } else {
      it++;
    }
  }
}

void PynativeExecutor::Clear(const std::string &flag) {
  if (!flag.empty()) {
    MS_LOG(DEBUG) << "Clear cell res";
    MapClear<std::unordered_map<std::string, ResourcePtr>>(&cell_resource_map_, flag);
    MapClear<std::unordered_map<std::string, bool>>(&cell_dynamic_map_, flag);
    MapClear<std::unordered_map<std::string, std::pair<FuncGraphPtr, bool>>>(&cell_graph_map_, flag);
    MapClear<std::unordered_map<std::string, std::pair<std::string, std::string>>>(&cell_sw_map_, flag);
    MapClear<std::unordered_map<std::string, std::pair<FuncGraphPtr, FuncGraphPtr>>>(&df_builder_map_, flag);

    // Maybe exit in the pynative runing op, so need reset pynative flag.
    auto ms_context = MsContext::GetInstance();
    if (ms_context != nullptr) {
      ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, false);
    }
    ConfigManager::GetInstance().ResetIterNum();
    if (top_graph_cells_.find(flag) != top_graph_cells_.end()) {
      Clean();
    }
    node_abs_map_.clear();
    return;
  }

  MS_LOG(DEBUG) << "Clear";
  grad_flag_ = false;
  top_g_ = nullptr;
  df_builder_ = nullptr;
  curr_g_ = nullptr;
  graph_info_map_.clear();
  obj_to_forward_id_.clear();
  node_abs_map_.clear();
  std::stack<FuncGraphPtr>().swap(graph_stack_);
  ConfigManager::GetInstance().ResetIterNum();
}

void PynativeExecutor::Clean() {
  MS_LOG(DEBUG) << "Clean all res";
  Clear();
  grad_flag_ = false;
  ad::CleanRes();
  pipeline::ReclaimOptimizer();
}

void PynativeExecutor::ClearRes() {
  MS_LOG(DEBUG) << "PynativeExecutor destruct";
  Clean();
  cell_sw_map_.clear();
  df_builder_map_.clear();
  cell_graph_map_.clear();
  cell_resource_map_.clear();
  cell_dynamic_map_.clear();
  node_abs_map_.clear();
  top_graph_cells_.clear();
}

void PynativeExecutor::NewGraph(const py::object &cell, const py::args &args) {
  PynativeExecutorTry(this, &PynativeExecutor::NewGraphInner, cell, args);
}

void PynativeExecutor::EndGraph(const py::object &cell, const py::object &out, const py::args &args) {
  PynativeExecutorTry(this, &PynativeExecutor::EndGraphInner, cell, out, args);
}

void PynativeExecutor::GradNet(const GradOperationPtr &grad, const py::object &cell, const py::object &weights,
                               const py::args &args) {
  PynativeExecutorTry(this, &PynativeExecutor::GradNetInner, grad, cell, weights, args);
}

REGISTER_PYBIND_DEFINE(PynativeExecutor_, ([](const py::module *m) {
                         (void)py::class_<PynativeExecutor, std::shared_ptr<PynativeExecutor>>(*m, "PynativeExecutor_")
                           .def_static("get_instance", &PynativeExecutor::GetInstance, "PynativeExecutor get_instance.")
                           .def("new_graph", &PynativeExecutor::NewGraph, "pynative new a graph.")
                           .def("end_graph", &PynativeExecutor::EndGraph, "pynative end a graph.")
                           .def("check_graph", &PynativeExecutor::CheckGraph, "pynative check a grad graph.")
                           .def("grad_net", &PynativeExecutor::GradNet, "pynative grad graph.")
                           .def("clear", &PynativeExecutor::Clear, "pynative clear status.")
                           .def("__call__", &PynativeExecutor::Run, py::arg("args"), py::arg("phase") = py::str(""),
                                "Executor run function.")
                           .def("set_grad_flag", &PynativeExecutor::set_grad_flag, py::arg("flag") = py::bool_(false),
                                "Executor set grad flag.");
                       }));
}  // namespace mindspore::pynative
