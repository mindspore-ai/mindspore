/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "pipeline/pynative/forward/forward.h"
#include <set>
#include <algorithm>
#include <vector>
#include "pipeline/pynative/pynative_utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/scoped_long_running.h"
#include "backend/graph_compiler/transform.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace pynative {
namespace {
// primitive unable to infer value for constant input in PyNative mode
const std::set<std::string> kVmOperators = {"InsertGradientOf", "stop_gradient", "mixed_precision_cast", "HookBackward",
                                            "CellBackwardHook"};
const std::set<std::string> kForceInferPrim = {"TopK", "DropoutGenMask"};
const std::set<std::string> kIgnoreInferPrim = {"mixed_precision_cast"};
enum class RunOpArgsEnum : size_t { PY_PRIM = 0, PY_NAME, PY_INPUTS, PY_ARGS_NUM };
std::set<std::string> kNotConstPrimOrConstInput;

// Shallow Copy Value and change shape
ValuePtr ShallowCopyValue(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(value);
  auto tensor_abs = op_run_info->base_op_run_info.abstract;
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
                         [op_run_info](const ValuePtr &elem) { return ShallowCopyValue(op_run_info, elem); });
    return std::make_shared<ValueTuple>(values);
  } else {
    return value;
  }
}

MsBackendPolicy GetBackendPolicy(const std::string &device_target) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  MsBackendPolicy backend_policy = kMsBackendVmOnly;
  if (device_target == kAscendDevice) {
    if (ms_context->backend_policy() == "ge") {
      MS_LOG(EXCEPTION) << "In PyNative mode, not support ge backend!";
    }
#ifdef WITH_BACKEND
    const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {kAscendDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());

    if (!device_context->GetDeprecatedInterface()->IsTsdOpened(ms_context)) {
      if (!device_context->GetDeprecatedInterface()->OpenTsd(ms_context)) {
        MS_LOG(EXCEPTION) << "Open tsd failed";
      }
    }
#endif
  }
  return backend_policy;
}

void GetSingleOpGraphInfo(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const std::vector<tensor::TensorPtr> &input_tensors = op_run_info->base_op_run_info.input_tensor;
  const std::vector<int64_t> &tensors_mask = op_run_info->base_op_run_info.input_mask;
  if (input_tensors.size() != tensors_mask.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size() << " should be equal to tensors mask size "
                      << tensors_mask.size();
  }
  std::ostringstream buf;
  buf << op_run_info->base_op_run_info.op_name;
  bool has_const_input = false;
  const auto &op_prim = op_run_info->op_prim;
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
    auto abstr = op_run_info->base_op_run_info.abstract;
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
  op_run_info->base_op_run_info.graph_info = buf.str();
}

void SetNonCostantValueAbs(const AbstractBasePtr &abs, const std::string &id) {
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
}

bool IsConstPrimOrConstInput(const FrontendOpRunInfoPtr &op_run_info, size_t index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  auto prim = op_run_info->op_prim;
  MS_EXCEPTION_IF_NULL(prim);
  if (kNotConstPrimOrConstInput.find(prim->name()) != kNotConstPrimOrConstInput.end()) {
    MS_LOG(DEBUG) << prim->name() << " is not const prim or have const input";
    return false;
  }
  bool is_const_prim = prim->is_const_prim();
  const auto &const_input_index = prim->get_const_input_indexes();
  bool have_const_input = !const_input_index.empty();
  bool is_const_input =
    have_const_input && std::find(const_input_index.begin(), const_input_index.end(), index) != const_input_index.end();
  MS_LOG(DEBUG) << prim->ToString() << " is const prim " << prim->is_const_prim() << ", is_const_input "
                << is_const_input;
  bool ret = is_const_prim || is_const_input;
  if (!ret) {
    (void)kNotConstPrimOrConstInput.emplace(prim->name());
  }
  return ret;
}

void PynativeInfer(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  auto prim = op_run_info->op_prim;
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(prim);
  MS_LOG(DEBUG) << "Prim " << prim->name() << " infer input: " << mindspore::ToString(op_run_info->input_abs);
  prim->BeginRecordAddAttr();
  auto eval_ret = EvalOnePrim(prim, op_run_info->input_abs);
  MS_EXCEPTION_IF_NULL(eval_ret);
  AbstractBasePtr infer_res = eval_ret->abstract();
  MS_EXCEPTION_IF_NULL(infer_res);
  prim->EndRecordAddAttr();
  MS_EXCEPTION_IF_NULL(op_run_info);
  op_run_info->base_op_run_info.abstract = infer_res;
  MS_EXCEPTION_IF_NULL(op_run_info->base_op_run_info.abstract);
  MS_LOG(DEBUG) << "Prim " << prim->name() << " infer result: " << op_run_info->base_op_run_info.abstract->ToString();
}
}  // namespace

GradExecutorPtr ForwardExecutor::grad() const {
  auto grad_executor = grad_executor_.lock();
  MS_EXCEPTION_IF_NULL(grad_executor);
  return grad_executor;
}

void ForwardExecutor::Init() {
  if (init_) {
    return;
  }
  MS_LOG(DEBUG) << "Init ForwardExecutor";
  compile::SetMindRTEnable();
  python_adapter::set_python_env_flag(true);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  device_target_ = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device_id_ = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  enable_mind_rt_ = ms_context->get_param<bool>(MS_CTX_ENABLE_MINDRT);
  init_ = true;
}

AbstractBasePtr ForwardExecutor::GetValueAbstract(const FrontendOpRunInfoPtr &op_run_info, size_t i,
                                                  const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(v);
  const auto &id = PyNativeAlgo::Common::GetIdByValue(v);
  AbstractBasePtr abs = nullptr;
  auto it = node_abs_map_.find(id);
  if (it != node_abs_map_.end()) {
    abs = it->second;
  }
  MS_LOG(DEBUG) << "Abstract cache hit " << (abs != nullptr);
  bool is_const_prim_or_input = IsConstPrimOrConstInput(op_run_info, i);
  if (abs == nullptr || is_const_prim_or_input) {
    abs = v->ToAbstract();
    MS_EXCEPTION_IF_NULL(abs);
    if (!is_const_prim_or_input) {
      SetNonCostantValueAbs(abs, id);
      node_abs_map_[id] = abs;
    }
  }
  return abs;
}

AbstractBasePtr ForwardExecutor::GetTupleInputAbstract(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v,
                                                       const std::string &id, size_t input_index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(v);
  abstract::AbstractBasePtrList abs_list;
  if (!IsConstPrimOrConstInput(op_run_info, input_index)) {
    auto it = node_abs_map_.find(id);
    if (it != node_abs_map_.end()) {
      return it->second;
    }
  }
  MS_LOG(DEBUG) << "Abstract cache not hit";
  auto tuple = v->cast<ValueSequencePtr>();
  auto tuple_size = tuple->size();
  for (size_t i = 0; i < tuple_size; ++i) {
    const auto &item_id = PyNativeAlgo::Common::GetIdByValue(tuple->value()[i]);
    const auto &dynamic_shape = PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->dynamic_shape();
    const auto item_it = dynamic_shape->id_with_dynamic_abs().find(item_id);
    if (item_it != dynamic_shape->id_with_dynamic_abs().end()) {
      (void)abs_list.emplace_back(item_it->second);
    } else {
      auto abs = GetValueAbstract(op_run_info, input_index, tuple->value()[i]);
      (void)abs_list.emplace_back(abs);
    }
  }
  abstract::AbstractBasePtr node_abs;
  if (v->isa<ValueTuple>()) {
    node_abs = std::make_shared<abstract::AbstractTuple>(abs_list);
  } else {
    node_abs = std::make_shared<abstract::AbstractList>(abs_list);
  }
  node_abs_map_[id] = node_abs;
  return node_abs;
}

AbstractBasePtr ForwardExecutor::GetInputAbs(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &v, size_t index) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(v);
  const auto &dynamic_shape = PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->dynamic_shape();
  const auto &id = PyNativeAlgo::Common::GetIdByValue(v);
  // Get tuple or list abs
  if (v->isa<ValueSequence>()) {
    auto abs = GetTupleInputAbstract(op_run_info, v, id, index);
    auto shape = abs->BuildShape();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->IsDynamic()) {
      MS_LOG(DEBUG) << "Input " << index << " get input of prev op dynamic output";
      op_run_info->base_op_run_info.has_dynamic_input = true;
      dynamic_shape->SaveIdWithDynamicShape(op_run_info, id, v, abs);
    }
    MS_LOG(DEBUG) << "Set " << index << "th input id " << id << " abs " << abs->ToString();
    return abs;
  }
  auto out_it = dynamic_shape->id_with_dynamic_abs().find(id);
  if (out_it != dynamic_shape->id_with_dynamic_abs().end()) {
    MS_LOG(DEBUG) << "Input " << index << " get input of prev op dynamic output";
    op_run_info->base_op_run_info.has_dynamic_input = true;
    dynamic_shape->SaveIdWithDynamicShape(op_run_info, id, v, out_it->second);
    MS_LOG(DEBUG) << "Set " << index << "th input id " << id << " abs " << out_it->second->ToString();
    return out_it->second;
  } else {
    const auto &input_abs = GetValueAbstract(op_run_info, index, v);
    const auto &shape = input_abs->BuildShape();
    MS_EXCEPTION_IF_NULL(shape);
    // For ms function
    if (shape->IsDynamic()) {
      MS_LOG(DEBUG) << "Input " << index << " get dynamic shape";
      op_run_info->base_op_run_info.has_dynamic_input = true;
    }
    MS_LOG(DEBUG) << "Set " << index << "th input id " << id << " abs " << input_abs->ToString();
    return input_abs;
  }
}

void ForwardExecutor::GetInputAbstract(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  for (size_t i = 0; i < op_run_info->input_value.size(); ++i) {
    (void)op_run_info->input_abs.emplace_back(GetInputAbs(op_run_info, op_run_info->input_value[i], i));
  }
}

bool ForwardExecutor::GetOutputAbstract(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  auto op_name = op_run_info->base_op_run_info.op_name;
  auto prim = op_run_info->op_prim;
  MS_EXCEPTION_IF_NULL(prim);

  bool prim_cache_hit = false;
  AbsCacheKey key{prim->name(), prim->Hash(), prim->attrs()};
  auto temp = prim_abs_list_.find(key);
  if (temp != prim_abs_list_.end()) {
    MS_LOG(DEBUG) << "Match prim input args " << op_name << mindspore::ToString(op_run_info->input_abs);
    auto iter = temp->second.find(op_run_info->input_abs);
    if (iter != temp->second.end()) {
      MS_LOG(DEBUG) << "Match prim ok " << iter->second.abs->ToString();
      op_run_info->base_op_run_info.abstract = iter->second.abs;
      prim->set_evaluate_added_attrs(iter->second.attrs);
      prim_cache_hit = true;
    }
  }

  if (op_run_info->base_op_run_info.abstract == nullptr || kForceInferPrim.find(op_name) != kForceInferPrim.end()) {
    // Use python infer method
    if (kIgnoreInferPrim.find(op_name) == kIgnoreInferPrim.end()) {
      PynativeInfer(op_run_info);
    }
  }
  // Get output dynamic shape info from infer steprr
  auto abstract = op_run_info->base_op_run_info.abstract;
  MS_EXCEPTION_IF_NULL(abstract);
  auto shape = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape);
  op_run_info->base_op_run_info.has_dynamic_output = shape->IsDynamic();
  if (PyNativeAlgo::Common::IsDynamicShape(op_run_info)) {
    MS_LOG(DEBUG) << "Set dynamic op " << op_name;
  }
  return prim_cache_hit;
}

void ForwardExecutor::RunOpInner(py::object *ret, const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(ret);
  MS_EXCEPTION_IF_NULL(op_run_info);
  Init();
  *ret = ValueToPyData(RunOpForward(op_run_info));
}

ValuePtr ForwardExecutor::RunOpForward(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_LOG(DEBUG) << "RunOp name: " << op_run_info->base_op_run_info.op_name;
  if (op_run_info->base_op_run_info.op_name == prim::kPrimMixedPrecisionCast->name()) {
    return RunMixedPrecisionCastOp(op_run_info);
  }

  // 1.Set cast for inputs
  SetCastForInputs(op_run_info);
  // 2. Get input abstract
  GetInputAbstract(op_run_info);
  // 3.Get output abstract
  bool prim_cache_hit = GetOutputAbstract(op_run_info);
  // 4.Get output
  const auto &out_value = GetOutput(op_run_info, prim_cache_hit);
  // 5. Do op grad
  grad()->ProcessOpGradInfo(op_run_info, out_value);
  return out_value;
}

FrontendOpRunInfoPtr ForwardExecutor::GenerateOpRunInfo(const py::args &args) {
  if (args.size() != static_cast<size_t>(RunOpArgsEnum::PY_ARGS_NUM)) {
    MS_LOG(EXCEPTION) << "Three args are needed by RunOp";
  }
  const auto &op_run_info = std::make_shared<FrontendOpRunInfo>();
  op_run_info->base_op_run_info.op_name = py::cast<std::string>(args[static_cast<size_t>(RunOpArgsEnum::PY_NAME)]);
  op_run_info->base_op_run_info.lazy_build = lazy_build_;
  PyNativeAlgo::PyParser::SetPrim(op_run_info, args[static_cast<size_t>(RunOpArgsEnum::PY_PRIM)]);
  PyNativeAlgo::PyParser::ParseOpInputByPythonObj(op_run_info, args[static_cast<size_t>(RunOpArgsEnum::PY_INPUTS)]);
  mindspore::ScopedLongRunning long_running;
  return op_run_info;
}

void ForwardExecutor::SetCastForInputs(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  // No need cast self
  if (op_run_info->base_op_run_info.op_name == prim::kPrimCast->name() || op_run_info->is_nop_prim) {
    return;
  }
  cast_operation()->DoCast(op_run_info);
}

ValuePtr ForwardExecutor::RunMixedPrecisionCastOp(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &res = RunOpWithBackendPolicy(op_run_info);
  auto res_v_list = res->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(res_v_list);
  if (res_v_list->size() == 1) {
    return res_v_list->value().front();
  }
  return res;
}

ValuePtr ForwardExecutor::DoNopOutput(const FrontendOpRunInfoPtr &op_run_info) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  // Get First input
  if (op_run_info->base_op_run_info.input_tensor.empty()) {
    MS_LOG(EXCEPTION) << "Inputs of " << op_run_info->base_op_run_info.op_name << " is empty";
  }
  const auto &value = op_run_info->base_op_run_info.input_tensor[0];
  if (!value->isa<tensor::Tensor>()) {
    MS_LOG(EXCEPTION) << "First input of " << op_run_info->base_op_run_info.op_name << " must be a tensor";
  }
  const auto &tensor_ptr = value->cast<tensor::TensorPtr>();
  const auto &v = ShallowCopyValue(op_run_info, tensor_ptr);
  MS_LOG(DEBUG) << "New copy value is " << v->ToString();
  return v;
}

ValuePtr ForwardExecutor::GetOutput(const FrontendOpRunInfoPtr &op_run_info, bool prim_cache_hit) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &prim = op_run_info->op_prim;
  MS_EXCEPTION_IF_NULL(prim);
  // Infer output value by constant folding
  py::dict output = abstract::ConvertAbstractToPython(op_run_info->base_op_run_info.abstract, true);
  if (!output[ATTR_VALUE].is_none()) {
    MS_LOG(DEBUG) << "Get output by constant folding, output is " << py::str(output[ATTR_VALUE]);
    op_run_info->output_get_by_infer_value = true;
    return PyNativeAlgo::DataConvert::PyObjToValue(output[ATTR_VALUE]);
  } else if (prim->is_const_prim()) {
    MS_LOG(DEBUG) << "Get const prim";
    op_run_info->output_get_by_infer_value = true;
    return MakeValue("");
  }

  // Add output abstract info into cache, the const value needs to infer evert step
  if (!prim_cache_hit && !PyNativeAlgo::Common::IsDynamicShape(op_run_info)) {
    AbsCacheKey key{prim->name(), prim->Hash(), prim->attrs()};
    auto &out = prim_abs_list_[key];
    out[op_run_info->input_abs].abs = op_run_info->base_op_run_info.abstract;
    out[op_run_info->input_abs].attrs = prim->evaluate_added_attrs();
  }

  // Run op with selected backend, nop is no need run backend
  ValuePtr out_real_value = nullptr;
  if (op_run_info->is_nop_prim) {
    out_real_value = DoNopOutput(op_run_info);
  } else {
    out_real_value = RunOpWithBackendPolicy(op_run_info);
    if (out_real_value->isa<ValueSequence>()) {
      const auto &result_v_list = out_real_value->cast<ValueSequencePtr>();
      if (result_v_list->size() == 1 && op_run_info->base_op_run_info.abstract != nullptr &&
          !op_run_info->base_op_run_info.abstract->isa<abstract::AbstractSequence>()) {
        out_real_value = result_v_list->value().front();
      }
    }
  }
  return out_real_value;
}

session::SessionPtr ForwardExecutor::GetCurrentSession(const std::string &device_target) {
  auto iter = session_backends_.find(device_target);
  if (iter == session_backends_.end()) {
    auto session = session::SessionFactory::Get().Create(device_target);
    MS_EXCEPTION_IF_NULL(session);
    session->Init(device_id_);
    session_backends_[device_target] = session;
    return session;
  } else {
    return iter->second;
  }
}

compile::MindRTBackendPtr ForwardExecutor::GetMindRtBackend(const std::string &device_target) {
  auto iter = mindrt_backends_.find(device_target);
  if (iter == mindrt_backends_.end()) {
    std::lock_guard<std::mutex> guard(pipeline::Resource::GetBackendInitMutex());
    auto backend = std::make_shared<compile::MindRTBackend>("ms", device_target, device_id_);
    MS_EXCEPTION_IF_NULL(backend);
    mindrt_backends_[device_target] = backend;
    return backend;
  } else {
    return iter->second;
  }
}

ValuePtr ForwardExecutor::RunOpWithBackendPolicy(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  ValuePtr result;
  auto backend_policy = GetBackendPolicy(device_target_);
  if (backend_policy == kMsBackendVmOnly) {
#ifndef ENABLE_TEST
    if (kVmOperators.find(op_run_info->base_op_run_info.op_name) != kVmOperators.end()) {
      result = RunOpInVM(op_run_info);
    } else {
      result = RunOpInMs(op_run_info);
    }
#else
    result = RunOpInVM(op_run_info);
#endif
  }
  return result;
}

ValuePtr ForwardExecutor::RunOpInVM(const FrontendOpRunInfoPtr &op_run_info) const {
  MS_LOG(DEBUG) << "RunOpInVM start";
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_EXCEPTION_IF_NULL(op_run_info->op_prim);
  op_run_info->run_in_vm = true;
  const auto &op_inputs = op_run_info->input_value;
  if (op_run_info->base_op_run_info.op_name == prim::kPrimInsertGradientOf->name() ||
      op_run_info->base_op_run_info.op_name == prim::kPrimStopGradient->name() ||
      op_run_info->base_op_run_info.op_name == prim::kPrimHookBackward->name() ||
      op_run_info->base_op_run_info.op_name == prim::kPrimCellBackwardHook->name()) {
    std::vector<ValuePtr> result(op_inputs.size());
    for (size_t i = 0; i < op_inputs.size(); i++) {
      auto tensor = op_inputs[i]->cast<TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      if (op_run_info->base_op_run_info.op_name == prim::kPrimHookBackward->name() ||
          op_run_info->base_op_run_info.op_name == prim::kPrimCellBackwardHook->name()) {
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
    auto result_v = std::make_shared<ValueTuple>(result);
    dynamic_shape()->SaveOutputDynamicShape(op_run_info, op_run_info->base_op_run_info.abstract, result_v);
    MS_LOG(DEBUG) << "RunOpInVM end";
    return result_v;
  }

  MS_EXCEPTION_IF_NULL(op_run_info->op_prim);
  op_run_info->op_inputs = py::list(op_inputs.size());
  for (size_t i = 0; i < op_inputs.size(); ++i) {
    op_run_info->op_inputs[i] = ValueToPyData(op_inputs[i]);
  }
  auto result = op_run_info->op_prim->RunPyComputeFunction(op_run_info->op_inputs);
  if (py::isinstance<py::none>(result)) {
    MS_LOG(EXCEPTION) << "VM op " << op_run_info->base_op_run_info.op_name << " run failed!";
  }
  ValuePtr result_v = PyNativeAlgo::DataConvert::PyObjToValue(result);
  dynamic_shape()->SaveOutputDynamicShape(op_run_info, op_run_info->base_op_run_info.abstract, result_v);
  MS_LOG(DEBUG) << "RunOpInVM end";
  if (result_v->isa<ValueSequence>()) {
    return result_v;
  }
  return std::make_shared<ValueTuple>(std::vector{result_v});
}

void ForwardExecutor::CheckIfNeedSyncForHeterogeneous(const std::string &cur_target) {
  if (last_target_ != "Unknown" && last_target_ != cur_target) {
    Sync();
  }
  last_target_ = cur_target;
}

bool ForwardExecutor::CellNotSetMixedPrecision(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &cur_cell = forward_cell_stack_.top();
  MS_EXCEPTION_IF_NULL(cur_cell);
  MixedPrecisionType mix_type = cur_cell->GetMixedPrecisionType();
  if (mix_type == kNotSet) {
    return true;
  }
  op_run_info->mix_type = mix_type;
  return false;
}

void ForwardExecutor::ExecuteLazyTask() {
  mindspore::ScopedLongRunning long_running;
  session::PynativeTaskManager::GetInstance().ExecuteRemainingTasks();
  for (const auto &item : mindrt_backends_) {
    MS_EXCEPTION_IF_NULL(item.second);
    item.second->WaitTaskFinish();
  }
}

void ForwardExecutor::ProcessBeforeNewGraph(const py::object &cell, const py::args &args) {
  if (py::isinstance<Cell>(cell)) {
    PushForwardCell(cell);
  }
  dynamic_shape()->SetFeedDynamicInputAbs(cell, args);
}

void ForwardExecutor::ProcessBeforeEndGraph(const py::object &cell, const py::args &args) {
  if (py::isinstance<Cell>(cell)) {
    PopForwardCell();
  }

  // Do some finishing work before end graph
  if (IsFirstCell()) {
    // Reset lazy build
    set_lazy_build(false);
    // Finish lazy task
    ExecuteLazyTask();
    if (!grad()->grad_flag()) {
      // Clean up some resources for dynamic shape
      dynamic_shape()->reset();
      ClearNodeAbsMap();
    }
  }
}

void ForwardExecutor::ProcessAfterEndGraph() {
  if (IsFirstCell()) {
    dynamic_shape()->reset();
    ClearNodeAbsMap();
  }
}

std::string ForwardExecutor::GetCurrentDeviceTarget(const PrimitivePtr &op_prim) {
  MS_EXCEPTION_IF_NULL(op_prim);
  const auto &attr_map = op_prim->attrs();
  auto iter = attr_map.find("primitive_target");
  if (iter != attr_map.end()) {
    return GetValue<std::string>(iter->second);
  }
  return device_target_;
}

void ForwardExecutor::Sync() {
  ExecuteLazyTask();
  if (!enable_mind_rt_) {
    for (auto &item : session_backends_) {
      MS_EXCEPTION_IF_NULL(item.second);
      item.second->SyncStream();
    }
  } else {
    for (auto &item : mindrt_backends_) {
      MS_EXCEPTION_IF_NULL(item.second);
      item.second->SyncStream();
    }
    for (auto &item : session_backends_) {
      MS_EXCEPTION_IF_NULL(item.second);
      item.second->SyncStream();
    }
  }
}

ValuePtr ForwardExecutor::RunOpInMs(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_LOG(DEBUG) << "RunOpInMs start";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, true);
  const auto &cur_target = GetCurrentDeviceTarget(op_run_info->op_prim);
  op_run_info->base_op_run_info.device_target = cur_target;
  CheckIfNeedSyncForHeterogeneous(cur_target);
  PyNativeAlgo::DataConvert::GetInputTensor(op_run_info, cur_target);
  dynamic_shape()->UpdateInputTensorToDynamicShape(op_run_info);
  // get graph info for checking it whether existing in the cache
  GetSingleOpGraphInfo(op_run_info);
  auto backend_op_run_info =
    std::make_shared<BackendOpRunInfo>(op_run_info->base_op_run_info, op_run_info->op_prim.get(), true, false);
#if defined(__APPLE__)
  backend_op_run_info->base_op_run_info.lazy_build = false;
#endif

  VectorRef outputs;
  if (enable_mind_rt_) {
    const auto &cur_mind_rt_backend = GetMindRtBackend(cur_target);
    MS_EXCEPTION_IF_NULL(cur_mind_rt_backend);
    cur_mind_rt_backend->RunOp(backend_op_run_info, &outputs);
  } else {
    auto cur_session = GetCurrentSession(cur_target);
    MS_EXCEPTION_IF_NULL(cur_session);
    cur_session->RunOp(backend_op_run_info, &outputs);
  }
  const auto &result_v = PyNativeAlgo::DataConvert::VectorRefToValue(outputs);
  dynamic_shape()->SaveOutputDynamicShape(op_run_info, op_run_info->base_op_run_info.abstract, result_v);
  ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER, false);
  MS_LOG(DEBUG) << "RunOpInMs end";
  return result_v;
}

void ForwardExecutor::ClearRes() {
  MS_LOG(DEBUG) << "Clear forward res";
  for (const auto &item : mindrt_backends_) {
    MS_EXCEPTION_IF_NULL(item.second);
    item.second->ClearOpExecutorResource();
  }
  node_abs_map_.clear();
  init_ = false;
  lazy_build_ = false;
  prim_abs_list_.clear();
  std::stack<CellPtr>().swap(forward_cell_stack_);
  session_backends_.clear();
  mindrt_backends_.clear();
  kNotConstPrimOrConstInput.clear();
}
}  // namespace pynative
}  // namespace mindspore
