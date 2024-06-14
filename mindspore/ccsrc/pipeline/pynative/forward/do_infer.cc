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

#include "pipeline/pynative/forward/do_infer.h"
#include "pipeline/pynative/pynative_utils.h"
#include "frontend/operator/ops_front_infer_function.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/common/profiler.h"
#include "ops/nn_op_name.h"
#include "ops/ops_frontend_func_impl.h"

namespace mindspore {
namespace pynative {
namespace {
constexpr size_t kCacheThreshold = 10000;

ValuePtr GetInferValueFromAbstract(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractTensor>()) {
    return abs->cast<abstract::AbstractTensorPtr>()->BuildValue();
  } else if (abs->isa<abstract::AbstractSlice>()) {
    return abs->cast<abstract::AbstractSlicePtr>()->BuildValue();
  } else if (abs->isa<abstract::AbstractScalar>() || abs->isa<abstract::AbstractType>()) {
    return abs->BuildValue();
  } else if (abs->isa<abstract::AbstractTuple>()) {
    auto tuple_abs = abs->cast<abstract::AbstractTuplePtr>();
    const auto &value = tuple_abs->BuildValue();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<ValueAny>()) {
      return value;
    }
    return tuple_abs->ElementsBuildValue<ValueTuple>();
  } else if (abs->isa<abstract::AbstractList>()) {
    auto list_abs = abs->cast<abstract::AbstractListPtr>();
    const auto &value = list_abs->BuildValue();
    if (value->isa<ValueAny>()) {
      return value;
    }
    return list_abs->ElementsBuildValue<ValueList>();
  } else if (abs->isa<abstract::AbstractRowTensor>()) {
    return abs->cast<abstract::AbstractRowTensorPtr>()->BuildValue();
  } else if (abs->isa<abstract::AbstractCOOTensor>()) {
    return abs->cast<abstract::AbstractCOOTensorPtr>()->BuildValue();
  } else if (abs->isa<abstract::AbstractCSRTensor>()) {
    return abs->cast<abstract::AbstractCSRTensorPtr>()->BuildValue();
  } else if (abs->isa<abstract::AbstractMapTensor>()) {
    return kValueAny;
  } else {
    MS_LOG(DEBUG) << "Unsupported abstract type for primitive, the abs is " << abs->ToString();
    return kValueAny;
  }
}

void CallPyInferFunc(const PrimitivePtr &primitive, const FrontendOpRunInfoPtr &op_run_info) {
  const AbstractBasePtrList &arg_spec = op_run_info->op_grad_info->input_abs;
  auto py_infer_args = PreparePyInputs(arg_spec);
  auto prim_py = dyn_cast<PrimitivePy>(primitive);
  MS_EXCEPTION_IF_NULL(prim_py);
  if (primitive->prim_type() == kPrimTypePyCheck) {
    prim_py->RunCheck(py_infer_args);
    return;
  }
  auto py_infer_result = prim_py->RunInfer(py_infer_args);
  auto abs = abstract::PyInferRes2Abstract(prim_py, py_infer_result);
  primitive->EndRecordAddAttr();
  op_run_info->base_op_run_info.abstract = abs;
}

std::optional<abstract::StandardPrimitiveImplReg> GetPyNativePrimitiveInferImpl(const PrimitivePtr &primitive) {
  auto iter = abstract::GetFrontendPrimitiveInferMap().find(primitive);
  if (iter != abstract::GetFrontendPrimitiveInferMap().end()) {
    return iter->second;
  }

  return abstract::GetPrimitiveInferImpl(primitive);
}
}  // namespace

bool InferByOpDef(const FrontendOpRunInfoPtr &op_run_info) {
  const auto &prim = op_run_info->op_grad_info->op_prim;
  auto frontend_func_impl = mindspore::ops::GetOpFrontendFuncImplPtr(prim->name());
  if (frontend_func_impl) {
    op_run_info->base_op_run_info.abstract =
      frontend_func_impl->InferAbstract(prim, op_run_info->op_grad_info->input_abs);
    if (op_run_info->base_op_run_info.abstract != nullptr) {
      MS_LOG(DEBUG) << "Pynative Infer by InferAbstract, got abstract: "
                    << op_run_info->base_op_run_info.abstract->ToString();
      return true;
    }
  }

  auto op_def = mindspore::ops::GetOpDef(prim->name());
  if (op_def) {
    (void)op_def->func_impl_.CheckValidation(prim, op_run_info->op_grad_info->input_abs);
    auto shape = op_def->func_impl_.InferShape(prim, op_run_info->op_grad_info->input_abs);
    auto type = op_def->func_impl_.InferType(prim, op_run_info->op_grad_info->input_abs);
    op_run_info->base_op_run_info.abstract = mindspore::abstract::MakeAbstract(shape, type);
    MS_LOG(DEBUG) << "Pynative Infer by OpDef, got abstract: " << op_run_info->base_op_run_info.abstract->ToString();
    return true;
  }

  return false;
}

void InferOperation::PynativeInfer(const FrontendOpRunInfoPtr &op_run_info) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_LOG(DEBUG) << "Op " << op_run_info->base_op_run_info.op_name
                << " infer input: " << mindspore::ToString(op_run_info->op_grad_info->input_abs);
  const auto &prim = op_run_info->op_grad_info->op_prim;
  MS_EXCEPTION_IF_NULL(prim);
  op_run_info->base_op_run_info.abstract = nullptr;

  prim->BeginRecordAddAttr();

  if (InferByOpDef(op_run_info)) {
    prim->EndRecordAddAttr();
    return;
  }

  auto eval_impl = GetPyNativePrimitiveInferImpl(prim);
  if (eval_impl.has_value()) {
    // the WhileList ops should be constant fold in Pynative mode.
    if (!eval_impl->IsInWhiteList() && eval_impl->IsImplInferValue()) {
      auto value = eval_impl->InferValue(prim, op_run_info->op_grad_info->input_abs);
      if (value != nullptr && !value->isa<ValueAny>()) {
        op_run_info->base_op_run_info.abstract = value->ToAbstract();
        prim->EndRecordAddAttr();
        return;
      }
    }

    op_run_info->base_op_run_info.abstract =
      eval_impl->InferShapeAndType(nullptr, prim, op_run_info->op_grad_info->input_abs);
    prim->EndRecordAddAttr();
    return;
  }

  // Only cache the abstract when the primitive should call the python code.
  if (GetOutputAbstractByCache(op_run_info)) {
    prim->EndRecordAddAttr();
    return;
  }

  // call python infer
  py::gil_scoped_acquire acquire;
  CallPyInferFunc(prim, op_run_info);
  MS_EXCEPTION_IF_NULL(op_run_info->base_op_run_info.abstract);
  prim->EndRecordAddAttr();
}

void InferOperation::DoInfer(const FrontendOpRunInfoPtr &op_run_info) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyNativeInfer,
                                     op_run_info->base_op_run_info.op_name, true);
  if (op_run_info->is_view_op) {
    if (op_run_info->requires_grad) {
      SetInputAbstract(op_run_info);
    }
    return;
  }
  SetInputAbstract(op_run_info);
  if (!op_run_info->is_view_op) {
    InferOutputAbstract(op_run_info);
  }
}

void InferOperation::SetInputAbstract(const FrontendOpRunInfoPtr &op_run_info) {
  // Get input abstract by input value and set it to `op_run_info`.
  MS_EXCEPTION_IF_NULL(op_run_info);
  op_run_info->input_value_id.resize(op_run_info->input_size);
  op_run_info->op_grad_info->input_abs.resize(op_run_info->input_size);
  for (size_t i = 0; i < op_run_info->input_size; ++i) {
    op_run_info->op_grad_info->input_abs[i] =
      GetInputValueAbs(op_run_info, op_run_info->op_grad_info->input_value[i], i);
  }
}

AbstractBasePtr InferOperation::GetInputValueAbs(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &input_value,
                                                 size_t input_index) {
  // Get tuple or list abs
  MS_EXCEPTION_IF_NULL(input_value);
  if (input_value->isa<tensor::BaseTensor>()) {
    op_run_info->input_value_id[input_index] = PyNativeAlgo::Common::GetIdByValue(input_value);
  }
  if (input_value->isa<ValueSequence>()) {
    const auto &tuple_value = input_value->cast<ValueSequencePtr>();
    return GetInputTupleValueAbstract(op_run_info, tuple_value, input_index);
  }
  // Get non-tuple and non-list abs.
  const auto &abs = GetAbstractByValue(input_value, input_index, op_run_info->input_value_id[input_index]);
  MS_LOG(DEBUG) << "Get abstract of input " << input_index << " is " << abs->ToString() << ", id "
                << op_run_info->input_value_id[input_index];
  return abs;
}

AbstractBasePtr InferOperation::GetInputTupleValueAbstract(const FrontendOpRunInfoPtr &op_run_info,
                                                           const ValueSequencePtr &tuple_value, size_t input_index) {
  // Create abstract list for tuple input.
  MS_EXCEPTION_IF_NULL(tuple_value);
  size_t tuple_value_size = tuple_value->size();
  abstract::AbstractBasePtrList abs_list(tuple_value_size);
  for (size_t i = 0; i < tuple_value_size; ++i) {
    const auto &item = tuple_value->value()[i];
    const auto &item_id = (item->isa<tensor::BaseTensor>() ? PyNativeAlgo::Common::GetIdByValue(item) : "");
    abs_list[i] = GetAbstractByValue(item, input_index, item_id);
  }
  // Create output abstract by value type.
  AbstractBasePtr abs;
  if (tuple_value->isa<ValueTuple>()) {
    abs = std::make_shared<abstract::AbstractTuple>(abs_list);
  } else {
    abs = std::make_shared<abstract::AbstractList>(abs_list);
  }
  MS_LOG(DEBUG) << "Get abstract of tuple input " << input_index << " is " << abs->ToString() << ", id "
                << op_run_info->input_value_id[input_index];
  return abs;
}

AbstractBasePtr InferOperation::GetAbstractByValue(const ValuePtr &value, size_t input_index,
                                                   const std::string &input_id) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>()) {
    auto cache_abs = GetNodeAbsById(input_id);
    if (cache_abs != nullptr) {
      MS_LOG(DEBUG) << "The abstract of input " << input_index << " hits cache.";
      return cache_abs;
    }
  }

  // Get abstract by input value.
  const auto &abs = value->ToAbstract();
  if (value->isa<tensor::BaseTensor>()) {
    SetNodeAbsById(input_id, PyNativeAlgo::Common::SetAbstractValueToAnyValue(abs));
  }
  return abs;
}

void InferOperation::InferOutputAbstract(const FrontendOpRunInfoPtr &op_run_info) {
  // Step 1 : Infer output abstract.
  MS_EXCEPTION_IF_NULL(op_run_info);
  PynativeInfer(op_run_info);
  MS_EXCEPTION_IF_NULL(op_run_info->base_op_run_info.abstract);
  MS_LOG(DEBUG) << "Op " << op_run_info->base_op_run_info.op_name
                << " infer result: " << op_run_info->base_op_run_info.abstract->ToString();
  // Step 2: Check whether output shape is dynamic.
  const auto &shape = op_run_info->base_op_run_info.abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape);
  op_run_info->base_op_run_info.has_dynamic_output = shape->IsDynamic();
  MS_LOG(DEBUG) << "Op " << op_run_info->base_op_run_info.op_name << " is dynamic "
                << op_run_info->base_op_run_info.has_dynamic_output;

  // Step 3: Get infer value from output abstract.
  auto infer_value = GetInferValueFromAbstract(op_run_info->base_op_run_info.abstract);
  MS_EXCEPTION_IF_NULL(infer_value);
  if (!infer_value->ContainsValueAny()) {
    MS_LOG(DEBUG) << "Get output by constant folding, output is " << infer_value->ToString();
    op_run_info->output_get_by_infer_value = true;
    op_run_info->should_be_cache = false;
  } else if (op_run_info->op_grad_info->op_prim->const_prim()) {
    MS_LOG(DEBUG) << "Get output by const prim.";
    op_run_info->output_get_by_infer_value = true;
    op_run_info->should_be_cache = false;
    infer_value = MakeValue("");
  } else if (op_run_info->should_be_cache) {
    // Cache output abstract, the const infer value needs to infer every step.
    SaveOutputAbstractToCache(op_run_info);
  }
  op_run_info->real_out = infer_value;
}

bool InferOperation::GetOutputAbstractByCache(const FrontendOpRunInfoPtr &op_run_info) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &prim = op_run_info->op_grad_info->op_prim;
  MS_EXCEPTION_IF_NULL(prim);

  AbsCacheKey key{prim->name(), prim->Hash(), prim->attrs()};
  auto prim_iter = prim_abs_list_.find(key);
  if (prim_iter != prim_abs_list_.end()) {
    MS_LOG(DEBUG) << "Output abstract cache matched prim " << prim->name();
    const auto &input_abs_map = prim_iter->second;
    auto abs_iter = input_abs_map.find(op_run_info->op_grad_info->input_abs);
    if (abs_iter != input_abs_map.end()) {
      MS_EXCEPTION_IF_NULL(abs_iter->second.abs);
      MS_LOG(DEBUG) << "From output abstract cache get output abs " << abs_iter->second.abs->ToString();
      op_run_info->base_op_run_info.abstract = abs_iter->second.abs;
      prim->set_evaluate_added_attrs(abs_iter->second.attrs);
      op_run_info->should_be_cache = false;
      return true;
    }
  }
  op_run_info->should_be_cache = true;
  return false;
}

void InferOperation::SaveOutputAbstractToCache(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &prim = op_run_info->op_grad_info->op_prim;
  MS_EXCEPTION_IF_NULL(prim);
  AbsCacheKey key{prim->name(), prim->Hash(), prim->attrs()};
  auto &out = prim_abs_list_[key];
  out[op_run_info->op_grad_info->input_abs].abs = op_run_info->base_op_run_info.abstract;
  out[op_run_info->op_grad_info->input_abs].attrs = prim->evaluate_added_attrs();
}

void InferOperation::SetNodeAbsCacheByValue(const FrontendOpRunInfoPtr &op_run_info) {
  SetNodeAbsById(op_run_info->out_value_id,
                 PyNativeAlgo::Common::SetAbstractValueToAnyValue(op_run_info->base_op_run_info.abstract));
  // If value is a `value tuple` or `value list`, cache the abstract of each element value.
  if (op_run_info->real_out->isa<ValueSequence>()) {
    const auto &seq_value = op_run_info->real_out->cast<ValueSequencePtr>();
    const auto &seq_abs = op_run_info->base_op_run_info.abstract->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq_abs);

    const auto &value_elems = seq_value->value();
    const auto &abs_elems = seq_abs->elements();
    size_t num = value_elems.size();
    if (num != abs_elems.size()) {
      SaveSpecifiedOutputToCache(op_run_info->base_op_run_info.op_name, value_elems, abs_elems);
      MS_LOG(DEBUG) << "The size of value " << num << " is not equal to the size of abstract " << abs_elems.size();
      return;
    }
    for (size_t i = 0; i < num; ++i) {
      SetNodeAbsById(PyNativeAlgo::Common::GetIdByValue(value_elems[i]), abs_elems[i]);
    }
  }
  // If Just call run op and have no cell or function running, node_abs_cache_ will not be clear.
  // So, set a threshold for clear it.
  if (node_abs_cache_.size() > kCacheThreshold) {
    std::unique_lock lock(abs_mutex_);
    auto half_it = node_abs_cache_.begin();
    std::advance(half_it, kCacheThreshold / 2);
    node_abs_cache_.erase(node_abs_cache_.begin(), half_it);
  }
}

void InferOperation::SaveSpecifiedOutputToCache(const std::string &op_name, const ValuePtrList &value_list,
                                                const AbstractBasePtrList &abs_list) {
  if (value_list.empty() || abs_list.empty()) {
    return;
  }
  // BatchNormal forward only use first output
  if (op_name == kBatchNormOpName) {
    SetNodeAbsById(PyNativeAlgo::Common::GetIdByValue(value_list[0]), abs_list[0]);
  }
}

void InferOperation::SetNodeAbsCacheById(const std::string &id, const abstract::AbstractBasePtr &abs) {
  SetNodeAbsById(id, PyNativeAlgo::Common::SetAbstractValueToAnyValue(abs));
}

void InferOperation::UpdateNodeAbsCacheById(const std::string &id, const abstract::AbstractBasePtr &abs) {
  std::unique_lock lock(abs_mutex_);
  (void)node_abs_cache_.erase(id);
  node_abs_cache_[id] = abs;
}

AbstractBasePtr InferOperation::GetNodeAbsById(const std::string &id) const {
  // GetNodeAbsById is used in NewGraph, need to release gil to avoid deadlock.
  GilReleaseWithCheck release_gil;
  std::shared_lock lock(abs_mutex_);
  auto iter = node_abs_cache_.find(id);
  if (iter == node_abs_cache_.end()) {
    return nullptr;
  }
  return iter->second;
}

void InferOperation::SetNodeAbsById(const std::string &id, const abstract::AbstractBasePtr &abs) {
  std::unique_lock lock(abs_mutex_);
  node_abs_cache_[id] = abs;
}

py::object InferOperation::CallConstantFolding(const py::args &args) const {
  const auto &op_run_info = std::make_shared<FrontendOpRunInfo>();
  PyNativeAlgo::PyParser::SetPrim(op_run_info, args[0]);
  op_run_info->base_op_run_info.op_name = op_run_info->op_grad_info->op_prim->name();
  const auto &v = PyNativeAlgo::DataConvert::PyObjToValue(args[1]);
  (void)op_run_info->op_grad_info->input_abs.emplace_back(v->ToAbstract());
  PynativeInfer(op_run_info);
  auto infer_value = GetInferValueFromAbstract(op_run_info->base_op_run_info.abstract);
  if (infer_value->ContainsValueAny()) {
    MS_LOG(EXCEPTION) << "Can not get value from abstract";
  }
  return PyNativeAlgo::DataConvert::ValueToPyObj(infer_value);
}
}  // namespace pynative
}  // namespace mindspore
