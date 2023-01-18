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
#include "include/common/utils/convert_utils_py.h"

namespace mindspore {
namespace pynative {
namespace {
const mindspore::HashSet<std::string> kForceInferPrim = {prim::kTopK, prim::kDropoutGenMask,
                                                         prim::kStatelessDropOutGenMask};
constexpr size_t kCacheThreshold = 10000;

void SetAnyValue(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractTensor>()) {
    abs->set_value(kAnyValue);
  } else if (abs->isa<abstract::AbstractTuple>() || abs->isa<abstract::AbstractList>()) {
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    for (auto &item_abs : abs_seq->elements()) {
      MS_EXCEPTION_IF_NULL(item_abs);
      if (item_abs->isa<abstract::AbstractTensor>()) {
        item_abs->set_value(kAnyValue);
      }
    }
  }
}

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
    if (value->isa<AnyValue>()) {
      return value;
    }
    return tuple_abs->ElementsBuildValue<ValueTuple>();
  } else if (abs->isa<abstract::AbstractList>()) {
    auto list_abs = abs->cast<abstract::AbstractListPtr>();
    const auto &value = list_abs->BuildValue();
    if (value->isa<AnyValue>()) {
      return value;
    }
    return list_abs->ElementsBuildValue<ValueList>();
  } else if (abs->isa<abstract::AbstractRowTensor>()) {
    return abs->cast<abstract::AbstractRowTensorPtr>()->BuildValue();
  } else if (abs->isa<abstract::AbstractCOOTensor>()) {
    return abs->cast<abstract::AbstractCOOTensorPtr>()->BuildValue();
  } else if (abs->isa<abstract::AbstractCSRTensor>()) {
    return abs->cast<abstract::AbstractCSRTensorPtr>()->BuildValue();
  } else {
    MS_LOG(DEBUG) << "Unsupported abstract type for primitive, the abs is " << abs->ToString();
    return kAnyValue;
  }
}

void PynativeInfer(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  MS_LOG(DEBUG) << "Op " << op_run_info->base_op_run_info.op_name << " input abs "
                << mindspore::ToString(op_run_info->input_abs);
  const auto &prim = op_run_info->op_prim;
  MS_EXCEPTION_IF_NULL(prim);
  prim->BeginRecordAddAttr();
  auto eval_ret = EvalOnePrim(prim, op_run_info->input_abs);
  MS_EXCEPTION_IF_NULL(eval_ret);
  const auto &infer_res = eval_ret->abstract();
  MS_EXCEPTION_IF_NULL(infer_res);
  prim->EndRecordAddAttr();
  op_run_info->base_op_run_info.abstract = infer_res;
  MS_LOG(DEBUG) << "Op " << op_run_info->base_op_run_info.op_name << " infer result " << infer_res->ToString();
}
}  // namespace

ValuePtr InferOperation::DoInfer(const FrontendOpRunInfoPtr &op_run_info) {
  SetInputAbstract(op_run_info);
  return InferOutputAbstract(op_run_info);
}

void InferOperation::SetInputAbstract(const FrontendOpRunInfoPtr &op_run_info) {
  // Check whether constant flag exists.
  const auto &input_const_flag = CheckPrimitiveConstFlag(op_run_info);
  // Get input abstract by input value and set it to `op_run_info`.
  MS_EXCEPTION_IF_NULL(op_run_info);
  size_t input_value_size = op_run_info->input_value.size();
  for (size_t i = 0; i < input_value_size; ++i) {
    (void)op_run_info->input_abs.emplace_back(
      GetInputValueAbs(op_run_info, op_run_info->input_value[i], i, input_const_flag[i]));
  }
}

AbstractBasePtr InferOperation::GetInputValueAbs(const FrontendOpRunInfoPtr &op_run_info, const ValuePtr &input_value,
                                                 size_t input_index, bool marked_const) {
  // Get tuple or list abs
  MS_EXCEPTION_IF_NULL(input_value);
  const auto &input_id = PyNativeAlgo::Common::GetIdByValue(input_value);
  if (input_value->isa<ValueSequence>()) {
    const auto &tuple_value = input_value->cast<ValueSequencePtr>();
    return GetInputTupleValueAbstract(op_run_info, tuple_value, input_index, input_id, marked_const);
  }
  // Get non-tuple and non-list abs.
  const auto &dynamic_shape_info = PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->dynamic_shape();
  const auto &id_with_dynamic_abs = dynamic_shape_info->id_with_dynamic_abs();
  auto iter = id_with_dynamic_abs.find(input_id);
  if (iter != id_with_dynamic_abs.end()) {
    MS_EXCEPTION_IF_NULL(op_run_info);
    op_run_info->base_op_run_info.has_dynamic_input = true;
    MS_LOG(DEBUG) << "Get abstract of input " << input_index << " value is " << iter->second->ToString() << ", id "
                  << input_id;
    return iter->second;
  } else {
    const auto &abs = GetAbstractByValue(input_value, input_index, input_id, marked_const);
    MS_EXCEPTION_IF_NULL(abs);
    const auto &shape = abs->BuildShape();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->IsDynamic()) {
      op_run_info->base_op_run_info.has_dynamic_input = true;
    }
    MS_LOG(DEBUG) << "Get abstract of input " << input_index << " value is " << abs->ToString() << ", id " << input_id;
    return abs;
  }
}

AbstractBasePtr InferOperation::GetInputTupleValueAbstract(const FrontendOpRunInfoPtr &op_run_info,
                                                           const ValueSequencePtr &tuple_value, size_t input_index,
                                                           const std::string &input_id, bool marked_const) {
  if (!marked_const) {
    auto iter = node_abs_cache_.find(input_id);
    if (iter != node_abs_cache_.end()) {
      MS_LOG(DEBUG) << "The abstract of tuple input " << input_index << " hits cache.";
      return iter->second;
    }
  }
  // Create abstract list for tuple input.
  MS_EXCEPTION_IF_NULL(tuple_value);
  size_t tuple_value_size = tuple_value->size();
  abstract::AbstractBasePtrList abs_list(tuple_value_size);
  const auto &dynamic_shape_info = PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->dynamic_shape();
  const auto &id_with_dynamic_abs = dynamic_shape_info->id_with_dynamic_abs();
  for (size_t i = 0; i < tuple_value_size; ++i) {
    const auto &item = tuple_value->value()[i];
    const auto &item_id = PyNativeAlgo::Common::GetIdByValue(item);
    auto item_iter = id_with_dynamic_abs.find(item_id);
    if (item_iter != id_with_dynamic_abs.end()) {
      abs_list[i] = item_iter->second;
    } else {
      abs_list[i] = GetAbstractByValue(item, input_index, item_id, marked_const);
    }
  }
  // Create output abstract by value type.
  AbstractBasePtr abs;
  if (tuple_value->isa<ValueTuple>()) {
    abs = std::make_shared<abstract::AbstractTuple>(abs_list);
  } else {
    abs = std::make_shared<abstract::AbstractList>(abs_list);
  }
  // Update dynamic shape info.
  MS_EXCEPTION_IF_NULL(abs);
  const auto &shape = abs->BuildShape();
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->IsDynamic()) {
    op_run_info->base_op_run_info.has_dynamic_input = true;
  }
  MS_LOG(DEBUG) << "Get abstract of tuple input " << input_index << " is " << abs->ToString() << ", id " << input_id;
  return abs;
}

AbstractBasePtr InferOperation::GetAbstractByValue(const ValuePtr &value, size_t input_index,
                                                   const std::string &input_id, bool marked_const) {
  if (!marked_const) {
    auto iter = node_abs_cache_.find(input_id);
    if (iter != node_abs_cache_.end()) {
      MS_LOG(DEBUG) << "The abstract of input " << input_index << " hits cache.";
      return iter->second;
    }
  }
  // Get abstract by input value.
  MS_EXCEPTION_IF_NULL(value);
  const auto &abs = value->ToAbstract();
  if (!marked_const) {
    if (value->isa<tensor::Tensor>() || value->isa<mindspore::Type>()) {
      SetNodeAbsCacheById(input_id, abs);
    }
  }
  return abs;
}

ValuePtr InferOperation::InferOutputAbstract(const FrontendOpRunInfoPtr &op_run_info) {
  // Step 1 : Get output abstract from cache.
  bool abs_cache_hit = GetOutputAbstractByCache(op_run_info);

  // Step 2 : Infer output abstract when cache not hit.
  MS_EXCEPTION_IF_NULL(op_run_info);
  if (!abs_cache_hit || kForceInferPrim.find(op_run_info->base_op_run_info.op_name) != kForceInferPrim.end()) {
    PynativeInfer(op_run_info);
  }

  // Step 3: Check whether output shape is dynamic.
  const auto &shape = op_run_info->base_op_run_info.abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape);
  op_run_info->base_op_run_info.has_dynamic_output = shape->IsDynamic();
  MS_LOG(DEBUG) << "Op " << op_run_info->base_op_run_info.op_name << " is dynamic "
                << PyNativeAlgo::Common::IsDynamicShape(op_run_info);

  // Step 4: Get infer value from output abstract.
  auto infer_value = GetInferValueFromAbstract(op_run_info->base_op_run_info.abstract);
  MS_EXCEPTION_IF_NULL(infer_value);
  if (!infer_value->isa<AnyValue>()) {
    MS_LOG(DEBUG) << "Get output by constant folding, output is " << infer_value->ToString();
    op_run_info->output_get_by_infer_value = true;
  } else if (op_run_info->op_prim->is_const_prim()) {
    MS_LOG(DEBUG) << "Get output by const prim.";
    op_run_info->output_get_by_infer_value = true;
    infer_value = MakeValue("");
  } else if (!abs_cache_hit) {
    // Cache output abstract, the const infer value needs to infer every step.
    SaveOutputAbstractToCache(op_run_info);
  }
  return infer_value;
}

bool InferOperation::GetOutputAbstractByCache(const FrontendOpRunInfoPtr &op_run_info) const {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &prim = op_run_info->op_prim;
  MS_EXCEPTION_IF_NULL(prim);

  AbsCacheKey key{prim->name(), prim->Hash(), prim->attrs()};
  auto prim_iter = prim_abs_list_.find(key);
  if (prim_iter != prim_abs_list_.end()) {
    MS_LOG(DEBUG) << "Output abstract cache matched prim " << prim->name();
    const auto &input_abs_map = prim_iter->second;
    auto abs_iter = input_abs_map.find(op_run_info->input_abs);
    if (abs_iter != input_abs_map.end()) {
      MS_EXCEPTION_IF_NULL(abs_iter->second.abs);
      MS_LOG(DEBUG) << "From output abstract cache get output abs " << abs_iter->second.abs->ToString();
      op_run_info->base_op_run_info.abstract = abs_iter->second.abs;
      prim->set_evaluate_added_attrs(abs_iter->second.attrs);
      return true;
    }
  }
  return false;
}

void InferOperation::SaveOutputAbstractToCache(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &prim = op_run_info->op_prim;
  MS_EXCEPTION_IF_NULL(prim);
  AbsCacheKey key{prim->name(), prim->Hash(), prim->attrs()};
  auto &out = prim_abs_list_[key];
  out[op_run_info->input_abs].abs = op_run_info->base_op_run_info.abstract;
  out[op_run_info->input_abs].attrs = prim->evaluate_added_attrs();
}

std::vector<bool> InferOperation::CheckPrimitiveConstFlag(const FrontendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  size_t input_value_num = op_run_info->input_value.size();
  const auto &prim = op_run_info->op_prim;
  MS_EXCEPTION_IF_NULL(prim);

  if (no_const_flag_prims_.find(prim->name()) != no_const_flag_prims_.end()) {
    return std::vector<bool>(input_value_num, false);
  }
  // Check whether primitive has constant flag.
  if (prim->is_const_prim()) {
    MS_LOG(DEBUG) << "Op " << op_run_info->base_op_run_info.op_name << " has const prim flag.";
    return std::vector<bool>(input_value_num, true);
  }
  // Check whether input position has been marked constant.
  std::vector<bool> input_const_flag(input_value_num, false);
  const auto &const_input_index = prim->get_const_input_indexes();
  for (const auto &index : const_input_index) {
    input_const_flag[index] = true;
    MS_LOG(DEBUG) << "The input " << index << " value of op " << op_run_info->base_op_run_info.op_name
                  << " marked constant.";
  }
  // Cache no constant flag primitive.
  if (const_input_index.empty()) {
    (void)no_const_flag_prims_.emplace(prim->name());
  }
  return input_const_flag;
}

void InferOperation::SetNodeAbsCacheByValue(const ValuePtr &value, const abstract::AbstractBasePtr &abs) {
  SetNodeAbsCacheById(PyNativeAlgo::Common::GetIdByValue(value), abs);
  // If value is a `value tuple` or `value list`, cache the abstract of each element value.
  if (value->isa<ValueSequence>()) {
    const auto &seq_value = value->cast<ValueSequencePtr>();
    const auto &seq_abs = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq_abs);

    const auto &value_elems = seq_value->value();
    const auto &abs_elems = seq_abs->elements();
    size_t num = value_elems.size();
    MS_EXCEPTION_IF_CHECK_FAIL(num == abs_elems.size(), "The size of value is not equal to the size of abstract.");
    for (size_t i = 0; i < num; ++i) {
      SetNodeAbsCacheById(PyNativeAlgo::Common::GetIdByValue(value_elems[i]), abs_elems[i]);
    }
  }
}

void InferOperation::SetNodeAbsCacheById(const std::string &id, const abstract::AbstractBasePtr &abs) {
  // If Just call run op and have no cell or function running, node_abs_cache_ will not be clear.
  // So, set a threshold for clear it.
  if (only_single_op_run_ && node_abs_cache_.size() > kCacheThreshold) {
    node_abs_cache_.clear();
  }
  SetAnyValue(abs);
  node_abs_cache_[id] = abs;
}

py::object InferOperation::CallConstantFolding(const py::args &args) const {
  const auto &op_run_info = std::make_shared<FrontendOpRunInfo>();
  PyNativeAlgo::PyParser::SetPrim(op_run_info, args[0]);
  const auto &v = PyNativeAlgo::DataConvert::PyObjToValue(args[1]);
  (void)op_run_info->input_abs.emplace_back(v->ToAbstract());
  PynativeInfer(op_run_info);
  auto infer_value = GetInferValueFromAbstract(op_run_info->base_op_run_info.abstract);
  if (infer_value->isa<AnyValue>()) {
    MS_LOG(EXCEPTION) << "Can not get value from abstract";
  }
  return ValueToPyData(infer_value);
}
}  // namespace pynative
}  // namespace mindspore
