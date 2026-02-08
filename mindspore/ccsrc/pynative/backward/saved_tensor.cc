/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "pynative/backward/saved_tensor.h"
#include <memory>
#include "pynative/backward/hook/custom_function.h"

namespace mindspore::pynative::autograd {
static constexpr const char *kOutputSaver = "_OutputSaver";
namespace {
inline bool isFromTensor(const TensorPtr &tensor) {
  return tensor->source_type() == ops::DT_BEGIN || tensor->source_type() == ops::DT_TENSOR;
}

TensorPtr UnwrapRecomputeTensor(const BackwardNodePtr &grad_node) {
  auto py_node = std::static_pointer_cast<PyBackwardNode>(grad_node);
  if (MS_UNLIKELY(py_node->GetSavedTensors().size() != kSizeOne)) {
    MS_LOG(EXCEPTION) << "Output Saver tensors size should be one but got " << py_node->GetSavedTensors().size();
  }
  const auto &saved_tensor = py_node->GetSavedTensors()[0];
  auto src_tensor = saved_tensor->UnWrapToTensor(grad_node);
  if (saved_tensor->saved_original()) {
    MS_LOG(DEBUG) << "Used cached tensor, " << src_tensor->ToString();
    return src_tensor;
  }
  // Cached real tensor to output saver node to solve recompute output tensor used by multi operations.
  auto cached_tensor = std::make_shared<SavedTensor>(src_tensor, false, false, py_node->seq_id(), true, true);
  py_node->SetSavedTensors({cached_tensor});
  return src_tensor;
}
}  // namespace

SavedTensor::SavedTensor(const TensorPtr &tensor, bool is_output, bool is_view_inplace, size_t seq_nr, bool is_custom,
                         bool force_no_recompute)
    : is_output_(is_output), is_view_inplace_(is_view_inplace), is_custom_(is_custom), seq_nr_(seq_nr) {
  is_leaf_ = tensor->is_leaf();
  version_ = tensor->version().current_version();
  MS_LOG(DEBUG) << "Pack Saved Tensor for tensor T" << std::to_string(tensor->id()) << ", is_output: " << is_output
                << ", is_view_inplace: " << is_view_inplace << ", is_leaf_: " << is_leaf_
                << ", version: " << std::to_string(version_) << ", seq_nr: " << std::to_string(seq_nr_);
  if (is_view_inplace) {
    weak_grad_node_ = impl::GetUnsafeGradNodeImpl(tensor);
  }
  // Saved tensor hook has higher priority than recompute output tensor.
  saved_tensor_hook_ = DefaultSavedTensorHookUtil::IsEnabled() ? DefaultSavedTensorHookUtil::GetTopHook() : nullptr;
  if (saved_tensor_hook_ && isFromTensor(tensor)) {
    SaveMetaData(tensor);
    saved_tensor_hook_->RunPackHook(tensor);
    if (tensor->version().current_version() != version_) {
      MS_LOG(EXCEPTION) << "Pack hook inputs cannot be modified in-place, as this may cause unexpected side effects.";
    }
    return;
  }
  auto grad_node = impl::GetUnsafeGradNodeImpl(tensor);
  if (!force_no_recompute && grad_node != nullptr && grad_node->name() == kOutputSaver) {
    is_from_recompute_ = true;
    grad_node_ = grad_node;
    return;
  }
  if (!is_output_ || is_leaf_) {
    data_ = tensor;
    saved_original_ = true;
    return;
  }

  SaveMetaData(tensor);
  data_ = CommonUtils::ShallowCopyAndDetachForTensor(tensor);
}

SavedTensor::SavedTensor(const TensorPtr &tensor, bool is_output, size_t seq_nr, bool is_custom)
    : SavedTensor(tensor, is_output, false, seq_nr, is_custom, false) {}

ValuePtr SavedTensor::UnWrap(const BackwardNodePtr &saved_for) { return UnWrapToTensor(saved_for); }

TensorPtr SavedTensor::UnWrapToTensor(const BackwardNodePtr &saved_for) {
  MS_EXCEPTION_IF_NULL(saved_for);
  MS_LOG(DEBUG) << "UnWrap Saved Tensor for " << saved_for->UniqueId();
  if (is_from_recompute_) {
    MS_LOG(DEBUG) << "Try to unwrap from output_recompute output tensor. " << grad_node_->ToString();
    return UnwrapRecomputeTensor(grad_node_);
  }
  BackwardNodePtr gn;
  if (is_view_inplace_) {
    gn = weak_grad_node_.lock();
  } else if (saved_tensor_hook_ == nullptr) {
    gn = saved_original_ ? impl::GetUnsafeGradNodeImpl(data_) : nullptr;
  } else {
    gn = grad_node_;
  }

  if (!is_leaf_ && gn == nullptr) {
    if (!is_output_) {
      MS_LOG(EXCEPTION) << "Trying to use a saved tensor that has been detached in-place, This is not supported, "
                           "please use out-of-place detach instead";
    }
    gn = saved_for;
  }

  if (saved_tensor_hook_ == nullptr) {
    CheckVersion(saved_for->name());
  }

  if (saved_original_) {
    return data_;
  }
  auto data =
    saved_tensor_hook_ ? CommonUtils::ShallowCopyAndDetachForTensor(saved_tensor_hook_->RunUnpackHook()) : data_;
  // recover
  auto auto_grad_meta_data = std::make_shared<AutoGradMetaData>(gn);
  if (is_leaf_) {
    auto_grad_meta_data->set_requires_grad(requires_grad_);
  }
  auto_grad_meta_data->set_output_index(output_index_);
  auto_grad_meta_data->set_grad_node(gn);
  data->set_auto_grad_meta_data(auto_grad_meta_data);
  return data;
}

void SavedTensor::Clear() {
  saved_tensor_hook_.reset();
  grad_node_.reset();
  data_.reset();
}

void SavedTensor::SaveMetaData(const TensorPtr &tensor) {
  output_index_ = tensor->output_index();
  if (is_leaf_) {
    requires_grad_ = tensor->requires_grad();
  }
  if (!is_output_) {
    grad_node_ = impl::GetUnsafeGradNodeImpl(tensor);
  }
}

void SavedTensor::CheckVersion(const std::string &grad_node_name) const {
  auto cur_version = data_->version().current_version();
  if (cur_version != version_) {
    std::stringstream error_msg;
    error_msg << "One of the Tensor needed for gradient computation has been "
                 "modified by an inplace operation, which is the ";
    if (is_custom_) {
      error_msg << std::to_string(seq_nr_) << " 's custom saved tensor of ";
    } else if (is_output_) {
      error_msg << std::to_string(seq_nr_) << " 's output of ";
    } else {
      error_msg << std::to_string(seq_nr_) << " 's input of ";
    }
    error_msg << grad_node_name << "; and its version is " << cur_version << " , expected version " << version_;
    MS_LOG(EXCEPTION) << error_msg.str();
  }
}

ValuePtr ValueToSavedValue(const ValuePtr &input, size_t seq_nr, bool is_output, bool is_view_inplace) {
  MS_EXCEPTION_IF_NULL(input);
  if (input->isa<tensor::Tensor>()) {
    auto tensor = input->cast<tensor::TensorPtr>();
    if (!tensor->used_in_bprop_graph()) {
      // when tensor is op output, must create a new tensor here:
      // Using the original tensor would introduce a cycle
      // (tensor -> grad_node -> tensor), causing a circular reference.
      // when the grad_node has not been executed,
      // the circular reference will not be broken, which can lead to a memory leak.
      return is_output ? CommonUtils::ShallowCopyAndDetach(tensor) : tensor;
    }
    return std::make_shared<SavedTensor>(tensor, is_output, is_view_inplace, seq_nr);
  }

  if (input->isa<ValueSequence>()) {
    auto &values = input->cast<ValueSequencePtr>()->value();

    ValuePtrList res;
    res.reserve(values.size());
    for (auto &value : values) {
      (void)res.emplace_back(ValueToSavedValue(value, seq_nr, is_output, is_view_inplace));
    }
    if (input->isa<ValueTuple>()) {
      return std::make_shared<ValueTuple>(res);
    }
    return std::make_shared<ValueList>(res);
  }

  if (input->isa<ValueDictionary>()) {
    auto &key_value = input->cast<ValueDictionaryPtr>()->value();
    std::vector<std::pair<ValuePtr, ValuePtr>> res;
    res.reserve(key_value.size());
    for (auto &[key, value] : key_value) {
      (void)res.emplace_back(key, ValueToSavedValue(value, seq_nr, is_output, is_view_inplace));
    }
    return std::make_shared<ValueDictionary>(res);
  }

  return input;
}

ValuePtr SavedValueToValue(const ValuePtr &saved_value, const BackwardNodePtr &grad_node) {
  MS_EXCEPTION_IF_NULL(saved_value);
  if (saved_value->isa<SavedTensor>()) {
    return saved_value->cast<SavedTensorPtr>()->UnWrap(grad_node);
  }

  if (saved_value->isa<Tensor>() || saved_value->isa<Scalar>()) {
    return saved_value;
  }

  if (saved_value->isa<ValueSequence>()) {
    auto values = saved_value->cast<ValueSequencePtr>()->value();

    ValuePtrList res;
    res.reserve(values.size());
    for (const auto &value : values) {
      (void)res.emplace_back(SavedValueToValue(value, grad_node));
    }

    if (saved_value->isa<ValueTuple>()) {
      return std::make_shared<ValueTuple>(res);
    }
    return std::make_shared<ValueList>(res);
  }

  if (saved_value->isa<ValueDictionary>()) {
    auto &key_value = saved_value->cast<ValueDictionaryPtr>()->value();
    std::vector<std::pair<ValuePtr, ValuePtr>> res;
    res.reserve(key_value.size());
    for (auto &[key, value] : key_value) {
      (void)res.emplace_back(key, SavedValueToValue(value, grad_node));
    }
    return std::make_shared<ValueDictionary>(res);
  }

  return saved_value;
}

ValuePtrList SavedValueListToValueList(const ValuePtrList &saved_value_list, const BackwardNodePtr &grad_node) {
  ValuePtrList res;
  res.reserve(saved_value_list.size());
  for (const auto &value : saved_value_list) {
    (void)res.emplace_back(SavedValueToValue(value, grad_node));
  }
  return res;
}

SavedTensorPtrList GenerateCustomSavedTensor(const std::vector<TensorPtr> &to_saved_tensors,
                                             const TensorPtrSet &dirty_tensor_set, const BackwardNodePtr &grad_node) {
  SavedTensorPtrList saved_tensors;
  saved_tensors.reserve(to_saved_tensors.size());
  for (size_t i = 0; i < to_saved_tensors.size(); i++) {
    const auto &saved_tensor = to_saved_tensors[i];
    if (saved_tensor == nullptr) {
      saved_tensors.emplace_back(nullptr);
      continue;
    }
    bool is_output = impl::GetUnsafeGradNodeImpl(saved_tensor) == grad_node;
    bool is_view_inplace = false;
    if (!is_output) {
      is_view_inplace =
        dirty_tensor_set.count(saved_tensor) > 0 && impl::GetViewAutogradMetaImpl(saved_tensor) != nullptr;
    }
    saved_tensors.emplace_back(
      std::make_shared<SavedTensor>(saved_tensor, is_view_inplace ? true : is_output, is_view_inplace, i, true));
  }
  return saved_tensors;
}

}  // namespace mindspore::pynative::autograd
