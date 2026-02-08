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

#ifndef MINDSPORE_CCSRC_PYNATIVE_BACKWARD_SAVED_TENSOR_H_
#define MINDSPORE_CCSRC_PYNATIVE_BACKWARD_SAVED_TENSOR_H_

#include <memory>
#include "ir/anf.h"
#include "ir/tensor.h"
#include "include/utils/pynative/variable.h"
#include "include/utils/pynative/common_utils.h"
#include "pynative/backward/hook/saved_tensor_hook.h"

namespace mindspore::pynative::autograd {
class SavedTensor final : public Value {
 public:
  SavedTensor(const TensorPtr &tensor, bool is_output, bool is_view_inplace, size_t seq_nr, bool is_custom = false,
              bool force_no_recompute = false);
  SavedTensor(const TensorPtr &tensor, bool is_output, size_t seq_nr, bool is_custom = false);
  ValuePtr UnWrap(const BackwardNodePtr &saved_for);
  TensorPtr UnWrapToTensor(const BackwardNodePtr &saved_for);
  bool saved_original() const { return saved_original_; }
  void Clear();
  ~SavedTensor() override = default;
  bool operator==(const Value &other) const override { return other.isa<SavedTensor>() && &other == this; }
  MS_DECLARE_PARENT(SavedTensor, Value);

 private:
  void SaveMetaData(const TensorPtr &tensor);
  void CheckVersion(const std::string &grad_node_name) const;

  bool is_output_;
  bool is_view_inplace_;
  bool is_leaf_;
  bool is_custom_;
  bool requires_grad_{false};
  bool saved_original_{false};
  bool is_from_recompute_{false};

  size_t output_index_{0};
  size_t version_;
  size_t seq_nr_;

  TensorPtr data_;
  BackwardNodePtr grad_node_;
  std::weak_ptr<BackwardNode> weak_grad_node_;
  std::unique_ptr<PySavedTensorHook> saved_tensor_hook_;
};

using SavedTensorPtr = std::shared_ptr<SavedTensor>;
using SavedTensorPtrList = std::vector<SavedTensorPtr>;

ValuePtr ValueToSavedValue(const ValuePtr &input, size_t seq_nr, bool is_output, bool is_view_inplace = false);

ValuePtr SavedValueToValue(const ValuePtr &saved_value, const BackwardNodePtr &grad_node);

ValuePtrList SavedValueListToValueList(const ValuePtrList &saved_value_list, const BackwardNodePtr &grad_node);

SavedTensorPtrList GenerateCustomSavedTensor(const std::vector<tensor::TensorPtr> &to_saved_tensors,
                                             const TensorPtrSet &dirty_tensor_set, const BackwardNodePtr &grad_node);
}  // namespace mindspore::pynative::autograd
#endif
