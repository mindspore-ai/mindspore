/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "mindspore/core/ops/infer_info/value_infer_info_adapter.h"
#include <algorithm>
#include <string>
#include "abstract/abstract_value.h"
#include "utils/log_adapter.h"

namespace mindspore::ops {
ShapeVector ValueInferInfoAdapter::GetShape() {
  if (MS_UNLIKELY(IsSequence())) {
    MS_LOG(EXCEPTION) << "Calling GetShape() on a sequence, " << BaseDebugInfo();
  }
  if (MS_UNLIKELY(IsNone())) {
    MS_LOG(EXCEPTION) << "Calling GetShape() on a None object, " << BaseDebugInfo();
  }
  if (value_->isa<tensor::Tensor>()) {
    auto tensor = value_->cast<tensor::TensorPtr>();
    return tensor->shape();
  } else if (value_->isa<Scalar>()) {
    return {};
  } else {
    MS_LOG(EXCEPTION) << "Calling GetShape() on unsupported value type '" << value_->type_name() << "',"
                      << BaseDebugInfo();
  }
}

bool ValueInferInfoAdapter::IsDynamic() { return false; }

bool ValueInferInfoAdapter::IsDynamicRank() { return false; }

TypeId ValueInferInfoAdapter::GetType() {
  if (MS_UNLIKELY(IsSequence())) {
    MS_LOG(EXCEPTION) << "Calling GetType() on a sequence, " << BaseDebugInfo();
  }
  if (MS_UNLIKELY(IsNone())) {
    MS_LOG(EXCEPTION) << "Calling GetType() on a None object, " << BaseDebugInfo();
  }
  if (value_->isa<tensor::Tensor>()) {
    auto tensor = value_->cast<tensor::TensorPtr>();
    return tensor->data_type();
  } else if (value_->isa<Scalar>()) {
    auto type_ptr = value_->type();
    MS_EXCEPTION_IF_NULL(type_ptr);
    return type_ptr->type_id();
  } else {
    MS_LOG(EXCEPTION) << "Calling GetType() on unsupported value type, " << BaseDebugInfo();
  }
}

bool ValueInferInfoAdapter::IsNone() {
  RETURN_IF_OPTIONAL_HAS_VALUE(is_none_);
  is_none_ = value_ == mindspore::kNone;
  return is_none_.value();
}

bool ValueInferInfoAdapter::IsSequence() {
  RETURN_IF_OPTIONAL_HAS_VALUE(is_sequence_);
  is_sequence_ = value_->isa<ValueSequence>();
  return is_sequence_.value();
}

std::vector<InferInfoPtr> ValueInferInfoAdapter::GetSequenceElements() {
  if (MS_UNLIKELY(!IsSequence())) {
    MS_LOG(EXCEPTION) << "Calling GetSequenceElements() on a non-sequence, " << BaseDebugInfo();
  }
  auto value_sequence = value_->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(value_sequence);
  auto elements = value_sequence->value();
  std::vector<InferInfoPtr> infer_infos;
  size_t idx = 0;
  std::transform(elements.begin(), elements.end(), std::back_inserter(infer_infos), [this, &idx](ValuePtr element) {
    return std::make_unique<ValueInferInfoAdapter>(element, op_type_, arg_name_ + "_" + std::to_string(idx++));
  });
  return infer_infos;
}

bool ValueInferInfoAdapter::IsDynamicSequence() { return false; }

InferInfoPtr ValueInferInfoAdapter::GetDynamicSequenceElement() {
  MS_LOG(EXCEPTION) << "Calling GetDynamicSequenceElement() on a non-dynamic sequence, " << BaseDebugInfo();
}

ValuePtr ValueInferInfoAdapter::GetValuePtr() { return value_; }

AbstractBasePtr ValueInferInfoAdapter::GetAbstractPtr() { return nullptr; }

const std::string &ValueInferInfoAdapter::BaseDebugInfo() { return base_debug_info_; }

std::string ValueInferInfoAdapter::DebugInfo() { return BaseDebugInfo() + " -> " + GetValuePtr()->ToString(); }
}  // namespace mindspore::ops
