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
#include "ops_utils/memory_overlap.h"
#include "ir/tensor.h"

namespace mindspore {
MemOverlap IsInternalOverlap(const TensorPtr &variable_tensor) {
  // For tensor is contiguous, never has overlap in tensor.
  if (variable_tensor->is_contiguous()) {
    return MemOverlap::No;
  }

  // For broadcast_to case, there is overlap in tensor of course.
  const auto &strides = variable_tensor->storage_info()->strides;
  const auto &shape = variable_tensor->storage_info()->shape;
  if (strides.size() != shape.size()) {
    MS_LOG(EXCEPTION) << "Size of strides and shape are not equal:" << strides.size() << ", " << shape.size();
  }
  for (uint32_t i = 0; i < strides.size(); i++) {
    if (shape[i] > 1 && strides[i] == 0) {
      return MemOverlap::Yes;
    }
  }

  // Others, to hard to judge.
  return MemOverlap::TooHard;
}

// This function used to assert the input of an inplace operator,
// for which case the result is uncertain. In the follow case, value of d is uncertain.
// a = [[1], [2], [3]]
// b = mint.broadcast_to(a, (3,2))
// c = [[1, 2], [1, 2], [1, 2]]
// d = b.copy_ext(c)
void ThrowExpectionWhenInternalOverlap(const TensorPtr &variable_tensor) {
  if (IsInternalOverlap(variable_tensor) == MemOverlap::Yes) {
    MS_LOG(EXCEPTION) << "This tensor has multi element reference to the same memory address,"
                         "which is forbidden.You can clone it before execute the operation.";
  }
}

namespace {
const char *GetTensorData(const TensorPtr &tensor) {
  const auto tensor_begin = static_cast<const char *>(tensor->unsafe_data());
  MS_EXCEPTION_IF_NULL(tensor_begin);
  return tensor_begin + tensor->storage_offset() * static_cast<size_t>(tensor->DataItemSize());
}
}  // namespace
MemOverlapStatus GetOverlapStatus(const TensorPtr &a, const TensorPtr &b) {
  MS_EXCEPTION_IF_NULL(a);
  MS_EXCEPTION_IF_NULL(b);
  if (a.get() == b.get()) {
    return MemOverlapStatus::FULL;
  }
  if (a->DataSize() == 0 || b->DataSize() == 0) {
    return MemOverlapStatus::NO;
  }
  if (!a->is_contiguous() || !b->is_contiguous()) {
    return MemOverlapStatus::TOO_HARD;
  }
  if (a->unsafe_data() == b->unsafe_data()) {
    const auto a_begin = GetTensorData(a);
    const auto a_end = a_begin + a->DataSize() * static_cast<size_t>(a->DataItemSize());
    const auto b_begin = GetTensorData(b);
    const auto b_end = b_begin + b->DataSize() * static_cast<size_t>(b->DataItemSize());
    if (a_begin == b_begin && a_end == b_end) {
      return MemOverlapStatus::FULL;
    }
    if (a_begin < b_end && b_begin < a_end) {
      return MemOverlapStatus::PARTIAL;
    }
  }
  return MemOverlapStatus::NO;
}

// This function used to assert the input of an inplace operator,
// for which case the result is uncertain. In the follow case, value of d is uncertain.
// x = mint.rand(4, 4)
// x[1:].add_(x[:-1])
void ThrowExpectionWhenPartialOverlap(const TensorPtr &a, const TensorPtr &b) {
  if (GetOverlapStatus(a, b) == MemOverlapStatus::PARTIAL) {
    MS_LOG(EXCEPTION) << "Unsupported operations: some elements of the input tensor and "
                      << "the written-to tensor refer to a single memory location. "
                      << "Please clone() the tensor before performing the operation.";
  }
}

void CheckMemory(const std::vector<TensorPtr> &inputs, const std::vector<TensorPtr> &outputs) {
  for (const auto &output : outputs) {
    for (const auto &input : inputs) {
      ThrowExpectionWhenPartialOverlap(output, input);
    }
  }
}
}  // namespace mindspore
