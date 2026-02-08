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

#ifndef MINDSPORE_OPS_OPS_UTILS_MEMORY_OVERLAP_H
#define MINDSPORE_OPS_OPS_UTILS_MEMORY_OVERLAP_H

#include <vector>
#include "ir/tensor.h"

namespace mindspore {
using TensorPtr = tensor::TensorPtr;
enum class MemOverlap { No, Yes, TooHard };
enum class MemOverlapStatus { FULL, PARTIAL, NO, TOO_HARD };

/// \brief To judge tensor whether there is memory over lap, only for view.
/// \param[in] variable_tensor The tensor to be judged.
/// \return No:no overlap, Yes:has overlap, TooHard:too hard to judge.
MS_CORE_API MemOverlap IsInternalOverlap(const TensorPtr &variable_tensor);

/// \brief throw expcetion when there is overlap in tensor, used for tensor of inplace operator.
MS_CORE_API void ThrowExpectionWhenInternalOverlap(const TensorPtr &variable_tensor);

MS_CORE_API MemOverlapStatus GetOverlapStatus(const TensorPtr &a, const TensorPtr &b);
MS_CORE_API void ThrowExpectionWhenPartialOverlap(const TensorPtr &a, const TensorPtr &b);
MS_CORE_API void CheckMemory(const std::vector<TensorPtr> &inputs, const std::vector<TensorPtr> &outputs);
}  // namespace mindspore

#endif  // MINDSPORE_OPS_OPS_UTILS_MEMORY_OVERLAP_H
