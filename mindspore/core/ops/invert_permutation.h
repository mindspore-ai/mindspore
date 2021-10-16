/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_INVERT_PERMUTATION_H_
#define MINDSPORE_CORE_OPS_INVERT_PERMUTATION_H_
#include <vector>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameInvertPermutation = "InvertPermutation";
/// \brief Computes the inverse of an index permutation.
/// Refer to Python API @ref mindspore.ops.InvertPermutation for more details.
class MS_CORE_API InvertPermutation : public PrimitiveC {
 public:
  /// \brief Constructor.
  InvertPermutation() : PrimitiveC(kNameInvertPermutation) {}
  /// \brief Destructor.
  ~InvertPermutation() = default;
  MS_DECLARE_PARENT(InvertPermutation, PrimitiveC);
};

using PrimInvertPermutationPtr = std::shared_ptr<InvertPermutation>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_INVERT_PERMUTATION_H_
