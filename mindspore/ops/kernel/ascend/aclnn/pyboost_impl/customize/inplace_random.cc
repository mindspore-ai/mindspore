/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_random.h"

#include <limits>

#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "ops_utils/type_dispatch.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
template <typename T>
void GetTypeLimit(int64_t *limit) {
  if constexpr (std::is_floating_point_v<T>) {
    constexpr int64_t base = 1;
    *limit = (base << std::numeric_limits<T>::digits) + 1;
  } else {
    *limit = std::numeric_limits<T>::max();
  }
}
}  // namespace

tensor::TensorPtr InplaceRandomAscendCustomize(const tensor::TensorPtr &tensor_tensor, const Int64ImmPtr from,
                                               const std::optional<Int64ImmPtr> &to, const tensor::TensorPtr &seed,
                                               const tensor::TensorPtr &offset) {
  auto [seed_imm, offset_imm] = GetGeneratorState(seed, offset);
  auto from_imm = GetValueWithCheck<int64_t>(from);
  int64_t to_imm;
  if (to.has_value()) {
    to_imm = GetValueWithCheck<int64_t>(to.value());
  } else {
    TYPE_DISPATCH_ALL(tensor_tensor->Dtype()->type_id(), "GetTypeLimit",
                      [&]() -> void { GetTypeLimit<scalar_t>(&to_imm); });
  }
  TensorToDevice(tensor_tensor);

  DISPATCH_ACLNN(aclnnInplaceRandom, tensor_tensor, from_imm, to_imm, seed_imm, offset_imm);
  return tensor_tensor;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
