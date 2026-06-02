/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "mindspore/ops/kernel/cpu/pyboost/customize/rand_like_ext.h"
#include "include/pynative/utils/pyboost/functions/composite/rand_like_ext.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr RandLikeExtCPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &tensor_tensor,
                                          const TensorPtr &seed, const TensorPtr &offset,
                                          const std::optional<Int64ImmPtr> &dtype,
                                          const std::optional<Int64ImmPtr> &device) {
  auto out = rand_like_ext(tensor_tensor, seed, offset, dtype, device);
  op->set_outputs({out});
  return out;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
