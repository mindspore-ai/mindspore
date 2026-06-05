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

#include "mindspore/ops/kernel/cpu/pyboost/customize/randn.h"
#include "include/pynative/utils/pyboost/functions/composite/randn.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr RandnCPUCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &shape,
                                    const tensor::TensorPtr &seed, const tensor::TensorPtr &offset,
                                    const std::optional<Int64ImmPtr> &dtype, const std::optional<Int64ImmPtr> &device) {
  auto out = randn(shape, seed, offset, dtype, device);
  op->set_outputs({out});
  return out;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
