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

#include "kernel/gpu/pyboost/customize/pixel_shuffle.h"
#include "include/pynative/utils/pyboost/customize/pixel_shuffle.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr PixelShuffleGPUCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                           const Int64ImmPtr &upscale_factor) {
  PixelShuffleCustomize(op, input, upscale_factor);
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
