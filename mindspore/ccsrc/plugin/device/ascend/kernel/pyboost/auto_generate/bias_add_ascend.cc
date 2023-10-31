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

#include "plugin/device/ascend/kernel/pyboost/auto_generate/bias_add_ascend.h"
#include "plugin/device/ascend/kernel/pyboost/call/bias_add.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr BiasAddAscend::Call(const tensor::TensorPtr &input_x, const tensor::TensorPtr &bias) {
  InferOutput(input_x, bias);
  DeviceMalloc(input_x, bias);
  return BiasAddAscendCall(primitive_, Contiguous(input_x), Contiguous(bias), output(0));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
