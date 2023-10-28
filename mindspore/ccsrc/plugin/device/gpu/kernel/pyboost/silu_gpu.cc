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

#include "plugin/device/gpu/kernel/pyboost/add_gpu.h"
#include "mindspore/ccsrc/plugin/device/gpu/kernel/math/binary_ops_gpu_kernel.h"
#include "runtime/device/device_address_utils.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "ops/add.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr SiLUGPU::Call(const tensor::TensorPtr &x) {
  // TODO
  InferOutput(x);
  return outputs_[0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
