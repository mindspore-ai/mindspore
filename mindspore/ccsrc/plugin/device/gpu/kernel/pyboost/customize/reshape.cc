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

#include "plugin/device/gpu/kernel/pyboost/customize/reshape.h"
#include "kernel/pyboost/customize/reshape.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr ReshapeGPUCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                          const ValueTuplePtr &shape) {
  MS_LOG(DEBUG) << "Call start";
  return ReshapeCustomize(op, input_tensor, shape, op->device_context()->device_context_key_.device_name_);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
