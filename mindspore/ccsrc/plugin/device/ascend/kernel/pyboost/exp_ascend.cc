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

#include "plugin/device/ascend/kernel/pyboost/exp_ascend.h"
#include <algorithm>
#include <functional>
#include <memory>
#include "ir/tensor.h"
#include "transform/acl_ir/op_api_exec.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr ExpAscend::Call(const tensor::TensorPtr &x) {
  MS_LOG(DEBUG) << "Call start";
  InferOutput(x);
  // Don't need to allocate memory for Scalar.
  DeviceMalloc(x);
  auto stream_ptr = device_context_->device_res_manager_->GetStream(kDefaultStreamIndex);
  LAUNCH_ACLNN(aclnnExp, stream_ptr, x, output(0));
  MS_LOG(DEBUG) << "Launch end";
  return outputs_[0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
