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

#include "plugin/device/ascend/kernel/pyboost/add_ascend.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "transform/acl_ir/op_api_exec.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr AddAscend::Call(const tensor::TensorPtr &x, const tensor::TensorPtr &y) {
  InferOutput(x, y);
  DeviceMalloc(x, y);
  auto stream = device_context_->device_res_manager_->GetStream(kDefaultStreamIndex);
  LAUNCH_ACLNN(aclnnAdd, stream, x, y, output(0));
  return output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
