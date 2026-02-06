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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_floor_divide.h"
#include <memory>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceFloorDivideAscendCustomize(const tensor::TensorPtr &input_tensor,
                                                    const tensor::TensorPtr &other_tensor) {
  TensorToDevice(input_tensor, other_tensor);
  // Check Memory Partial Overlap
  mindspore::pyboost::CheckMemoryOverlap({input_tensor, other_tensor}, {input_tensor});
  // Inplace output need be front
  DISPATCH_ACLNN(aclnnInplaceFloorDivide, input_tensor, other_tensor);
  return input_tensor;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
