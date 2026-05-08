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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_fill_diagonal.h"

#include <memory>

#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceFillDiagonalAscendCustomize(const tensor::TensorPtr &input, const ScalarPtr &fill_value,
                                                     const BoolImmPtr &wrap) {
  TensorToDevice(input);
  auto wrap_imm = GetValue<bool>(wrap);
  DISPATCH_ACLNN(aclnnInplaceFillDiagonal, input, fill_value, wrap_imm);
  return input;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
