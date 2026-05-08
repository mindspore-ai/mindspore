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

#include "kernel/ascend/aclnn/pyboost_impl/customize/inplace_remainder_tensor_scalar.h"
#include <memory>
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr InplaceRemainderTensorScalarAscendCustomize(const tensor::TensorPtr &input_tensor,
                                                              const ScalarPtr &other) {
  TensorToDevice(input_tensor);
  DISPATCH_ACLNN(aclnnInplaceRemainderTensorScalar, input_tensor, other);
  return input_tensor;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
