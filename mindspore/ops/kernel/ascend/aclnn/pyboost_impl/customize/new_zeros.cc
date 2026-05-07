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

#include "kernel/ascend/aclnn/pyboost_impl/customize/new_zeros.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/zeros.h"
#include "include/pynative/utils/pyboost/op_register.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr NewZerosAscendCustomize(const TensorPtr &input_tensor, const ValueTuplePtr &size,
                                          const std::optional<Int64ImmPtr> &dtype) {
  MS_LOG(DEBUG) << "NewZeros Call start";
  std::optional<Int64ImmPtr> use_dtype;
  if (dtype.has_value()) {
    use_dtype.emplace(dtype.value());
  } else {
    auto tensor_type = input_tensor->Dtype()->type_id();
    use_dtype.emplace(std::make_shared<Int64Imm>(tensor_type));
  }

  auto zeros_op = CREATE_PYBOOST_OP(Zeros, device::DeviceType::kAscend);
  auto zeros_out = zeros_op->Call(size, use_dtype);

  MS_LOG(DEBUG) << "NewZeros Call end";
  return zeros_out;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
