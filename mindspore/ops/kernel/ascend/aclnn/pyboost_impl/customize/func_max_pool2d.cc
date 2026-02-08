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

#include "kernel/ascend/aclnn/pyboost_impl/customize/func_max_pool2d.h"

#include <memory>

#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void FuncMaxPool2DAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                  const ValueTuplePtr &kernel_size, const std::optional<ValueTuplePtr> &stride,
                                  const ValueTuplePtr &padding, const ValueTuplePtr &dilation,
                                  const BoolImmPtr &ceil_mode, const BoolImmPtr &return_indices) {
  TensorPtr out;
  TensorPtr indices;
  const auto &real_stride = stride.value_or(kernel_size);
  auto return_indices_value = return_indices->value();
  static const auto argmax_type = std::make_shared<Int64Imm>(static_cast<int64_t>(kNumberTypeInt64));
  if (return_indices_value) {
    std::tie(out, indices) =
      max_pool_with_indices(input_tensor, kernel_size, real_stride, padding, dilation, ceil_mode, argmax_type);
  } else {
    std::tie(out, indices) =
      max_pool_with_mask(input_tensor, kernel_size, real_stride, padding, dilation, ceil_mode, argmax_type);
  }
  op->set_outputs({out, indices});
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
