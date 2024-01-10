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

#include "plugin/device/ascend/kernel/pyboost/customize/contiguous.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "kernel/pyboost/auto_generate/copy.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr ContiguousAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "Call start";
  MS_EXCEPTION_IF_NULL(input_tensor);

  auto old_storage_info = input_tensor->storage_info();
  if (old_storage_info == nullptr || old_storage_info->is_contiguous) {
    // tensor is contiguous, no need contiguous
    op->set_outputs({input_tensor});
    op->set_output_abs(input_tensor->ToAbstract());
    return input_tensor;
  }

  auto copy_op = CREATE_PYBOOST_OP(Copy, kAscendDevice);
  auto output_tensor = copy_op->Call(input_tensor);
  op->set_outputs(copy_op->outputs());
  op->set_output_abs(copy_op->output_abs());
  return output_tensor;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
