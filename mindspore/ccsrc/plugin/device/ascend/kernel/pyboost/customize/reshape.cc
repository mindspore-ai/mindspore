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

#include "plugin/device/ascend/kernel/pyboost/customize/reshape.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "kernel/pyboost/auto_generate/copy.h"
#include "kernel/pyboost/auto_generate/view.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr ReshapeAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                         const ValueTuplePtr &shape) {
  MS_LOG(DEBUG) << "Call start";
  MS_EXCEPTION_IF_NULL(input_tensor);

  auto view_op = CREATE_PYBOOST_OP(View, kAscendDevice);
  auto old_storage_info = input_tensor->storage_info();
  if (old_storage_info == nullptr || old_storage_info->is_contiguous) {
    // Tensor is contiguous, reshape by view
    auto output_tensor = view_op->Call(input_tensor, shape);
    op->set_output_abs(view_op->output_abs());
    op->set_outputs(view_op->outputs());
    return output_tensor;
  }

  // Tensor is not contiguous, need call copy first
  auto copy_op = CREATE_PYBOOST_OP(Copy, kAscendDevice);
  const auto copy_tensor = copy_op->Call(input_tensor);
  auto output_tensor = view_op->Call(copy_tensor, shape);
  op->set_output_abs(view_op->output_abs());
  op->set_outputs(view_op->outputs());
  return output_tensor;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
