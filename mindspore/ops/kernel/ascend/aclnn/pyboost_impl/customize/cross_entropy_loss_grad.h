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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_CROSS_ENTROPY_LOSS_GRAD_GRAD_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_CROSS_ENTROPY_LOSS_GRAD_H_

#include <memory>
#include <tuple>
#include "ir/tensor.h"
#include "ir/scalar.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/pynative/utils/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr CrossEntropyLossGradAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &grad_loss, const TensorPtr &log_prob, const TensorPtr &target,
  const std::optional<TensorPtr> &weight, const std::optional<TensorPtr> &grad_zloss,
  const std::optional<TensorPtr> &lse_for_zloss, const Int64ImmPtr &reduction, const Int64ImmPtr &ignore_index,
  const FP32ImmPtr &label_smoothing, const FP32ImmPtr &lse_square_scale_for_zloss);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_CROSS_ENTROPY_LOSS_GRAD_H_
