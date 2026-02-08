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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_BATCH_NORM_GATHER_STATS_WITH_COUNTS_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_BATCH_NORM_GATHER_STATS_WITH_COUNTS_H_
#include <vector>
#include <memory>
#include <tuple>
#include "ir/tensor.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/pynative/utils/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr> BatchNormGatherStatsWithCountsAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor, const TensorPtr &mean_tensor,
  const TensorPtr &invstd_tensor, const std::optional<TensorPtr> &running_mean_tensor,
  const std::optional<TensorPtr> &running_var_tensor, const FP32ImmPtr &momentum, const FP32ImmPtr &eps,
  const std::optional<TensorPtr> &counts_tensor);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_BATCH_NORM_GATHER_STATS_WITH_COUNTS_H_
