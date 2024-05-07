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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_FFNEXT_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_FFNEXT_H_

#include <vector>
#include <memory>
#include "ir/tensor.h"
#include "ir/value.h"
#include "runtime/hardware/device_context_manager.h"
#include "kernel/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr FFNExtAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x_tensor, const BaseTensorPtr &weight1_tensor,
  const BaseTensorPtr &weight2_tensor, const std::optional<ValueTuplePtr> &expertTokens,
  const std::optional<BaseTensorPtr> &bias1_tensor, const std::optional<BaseTensorPtr> &bias2_tensor,
  const std::optional<BaseTensorPtr> &scale_tensor, const std::optional<BaseTensorPtr> &offset_tensor,
  const std::optional<BaseTensorPtr> &deqScale1_tensor, const std::optional<BaseTensorPtr> &deqScale2_tensor,
  const std::optional<BaseTensorPtr> &antiquant_scale1, const std::optional<BaseTensorPtr> &antiquant_scale2,
  const std::optional<BaseTensorPtr> &antiquant_offset1, const std::optional<BaseTensorPtr> &antiquant_offset2,
  const Int64ImmPtr &activation, const Int64ImmPtr &inner_precise);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_FFNEXT_H_
