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
 * limitations under the License.plugin/device/cpu/hal/device
 */

#include "plugin/device/gpu/kernel/pyboost/customize/grouped_matmul.h"
#include <memory>
#include <functional>
#include "kernel/pyboost/pyboost_utils.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void GroupedMatmulGPUCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &x_tensor_list,
                               const ValueTuplePtr &weight_tensor_list,
                               const std::optional<ValueTuplePtr> &bias_tensor_list,
                               const std::optional<ValueTuplePtr> &scale_tensor_list,
                               const std::optional<ValueTuplePtr> &offset_tensor_list,
                               const std::optional<ValueTuplePtr> &antiquant_scale_tensor_list,
                               const std::optional<ValueTuplePtr> &antiquant_offset_tensor_list,
                               const std::optional<BaseTensorPtr> &group_list, const Int64ImmPtr &split_item,
                               const Int64ImmPtr &group_type) {
  MS_LOG(DEBUG) << "Call start";

  MS_LOG(DEBUG) << "Launch end";
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
