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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_CUSTOMIZE_GROUPED_MATMUL_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_CUSTOMIZE_GROUPED_MATMUL_H_

#include <memory>
#include "ir/tensor.h"
#include "ir/value.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/pynative/utils/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void GroupedMatmulCPUCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &x_tensor_list,
                               const ValueTuplePtr &weight_tensor_list,
                               const std::optional<ValueTuplePtr> &bias_tensor_list,
                               const std::optional<ValueTuplePtr> &scale_tensor_list,
                               const std::optional<ValueTuplePtr> &offset_tensor_list,
                               const std::optional<ValueTuplePtr> &antiquant_scale_tensor_list,
                               const std::optional<ValueTuplePtr> &antiquant_offset_tensor_list,
                               const std::optional<TensorPtr> &group_list, const Int64ImmPtr &split_item,
                               const Int64ImmPtr &group_type, const mindspore::BoolImmPtr &transpose_a,
                               const mindspore::BoolImmPtr &transpose_b);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_CUSTOMIZE_GROUPED_MATMUL_H_
