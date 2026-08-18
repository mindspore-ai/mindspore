/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_CUSTOMIZE_GROUPED_MATMUL_V5_H_
#define MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_CUSTOMIZE_GROUPED_MATMUL_V5_H_

#include <memory>
#include <vector>
#include "ir/tensor.h"
#include "ir/value.h"

namespace mindspore::kernel::pyboost {
std::vector<tensor::TensorPtr> GroupedMatmulV5AscendCustomize(
  const ValueTuplePtr &x, const ValueTuplePtr &weight, const std::optional<ValueTuplePtr> &bias,
  const std::optional<ValueTuplePtr> &scale, const std::optional<ValueTuplePtr> &offset,
  const std::optional<ValueTuplePtr> &antiquant_scale, const std::optional<ValueTuplePtr> &antiquant_offset,
  const std::optional<ValueTuplePtr> &per_token_scale, const std::optional<tensor::TensorPtr> &group_list,
  const std::optional<ValueTuplePtr> &activation_input, const std::optional<ValueTuplePtr> &activation_quant_scale,
  const std::optional<ValueTuplePtr> &activation_quant_offset, const Int64ImmPtr &split_item,
  const Int64ImmPtr &group_type, const Int64ImmPtr &group_list_type, const Int64ImmPtr &act_type,
  const std::optional<ValueTuplePtr> &tuning_config, const std::optional<Int64ImmPtr> &output_dtype,
  const std::optional<Int64ImmPtr> &scale_dtype, const std::optional<Int64ImmPtr> &per_token_scale_dtype);
}  // namespace mindspore::kernel::pyboost

#endif  // MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_CUSTOMIZE_GROUPED_MATMUL_V5_H_
