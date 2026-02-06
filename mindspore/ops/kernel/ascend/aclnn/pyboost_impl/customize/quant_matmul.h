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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CALL_QUANT_MATMUL_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CALL_QUANT_MATMUL_H_

#include <vector>
#include <memory>
#include "ir/tensor.h"
#include "ir/value.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/pynative/utils/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void QuantMatmulAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &x1, const TensorPtr &x2,
                                const TensorPtr &scale, const std::optional<TensorPtr> &offset,
                                const std::optional<TensorPtr> &pertoken_scale, const std::optional<TensorPtr> &bias,
                                const std::optional<Int64ImmPtr> &output_dtype,
                                const std::optional<Int64ImmPtr> &x1_dtype, const std::optional<Int64ImmPtr> &x2_dtype,
                                const std::optional<Int64ImmPtr> &pertoken_scale_dtype,
                                const std::optional<Int64ImmPtr> &scale_dtype,
                                const std::optional<ValueTuplePtr> &group_sizes);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CALL_QUANT_MATMUL_H_
