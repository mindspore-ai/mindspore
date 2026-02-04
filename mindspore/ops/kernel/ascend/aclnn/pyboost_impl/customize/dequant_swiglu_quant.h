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

#ifndef MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_DEQUANT_SWIGLU_QUANT_H_
#define MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_DEQUANT_SWIGLU_QUANT_H_
#include <vector>
#include <memory>
#include "ir/tensor.h"
#include "ir/value.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/pynative/utils/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr DequantSwigluQuantAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &x, const std::optional<TensorPtr> &weight_scale,
  const std::optional<TensorPtr> &activation_scale, const std::optional<TensorPtr> &bias,
  const std::optional<TensorPtr> &quant_scale, const std::optional<TensorPtr> &quant_offset,
  const std::optional<TensorPtr> &group_index, const BoolImmPtr &activate_left, const Int64ImmPtr &quant_mode);

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_PYBOOST_CUSTOMIZE_DEQUANT_SWIGLU_QUANT_H_
