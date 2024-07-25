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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_SILENT_CHECK_V2_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_SILENT_CHECK_V2_H_

#include <vector>
#include <memory>
#include "ir/scalar.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "runtime/hardware/device_context_manager.h"
#include "kernel/pyboost/op_runner.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::BaseTensorPtr> SilentCheckV2AscendCustomize(
  const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &val, const BaseTensorPtr &input_grad,
  const BaseTensorPtr &sfda, const BaseTensorPtr &step, const Int64ImmPtr &c_min_steps_ptr,
  const FloatImmPtr &c_thresh_l1_ptr, const FloatImmPtr &c_coeff_l1_ptr, const FloatImmPtr &c_thresh_l2_ptr,
  const FloatImmPtr &c_coeff_l2_ptr, const Int64ImmPtr &npu_asd_detect_ptr);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_PYBOOST_CUSTOMIZE_SILENT_CHECK_V2_H_
