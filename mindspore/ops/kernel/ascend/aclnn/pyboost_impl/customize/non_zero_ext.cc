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

#include "kernel/ascend/aclnn/pyboost_impl/customize/non_zero_ext.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/customize/op_common.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "include/pynative/utils/pyboost/op_runner.h"
#include "include/pynative/utils/pyboost/op_register.h"
#include "include/pynative/utils/pyboost/auto_generate/non_zero.h"
#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"
#include "include/pynative/utils/pyboost/functions/auto_grad_guard.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::TensorPtr> NonZeroExtAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                         const TensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "NonZeroExt call start";
  MS_EXCEPTION_IF_NULL(input_tensor);
  kernel::pyboost::RequireGradGuard require_grad_guard(false);
  auto nonzero_op = CREATE_PYBOOST_OP(NonZero, device::DeviceType::kAscend);
  TensorPtr output_tensor = nullptr;
  if (input_tensor->shape().size() == kDim0) {
    std::vector<int64_t> unsqueeze_shape;
    unsqueeze_shape.emplace_back(kIndex1);
    auto expanded_input = reshape(input_tensor, unsqueeze_shape);
    output_tensor = nonzero_op->Call(expanded_input);
  } else {
    output_tensor = nonzero_op->Call(input_tensor);
  }
  auto output_tuple = unstack_ext_view(output_tensor, 1);
  op->set_outputs(output_tuple);
  MS_LOG(DEBUG) << "NonZeroExt call end";
  return output_tuple;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
