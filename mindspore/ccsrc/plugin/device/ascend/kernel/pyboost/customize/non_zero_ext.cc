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

#include "plugin/device/ascend/kernel/pyboost/customize/non_zero_ext.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/customize/op_common.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "runtime/device/device_address_utils.h"
#include "kernel/pyboost/op_runner.h"
#include "kernel/pyboost/op_register.h"
#include "kernel/pyboost/auto_generate/non_zero.h"
#include "kernel/pyboost/auto_generate/unstack_ext.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::BaseTensorPtr> NonZeroExtAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                             const BaseTensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "NonZeroExt call start";
  MS_EXCEPTION_IF_NULL(input_tensor);
  op->GenerateInputAbstract(input_tensor);
  auto nonzero_op = CREATE_PYBOOST_OP(NonZero, kAscendDevice);
  auto unstack_op = CREATE_PYBOOST_OP(UnstackExt, kAscendDevice);
  const auto output_tensor = nonzero_op->Call(input_tensor);
  auto output_tuple = unstack_op->Call(output_tensor, std::make_shared<Int64Imm>(1));
  op->set_output_abs(unstack_op->output_abs());
  op->set_outputs(unstack_op->outputs());
  MS_LOG(DEBUG) << "NonZeroExt call end";
  return output_tuple;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
