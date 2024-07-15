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

#include "plugin/device/ascend/kernel/pyboost/customize/silent_check_v2.h"
#include <cassert>
#include <memory>
#include <vector>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/pyboost_utils.h"
#include "mindapi/base/types.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::BaseTensorPtr> SilentCheckV2AscendCustomize(
  const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &val, const BaseTensorPtr &input_grad,
  const BaseTensorPtr &sfda, const BaseTensorPtr &step, const Int64ImmPtr &c_min_steps_ptr,
  const FloatImmPtr &c_thresh_l1_ptr, const FloatImmPtr &c_coeff_l1_ptr, const FloatImmPtr &c_thresh_l2_ptr,
  const FloatImmPtr &c_coeff_l2_ptr, const Int64ImmPtr &npu_asd_detect_ptr) {
  MS_LOG(INFO) << op->primitive()->name() << "Call start";
  OpRunner::InferOpOutput(op, val, input_grad, sfda, step, c_min_steps_ptr, c_thresh_l1_ptr, c_coeff_l1_ptr,
                          c_thresh_l2_ptr, c_coeff_l2_ptr, npu_asd_detect_ptr);

  auto c_min_steps = GetValue<int64_t>(c_min_steps_ptr);
  auto c_thresh_l1 = GetValue<pyfloat>(c_thresh_l1_ptr);
  auto c_coeff_l1 = GetValue<pyfloat>(c_coeff_l1_ptr);
  auto c_thresh_l2 = GetValue<pyfloat>(c_thresh_l2_ptr);
  auto c_coeff_l2 = GetValue<pyfloat>(c_coeff_l2_ptr);
  auto npu_asd_detect = GetValue<int64_t>(npu_asd_detect_ptr);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), val, input_grad, sfda, step);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, val, input_grad, sfda, step, c_min_steps, c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2, npu_asd_detect]() {
      auto device_context = op->device_context();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(op->device_context(), val, input_grad, sfda, step);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(op->device_context(), op->outputs());
      LAUNCH_ACLNN(aclnnSilentCheck, device_context, op->stream_id(), val, input_grad, sfda, step, c_min_steps,
                   c_thresh_l1, c_coeff_l1, c_thresh_l2, c_coeff_l2, npu_asd_detect, op->output(kIndex3));
    }));
  op->set_outputs(std::vector<tensor::BaseTensorPtr>{input_grad, sfda, step, op->output(kIndex3)});
  MS_LOG(INFO) << op->primitive()->name() << " Launch end";
  return op->outputs();
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
