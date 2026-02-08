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

#include "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/silent_check_v2.h"
#include <cassert>
#include <memory>
#include <vector>
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::TensorPtr> SilentCheckV2AscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &val, const TensorPtr &input_grad, const TensorPtr &sfda,
  const TensorPtr &step, const Int64ImmPtr &c_min_steps_ptr, const FloatImmPtr &c_thresh_l1_ptr,
  const FloatImmPtr &c_coeff_l1_ptr, const FloatImmPtr &c_thresh_l2_ptr, const FloatImmPtr &c_coeff_l2_ptr,
  const Int64ImmPtr &npu_asd_detect_ptr) {
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
  op->set_outputs(std::vector<tensor::TensorPtr>{input_grad, sfda, step, op->output(kIndex3)});
  MS_LOG(INFO) << op->primitive()->name() << " Launch end";
  return op->outputs();
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
