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

#include "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/silent_check_v3.h"
#include <cassert>
#include <memory>
#include <vector>
#include "mindapi/base/shape_vector.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::vector<tensor::TensorPtr> SilentCheckV3AscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &val,
                                                            const TensorPtr &max, const TensorPtr &avg,
                                                            const TensorPtr &input_grad, const TensorPtr &step,
                                                            const FloatImmPtr &c_thresh_l1,
                                                            const FloatImmPtr &c_thresh_l2, const FloatImmPtr &beta1,
                                                            const Int64ImmPtr &npu_asd_detect) {
  MS_LOG(INFO) << op->primitive()->name() << "Call start";
  OpRunner::InferOpOutput(op, val, max, avg, input_grad, step, c_thresh_l1, c_thresh_l2, beta1, npu_asd_detect);

  auto c_thresh_l1_value = GetValue<pyfloat>(c_thresh_l1);
  auto c_thresh_l2_value = GetValue<pyfloat>(c_thresh_l2);
  auto beta1_value = GetValue<pyfloat>(beta1);
  auto npu_asd_detect_value = GetValue<int64_t>(npu_asd_detect);

  op->set_outputs(std::vector<tensor::TensorPtr>{avg, input_grad, step, op->output(kIndex3)});
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), val, max, avg, input_grad, step);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), std::vector<TensorPtr>{op->output(kIndex3)});

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, val, max, avg, input_grad, step, c_thresh_l1_value, c_thresh_l2_value, beta1_value, npu_asd_detect_value]() {
      auto device_context = op->device_context();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(op->device_context(), val, max, avg, input_grad, step);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(op->device_context(), std::vector<TensorPtr>{op->output(kIndex3)});
      LAUNCH_ACLNN(aclnnSilentCheckV2, device_context, op->stream_id(), val, max, avg, input_grad, step,
                   input_grad->shape_c(), input_grad->stride(), ShapeVector({SizeToLong(input_grad->storage_offset())}),
                   c_thresh_l1_value, c_thresh_l2_value, beta1_value, npu_asd_detect_value, op->output(kIndex3));
    }));

  MS_LOG(INFO) << op->primitive()->name() << " Launch end";
  return op->outputs();
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
