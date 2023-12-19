/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/pyboost/customize/copy.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/pyboost/py_boost_utils.h"
#include "plugin/device/ascend/kernel/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr CopyAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "Call start";
  auto input_abs = input_tensor->ToAbstract();
  auto output_abs = input_abs->Clone();
  op->set_input_abs({input_abs});
  op->set_output_abs(output_abs);

  std::vector<tensor::TensorPtr> outputs;
  PyBoostUtils::CreateOutputTensor(output_abs, &outputs);
  op->set_outputs(outputs);

  PyBoostUtils::PrepareOpInputs(op->device_context(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<pynative::PyBoostDeviceTask>([op, input_tensor]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    auto stream_ptr = device::ascend::AscendStreamMng::GetInstance().GetStream(kDefaultStreamIndex);
    // Inplace output need be front
    LAUNCH_ACLNN(aclnnInplaceCopy, device_context, stream_ptr, outputs[0], input_tensor);
    MS_LOG(DEBUG) << "Launch end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
