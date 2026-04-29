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
#include "kernel/ascend/aclnn/pyboost_impl/customize/copy.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/device_put.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr CopyAscendCustomize(const tensor::TensorPtr &input_tensor) {
  MS_LOG(DEBUG) << "Call start";

  std::vector<tensor::TensorPtr> outputs;
  PyBoostUtils::CreateOutputTensor(input_tensor->data_type(), input_tensor->shape(), &outputs);
  TensorToDevice(input_tensor);
  TensorAllocate(outputs);
  const auto &output = outputs[0];
  DISPATCH_ACLNN(aclnnInplaceCopy, output, input_tensor);
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
