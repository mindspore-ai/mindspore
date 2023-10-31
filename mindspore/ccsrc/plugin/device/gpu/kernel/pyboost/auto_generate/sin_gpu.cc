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

#include "plugin/device/gpu/kernel/pyboost/auto_generate/sin_gpu.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr SinGPU::Call(const tensor::TensorPtr &x) {
  // TODO: add op
  MS_LOG(DEBUG) << "Call start";
  // Step1: add infer func: CPU/GPU/Ascend are same
  // InferOutput(input, batch1, batch2, beta, alpha);
  // Step2: add malloc func: CPU/GPU/Ascend are same
  // Don't need to allocate memory for Scalar.
  // DeviceMalloc(input, batch1, batch2);
  // Step3: add stream func: CPU/GPU/Ascend are same
  // auto stream_ptr = device_context_->device_res_manager_->GetStream(kDefaultStreamIndex);
  // Step4: add launc func: CPU/GPU/Ascend are different
  // Ascend: need to check cube_math_type, please refer to op document of CANN
  // Ascend: LAUNCH_ACLNN_CUBE(aclnnBaddbmm, stream_ptr, input, batch1, batch2, beta, alpha, output(0));
  // Ascend: LAUNCH_ACLNN(aclnnBaddbmm, stream_ptr, input, batch1, batch2, beta, alpha, output(0));
  // CPU: framework will provide the func
  // GPU: framework will provide the func
  MS_LOG(DEBUG) << "Launch end";
  return outputs_[0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
