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

#include "plugin/device/ascend/kernel/pyboost/baddbmm_ascend.h"
#include "plugin/device/ascend/kernel/opapi/aclnn/baddbmm_aclnn_kernel.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
bool BaddbmmAscend::Launch(const tensor::TensorPtr &input, const tensor::TensorPtr &batch1,
                           const tensor::TensorPtr &batch2, const ScalarPtr &beta, const ScalarPtr &alpha,
                           const tensor::TensorPtr &output) {
  BaddbmmAclnnFunctionalKernelMod kernel;
  kernel.Init(nullptr, false);
  return kernel.Call(input, batch1, batch2, beta, alpha, output);
}

tensor::TensorPtr BaddbmmAscend::Call(const tensor::TensorPtr &input, const tensor::TensorPtr &batch1,
                                      const tensor::TensorPtr &batch2, const ScalarPtr &beta, const ScalarPtr &alpha) {
  InferOutput(input, batch1, batch2, beta, alpha);
  Launch(input, batch1, batch2, beta, alpha, output_);
  return output_;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
