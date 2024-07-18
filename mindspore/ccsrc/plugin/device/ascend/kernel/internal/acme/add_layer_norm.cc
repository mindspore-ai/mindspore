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

#include "plugin/device/ascend/kernel/internal/acme/add_layer_norm.h"

#include <memory>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
acme::AcmeOpPtr AcmeAddLayerNorm::CreateKernel(const acme::InputsImmutableInfoList &inputs_ii,
                                               const acme::OutputsImmutableInfoList &outputs_ii,
                                               const std::vector<KernelTensor *> &ms_inputs,
                                               const std::vector<KernelTensor *> &ms_outputs) {
  auto beginNormAxis = ms_inputs[kIndex4]->GetValueWithCheck<int64_t>();
  auto beginParamsAxis = ms_inputs[kIndex5]->GetValueWithCheck<int64_t>();
  if (beginNormAxis != -1 || beginParamsAxis != -1) {
    MS_LOG(EXCEPTION) << "beginNormAxis and beginParamsAxis must both be -1, but get beginNormAxis: '" << beginNormAxis
                      << " and beginParamsAxis: " << beginParamsAxis;
  }
  acme::NormParam param;
  param.eps = ms_inputs[kIndex6]->GetValueWithCheck<float>();

  MS_LOG(INFO) << "Create kernel: " << acme::kAcmeAddLayerNormOpName << " eps: " << param.eps;
  return acme::CreateAddLayerNormOp(inputs_ii, outputs_ii, param, acme::kAcmeAddLayerNormOpName);
}
}  // namespace kernel
}  // namespace mindspore
