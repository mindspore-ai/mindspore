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
#include "plugin/device/ascend/kernel/internal/layernorm.h"

#include <memory>
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalLayerNorm::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  internal::NormParam op_param;
  op_param.normType = internal::NormParam::LAYER_NORM;
  op_param.beginNormAxis = inputs[kIndex3]->GetValueWithCheck<int64_t>();
  op_param.beginParamsAxis = inputs[kIndex4]->GetValueWithCheck<int64_t>();
  op_param.epsilon = inputs[kIndex5]->GetValueWithCheck<float>();
  op_param.inGamma = true;
  op_param.inBeta = true;
  op_param.outMean = true;
  op_param.outVarience = true;

  param_ptr->specificParam = op_param;
  param_ptr->opId = internal::OpId::LayerNorm;
  return param_ptr;
}

void InternalLayerNorm::SetInOutIdx() {
  inputsIdxMap_[0] = 0;
  inputsIdxMap_[1] = 1;
  inputsIdxMap_[2] = 2;
  outputsIdxMap_[0] = 0;
  outputsIdxMap_[1] = 1;
  outputsIdxMap_[2] = 2;
}

MS_INTERNAL_KERNEL_FACTORY_REG(LayerNorm, InternalLayerNorm);
}  // namespace kernel
}  // namespace mindspore
