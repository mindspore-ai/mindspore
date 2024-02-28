/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#include <memory>
#include "plugin/device/ascend/kernel/internal/rmsnorm.h"
namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalRmsNorm::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  internal::NormParam rmsnorm_param;
  param_ptr->opId = internal::OpId::RmsNorm;

  if (primitive_->HasAttr("epsilon")) {
    auto value_str = primitive_->GetAttr("epsilon");
    MS_EXCEPTION_IF_NULL(value_str);

    rmsnorm_param.normType = internal::NormParam::RMS_NORM_FORWARD;
    float epsilon = GetValue<float>(value_str);
    rmsnorm_param.epsilon = epsilon;
    rmsnorm_param.inGamma = true;
  }

  param_ptr->specificParam = rmsnorm_param;
  return param_ptr;
}
void InternalRmsNorm::SetInOutIdx() {
  inputsIdxMap_[0] = 0;
  inputsIdxMap_[1] = 1;
  outputsIdxMap_[0] = 0;
  outputsIdxMap_[1] = 1;
}

MS_INTERNAL_KERNEL_FACTORY_REG(RmsNorm, InternalRmsNorm);
}  // namespace kernel
}  // namespace mindspore
