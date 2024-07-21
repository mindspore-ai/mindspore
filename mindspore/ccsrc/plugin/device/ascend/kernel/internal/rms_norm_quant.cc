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

#include "plugin/device/ascend/kernel/internal/rms_norm_quant.h"

#include <memory>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalRmsNormQuant::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                         const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  internal::NormParam rmsnorm_param;
  param_ptr->opId = internal::OpId::RmsNormQuant;
  rmsnorm_param.normType = internal::NormParam::RMS_NORM;
  rmsnorm_param.inGamma = true;
  rmsnorm_param.inBeta = true;
  rmsnorm_param.outRes = false;
  rmsnorm_param.inRes = false;
  rmsnorm_param.inNormBias = false;

  if (primitive_->HasAttr("epsilon")) {
    auto value_str = primitive_->GetAttr("epsilon");
    MS_EXCEPTION_IF_NULL(value_str);
    float epsilon = GetValue<float>(value_str);
    rmsnorm_param.epsilon = epsilon;
  }

  param_ptr->specificParam = rmsnorm_param;
  return param_ptr;
}

MS_INTERNAL_KERNEL_FACTORY_REG(RmsNormQuant, InternalRmsNormQuant);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(RmsNormQuant, INPUT_NUM_5, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_4);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(RmsNormQuant, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
