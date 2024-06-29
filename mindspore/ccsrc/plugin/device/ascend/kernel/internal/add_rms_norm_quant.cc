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

#include "plugin/device/ascend/kernel/internal/add_rms_norm_quant.h"

#include <memory>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalAddRmsNormQuant::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                            const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  // setup param from inputs
  internal::AddRmsNormParam op_param;
  op_param.eps = inputs[kIndex5]->GetValueWithCheck<float>();

  param_ptr->specificParam = op_param;
  param_ptr->opId = internal::OpId::AddRmsNormQuant;
  MS_LOG(INFO) << "Create kernel: " << param_ptr->opId << " eps: " << op_param.eps;
  return param_ptr;
}

MS_INTERNAL_KERNEL_FACTORY_REG(AddRmsNormQuantV2, InternalAddRmsNormQuant);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(AddRmsNormQuantV2, INPUT_NUM_5, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_4);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(AddRmsNormQuantV2, OUTPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);
}  // namespace kernel
}  // namespace mindspore
