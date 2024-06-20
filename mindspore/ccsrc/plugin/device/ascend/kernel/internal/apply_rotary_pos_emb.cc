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

#include "plugin/device/ascend/kernel/internal/apply_rotary_pos_emb.h"
#include "param/apply_rotary_pos_emb_param.h"
#include <memory>
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr ApplyRotaryPosEmb::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  auto param_ptr = std::make_shared<internal::ApplyRotaryPosEmbParam>();
  param_ptr->opId = internal::OpId::ApplyRotaryPosEmb;

  auto cos_format_ptr = inputs.at(kIndex5);
  if (cos_format_ptr->dtype_id() == TypeId::kNumberTypeInt64) {
    param_ptr->cosFormat = static_cast<int32_t>(cos_format_ptr->GetValue<int64_t>().value());
  } else {
    MS_LOG(EXCEPTION) << "ApplyRotaryPosEmb input[5] dtype is not kNumberTypeInt64";
  }
  auto rotary_coeff_ptr = inputs.at(kIndex6);
  if (rotary_coeff_ptr->dtype_id() == TypeId::kNumberTypeInt64) {
    param_ptr->rotaryCoeff = static_cast<int32_t>(rotary_coeff_ptr->GetValue<int64_t>().value());

  } else {
    MS_LOG(EXCEPTION) << "ApplyRotaryPosEmb input[6] dtype is not kNumberTypeInt64";
  }

  param_ptr->queryDims = internal::VecToSVec<int64_t>(inputs[kIndex0]->GetShapeVector());
  param_ptr->keyDims = internal::VecToSVec<int64_t>(inputs[kIndex1]->GetShapeVector());

  internal::MixParam op_param;
  op_param.mixType = internal::MixParam::MixType::MIX_ROPE;
  op_param.cosFormat = param_ptr->cosFormat;
  op_param.rotaryCoeff = param_ptr->rotaryCoeff;
  // setup rope param from inputs
  param_ptr->specificParam = op_param;
  return param_ptr;
}

MS_INTERNAL_KERNEL_FACTORY_REG(ApplyRotaryPosEmb, ApplyRotaryPosEmb);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(ApplyRotaryPosEmb, INPUT_NUM_5, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_4);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(ApplyRotaryPosEmb, OUTPUT_NUM_2, INDEX_0, INDEX_1);
}  // namespace kernel
}  // namespace mindspore
