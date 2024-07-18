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

#include "plugin/device/ascend/kernel/internal/flash_attention_score.h"

#include <memory>
#include <string>
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "param/attention_param.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {

internal::OpParamPtr InternalFlashAttentionScore::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                                const std::vector<KernelTensor *> &outputs) {
  auto param_ptr = std::make_shared<internal::FlashAttentionScoreParam>();
  // setup param from inputs
  auto q_shape = inputs[kIndex0]->GetShapeVector();
  auto kv_shape = inputs[kIndex1]->GetShapeVector();

  int64_t head_num = primitive_->HasAttr("head_num") ? GetValue<int64_t>(primitive_->GetAttr("head_num"))
                                                     : inputs[kIndex10]->GetValueWithCheck<int64_t>();
  int64_t head_dim = q_shape[kDim2] / head_num;

  int64_t pre_tokens = primitive_->HasAttr("pre_tokens") ? GetValue<int64_t>(primitive_->GetAttr("pre_tokens"))
                                                         : inputs[kIndex13]->GetValueWithCheck<int64_t>();
  int64_t next_tokens = primitive_->HasAttr("next_tokens") ? GetValue<int64_t>(primitive_->GetAttr("next_tokens"))
                                                           : inputs[kIndex14]->GetValueWithCheck<int64_t>();

  param_ptr->head_num = head_num;
  param_ptr->inner_precise = primitive_->HasAttr("inner_precise")
                               ? GetValue<int64_t>(primitive_->GetAttr("inner_precise"))
                               : inputs[kIndex15]->GetValueWithCheck<int64_t>();
  param_ptr->pre_tokens = pre_tokens;
  param_ptr->next_tokens = next_tokens;
  param_ptr->sparse_mode = primitive_->HasAttr("sparse_mode") ? GetValue<int64_t>(primitive_->GetAttr("sparse_mode"))
                                                              : inputs[kIndex17]->GetValueWithCheck<int64_t>();

  param_ptr->mask_dtype_ = InternalKernelUtils::ToInternalDType(inputs[kIndex6]->dtype_id());
  param_ptr->mask_dims_ = internal::VecToSVec<int64_t>(inputs[kIndex6]->GetShapeVector());

  internal::MixParam op_param;

  if (soc_ == "ascend310p") {
    op_param.mixType = internal::MixParam::MixType::MIX_UNPAD_FLASH_ATTENTION_NZ_ENCODER_NOCACHE;
    op_param.maskType = internal::MixParam::MaskType::MASK_TYPE_NORM;
  } else {
    op_param.mixType = internal::MixParam::MixType::MIX_UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION;
  }

  op_param.headSize = head_num;
  op_param.preTokens = pre_tokens;
  op_param.nextTokens = next_tokens;
  op_param.tor = primitive_->HasAttr("scale_value") ? GetValue<float>(primitive_->GetAttr("scale_value"))
                                                    : inputs[kIndex12]->GetValueWithCheck<float>();
  op_param.kvHead = kv_shape[kDim2] / head_dim;

  for (int64_t i = 0; i < kv_shape[kDim0]; i++) {
    op_param.qSeqLen.emplace_back(q_shape[kDim1]);
    op_param.kvSeqLen.emplace_back(q_shape[kDim1]);
    op_param.batchRunStatus.emplace_back(1);
  }
  param_ptr->specificParam = op_param;

  param_ptr->opId = internal::OpId::FlashAttentionScore;
  return param_ptr;
}

bool InternalFlashAttentionScore::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  const std::string op_name = "FlashAttentionScore";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto &enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  enable_internal_fa_ = (std::find(enable_op_list.begin(), enable_op_list.end(), op_name) != enable_op_list.end());
  return InternalKernelMod::Init(inputs, outputs);
}

int InternalFlashAttentionScore::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  if (!enable_internal_fa_) {
    impl_ = nullptr;
  }
  auto ret = InternalKernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    MS_LOG(ERROR) << "op " << op_type_ << " invoke resize failed";
    return KRET_RESIZE_FAILED;
  }
  return 0;
}

MS_INTERNAL_KERNEL_FACTORY_REG(FlashAttentionScore, InternalFlashAttentionScore);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(FlashAttentionScore, INPUT_NUM_5, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_6);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(FlashAttentionScore, OUTPUT_NUM_1, INDEX_3);
}  // namespace kernel
}  // namespace mindspore
