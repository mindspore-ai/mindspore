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

#include "plugin/device/ascend/kernel/internal/paged_attention.h"

#include <memory>
#include <string>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "utils/llm_manager.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalPagedAttention::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                           const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::PagedAttention;
  internal::MixParam op_param;
  op_param.maskType = internal::MixParam::MaskType::MASK_TYPE_NONE;

  auto context_ptr = mindspore::MsContext::GetInstance();
  if (context_ptr->ascend_soc_version() == "ascend310p") {
    op_param.mixType = internal::MixParam::MixType::MIX_PAGED_ATTENTION_NZ_MASK;
  } else {
    op_param.mixType = internal::MixParam::MixType::MIX_PAGED_ATTENTION_MASK_ND;
  }

  op_param.headSize = static_cast<int32_t>(inputs[kIndex7]->GetValueWithCheck<int64_t>());
  op_param.tor = inputs[kIndex8]->GetValueWithCheck<float>();
  op_param.kvHead = static_cast<int32_t>(inputs[kIndex9]->GetValueWithCheck<int64_t>());
  // set kvSeqLen with llm_manager's round_up_max_seq_length
  auto &llm_manager = LLMManager::GetInstance();
  max_seq_len_ = llm_manager.get_current_round_up_max_seq_length();
  // reset kvSeqLen with env value if exists
  std::string max_seq_len_env = common::GetEnv("MS_INTERNAL_MAX_SEQ_LEN");
  if (!max_seq_len_env.empty()) {
    max_seq_len_ = std::stoi(max_seq_len_env);
  }
  op_param.kvSeqLen = {max_seq_len_};
  MS_LOG(DEBUG) << "For op PagedAttention, set param.kvSeqlen = " << max_seq_len_;
  param_ptr->specificParam = op_param;
  return param_ptr;
}

uint64_t InternalPagedAttention::GenTilingCacheKey(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return TilingCacheMgr::GetInstance().GenTilingCacheKey(kernel_name_, inputs, max_seq_len_);
}

MS_INTERNAL_KERNEL_FACTORY_REG(PagedAttention, InternalPagedAttention);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(PagedAttention, INPUT_NUM_7, INDEX_0, INDEX_1, INDEX_2, INDEX_4, INDEX_3, INDEX_5,
                                     INDEX_6);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(PagedAttention, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
