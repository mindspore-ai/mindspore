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

#include "plugin/device/ascend/kernel/internal/reshape_and_cache.h"

#include <memory>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr ReshapeAndCache::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &outputs) {
  auto param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::ReshapeAndCache;

  auto context_ptr = mindspore::MsContext::GetInstance();
  if (context_ptr->ascend_soc_version() == "ascend310p") {
    param_ptr->opId = internal::OpId::ReshapeAndCacheNz;
  }

  internal::MixParam mix_param;
  mix_param.mixType = internal::MixParam::MixType::MIX_RESHAPE_AND_CACHE_ND;
  param_ptr->specificParam = mix_param;
  return param_ptr;
}

MS_INTERNAL_KERNEL_FACTORY_REG(ReshapeAndCache, ReshapeAndCache);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(ReshapeAndCache, INPUT_NUM_5, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_4);
}  // namespace kernel
}  // namespace mindspore
