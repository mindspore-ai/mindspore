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
#include <memory>
#include "plugin/device/ascend/kernel/internal/reshape_and_cache.h"
namespace mindspore {
namespace kernel {
internal::OpParamPtr ReshapeAndCache::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &outputs) {
  auto param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::ReshapeAndCache;

  internal::MixParam mix_param;
  mix_param.mixType = internal::MixParam::MixType::MIX_RESHAPE_AND_CACHE_ND;
  param_ptr->specificParam = mix_param;
  return param_ptr;
}
void ReshapeAndCache::SetInOutIdx() {
  inputsIdxMap_[0] = 0;
  inputsIdxMap_[1] = 1;
  inputsIdxMap_[2] = 2;
  inputsIdxMap_[3] = 3;
  inputsIdxMap_[4] = 4;
}

MS_INTERNAL_KERNEL_FACTORY_REG(ReshapeAndCache, ReshapeAndCache);
}  // namespace kernel
}  // namespace mindspore
