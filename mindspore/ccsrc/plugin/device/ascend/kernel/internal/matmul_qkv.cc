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

#include "plugin/device/ascend/kernel/internal/matmul_qkv.h"

#include <memory>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "param/matmul_qkv_param.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalMatmulQkv::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  auto param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::MatmulQkv;
  bool transpose_a = false;
  bool transpose_b = true;
  internal::MatmulQkvParam op_param = {transpose_a, transpose_b};
  param_ptr->specificParam = op_param;
  return param_ptr;
}

void InternalMatmulQkv::SetInOutIdx() {
  inputsIdxMap_[kIndex0] = kIndex0;
  inputsIdxMap_[kIndex1] = kIndex1;
  inputsIdxMap_[kIndex2] = kIndex2;
  inputsIdxMap_[kIndex3] = kIndex3;
  outputsIdxMap_[kIndex0] = kIndex0;
  outputsIdxMap_[kIndex1] = kIndex1;
  outputsIdxMap_[kIndex2] = kIndex2;
}

uint64_t InternalMatmulQkv::GenTilingCacheKey(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return TilingCacheMgr::GetInstance().GenTilingCacheKey(
    kernel_name_, inputs[kIndex0]->GetShapeVector(), inputs[kIndex0]->dtype_id(), inputs[kIndex1]->GetShapeVector(),
    inputs[kIndex1]->dtype_id(), inputs[kIndex2]->GetShapeVector(), inputs[kIndex2]->dtype_id(),
    inputs[kIndex3]->GetShapeVector(), inputs[kIndex3]->dtype_id());
}

MS_INTERNAL_KERNEL_FACTORY_REG(MatmulQkv, InternalMatmulQkv);
}  // namespace kernel
}  // namespace mindspore
