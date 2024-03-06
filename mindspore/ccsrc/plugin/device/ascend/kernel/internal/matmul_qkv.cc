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
#include "include/param/matmul_qkv_param.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalMatmulQkv::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  auto param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::MatmulQkv;
  bool tranpose_a = false;
  bool transpose_b = true
  internal::MatmulQkvParam op_param = {transpose_a, transpose_b};
  param_ptr->specificParam = op_param;
  return param_ptr;
}

void InternalMatmulQkv::SetInOutIdx() {
  inputsIdxMap_[0] = 0;
  inputsIdxMap_[1] = 1;
  inputsIdxMap_[2] = 2;
  inputsIdxMap_[3] = 3;
  outputsIdxMap_[0] = 0;
  outputsIdxMap_[1] = 1;
  outputsIdxMap_[2] = 2;
}

uint64_t InternalMatmulQkv::GenTilingCacheKey(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return TilingCacheMgr::GetInstance().GenTilingCacheKey(
    kernel_name_, inputs[0]->GetShapeVector(), inputs[0]->dtype_id(), inputs[1]->GetShapeVector(),
    inputs[1]->dtype_id(), inputs[2]->GetShapeVector(), inputs[2]->dtype_id(), inputs[3]->GetShapeVector(),
    inputs[3]->dtype_id());
}
MS_INTERNAL_KERNEL_FACTORY_REG(MatmulQkv, InternalMatmulQkv);
}  // namespace kernel
}  // namespace mindspore
