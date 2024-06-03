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

#include "plugin/device/ascend/kernel/internal/matmul_ffn.h"

#include <memory>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "param/matmul_qkv_param.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalMatmulFfn::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  auto param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::MatmulQkv;
  bool transpose_a = false;
  bool transpose_b = true;
  auto n_lens = primitive_->GetAttr("n_lens");
  MS_EXCEPTION_IF_NULL(n_lens);
  auto n_list = GetValue<std::vector<int64_t>>(n_lens);
  internal::MatmulQkvParam op_param = {static_cast<uint32_t>(n_list[0]), static_cast<uint32_t>(n_list[1]),
                                       static_cast<uint32_t>(0), transpose_a, transpose_b};
  param_ptr->specificParam = op_param;
  return param_ptr;
}

uint64_t InternalMatmulFfn::GenTilingCacheKey(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return TilingCacheMgr::GetInstance().GenTilingCacheKey(kernel_name_, inputs[kIndex0]->GetShapeVector(),
                                                         inputs[kIndex0]->dtype_id(), inputs[kIndex1]->GetShapeVector(),
                                                         inputs[kIndex1]->dtype_id());
}

MS_INTERNAL_KERNEL_FACTORY_REG(MatmulFfn, InternalMatmulFfn);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatmulFfn, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatmulFfn, OUTPUT_NUM_2, INDEX_0, INDEX_1);
}  // namespace kernel
}  // namespace mindspore
