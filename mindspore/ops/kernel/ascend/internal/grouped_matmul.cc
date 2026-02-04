/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/internal/grouped_matmul.h"

#include <memory>
#include "kernel/ascend/internal/internal_kernel_in_out_map.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalGroupedMatmul::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                            const internal::OutputsImmutableInfoList &outputs,
                                                            const std::vector<KernelTensor *> &ms_inputs,
                                                            const std::vector<KernelTensor *> &ms_outputs) {
  auto input_len = ms_inputs.size();
  param_.transpose_a = ms_inputs[input_len - kIndex2]->GetValueWithCheck<bool>();
  param_.transpose_b = ms_inputs[input_len - kIndex1]->GetValueWithCheck<bool>();
  param_.with_bias = !(ms_inputs[kIndex2]->GetType()->isa<TypeNone>());
  param_.enable_shuffle = false;  // the real definition is in internal
  output_format_ = outputs[0].GetFormat();
  return internal::CreateGroupedMatmulOp(inputs, outputs, param_, internal::kInternalGroupedMatmulOpName);
}

uint64_t InternalGroupedMatmul::GenerateTilingKey(const std::vector<KernelTensor *> &inputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return InternalTilingCache::GenerateKey(kernel_name_, inputs, param_, output_format_);
}

MS_INTERNAL_KERNEL_FACTORY_REG(GroupedMatmul, internal::kInternalGroupedMatmulOpName, InternalGroupedMatmul);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(GroupedMatmul, INPUT_NUM_5, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_7);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(GroupedMatmul, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
