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

#include "plugin/device/ascend/kernel/internal/quant_batch_matmul.h"

#include <memory>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalQuantBatchMatmul::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                             const std::vector<KernelTensor *> &outputs) {
  auto param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::MatMul;
  bool transpose_x1 = false;
  bool transpose_x2 = false;
  transpose_x1 = static_cast<bool>(inputs[kIndex5]->GetValueWithCheck<bool>());
  transpose_x2 = static_cast<bool>(inputs[kIndex6]->GetValueWithCheck<bool>());

  bool has_bias = !(inputs[kIndex4]->GetType()->isa<TypeNone>());
  internal::MatMulParam op_param = {transpose_x1, transpose_x2, {0, 0, 0}, has_bias, true};
  param_ptr->specificParam = op_param;
  return param_ptr;
}

MS_INTERNAL_KERNEL_FACTORY_REG(QuantBatchMatmul, InternalQuantBatchMatmul);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(QuantBatchMatmul, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_4, INDEX_2);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(QuantBatchMatmul, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
