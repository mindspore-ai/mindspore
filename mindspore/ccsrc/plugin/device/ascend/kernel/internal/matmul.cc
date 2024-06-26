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

#include "plugin/device/ascend/kernel/internal/matmul.h"

#include <memory>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "param/matmul_ext_param.h"

namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalMatMul::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs) {
  auto param_ptr = std::make_shared<internal::MatMulExtParam>();
  internal::MatMulParam matmul_param;
  param_ptr->opId = internal::OpId::MatMul;
  // setup matmul param from inputs
  bool transpose_a = false;
  bool transpose_b = false;
  if (primitive_->HasAttr("transpose_a") && primitive_->HasAttr("transpose_b")) {
    transpose_a = GetValue<bool>(primitive_->GetAttr("transpose_a"));
    transpose_b = GetValue<bool>(primitive_->GetAttr("transpose_b"));
  } else {
    transpose_a = inputs[inputs.size() - 2]->GetValueWithCheck<bool>();
    transpose_b = inputs[inputs.size() - 1]->GetValueWithCheck<bool>();
  }
  auto shape_a = inputs[kIndex0]->GetShapeVector();
  auto shape_b = inputs[kIndex1]->GetShapeVector();
  int m = (!transpose_a) ? shape_a[kIndex0] : shape_a[kIndex1];
  int k = (!transpose_a) ? shape_a[kIndex1] : shape_a[kIndex0];
  int n = (!transpose_b) ? shape_b[kIndex1] : shape_b[kIndex0];

  param_ptr->input_dtype = InternalKernelUtils::ToInternalDType(inputs[kIndex0]->dtype_id());
  param_ptr->weight_dtype = InternalKernelUtils::ToInternalDType(inputs[kIndex1]->dtype_id());
  param_ptr->output_dtype = InternalKernelUtils::ToInternalDType(outputs[kIndex0]->dtype_id());

  matmul_param = {
    transpose_a,  // transposeA
    transpose_b,  // transposeB
    {m, k, n},    // oriShape
    false,        // withBias
    false,        // enDequant
    0,            // tilingN
    0,            // tilingK
    false,        // enShuffleK
  };
  param_ptr->specificParam = matmul_param;
  return std::static_pointer_cast<internal::OpParam>(param_ptr);
}

MS_INTERNAL_KERNEL_FACTORY_REG(MatMul, InternalMatMul);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MatMul, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MatMul, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
