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
#include <memory>
#include "plugin/device/ascend/kernel/internal/matmul.h"
namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalMatMul::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  internal::MatMulParam matmul_param;
  param_ptr->opId = internal::OpId::MatMul;
  // setup matmul param from inputs
  bool transpose_a = primitive_->HasAttr("transpose_a") ? GetValue<bool>(primitive_->GetAttr("transpose_a")) : false;
  bool transpose_b = primitive_->HasAttr("transpose_b") ? GetValue<bool>(primitive_->GetAttr("transpose_b")) : false;
  auto shape_a = inputs[0]->GetShapeVector();
  auto shape_b = inputs[1]->GetShapeVector();
  int m = (!transpose_a) ? shape_a[0] : shape_a[1];
  int k = (!transpose_a) ? shape_a[1] : shape_a[0];
  int n = (!transpose_b) ? shape_b[1] : shape_b[0];

  matmul_param = {transpose_a, transpose_b, {m, k, n}};
  param_ptr->specificParam = matmul_param;
  return param_ptr;
}
void InternalMatMul::SetInOutIdx() {
  inputsIdxMap_[0] = 0;
  inputsIdxMap_[1] = 1;
  outputsIdxMap_[0] = 0;
}

MS_INTERNAL_KERNEL_FACTORY_REG(MatMul, InternalMatMul);
}  // namespace kernel
}  // namespace mindspore
