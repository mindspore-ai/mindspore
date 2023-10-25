/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/pyboost/baddbmm_gpu.h"
#include "kernel/pyboost/op/add.h"
#include "kernel/pyboost/op/mul.h"
#include "kernel/pyboost/op/batch_matmul.h"
#include "ops/math_ops.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr BaddbmmGPU::Call(const tensor::TensorPtr &input, const tensor::TensorPtr &batch1,
                                   const tensor::TensorPtr &batch2, const ScalarPtr &beta, const ScalarPtr &alpha) {
  // input * beta + alpha * (batch1 @ batch2)
  auto add = CREATE_PYBOOST_OP(Add, "GPU");
  add->set_primitive(prim::kPrimAdd->Clone());
  auto mul = CREATE_PYBOOST_OP(Mul, "GPU");
  mul->set_primitive(prim::kPrimMul->Clone());
  auto bmm = CREATE_PYBOOST_OP(BatchMatmul, "GPU");
  auto prim_batch_matmul = prim::kPrimBatchMatMul->Clone();
  prim_batch_matmul->set_attr("transpose_a", MakeValue(false));
  prim_batch_matmul->set_attr("transpose_b", MakeValue(false));
  bmm->set_primitive(prim_batch_matmul);

  return add->Call(mul->Call(input, beta), mul->Call(bmm->Call(batch1, batch2), alpha));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
