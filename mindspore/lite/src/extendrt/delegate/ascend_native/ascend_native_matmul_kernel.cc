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
#include <vector>
#include "extendrt/delegate/ascend_native/ascend_native_matmul_kernel.h"
#include "extendrt/delegate/ascend_native/ascend_native_kernel_registry.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/ai_core/matmul.h"
#include "ops/fusion/mat_mul_fusion.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/tiling.h"
#include "ascend_native_impl/gemm.h"
namespace mindspore::kernel {
using mindspore::ops::kNameMatMulFusion;

int AscendNativeMatmulKernel::InferShape() {
  if (in_tensors_[0] != nullptr && in_tensors_[1] != nullptr) {
    bool is_bmm = (in_tensors_[0]->shape().size() == C3NUM);
    std::vector<int> shape;
    if (is_bmm) shape.push_back(in_tensors_[0]->shape()[0]);
    shape.push_back(m_);
    shape.push_back(n_);
    out_tensors_[0]->set_shape(shape);
  }
  return kSuccess;
}

int AscendNativeMatmulKernel::Prepare() {
  auto primitive = AsOps<ops::MatMulFusion>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "convert to primitive matmul failed for " << get_name();
    return kLiteError;
  }
  transpose_a_ = primitive->get_transpose_a();
  transpose_b_ = primitive->get_transpose_b();

  auto zeroth_mm_dim = in_tensors_[0]->shape().size() - 2;
  m_ = (transpose_a_) ? in_tensors_[0]->shape()[zeroth_mm_dim + C1NUM] : in_tensors_[0]->shape()[zeroth_mm_dim];
  k_ = (transpose_a_) ? in_tensors_[0]->shape()[zeroth_mm_dim] : in_tensors_[0]->shape()[zeroth_mm_dim + C1NUM];
  n_ = (transpose_b_) ? in_tensors_[C1NUM]->shape()[zeroth_mm_dim] : in_tensors_[C1NUM]->shape()[zeroth_mm_dim + C1NUM];
  return kSuccess;
}

int AscendNativeMatmulKernel::Run() {
  bool is_bias = (in_tensors_.size() == C3NUM);
  ascend_native::Gemm gemm;
  auto bias = is_bias ? in_tensors_.at(C2NUM) : nullptr;
  gemm.init(extra_h_.bmm_num_, m_, n_, k_, const_cast<void *>(stream_), in_tensors_[0]->device_data(),
            in_tensors_[C1NUM]->device_data(), out_tensors_[0]->device_data(), transpose_a_, transpose_b_, bias);
  gemm.compute(get_sys_workspace(), SYS_WS_RESERVED, const_cast<void *>(stream_));
  return kSuccess;
}

int AscendNativeMatmulKernel::ReSize() {
  if (in_tensors_[0]->shape()[1] != in_tensors_[1]->shape()[0]) {
    MS_LOG(ERROR) << "matmul ReSize failed";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}
REGISTER_ASCEND_NATIVE_CREATOR(kNameMatMulFusion, AscendNativeMatmulKernel)
}  // namespace mindspore::kernel
