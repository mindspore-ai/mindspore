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
#include "extendrt/delegate/ascend_native/ascend_native_impl/gemm.h"
#include "ops/fusion/mat_mul_fusion.h"

namespace mindspore::kernel {
using mindspore::ops::kNameMatMulFusion;

int AscendNativeMatmulKernel::InferShape() {
  if (in_tensors_[0] != nullptr && in_tensors_[1] != nullptr) {
    std::vector<int> shape;
    shape.push_back(in_tensors_[0]->shape()[0]);
    shape.push_back(in_tensors_[1]->shape()[1]);
    out_tensors_[0]->set_shape(shape);
  }
  return kSuccess;
}

int AscendNativeMatmulKernel::Prepare() { return kSuccess; }

int AscendNativeMatmulKernel::Run() {
  MS_LOG(INFO) << "AscendNativeMatmulKernel::Execute";
  const std::vector<InferTensor *> &in_tensors = this->in_tensors();
  const std::vector<InferTensor *> &out_tensors = this->out_tensors();

  auto primitive = AsOps<ops::MatMulFusion>();
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "convert to primitive matmul failed for " << get_name();
    return kLiteError;
  }
  bool transpose_a = primitive->get_transpose_a();
  bool transpose_b = primitive->get_transpose_b();
  auto shape_a = in_tensors.at(FIRST_INPUT)->shape();
  auto shape_b = in_tensors.at(SECOND_INPUT)->shape();
  if (shape_a.size() != shape_b.size() || shape_a.size() < 2) {
    std::cout << "AscendNativeBatchMatMulKernel::Execute Error -- tensors have different dims or too short\n";
    return kLiteInputTensorError;
  }
  size_t tiles = 1;
  for (size_t i = 0; i < shape_a.size() - 2; i++) {
    if (shape_a.at(i) != shape_b.at(i)) {
      std::cout << "AscendNativeBatchMatMulKernel::Execute Error -- tensors have different shapes\n";
      return kLiteInputTensorError;
    }
    tiles *= shape_a.at(i);
  }
  auto zeroth_mm_dim = shape_a.size() - 2;
  auto m = static_cast<size_t>(shape_a.at(zeroth_mm_dim));
  auto k = static_cast<size_t>(shape_a.at(zeroth_mm_dim + 1));
  auto n = static_cast<size_t>(shape_b.at(zeroth_mm_dim + 1));

  ascend_native::BGemmFp16(const_cast<void *>(get_stream()), transpose_a, transpose_b, m, n, k, 1.0f,
                           in_tensors[0]->device_data(), k, in_tensors[1]->device_data(), n, 0.0f,
                           out_tensors[0]->device_data(), n, tiles, 1);

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
