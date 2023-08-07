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
#include "extendrt/kernel/ascend_native/ascend_native_matmul_kernel.h"
#include "extendrt/kernel/ascend_native/ascend_native_kernel_registry.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/gemm.h"
#include "ops/fusion/mat_mul_fusion.h"

namespace mindspore::kernel {
using mindspore::ops::kNameMatMulFusion;

int AscendNativeMatmulKernel::Prepare() {
  if (out_tensors_[0]->shape().size() == 0) {
    if (in_tensors_[0] != nullptr && in_tensors_[1] != nullptr) {
      std::vector<int> shape;
      shape.push_back(in_tensors_[0]->shape()[0]);
      shape.push_back(in_tensors_[1]->shape()[1]);
      out_tensors_[0]->set_shape(shape);
    }
  }
  return kSuccess;
}

int AscendNativeMatmulKernel::Execute() {
  MS_LOG(INFO) << "AscendNativeMatmulKernel::Execute";
  const std::vector<InferTensor *> &in_tensors = this->in_tensors();
  const std::vector<InferTensor *> &out_tensors = this->out_tensors();
  auto m = static_cast<size_t>(in_tensors[0]->shape()[0]);
  auto k = static_cast<size_t>(in_tensors[0]->shape()[1]);
  auto n = static_cast<size_t>(in_tensors[1]->shape()[1]);
  auto prim = GetValueNode<PrimitivePtr>(primitive_.cnode->input(0));
  bool transpose_a = GetValue<bool>(prim->GetAttr("transpose_a"));
  bool transpose_b = GetValue<bool>(prim->GetAttr("transpose_b"));
  ascend_native::GemmFp16(const_cast<void *>(get_stream()), transpose_a, transpose_b, m, n, k, 1.0f,
                          in_tensors[0]->device_data(), k, in_tensors[1]->device_data(), n, 0.0f,
                          out_tensors[0]->device_data(), n, 1);
  return kSuccess;
}
REGISTER_ASCEND_NATIVE_CREATOR(kNameMatMulFusion, AscendNativeMatmulKernel)
}  // namespace mindspore::kernel
