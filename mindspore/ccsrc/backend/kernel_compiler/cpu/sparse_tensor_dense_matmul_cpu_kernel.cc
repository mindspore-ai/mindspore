/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/cpu/sparse_tensor_dense_matmul_cpu_kernel.h"

namespace mindspore {
namespace kernel {
template <typename I, typename T>
void SparseTensorDenseMatmulCPUKernel<I, T>::InitKernel(const CNodePtr &kernel_node) {
  output_size_ = 1;
  auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  for (auto &dim : output_shape) {
    output_size_ *= dim;
  }

  aValues_size_ = 1;
  auto aValues_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  for (auto &dim : aValues_shape) {
    aValues_size_ *= dim;
  }

  b_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 3);
  output_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
}

template <typename I, typename T>
bool SparseTensorDenseMatmulCPUKernel<I, T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                    const std::vector<kernel::AddressPtr> & /*workspace*/,
                                                    const std::vector<kernel::AddressPtr> &outputs) {
  auto a_indices = reinterpret_cast<I *>(inputs[0]->addr);
  auto a_values = reinterpret_cast<T *>(inputs[1]->addr);
  auto b = reinterpret_cast<T *>(inputs[3]->addr);
  auto out = reinterpret_cast<T *>(outputs[0]->addr);

  memset(out, 0, output_size_);

  const size_t nnz = aValues_size_;
  const size_t rhs_right = b_shape_[1];
  const size_t lhs_right = b_shape_[0];

  for (size_t i = 0; i < nnz; ++i) {
    const size_t m = a_indices[i * 2];
    const size_t k = a_indices[i * 2 + 1];

    if (k > lhs_right) {
      MS_LOG(ERROR) << "Invalid value: k: " << k << ", lhs_right: " << lhs_right;
      return false;
    }
    if (m > output_shape_[0]) {
      MS_LOG(ERROR) << "Invalid value: m: " << m << ", output_shape: " << output_shape_[0];
      return false;
    }

    for (size_t n = 0; n < rhs_right; ++n) {
      const float b_value = b[k * lhs_right + n];
      out[m * output_shape_[0] + n] += a_values[i] * b_value;
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
