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

#include <functional>

#include "backend/kernel_compiler/cpu/sparse_tensor_dense_matmul_cpu_kernel.h"

namespace mindspore {
namespace kernel {
template <typename I, typename T>
void SparseTensorDenseMatmulCPUKernel<I, T>::InitKernel(const CNodePtr &kernel_node) {
  adj_st_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, ADJ_ST);
  adj_dt_ = AnfAlgo::GetNodeAttr<bool>(kernel_node, ADJ_dT);
  output_shape_ = AnfAlgo::GetOutputInferShape(kernel_node, 0);
  output_size_ = std::accumulate(output_shape_.begin(), output_shape_.end(), size_t(1), std::multiplies<size_t>());
  auto values_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  if (values_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "SparseTensorDenseMatmul requires the values must be a 1-D tensor, but got "
                      << values_shape.size() << "-D";
  }
  values_size_ = values_shape[0];
  b_shape_ = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
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

  const size_t out_dim_0 = output_shape_[0];
  const size_t out_dim_1 = output_shape_[1];
  const size_t b_dim_0 = b_shape_[0];
  const size_t b_dim_1 = b_shape_[1];
  const size_t same_dim = adj_dt_ ? b_dim_1 : b_dim_0;

  for (size_t i = 0; i < values_size_; ++i) {
    const int row = adj_st_ ? a_indices[i * 2 + 1] : a_indices[i * 2];
    const int col = adj_st_ ? a_indices[i * 2] : a_indices[i * 2 + 1];
    if (row >= SizeToInt(out_dim_0) || row < 0 || col >= SizeToInt(same_dim) || col < 0) {
      MS_LOG(ERROR) << "The indices including out of bounds index, row range: [0, " << out_dim_0 << "), col range: [0, "
                    << same_dim << "), but got row: " << row << ", col: " << col;
      return false;
    }

    for (size_t n = 0; n < out_dim_1; ++n) {
      if (adj_dt_) {
        const T b_value = b[n * b_dim_1 + col];
        out[row * out_dim_1 + n] += a_values[i] * b_value;
      } else {
        const T b_value = b[col * b_dim_1 + n];
        out[row * out_dim_1 + n] += a_values[i] * b_value;
      }
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
