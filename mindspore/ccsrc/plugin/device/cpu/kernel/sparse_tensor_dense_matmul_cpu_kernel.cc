/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/sparse_tensor_dense_matmul_cpu_kernel.h"
#include <functional>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSparseTensorDenseMatmulInputsNum = 4;
constexpr size_t kSparseTensorDenseMatmulOutputsNum = 1;
constexpr size_t kSparseTensorDenseMatmulOutputShapeSize = 2;
constexpr size_t kSparseTensorDenseMatmulDenseShapeSize = 2;
constexpr size_t kIndicesSizeNum = 2;
constexpr size_t kIndices2rdDimNum = 2;
}  // namespace

template <typename I, typename T>
void SparseTensorDenseMatmulCpuKernelMod<I, T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  adj_st_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, ADJ_ST);
  adj_dt_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, ADJ_dT);
  auto indices_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, INDICES);
  if (indices_shape.size() != kIndicesSizeNum && indices_shape[1] != kIndices2rdDimNum) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', it requires 'indices' should be a 2-D Tensor and the second dimension length "
                         "should be 2, but got 'indices' shape: "
                      << Vector2Str(indices_shape);
  }
  auto values_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, VALUES);
  if (values_shape.size() != 1 || values_shape[0] != indices_shape[0]) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', it requires 'values' should be a 1-D Tensor and the first dimension length "
                         " should be equal to the first dimension length of 'indices', but got 'values' shape: "
                      << Vector2Str(values_shape) << " and 'indices' shape: " << Vector2Str(indices_shape);
  }
  output_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  values_size_ = values_shape[0];
  b_shape_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, DENSE);
  if (b_shape_.size() != kSparseTensorDenseMatmulDenseShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'dense' should be "
                      << kSparseTensorDenseMatmulDenseShapeSize << "-D, but got " << b_shape_.size() << "-D";
  }
  if (output_shape_.size() != kSparseTensorDenseMatmulOutputShapeSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of output should be "
                      << kSparseTensorDenseMatmulOutputShapeSize << "-D, but got " << output_shape_.size() << "-D";
  }
}

template <typename I, typename T>
bool SparseTensorDenseMatmulCpuKernelMod<I, T>::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                       const std::vector<kernel::AddressPtr> & /* workspace */,
                                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseTensorDenseMatmulInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseTensorDenseMatmulOutputsNum, kernel_name_);
  if (outputs[0]->size == 0) {
    MS_LOG(WARNING) << "For '" << kernel_name_ << "', output memory size should be greater than 0, but got 0.";
    return true;
  }
  auto ret = memset_s(outputs[0]->addr, outputs[0]->size, 0, outputs[0]->size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output failed. Error no: " << ret;
  }

  const size_t b_index = 3;
  const auto *a_indices = reinterpret_cast<I *>(inputs[0]->addr);
  const auto *a_values = reinterpret_cast<T *>(inputs[1]->addr);
  const auto *b = reinterpret_cast<T *>(inputs[b_index]->addr);
  auto *out = reinterpret_cast<T *>(outputs[0]->addr);
  const size_t indices_length = inputs[0]->size / sizeof(I);
  const size_t values_length = inputs[1]->size / sizeof(T);
  const size_t b_length = inputs[b_index]->size / sizeof(T);

  const size_t dim_num = 2;
  const size_t out_dim_0 = output_shape_[0];
  const size_t out_dim_1 = output_shape_[1];
  const size_t b_dim_0 = b_shape_[0];
  const size_t b_dim_1 = b_shape_[1];
  const size_t same_dim = adj_dt_ ? b_dim_1 : b_dim_0;

  for (size_t i = 0; i < values_size_; ++i) {
    if (i * dim_num + 1 >= indices_length) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the index of 'indices' out of bounds.";
    }
    if (i >= values_length) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the index of 'values' out of bounds.";
    }
    const int row = adj_st_ ? a_indices[i * dim_num + 1] : a_indices[i * dim_num];
    const int col = adj_st_ ? a_indices[i * dim_num] : a_indices[i * dim_num + 1];
    if (row >= SizeToInt(out_dim_0) || row < 0 || col >= SizeToInt(same_dim) || col < 0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', the indices including out of bounds index, row range: [0, " << out_dim_0
                               << "), col range: [0, " << same_dim << "), but got row: " << row << ", col: " << col;
    }
    const size_t row_s = IntToSize(row);
    const size_t col_s = IntToSize(col);
    for (size_t n = 0; n < out_dim_1; ++n) {
      if (adj_dt_) {
        if (n * b_dim_1 + col_s >= b_length) {
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the index of 'b' out of bounds.";
        }
        const T b_value = b[n * b_dim_1 + col_s];
        out[row_s * out_dim_1 + n] += a_values[i] * b_value;
      } else {
        if (col_s * b_dim_1 + n >= b_length) {
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the index of 'b' out of bounds.";
        }
        const T b_value = b[col_s * b_dim_1 + n];
        out[row_s * out_dim_1 + n] += a_values[i] * b_value;
      }
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
