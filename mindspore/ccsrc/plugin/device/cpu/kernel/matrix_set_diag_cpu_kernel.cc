/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/matrix_set_diag_cpu_kernel.h"
#include <algorithm>
#include <tuple>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "common/thread_pool.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
void MatrixSetDiagCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
  constexpr size_t required_input_nums = 3;
  constexpr size_t required_output_nums = 1;
  if (AnfAlgo::GetInputNum(kernel_node) != required_input_nums ||
      AnfAlgo::GetOutputTensorNum(kernel_node) != required_output_nums) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the input nums are required to [input, diagonal, "
                         "k, alignment] for 3 and the output nums is require to 1.";
  }

  // invalid alignment will throw an exception.
  auto alignment = AnfAlgo::GetNodeAttr<std::string>(kernel_node, ALIGNMENT);
  alignment_ = GetAlignments(alignment);
  constexpr size_t input_index = 0;
  constexpr size_t diag_index = 1;
  constexpr size_t diag_k_index = 2;
  constexpr size_t output_index = 0;
  auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, input_index);
  auto diag_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, diag_index);
  auto diag_k_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, diag_k_index);
  auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, output_index);

  constexpr size_t temporary_2d_dim = 2;
  constexpr size_t temporary_1d_dim = 1;
  if (input_shape.size() < temporary_2d_dim || diag_shape.size() < temporary_1d_dim || input_shape != output_shape) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of input is invalid for input shape greater than 2D, diag shape "
                         "greater than 1D, input shape should equal to output shape.";
  }
  if (diag_k_shape.size() != temporary_1d_dim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of diag_region's dim should be limited to range (k[0],k[1]).";
  }
  size_t input_rank = input_shape.size();
  for (size_t i = 0; i < input_rank - temporary_2d_dim; ++i) {
    outer_batch_ *= SizeToInt(input_shape.at(i));
  }
  input_shape_ = input_shape;
  inner_rows_ = SizeToInt(input_shape.at(input_rank - temporary_2d_dim));
  inner_cols_ = SizeToInt(input_shape.at(input_rank - temporary_1d_dim));

  expected_num_diags_ = diag_shape.size() == input_rank ? SizeToInt(diag_shape.at(input_rank - temporary_2d_dim)) : 1;

  data_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
}

bool MatrixSetDiagCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &workspaces,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  if (data_type_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, workspaces, outputs);
  } else if (data_type_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, workspaces, outputs);
  } else if (data_type_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, workspaces, outputs);
  } else if (data_type_ == kNumberTypeInt32) {
    LaunchKernel<int>(inputs, workspaces, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the data_type of input should be float16, float32, float64, int, int32."
                         " but got "
                      << TypeIdToType(data_type_)->ToString();
  }
  return true;
}

template <typename T>
void MatrixSetDiagCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                             const std::vector<AddressPtr> &outputs) {
  auto input = inputs.at(kDim0);
  auto diag = inputs.at(kDim1);
  constexpr int diag_k_index = 2;
  auto k = inputs.at(diag_k_index);
  auto output = outputs.at(kDim0);

  T *input_addr = reinterpret_cast<T *>(input->addr);
  T *diag_addr = reinterpret_cast<T *>(diag->addr);
  int *diag_k_addr = reinterpret_cast<int *>(k->addr);
  T *output_addr = reinterpret_cast<T *>(output->addr);
  lower_diag_index_ = diag_k_addr[0];
  upper_diag_index_ = diag_k_addr[1];
  is_single_diag_ = (lower_diag_index_ == upper_diag_index_);
  if (lower_diag_index_ <= -inner_rows_ || lower_diag_index_ >= inner_cols_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of diag_region's lower_diag_index is invalid, which must be between "
                      << -inner_rows_ << " and " << inner_cols_;
  }
  if (upper_diag_index_ <= -inner_rows_ || upper_diag_index_ >= inner_cols_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of diag_region's upper_diag_index is invalid, which must be between "
                      << -inner_rows_ << " and " << inner_cols_;
  }

  if (lower_diag_index_ > upper_diag_index_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dimension of diag_region's lower_diag_index_ must less than upper_diag_index "
                      << lower_diag_index_ << " < " << upper_diag_index_;
  }
  num_diags_ = upper_diag_index_ - lower_diag_index_ + 1;
  if (lower_diag_index_ != upper_diag_index_ && expected_num_diags_ != num_diags_) {
    MS_LOG(EXCEPTION)
      << "For '" << kernel_name_
      << "', the dimension of diag_region's lower_diag_index and upper_diag_index are not consistent with input shape.";
  }
  max_diag_len_ = std::min(inner_rows_ + std::min(upper_diag_index_, 0), inner_cols_ - std::max(lower_diag_index_, 0));

  // Copy input to output first, then set diagonal value to output.
  (void)memcpy_s(output_addr, output->size, input_addr, input->size);
  size_t max_index = IntToSize(outer_batch_ * inner_rows_ * inner_cols_);
  auto task = [this, max_index, diag_addr, output_addr](size_t start, size_t end) {
    MatrixInfoPtr matrix_info = std::make_shared<MatrixInfo>(max_index, input_shape_);
    if (!matrix_info->SetIndex(start, end)) {
      MS_LOG(EXCEPTION) << "current data indexes are invalid : [" << start << ", " << end
                        << "]. you should limit them in [0, " << max_index << "].";
    }
    auto get_out_batch = [](const std::vector<size_t> &current_indexes) {
      constexpr size_t last_two_dims = 2;
      int out_batch = 1;
      for (size_t i = 0; i < current_indexes.size() - last_two_dims; ++i) {
        out_batch *= (SizeToInt(current_indexes.at(i)) + 1);
      }
      size_t inner_row = current_indexes.at(current_indexes.size() - last_two_dims);
      size_t inner_col = current_indexes.at(current_indexes.size() - 1);
      std::tuple<int, int, int> flatten_3d_shape = std::make_tuple(out_batch - 1, inner_row, inner_col);
      return flatten_3d_shape;
    };
    for (size_t inner = start; inner < end; ++inner) {
      std::vector<size_t> current_indexes = matrix_info->IndexIterator();
      auto flatten_3d_shape = get_out_batch(current_indexes);
      int batch = std::get<0>(flatten_3d_shape);
      int m = std::get<1>(flatten_3d_shape);
      constexpr size_t col_index = 2;
      int n = std::get<col_index>(flatten_3d_shape);
      int d = n - m;
      if (is_single_diag_) {
        if (d == upper_diag_index_) {
          output_addr[inner] = diag_addr[batch * max_diag_len_ + n - std::max(upper_diag_index_, 0)];
        }
      } else {
        int diag_index = upper_diag_index_ - d;
        int offset = CalDiagOffset(d, max_diag_len_, inner_rows_, inner_cols_, alignment_);
        int index_in_diag = n - std::max(d, 0) + offset;
        if (d >= lower_diag_index_ && d <= upper_diag_index_) {
          output_addr[inner] =
            diag_addr[batch * num_diags_ * max_diag_len_ + diag_index * max_diag_len_ + index_in_diag];
        }
      }
    }
  };
  CPUKernelUtils::ParallelFor(task, max_index);
}
}  // namespace kernel
}  // namespace mindspore
