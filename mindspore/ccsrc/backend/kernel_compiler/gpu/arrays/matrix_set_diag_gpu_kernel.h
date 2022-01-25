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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MATRIX_SET_DIAG_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MATRIX_SET_DIAG_KERNEL_H_

#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <utility>
#include "backend/kernel_compiler/gpu//gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/common_utils.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "backend/kernel_compiler/gpu/cuda_impl/matrix_set_diag_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T>
class MatrixSetDiagGpuKernelMod : public NativeGpuKernelMod {
 public:
  MatrixSetDiagGpuKernelMod() = default;
  ~MatrixSetDiagGpuKernelMod() override = default;

  bool Init(const CNodePtr &kernel_node) override {
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
    auto alignment = AnfAlgo::GetNodeAttr<std::string>(kernel_node, kAlignment);
    alignment_ = GetAlignments(alignment);
    constexpr int input_index = 0;
    constexpr int diag_index = 1;
    constexpr int diag_k_index = 2;
    constexpr int output_index = 0;
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, input_index);
    auto diag_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, diag_index);
    auto diag_k_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, diag_k_index);
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, output_index);

    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input_shape") ||
                     CHECK_SHAPE_NULL(diag_shape, kernel_name_, "diag_shape") ||
                     CHECK_SHAPE_NULL(diag_k_shape, kernel_name_, "diag_k_shape") ||
                     CHECK_SHAPE_NULL(output_shape, kernel_name_, "output_shape");
    if (is_null_input_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the input shape contains empty, which is invalid, please check, it's input.";
    }

    constexpr int temporary_2d_dim = 2;
    constexpr int temporary_1d_dim = 1;
    if (SizeToInt(input_shape.size()) < temporary_2d_dim || SizeToInt(diag_shape.size()) < temporary_1d_dim ||
        input_shape != output_shape) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of input is invalid for input shape greater than 2D, diag shape "
                           "greater than 1D, input shape should equal to output shape.";
    }
    if (SizeToInt(diag_k_shape.size()) != temporary_1d_dim) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of diag_region's dim should be limited to range (k[0],k[1]).";
    }
    int input_rank = SizeToInt(input_shape.size());
    for (int i = 0; i < input_rank - temporary_2d_dim; ++i) {
      outer_batch_ *= SizeToInt(input_shape.at(i));
    }
    inner_rows_ = SizeToInt(input_shape.at(input_rank - temporary_2d_dim));
    inner_cols_ = SizeToInt(input_shape.at(input_rank - temporary_1d_dim));

    expected_num_diags_ =
      SizeToInt(diag_shape.size()) == input_rank ? SizeToInt(diag_shape.at(input_rank - temporary_2d_dim)) : 1;
    for (const auto &diag_sh : diag_shape) {
      diagonal_count_ *= diag_sh;
    }
    for (const auto &k_sh : diag_k_shape) {
      k_count_ *= k_sh;
    }
    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto input = inputs.at(kDim0);
    auto diag = inputs.at(kDim1);
    constexpr int diag_k_index = 2;
    auto k = inputs.at(diag_k_index);
    auto output = outputs.at(kDim0);

    T *input_addr = reinterpret_cast<T *>(input->addr);
    T *diag_addr = reinterpret_cast<T *>(diag->addr);
    int *diag_k_addr = reinterpret_cast<int *>(k->addr);
    T *output_addr = reinterpret_cast<T *>(output->addr);
    std::vector<int> host_k_vec(diag_k_index, 0);
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(host_k_vec.data(), diag_k_addr, k->size, cudaMemcpyDeviceToHost,
                                              reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "matrix_set_diag cuda memcopy device to host Fail");
    lower_diag_index_ = host_k_vec.at(kDim0);
    upper_diag_index_ = host_k_vec.at(kDim1);
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
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of diag_region's lower_diag_index and upper_diag_index are not consistent "
                           "with input shape.";
    }
    max_diag_len_ =
      std::min(inner_rows_ + std::min(upper_diag_index_, 0), inner_cols_ - std::max(lower_diag_index_, 0));

    // copy input to output first, then set diagonal value to output.
    CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                              cudaMemcpyAsync(output_addr, input_addr, input->size, cudaMemcpyDeviceToDevice,
                                              reinterpret_cast<cudaStream_t>(stream_ptr)),
                              "matrix_set_diag cuda memcopy input to output Fail");

    bool right_align_super_diagonal = (alignment_.first == MatrixDiag::RIGHT);
    bool right_align_sub_diagonal = (alignment_.second == MatrixDiag::RIGHT);
    MatrixSetDiag(outer_batch_, inner_rows_, inner_cols_, num_diags_, max_diag_len_, lower_diag_index_,
                  upper_diag_index_, right_align_super_diagonal, right_align_sub_diagonal, is_single_diag_, diag_addr,
                  output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 private:
  void InitSizeLists() override {
    input_size_list_.emplace_back(outer_batch_ * inner_rows_ * inner_cols_ * sizeof(T));
    input_size_list_.emplace_back(diagonal_count_ * sizeof(T));
    input_size_list_.emplace_back(k_count_ * sizeof(int));
    output_size_list_.emplace_back(outer_batch_ * inner_rows_ * inner_cols_ * sizeof(T));
  };
  int lower_diag_index_{0};
  int upper_diag_index_{0};
  int inner_rows_{0};
  int inner_cols_{0};
  int num_diags_{0};
  int expected_num_diags_{0};
  int max_diag_len_{0};
  int outer_batch_{1};
  size_t diagonal_count_{1};
  size_t k_count_{1};
  bool is_single_diag_{true};
  bool is_null_input_{true};
  // <super_matrix_diag_align, sub_matrix_diag_align>
  std::pair<MatrixDiag::Alignment, MatrixDiag::Alignment> alignment_{MatrixDiag::RIGHT, MatrixDiag::LEFT};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_MATRIX_SET_DIAG_KERNEL_H_
