/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_EXTRACT_IMAGE_PATCHES_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_EXTRACT_IMAGE_PATCHES_GPU_KERNEL_H_

#include <string>
#include <algorithm>
#include <cmath>
#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/transpose_impl_opt.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/extract_image_patches_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class ExtractImagePatchesKernel : public GpuKernel {
 public:
  ExtractImagePatchesKernel() { ResetResource(); }
  ~ExtractImagePatchesKernel() override = default;

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);
    T *t_input = GetDeviceAddress<T>(workspace, 0);
    T *t_output = GetDeviceAddress<T>(workspace, 1);
    size_t *input_shape = GetDeviceAddress<size_t>(workspace, 2);
    size_t *input_to_nhwc_axis = GetDeviceAddress<size_t>(workspace, 3);
    size_t *t_output_shape = GetDeviceAddress<size_t>(workspace, 4);
    size_t *t_output_to_nchw_axis = GetDeviceAddress<size_t>(workspace, 5);

    const size_t shape_size = 4 * sizeof(size_t);
    std::vector<size_t> to_nhwc_axis = {0, 2, 3, 1};
    std::vector<size_t> to_nchw_axis = {0, 3, 1, 2};

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(input_shape, &input_shape_[0], shape_size, cudaMemcpyHostToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync input_shape_ failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(input_to_nhwc_axis, &to_nhwc_axis[0], shape_size, cudaMemcpyHostToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync to_nhwc_axis failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(t_output_shape, &t_output_shape_[0], shape_size, cudaMemcpyHostToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync t_output_shape_ failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(t_output_to_nchw_axis, &to_nchw_axis[0], shape_size,
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync to_nchw_axis failed");
    CalNCHW2NHWCInterface(input_size_, shape_size / sizeof(size_t), input, &input_shape_[0], &to_nhwc_axis[0],
                          input_shape, input_to_nhwc_axis, t_input, reinterpret_cast<cudaStream_t>(stream_ptr));
    CalExtractImagePatchesNHWC(output_size_, stride_row_, stride_col_, rate_row_, rate_col_, output_cols_, need_batch_,
                               row_stride_, patch_stride_, other_stride_, input_row_size_, input_col_size_,
                               row_padding_top_, col_padding_left_, col_input_stride_, row_input_stride_,
                               patch_input_stride_, output_depth_, t_input, t_output,
                               reinterpret_cast<cudaStream_t>(stream_ptr));
    CalNHWC2NCHWInterface(output_size_, shape_size / sizeof(size_t), t_output, &t_output_shape_[0], &to_nchw_axis[0],
                          t_output_shape, t_output_to_nchw_axis, output, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "Input number is " << input_num << ", but ExtractImagePatches needs 1 inputs.";
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "Output number is " << output_num << ", but ExtractImagePatches has 1 output.";
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape) || CHECK_NULL_INPUT(output_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'ExtractImagePatchesGpuKernel', input or output is null.";
      InitSizeLists();
      return true;
    }
    input_size_ = 1;
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size_ *= input_shape[i];
      input_shape_.push_back(input_shape[i]);
    }

    output_size_ = 1;
    for (size_t i = 0; i < output_shape.size(); i++) {
      output_size_ *= output_shape[i];
    }
    if (input_shape.size() != 4 || output_shape.size() != 4) {
      MS_LOG(EXCEPTION) << "For 'ExtractImagePatchesGpuKernel', the rank of input and output should be 4, "
                        << "but got the rank of input: " << input_shape.size()
                        << ", the rank of output: " << output_shape.size();
    }
    // transposed NHWC shape
    t_output_shape_ = {output_shape[0], output_shape[2], output_shape[3], output_shape[1]};

    auto padding = GetAttr<std::string>(kernel_node, "padding");
    auto ksizes = GetAttr<std::vector<int64_t>>(kernel_node, "ksizes");
    auto strides = GetAttr<std::vector<int64_t>>(kernel_node, "strides");
    auto rates = GetAttr<std::vector<int64_t>>(kernel_node, "rates");
    if (ksizes.size() != 4 || strides.size() != 4 || rates.size() != 4) {
      MS_LOG(EXCEPTION) << "For 'ExtractImagePatchesGpuKernel', the rank of ksizes, strides and rates should be 4, "
                        << "but got the rank of ksizes: " << ksizes.size()
                        << ", the rank of strides: " << strides.size() << ", the rank of rates: " << rates.size();
    }

    ksize_row_ = ksizes[2];
    ksize_col_ = ksizes[3];
    stride_row_ = strides[2];
    stride_col_ = strides[3];
    rate_row_ = rates[2];
    rate_col_ = rates[3];

    // transposed NHWC shape
    std::vector<size_t> t_input_shape = {input_shape_[0], input_shape_[2], input_shape_[3], input_shape_[1]};

    int64_t input_depth = static_cast<int64_t>(t_input_shape[3]);
    input_col_size_ = static_cast<int64_t>(t_input_shape[2]);
    input_row_size_ = static_cast<int64_t>(t_input_shape[1]);

    int64_t patch_rows_eff = ksize_row_ + (ksize_row_ - 1) * (rate_row_ - 1);
    int64_t patch_cols_eff = ksize_col_ + (ksize_col_ - 1) * (rate_col_ - 1);

    MS_EXCEPTION_IF_ZERO("stride row", stride_row_);
    MS_EXCEPTION_IF_ZERO("stride col", stride_col_);

    if (padding == "VALID") {
      output_rows_ = std::ceil((input_row_size_ - patch_rows_eff + 1.f) / static_cast<float>(stride_row_));
      output_cols_ = std::ceil((input_col_size_ - patch_cols_eff + 1.f) / static_cast<float>(stride_col_));
      row_padding_top_ = std::max(0l, ((output_rows_ - 1) * stride_row_ + patch_rows_eff - input_row_size_) / 2);
      col_padding_left_ = std::max(0l, ((output_cols_ - 1) * stride_col_ + patch_cols_eff - input_col_size_) / 2);
    } else if (padding == "SAME") {
      output_rows_ = std::ceil(input_row_size_ / static_cast<float>(stride_row_));
      output_cols_ = std::ceil(input_col_size_ / static_cast<float>(stride_col_));
      row_padding_top_ = ((output_rows_ - 1) * stride_row_ + patch_rows_eff - input_row_size_) / 2;
      col_padding_left_ = ((output_cols_ - 1) * stride_col_ + patch_cols_eff - input_col_size_) / 2;
    } else {
      MS_LOG(EXCEPTION) << "Invalid padding value: " << padding << ".";
    }

    row_stride_ = ksize_col_;
    patch_stride_ = row_stride_ * ksize_row_ * input_depth;
    other_stride_ = patch_stride_ * output_rows_ * output_cols_;
    col_input_stride_ = input_depth;
    row_input_stride_ = input_depth * input_col_size_;
    patch_input_stride_ = input_depth * input_col_size_ * input_row_size_;
    output_depth_ = input_depth;
    MS_EXCEPTION_IF_ZERO("other stride", other_stride_);
    need_batch_ = (output_size_ - 1) / other_stride_;

    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 1;
    output_size_ = 1;
    ksize_row_ = 1;
    ksize_col_ = 1;
    stride_row_ = 1;
    stride_col_ = 1;
    rate_row_ = 1;
    rate_col_ = 1;
    output_rows_ = 1;
    output_cols_ = 1;
    need_batch_ = 1;
    row_stride_ = 1;
    patch_stride_ = 1;
    other_stride_ = 1;
    input_row_size_ = 1;
    input_col_size_ = 1;
    row_padding_top_ = 1;
    col_padding_left_ = 1;
    col_input_stride_ = 1;
    row_input_stride_ = 1;
    patch_input_stride_ = 1;
    output_depth_ = 1;
    is_null_input_ = false;
    input_shape_.clear();
    t_output_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_ * sizeof(T));
    output_size_list_.push_back(output_size_ * sizeof(T));
    workspace_size_list_.push_back(input_size_ * sizeof(T));
    workspace_size_list_.push_back(output_size_ * sizeof(T));
    workspace_size_list_.push_back(4 * sizeof(size_t));
    workspace_size_list_.push_back(4 * sizeof(size_t));
    workspace_size_list_.push_back(4 * sizeof(size_t));
    workspace_size_list_.push_back(4 * sizeof(size_t));
  }

 private:
  size_t input_size_;
  size_t output_size_;
  int64_t ksize_row_;
  int64_t ksize_col_;
  int64_t stride_row_;
  int64_t stride_col_;
  int64_t rate_row_;
  int64_t rate_col_;
  int64_t output_rows_;
  int64_t output_cols_;
  bool need_batch_;
  bool is_null_input_;
  int64_t row_stride_;
  int64_t patch_stride_;
  int64_t other_stride_;
  int64_t input_row_size_;
  int64_t input_col_size_;
  int64_t row_padding_top_;
  int64_t col_padding_left_;
  int64_t col_input_stride_;
  int64_t row_input_stride_;
  int64_t patch_input_stride_;
  int64_t output_depth_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> t_output_shape_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_EXTRACT_IMAGE_PATCHES_GPU_KERNEL_H_
