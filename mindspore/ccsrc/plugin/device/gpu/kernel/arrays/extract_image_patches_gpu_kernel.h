/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl_opt.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/extract_image_patches_impl.cuh"
#include "ops/extract_image_patches.h"

namespace mindspore {
namespace kernel {
constexpr int64_t kMidDividend = 2;
class ExtractImagePatchesKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<ExtractImagePatchesKernelMod> {
 public:
  ExtractImagePatchesKernelMod() { ResetResource(); }
  ~ExtractImagePatchesKernelMod() override = default;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    stream_ptr_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  void ResetResource() noexcept;
  template <class T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs) {
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

    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(input_shape, &input_shape_[0], shape_size, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr_)),
      "cudaMemcpyAsync input_shape_ failed");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(input_to_nhwc_axis, &to_nhwc_axis[0], shape_size, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr_)),
      "cudaMemcpyAsync to_nhwc_axis failed");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(t_output_shape, &t_output_shape_[0], shape_size, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr_)),
      "cudaMemcpyAsync t_output_shape_ failed");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(t_output_to_nchw_axis, &to_nchw_axis[0], shape_size, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr_)),
      "cudaMemcpyAsync to_nchw_axis failed");
    CalNCHW2NHWCInterface(input_size_, shape_size / sizeof(size_t), input, &input_shape_[0], &to_nhwc_axis[0],
                          input_shape, input_to_nhwc_axis, t_input, reinterpret_cast<cudaStream_t>(stream_ptr_));
    CalExtractImagePatchesNHWC(output_size_, stride_row_, stride_col_, rate_row_, rate_col_, output_cols_, need_batch_,
                               row_stride_, patch_stride_, other_stride_, input_row_size_, input_col_size_,
                               row_padding_top_, col_padding_left_, col_input_stride_, row_input_stride_,
                               patch_input_stride_, output_depth_, t_input, t_output,
                               reinterpret_cast<cudaStream_t>(stream_ptr_));
    CalNHWC2NCHWInterface(output_size_, shape_size / sizeof(size_t), t_output, &t_output_shape_[0], &to_nchw_axis[0],
                          t_output_shape, t_output_to_nchw_axis, output, reinterpret_cast<cudaStream_t>(stream_ptr_));
    return true;
  }

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
  void *stream_ptr_ = nullptr;
  std::vector<size_t> input_shape_;
  std::vector<size_t> t_output_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_EXTRACT_IMAGE_PATCHES_GPU_KERNEL_H_
