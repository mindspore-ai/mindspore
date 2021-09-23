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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_TRANSPOSE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_TRANSPOSE_H_

#include <vector>
#include <algorithm>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/transpose_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/transpose_impl_opt.cuh"
namespace mindspore {
namespace kernel {
template <typename T>
class TransposeGpuFwdKernel : public GpuKernel {
 public:
  TransposeGpuFwdKernel() { ResetResource(); }
  ~TransposeGpuFwdKernel() = default;

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
    size_t *input_shape = GetDeviceAddress<size_t>(workspace, 0);
    size_t *input_axis = GetDeviceAddress<size_t>(workspace, 1);
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(input_shape, &input_shape_[0], workspace_size_, cudaMemcpyHostToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync input_shape failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(input_axis, &input_axis_[0], workspace_size_, cudaMemcpyHostToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync input_axis failed");
    size_t size = input_size_ / sizeof(T);

    size_t *h_input_shape = &input_shape_[0];
    size_t *h_input_axis = &input_axis_[0];
    // nhwc->nchw: 0,3,1,2
    if (shape_size_ == 4 && h_input_axis[0] == 0 && h_input_axis[1] == 3 && h_input_axis[2] == 1 &&
        h_input_axis[3] == 2) {
      CalNHWC2NCHWInterface(size, shape_size_, input, h_input_shape, h_input_axis, input_shape, input_axis, output,
                            reinterpret_cast<cudaStream_t>(stream_ptr));
    } else if (shape_size_ == 4 && h_input_axis[0] == 0 && h_input_axis[1] == 2 && h_input_axis[2] == 3 &&
               h_input_axis[3] == 1) {
      // nchw->nhwc: 0,2,3,1
      CalNCHW2NHWCInterface(size, shape_size_, input, h_input_shape, h_input_axis, input_shape, input_axis, output,
                            reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      CalTranspose(size, input, input_shape, input_axis, shape_size_, output,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but transpose needs 1 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but transpose needs 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'TransposeGpuKernel', input is null";
      InitSizeLists();
      return true;
    }
    shape_size_ = input_shape.size();
    if (shape_size_ > TRANSPOSE_MAX_DIMENSION) {
      MS_LOG(EXCEPTION) << "Input is " << shape_size_ << "-D, but transpose supports max " << TRANSPOSE_MAX_DIMENSION
                        << "-D inputs.";
    }

    input_size_ = 1;
    for (size_t i = 0; i < shape_size_; i++) {
      input_size_ *= input_shape[i];
      input_shape_.push_back(input_shape[i]);
    }
    input_size_ *= sizeof(T);
    output_size_ = input_size_;
    std::vector<int64_t> perm = GetAttr<std::vector<int64_t>>(kernel_node, "perm");
    for (size_t j = 0; j < perm.size(); j++) {
      input_axis_.push_back(perm[j]);
    }
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    shape_size_ = 0;
    input_size_ = 0;
    output_size_ = 0;
    workspace_size_ = 0;
    is_null_input_ = false;
    input_shape_.clear();
    input_axis_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    workspace_size_ = shape_size_ * sizeof(size_t);
    workspace_size_list_.push_back(workspace_size_);
    workspace_size_list_.push_back(workspace_size_);
    return;
  }

 private:
  std::vector<size_t> input_shape_;
  std::vector<size_t> input_axis_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  size_t shape_size_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_TRANSPOSE_H_
