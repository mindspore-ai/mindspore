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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MEDIAN_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MEDIAN_GRAD_GPU_KERNEL_H_

#include <vector>
#include <map>
#include <memory>
#include "mindspore/ops/infer/grad/median_grad.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/median_grad_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t kMedianOutputsNum = 1;
constexpr size_t kInputsNum4 = 4;
constexpr size_t kInputsNum3 = 3;
template <typename T, typename S, typename V>
class MedianGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  MedianGradGpuKernelMod() : global_median_(false), keep_dims_(false), axis_(0) {}
  ~MedianGradGpuKernelMod() = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    T *y_grad = GetDeviceAddress<T>(inputs, kIndex0);
    T *x = GetDeviceAddress<T>(inputs, kIndex1);
    T *y = GetDeviceAddress<T>(inputs, kIndex2);
    S *indices = nullptr;
    V *output0_addr = GetDeviceAddress<V>(outputs, kIndex0);
    if (!global_median_) {
      indices = GetDeviceAddress<S>(inputs, kIndex3);
    }

    int *elem_num_each_dim_x = GetDeviceAddress<int>(workspace, kIndex0);
    int *elem_num_each_dim_y = GetDeviceAddress<int>(workspace, kIndex1);
    int *repeat_val = GetDeviceAddress<int>(workspace, kIndex2);

    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemsetAsync(output0_addr, 0, outputs[0]->size(), reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemSet Failed");

    if (!global_median_) {
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(elem_num_each_dim_x, &elem_num_each_dim_x_[0], sizeof(int) * input1_dim_,
                        cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cudaMemcpyAsync elem_num_each_dim_x failed");
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(elem_num_each_dim_y, &elem_num_each_dim_y_[0], sizeof(int) * input1_dim_,
                        cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cudaMemcpyAsync elem_num_each_dim_y failed");
    }

    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemsetAsync(repeat_val, 0, sizeof(int), reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemset failed in repeat_val.");

    auto status =
      MedianGrad(y_grad, x, y, indices, output0_addr, axis_, global_median_, input0_size_, input1_size_, input1_dim_,
                 elem_num_each_dim_x, elem_num_each_dim_y, repeat_val, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    if (((inputs.size() != kInputsNum4) && (inputs.size() != kInputsNum3)) || outputs.size() > kMedianOutputsNum) {
      MS_LOG(ERROR) << kernel_name_ << ": input size should be 4 or 3"
                    << "but get " << inputs.size() << " and output size should be 1, but get " << outputs.size();
      return false;
    }
    global_median_ = GetValue<bool>(primitive_->GetAttr(ops::kGlobalMedian));
    keep_dims_ = GetValue<bool>(primitive_->GetAttr(ops::kKeepDims));
    axis_ = GetValue<int64_t>(primitive_->GetAttr(ops::kAxis));
    input_shape_ = inputs[1]->GetShapeVector();
    input1_dim_ = input_shape_.size();
    std::vector<int64_t> input0_shape = inputs[0]->GetShapeVector();
    input1_size_ = 1;
    input0_size_ = 1;
    for (size_t i = 0; i < input_shape_.size(); i++) {
      input1_size_ *= input_shape_[i];
    }
    for (size_t i = 0; i < input0_shape.size(); i++) {
      input0_size_ *= input0_shape[i];
    }
    if (global_median_) {
      input_shape_.clear();
      input_shape_.push_back(input1_size_);
    } else {
      std::vector<int64_t> shape_keepdim;
      for (int64_t i = 0; i < input1_dim_; i++) {
        if (i == axis_) {
          shape_keepdim.push_back(1);
        } else {
          shape_keepdim.push_back(input_shape_[i]);
        }
      }
      int elem_num_x = 1;
      int elem_num_y = 1;
      for (size_t i = 0; i < shape_keepdim.size(); i++) {
        elem_num_each_dim_x_.insert(elem_num_each_dim_x_.begin(), elem_num_x);
        elem_num_x *= input_shape_[shape_keepdim.size() - 1 - i];
        elem_num_each_dim_y_.insert(elem_num_each_dim_y_.begin(), elem_num_y);
        elem_num_y *= shape_keepdim[shape_keepdim.size() - 1 - i];
      }
    }
    ResetResource();
    InitWorkSpaceSizeList();
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    int ret = KernelMod::Resize(inputs, outputs);
    if (ret != 0) {
      return ret;
    }
    input_shape_ = inputs[1]->GetShapeVector();
    std::vector<int64_t> input0_shape = inputs[0]->GetShapeVector();
    input1_dim_ = input_shape_.size();
    input1_size_ = 1;
    input0_size_ = 1;
    if (input1_dim_ == 0) {
      if (axis_ < -1 || axis_ > 0) {
        MS_LOG(EXCEPTION) << "For 'MedianGrad'"
                          << "', the 'axis' must be in the range [-1,1), but got " << axis_;
      }
    } else if (axis_ < -input1_dim_ || axis_ >= input1_dim_) {
      MS_LOG(EXCEPTION) << "For 'MedianGrad'"
                        << "', the 'axis' must be in the range [-" << input1_dim_ << "," << input1_dim_ << "), but got "
                        << axis_;
    }
    if (axis_ < 0) {
      if (input1_dim_ == 0) {
        axis_ = 0;
      } else {
        axis_ += input1_dim_;
      }
    }
    for (size_t i = 0; i < input_shape_.size(); i++) {
      input1_size_ *= input_shape_[i];
    }
    for (size_t i = 0; i < input0_shape.size(); i++) {
      input0_size_ *= input0_shape[i];
    }
    if (global_median_) {
      input_shape_.clear();
      input_shape_.push_back(input1_size_);
    } else {
      std::vector<int64_t> shape_keepdim;
      for (int64_t i = 0; i < input1_dim_; i++) {
        if (i == axis_) {
          shape_keepdim.push_back(1);
        } else {
          shape_keepdim.push_back(input_shape_[i]);
        }
      }
      int elem_num_x = 1;
      int elem_num_y = 1;
      elem_num_each_dim_x_.clear();
      elem_num_each_dim_y_.clear();
      for (size_t i = 0; i < shape_keepdim.size(); i++) {
        elem_num_each_dim_x_.insert(elem_num_each_dim_x_.begin(), elem_num_x);
        elem_num_x *= input_shape_[shape_keepdim.size() - 1 - i];
        elem_num_each_dim_y_.insert(elem_num_each_dim_y_.begin(), elem_num_y);
        elem_num_y *= shape_keepdim[shape_keepdim.size() - 1 - i];
      }
    }
    InitWorkSpaceSizeList();
    return KRET_OK;
  }

  std::vector<KernelAttr> GetOpSupport() {
    static std::vector<KernelAttr> support_list = {KernelAttr()
                                                     .AddInputAttr(kNumberTypeInt16)
                                                     .AddInputAttr(kNumberTypeInt16)
                                                     .AddInputAttr(kNumberTypeInt16)
                                                     .AddOutputAttr(kNumberTypeInt64)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeInt32)
                                                     .AddInputAttr(kNumberTypeInt32)
                                                     .AddInputAttr(kNumberTypeInt32)
                                                     .AddOutputAttr(kNumberTypeInt64)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddInputAttr(kNumberTypeInt64)
                                                     .AddOutputAttr(kNumberTypeInt64)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddInputAttr(kNumberTypeFloat32)
                                                     .AddOutputAttr(kNumberTypeInt64)
                                                     .AddOutputAttr(kNumberTypeFloat32),
                                                   KernelAttr()
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddInputAttr(kNumberTypeFloat64)
                                                     .AddOutputAttr(kNumberTypeInt64)
                                                     .AddOutputAttr(kNumberTypeFloat32)};
    return support_list;
  }

 protected:
  void ResetResource() noexcept {
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 private:
  void InitWorkSpaceSizeList() {
    workspace_size_list_.push_back(input1_dim_ * sizeof(int));
    workspace_size_list_.push_back(input1_dim_ * sizeof(int));
    workspace_size_list_.push_back(sizeof(int));
  }

  bool global_median_;
  bool keep_dims_;
  int64_t axis_;
  int64_t input1_dim_;
  int64_t input0_size_;
  int64_t input1_size_;
  std::vector<int64_t> input_shape_;
  std::vector<int> elem_num_each_dim_x_;
  std::vector<int> elem_num_each_dim_y_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MEDIAN_GRAD_GPU_KERNEL_H_
