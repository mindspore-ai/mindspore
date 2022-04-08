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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TENSOR_TENSOR_SCATTER_ARITHMETIC_GPU_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TENSOR_TENSOR_SCATTER_ARITHMETIC_GPU_KERNEL_H

#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/tensor_scatter_arithmetic.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T, typename S>
class TensorScatterArithmeticGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  TensorScatterArithmeticGpuKernelMod() = default;
  ~TensorScatterArithmeticGpuKernelMod() {
    if (indices_stride_ != nullptr) {
      device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(static_cast<void *>(indices_stride_));
    }
    if (work_shape_ != nullptr) {
      device::gpu::GPUMemoryAllocator::GetInstance().FreeTensorMem(static_cast<void *>(work_shape_));
    }
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    GetOpType();
    kernel_node_ = kernel_node;
    memcpy_flag_ = false;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != kIndex3) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 3, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != kIndex1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
    input_shapes_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex0);
    indices_shapes_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex1);
    update_shapes_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, kIndex2);
    output_shapes_ = common::AnfAlgo::GetOutputInferShape(kernel_node, kIndex0);
    is_null_input_ = CHECK_SHAPE_NULL(update_shapes_, kernel_name_, "update") ||
                     CHECK_SHAPE_NULL(indices_shapes_, kernel_name_, "indices") ||
                     CHECK_SHAPE_NULL(input_shapes_, kernel_name_, "input_x") ||
                     CHECK_SHAPE_NULL(output_shapes_, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    std::vector<size_t> shape_me = input_shapes_;
    (void)std::transform(shape_me.begin(), shape_me.end(), std::back_inserter(vec_work_shape_),
                         [](const size_t &value) { return static_cast<S>(value); });
    GetSize();
    const size_t indices_len = sizeof(S) * vec_indices_stride_.size();
    void *indices_stride_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(indices_len);
    if (indices_stride_work == nullptr) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the memory alloc of indices_stride_work should be successful, but failed, got size: "
                        << indices_len;
    }
    indices_stride_ = static_cast<S *>(indices_stride_work);
    const size_t vec_work_len = sizeof(S) * vec_work_shape_.size();
    void *work_shape_work = device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(vec_work_len);
    if (work_shape_work == nullptr) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the memory alloc of work_shape_work should be successful, but failed, got size: "
                        << vec_work_len;
    }
    work_shape_ = static_cast<S *>(work_shape_work);
    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    VARIABLE_NOT_USED(workspace);
    T *input = GetDeviceAddress<T>(inputs, kIndex0);
    S *indices = GetDeviceAddress<S>(inputs, kIndex1);
    T *update = GetDeviceAddress<T>(inputs, kIndex2);
    T *output = GetDeviceAddress<T>(outputs, kIndex0);

    if (!memcpy_flag_) {
      const size_t indices_len = sizeof(S) * vec_indices_stride_.size();
      const size_t vec_work_len = sizeof(S) * vec_work_shape_.size();
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(indices_stride_, &vec_indices_stride_[0], indices_len,
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpy failed in TensorScatterArithmeticGpuKernelMod::Launch.");
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(work_shape_, &vec_work_shape_[0], vec_work_len, cudaMemcpyHostToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpy failed in TensorScatterArithmeticGpuKernelMod::Launch.");
      memcpy_flag_ = true;
    }

    const size_t update_size = update_size_ / sizeof(T);
    const size_t output_size = output_size_ / sizeof(T);

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(&output[0], &input[0], input_size_, cudaMemcpyDeviceToDevice,
                                               reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemcpyAsync output failed");

    TensorScatterArithmetic(op_func_type_, input, indices, update, output, block_size_, update_size, output_size,
                            indices_dim_0_, indices_dim_1_, indices_stride_, work_shape_,
                            reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(indices_size_);
    input_size_list_.push_back(update_size_);
    output_size_list_.push_back(output_size_);
  }

  void GetSize() {
    input_size_ = sizeof(T);
    for (const auto &shape_item : input_shapes_) {
      input_size_ *= shape_item;
    }
    indices_size_ = sizeof(S);
    for (const auto &shape_item : indices_shapes_) {
      indices_size_ *= shape_item;
    }
    update_size_ = sizeof(T);
    for (const auto &shape_item : update_shapes_) {
      update_size_ *= shape_item;
    }
    output_size_ = sizeof(T);
    for (const auto &shape_item : output_shapes_) {
      output_size_ *= shape_item;
    }

    // calculate indices dim 0/1
    indices_dim_0_ = indices_shapes_[0];
    indices_dim_1_ = indices_shapes_[indices_shapes_.size() - 1];

    // calculate block_size
    for (size_t i = indices_dim_1_; i < output_shapes_.size(); i++) {
      block_size_ *= output_shapes_[i];
    }

    // calculate indices_stride
    vec_indices_stride_.resize(indices_dim_1_, 0);
    vec_indices_stride_[indices_dim_1_ - 1] = block_size_;

    for (size_t i = indices_dim_1_ - 1; i > 0; --i) {
      vec_indices_stride_[i - 1] = vec_indices_stride_[i] * output_shapes_[i];
    }
  }

 private:
  void GetOpType() {
    static const std::map<std::string, TensorScatterArithmeticFunctionType> kTensorScatterOpTypeMap = {
      {"TensorScatterUpdate", TENSOR_SCATTER_FUNC_UPDATE}, {"TensorScatterMin", TENSOR_SCATTER_FUNC_MIN},
      {"TensorScatterMax", TENSOR_SCATTER_FUNC_MAX},       {"TensorScatterAdd", TENSOR_SCATTER_FUNC_ADD},
      {"TensorScatterSub", TENSOR_SCATTER_FUNC_SUB},       {"TensorScatterMul", TENSOR_SCATTER_FUNC_MUL},
      {"TensorScatterDiv", TENSOR_SCATTER_FUNC_DIV}};
    auto op_type_iter = kTensorScatterOpTypeMap.find(kernel_name_);
    if (op_type_iter == kTensorScatterOpTypeMap.end()) {
      MS_LOG(EXCEPTION) << "Only support these tensor_scatter function: TensorScatterUpdate, TensorScatterMin, "
                           "TensorScatterMax, TensorScatterAdd, TensorScatterSub, TensorScatterMul or TensorScatterDiv "
                           "currently, but got "
                        << kernel_name_;
    }
    op_func_type_ = op_type_iter->second;
  }
  std::vector<size_t> update_shapes_;
  std::vector<size_t> indices_shapes_;
  std::vector<size_t> input_shapes_;
  std::vector<size_t> output_shapes_;
  std::vector<S> vec_indices_stride_;
  std::vector<S> vec_work_shape_;

  std::string kernel_name_{};

  bool memcpy_flag_{false};
  bool is_null_input_{false};
  TensorScatterArithmeticFunctionType op_func_type_{TENSOR_SCATTER_FUNC_INVALID_TYPE};
  size_t input_size_{1};
  size_t update_size_{1};
  size_t indices_size_{1};
  size_t output_size_{1};
  size_t block_size_{1};
  size_t indices_dim_0_{0};
  size_t indices_dim_1_{0};
  S *indices_stride_{nullptr};
  S *work_shape_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_TENSOR_TENSOR_SCATTER_ARITHMETIC_GPU_KERNEL_H
