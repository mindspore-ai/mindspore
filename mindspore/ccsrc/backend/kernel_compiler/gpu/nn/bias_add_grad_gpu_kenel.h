/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BIAS_ADD_GRAD_GPU_KENEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BIAS_ADD_GRAD_GPU_KENEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "backend/kernel_compiler/gpu/cuda_impl/bias_add_grad_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class BiasAddGradGpuKernel : public GpuKernel {
 public:
  BiasAddGradGpuKernel()
      : same_dims_(true),
        use_cudnn_(false),
        dy_num_(1),
        db_num_(1),
        bias_size_(0),
        cudnn_handle_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        cudnn_compute_format_(CUDNN_TENSOR_NCHW),
        dy_desc_(nullptr),
        db_desc_(nullptr),
        op_desc_(nullptr) {}
  ~BiasAddGradGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *dy_addr = GetDeviceAddress<T>(inputs, 0);
    T *db_addr = GetDeviceAddress<T>(outputs, 0);
    if (same_dims_) {
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(db_addr, dy_addr, output_size_list_[0], cudaMemcpyDeviceToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync failed.");
    } else {
      if (use_cudnn_) {  // shared memory not satisfied or num_dim > 4
        T *indices_addr = GetPossiblyNullDeviceAddress<T>(workspace, 0);
        T *workspace_addr = GetPossiblyNullDeviceAddress<T>(workspace, 1);
        const float alpha = 1;
        const float beta = 0;
        CHECK_CUDNN_RET_WITH_EXCEPT(
          kernel_node_,
          cudnnReduceTensor(cudnn_handle_, op_desc_, indices_addr, workspace_size_list_[0], workspace_addr,
                            workspace_size_list_[1], &alpha, dy_desc_, dy_addr, &beta, db_desc_, db_addr),
          "cudnnReduceTensor failed");
      } else {  // use own implementation which is more efficient but cannot process num_dim > 4
        if (data_format_ == kOpFormat_NHWC) {
          CalBiasAddGradNHWC(dy_num_, bias_size_, dy_addr, db_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
        } else {
          CalBiasAddGradNCHW(dy_num_, bias_size_, SizeToInt(dy_shape_[2]), SizeToInt(dy_shape_[3]), dy_addr, db_addr,
                             reinterpret_cast<cudaStream_t>(stream_ptr));
        }
      }
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    auto dy_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(dy_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'BiasAddGradGpuKernel', input is null";
      InitSizeLists();
      return true;
    }
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    auto input_device_format = AnfAlgo::GetInputFormat(kernel_node, 0);
    cudnn_compute_format_ = (input_device_format == kOpFormat_NHWC) ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
    num_dims_ = dy_shape.size();
    if (num_dims_ < 2) {
      MS_LOG(EXCEPTION) << "input dims must be at least 2, but got " << num_dims_;
    }
    std::string format = GetAttr<std::string>(kernel_node, "format");
    string::size_type pos = format.find("C");
    if (pos == std::string::npos || pos >= num_dims_) {
      MS_LOG(EXCEPTION) << "format '" << format << "' invalid";
    }
    bias_size_ = dy_shape[pos];
    auto num_dims_fix = std::max(num_dims_, 4UL);
    for (size_t i = 0; i < num_dims_fix; i++) {
      dy_shape_.push_back((i < num_dims_) ? dy_shape[i] : 1);
      db_shape_.push_back((i == pos) ? dy_shape[i] : 1);
      if (dy_shape_[i] != db_shape_[i]) {
        same_dims_ = false;
      }
    }
    for (size_t i = 0; i < dy_shape_.size(); i++) {
      dy_num_ *= dy_shape_[i];
    }
    for (size_t i = 0; i < db_shape_.size(); i++) {
      db_num_ *= db_shape_[i];
    }
    data_format_ = input_device_format;  // for opt implementation
    if (format == kOpFormat_NHWC) {
      data_format_ = kOpFormat_NHWC;
    }
    MethodSelection();
    InitResource();
    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    if (use_cudnn_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnDestroyReduceTensorDescriptor(op_desc_),
                                  "cudnnDestroyReduceTensorDescriptor failed");
      CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(db_desc_),
                                 "cudnnDestroyTensorDescriptor failed");
      CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dy_desc_),
                                 "cudnnDestroyOpTensorDescriptor failed");
    }
  }

 protected:
  void MethodSelection() {
    // opt implementation can only process num_dims_ <= 4
    // for num_dims_ = 2, not time-consuming, use cudnn
    if (num_dims_ > 4 || num_dims_ == 2) {
      use_cudnn_ = true;
      return;
    }
    if (data_format_ == kOpFormat_NHWC) {
      const size_t required_sharedmem_size = 32 * 33 * sizeof(float);
      // nhwc opt implementation performs not so well when bias_size_ <= 6
      if (required_sharedmem_size > SHARED_MEM_PER_BLOCK || bias_size_ <= 6) {
        use_cudnn_ = true;
        return;
      }
    }
  }
  void InitResource() override {
    if (use_cudnn_) {
      cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dy_desc_),
                                  "cudnnCreateTensorDescriptor failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&db_desc_),
                                  "cudnnCreateTensorDescriptor failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateReduceTensorDescriptor(&op_desc_),
                                  "cudnnCreateOpTensorDescriptor failed");
      // Expand to 4 dims for cudnnSetTensorNdDescriptorEx.
      auto cudnn_dims = std::max(num_dims_, 4UL);
      std::unique_ptr<int[]> dy_dims = std::make_unique<int[]>(cudnn_dims);
      std::unique_ptr<int[]> db_dims = std::make_unique<int[]>(cudnn_dims);
      for (size_t i = 0; i < cudnn_dims; i++) {
        dy_dims[i] = SizeToInt(dy_shape_[i]);
        db_dims[i] = SizeToInt(db_shape_[i]);
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetTensorNdDescriptorEx(dy_desc_, cudnn_compute_format_, cudnn_data_type_,
                                                               SizeToInt(cudnn_dims), dy_dims.get()),
                                  "cudnnSetTensorNdDescriptor failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetTensorNdDescriptorEx(db_desc_, cudnn_compute_format_, cudnn_data_type_,
                                                               SizeToInt(cudnn_dims), db_dims.get()),
                                  "cudnnSetTensorNdDescriptor failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSetReduceTensorDescriptor(op_desc_, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN,
                                       CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES),
        "cudnnSetReduceTensorDescriptor failed");
    }
  }
  void InitSizeLists() override {
    if (use_cudnn_) {
      size_t dy_size, db_size;
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(dy_desc_, &dy_size),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(db_desc_, &db_size),
                                  "cudnnGetTensorSizeInBytes failed");
      input_size_list_.push_back(dy_size);
      output_size_list_.push_back(db_size);
      size_t indices_size, workspace_size;
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnGetReductionIndicesSize(cudnn_handle_, op_desc_, dy_desc_, db_desc_, &indices_size),
        "cudnnGetReductionIndicesSize failed")
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnGetReductionWorkspaceSize(cudnn_handle_, op_desc_, dy_desc_, db_desc_, &workspace_size),
        "cudnnGetReductionWorkspaceSize failed")
      workspace_size_list_.push_back(indices_size);
      workspace_size_list_.push_back(workspace_size);
    } else {
      input_size_list_.push_back(dy_num_ * sizeof(T));
      output_size_list_.push_back(db_num_ * sizeof(T));
    }
  }

 private:
  bool same_dims_;
  bool is_null_input_;
  bool use_cudnn_;
  size_t dy_num_;  // for own implementation
  size_t db_num_;
  size_t num_dims_;
  size_t bias_size_;              // for own implementation
  std::vector<size_t> dy_shape_;  // for own implementation
  std::vector<size_t> db_shape_;  // for own implementation
  std::string data_format_ = kOpFormat_NCHW;
  // for cudnn implementation
  cudnnHandle_t cudnn_handle_;
  cudnnDataType_t cudnn_data_type_;
  cudnnTensorFormat_t cudnn_compute_format_;
  cudnnTensorDescriptor_t dy_desc_;
  cudnnTensorDescriptor_t db_desc_;
  cudnnReduceTensorDescriptor_t op_desc_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BIAS_ADD_GRAD_GPU_KENEL_H_
