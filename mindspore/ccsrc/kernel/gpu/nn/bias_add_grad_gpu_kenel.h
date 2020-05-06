/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_BIAS_ADD_GRAD_GPU_KENEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_BIAS_ADD_GRAD_GPU_KENEL_H_

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class BiasAddGradGpuKernel : public GpuKernel {
 public:
  BiasAddGradGpuKernel()
      : same_dims_(true),
        cudnn_handle_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        dy_desc_(nullptr),
        db_desc_(nullptr),
        op_desc_(nullptr) {}
  ~BiasAddGradGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override {
    T *dy_addr = GetDeviceAddress<T>(inputs, 0);
    T *db_addr = GetDeviceAddress<T>(outputs, 0);
    T *indices_addr = GetDeviceAddress<T>(workspace, 0);
    T *workspace_addr = GetDeviceAddress<T>(workspace, 1);

    const float alpha = 1;
    const float beta = 0;
    if (same_dims_) {
      CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(db_addr, dy_addr, output_size_list_[0], cudaMemcpyDeviceToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync failed.");
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnReduceTensor(cudnn_handle_, op_desc_, indices_addr, workspace_size_list_[0], workspace_addr,
                          workspace_size_list_[1], &alpha, dy_desc_, dy_addr, &beta, db_desc_, db_addr),
        "cudnnReduceTensor failed");
    }

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    cudnn_data_type_ = kCudnnDtypeMap[TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0))];
    auto dy_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto num_dims = dy_shape.size();
    if (num_dims < 2) {
      MS_LOG(EXCEPTION) << "input dims must be at least 2, but got " << num_dims;
    }

    std::string format = GetAttr<std::string>(kernel_node, "data_format");
    string::size_type pos = format.find("C");
    if (pos == std::string::npos || pos >= num_dims) {
      MS_LOG(EXCEPTION) << "format '" << format << "' invalid";
    }

    // Expand to 4 dims for cudnnSetTensorNdDescriptorEx.
    auto cudnn_dims = std::max(num_dims, 4UL);
    std::unique_ptr<int[]> dy_dims = std::make_unique<int[]>(cudnn_dims);
    std::unique_ptr<int[]> db_dims = std::make_unique<int[]>(cudnn_dims);
    for (size_t i = 0; i < cudnn_dims; i++) {
      dy_dims[i] = (i < num_dims) ? SizeToInt(dy_shape[i]) : 1;
      db_dims[i] = (i == pos) ? SizeToInt(dy_shape[i]) : 1;

      if (dy_dims[i] != db_dims[i]) {
        same_dims_ = false;
      }
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensorNdDescriptorEx(dy_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(cudnn_dims), dy_dims.get()),
      "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensorNdDescriptorEx(db_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(cudnn_dims), db_dims.get()),
      "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetReduceTensorDescriptor(op_desc_, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN,
                                     CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES),
      "cudnnSetReduceTensorDescriptor failed");

    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&dy_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&db_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateReduceTensorDescriptor(&op_desc_), "cudnnCreateOpTensorDescriptor failed");
  }
  void InitSizeLists() override {
    size_t dy_size, db_size;
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(dy_desc_, &dy_size), "cudnnGetTensorSizeInBytes failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(db_desc_, &db_size), "cudnnGetTensorSizeInBytes failed");
    input_size_list_.push_back(dy_size);
    output_size_list_.push_back(db_size);

    size_t indices_size, workspace_size;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnGetReductionIndicesSize(cudnn_handle_, op_desc_, dy_desc_, db_desc_, &indices_size),
      "cudnnGetReductionIndicesSize failed")
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnGetReductionWorkspaceSize(cudnn_handle_, op_desc_, dy_desc_, db_desc_, &workspace_size),
      "cudnnGetReductionWorkspaceSize failed")
    workspace_size_list_.push_back(indices_size);
    workspace_size_list_.push_back(workspace_size);
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnDestroyReduceTensorDescriptor(op_desc_),
                                "cudnnDestroyReduceTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(db_desc_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(dy_desc_), "cudnnDestroyOpTensorDescriptor failed");
  }

  bool same_dims_;
  cudnnHandle_t cudnn_handle_;
  cudnnDataType_t cudnn_data_type_;
  cudnnTensorDescriptor_t dy_desc_;
  cudnnTensorDescriptor_t db_desc_;
  cudnnReduceTensorDescriptor_t op_desc_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_BIAS_ADD_GRAD_GPU_KENEL_H_
