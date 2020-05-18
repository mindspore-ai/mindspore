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

#ifndef MINDSPORE_BIAS_ADD_GPU_KERNEL_H
#define MINDSPORE_BIAS_ADD_GPU_KERNEL_H
#include <cuda_runtime_api.h>
#include <string>
#include <algorithm>
#include <memory>
#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class BiasAddGpuKernel : public GpuKernel {
 public:
  BiasAddGpuKernel()
      : cudnn_handle_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        x_desc_(nullptr),
        b_desc_(nullptr),
        op_desc_(nullptr) {}
  ~BiasAddGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    VARIABLE_NOT_USED(stream_ptr);
    T *x_addr = GetDeviceAddress<T>(inputs, 0);
    T *b_addr = GetDeviceAddress<T>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    try {
      const float alpha = 1;
      const float beta = 0;
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnOpTensor(cudnn_handle_, op_desc_, &alpha, x_desc_, x_addr, &alpha, b_desc_,
                                                b_addr, &beta, x_desc_, output_addr),
                                  "cudnnOpTensor failed");
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "Encountered an exception: " << e.what() << " when invoke cudnnOpTensor";
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    cudnn_data_type_ = kCudnnDtypeMap[TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0))];
    auto x_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto num_dims = x_shape.size();
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
    std::unique_ptr<int[]> x_dims = std::make_unique<int[]>(cudnn_dims);
    std::unique_ptr<int[]> b_dims = std::make_unique<int[]>(cudnn_dims);
    for (size_t i = 0; i < cudnn_dims; i++) {
      x_dims[i] = (i < num_dims) ? SizeToInt(x_shape[i]) : 1;
      b_dims[i] = (i == pos) ? SizeToInt(x_shape[i]) : 1;
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensorNdDescriptorEx(x_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(cudnn_dims), x_dims.get()),
      "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensorNdDescriptorEx(b_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(cudnn_dims), b_dims.get()),
      "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetOpTensorDescriptor(op_desc_, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN),
      "cudnnSetOpTensorDescriptor failed");

    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&x_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&b_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateOpTensorDescriptor(&op_desc_), "cudnnCreateOpTensorDescriptor failed");
  }
  void InitSizeLists() override {
    size_t x_size, b_size;
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(x_desc_, &x_size), "cudnnGetTensorSizeInBytes failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(b_desc_, &b_size), "cudnnGetTensorSizeInBytes failed.");
    input_size_list_.push_back(x_size);
    input_size_list_.push_back(b_size);
    output_size_list_.push_back(x_size);
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyOpTensorDescriptor(op_desc_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(b_desc_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(x_desc_), "cudnnDestroyOpTensorDescriptor failed");
  }

  cudnnHandle_t cudnn_handle_;
  cudnnDataType_t cudnn_data_type_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t b_desc_;
  cudnnOpTensorDescriptor_t op_desc_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_BIAS_ADD_GPU_KERNEL_H
