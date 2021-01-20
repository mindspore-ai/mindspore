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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BIAS_ADD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BIAS_ADD_GPU_KERNEL_H_
#include <cuda_runtime_api.h>
#include <string>
#include <algorithm>
#include <memory>
#include <vector>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class BiasAddGpuKernel : public GpuKernel {
 public:
  BiasAddGpuKernel() { ResetResource(); }
  ~BiasAddGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    VARIABLE_NOT_USED(stream_ptr);
    if (is_null_input_) {
      return true;
    }

    T *x_addr = GetDeviceAddress<T>(inputs, 0);
    T *b_addr = GetDeviceAddress<T>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    try {
      const float alpha = 1;
      const float beta = 0;
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnOpTensor(cudnn_handle_, op_desc_, &alpha, x_desc_, x_addr, &alpha, b_desc_,
                                                b_addr, &beta, x_desc_, output_addr),
                                  "cudnnOpTensor failed");
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "Encountered an exception: " << e.what() << " when invoke cudnnOpTensor";
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    InitResource();
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    auto x_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto num_dims = x_shape.size();
    is_null_input_ = CHECK_NULL_INPUT(x_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "input is null";
      InitSizeLists();
      return true;
    }

    if (num_dims < 2) {
      MS_LOG(EXCEPTION) << "input dims must be at least 2, but got " << num_dims;
    }

    std::string format = GetAttr<std::string>(kernel_node, "format");
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

    auto input_device_format = AnfAlgo::GetInputFormat(kernel_node, 0);
    auto cudnn_cal_format = (input_device_format == kOpFormat_NHWC) ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensorNdDescriptorEx(x_desc_, cudnn_cal_format, cudnn_data_type_, SizeToInt(cudnn_dims), x_dims.get()),
      "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensorNdDescriptorEx(b_desc_, cudnn_cal_format, cudnn_data_type_, SizeToInt(cudnn_dims), b_dims.get()),
      "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetOpTensorDescriptor(op_desc_, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN),
      "cudnnSetOpTensorDescriptor failed");

    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    cudnn_handle_ = nullptr;
    cudnn_data_type_ = CUDNN_DATA_FLOAT;
    x_desc_ = nullptr;
    b_desc_ = nullptr;
    op_desc_ = nullptr;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyOpTensorDescriptor(op_desc_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(b_desc_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(x_desc_),
                               "cudnnDestroyOpTensorDescriptor failed");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&x_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&b_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateOpTensorDescriptor(&op_desc_),
                                "cudnnCreateOpTensorDescriptor failed");
  }

  void InitSizeLists() override {
    size_t x_size, b_size;
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(x_desc_, &x_size),
                                "cudnnGetTensorSizeInBytes failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(b_desc_, &b_size),
                                "cudnnGetTensorSizeInBytes failed.");
    input_size_list_.push_back(x_size);
    input_size_list_.push_back(b_size);
    output_size_list_.push_back(x_size);
  }

 private:
  cudnnHandle_t cudnn_handle_;
  cudnnDataType_t cudnn_data_type_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t b_desc_;
  cudnnOpTensorDescriptor_t op_desc_;
  bool is_null_input_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_BIAS_ADD_GPU_KERNEL_H_
