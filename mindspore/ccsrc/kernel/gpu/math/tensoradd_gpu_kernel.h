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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_TENSORADD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_TENSORADD_GPU_KERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
namespace mindspore {
namespace kernel {
template <typename T>
class TensorAddGpuFwdKernel : public GpuKernel {
 public:
  TensorAddGpuFwdKernel()
      : cudnn_handle_(nullptr),
        inputA_descriptor_(nullptr),
        inputB_descriptor_(nullptr),
        opTensor_descriptor_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        input_size_(0),
        output_size_(0),
        workspace_size_(0),
        is_null_input_(false) {}
  ~TensorAddGpuFwdKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, uintptr_t) {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *input_addr2 = GetDeviceAddress<T>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    const float alpha = 1;
    const float beta = 0;
    // A + B = C. [ C = op(alpha1[0] * A, alpha2[0] * B) + beta[0] * C ]
    // InputA must match the corresponding dimension of the destination tensor outC, and each dimension of the inputB
    // must match the corresponding dimension of outC or must be equal to 1.
    if (inputs[0]->size > inputs[1]->size) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnOpTensor(cudnn_handle_, opTensor_descriptor_, &alpha, inputA_descriptor_, input_addr, &alpha,
                      inputB_descriptor_, input_addr2, &beta, inputA_descriptor_, output_addr),
        "cudnnOpTensor Add failed");
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnOpTensor(cudnn_handle_, opTensor_descriptor_, &alpha, inputB_descriptor_, input_addr2, &alpha,
                      inputA_descriptor_, input_addr, &beta, inputB_descriptor_, output_addr),
        "cudnnOpTensor Add failed");
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) {
    InitResource();
    cudnn_data_type_ = kCudnnDtypeMap[TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0))];
    if (cudnn_data_type_ == CUDNN_DATA_INT32) {
      cudnn_data_type_ = CUDNN_DATA_FLOAT;
    }
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but cudnnAddTensor needs 2 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but cudnnAddTensor needs 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto input_shapeB = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    is_null_input_ = CHECK_NULL_INPUT(input_shape) || CHECK_NULL_INPUT(input_shapeB);
    if (is_null_input_) {
      MS_LOG(WARNING) << "TensorAddGpuFwdKernel input is null";
      InitSizeLists();
      return true;
    }
    int shape_n = input_shape.size() < 4 ? 1 : SizeToInt(input_shape[input_shape.size() - 4]);
    int shape_c = input_shape.size() < 3 ? 1 : SizeToInt(input_shape[input_shape.size() - 3]);
    int shape_h = input_shape.size() < 2 ? 1 : SizeToInt(input_shape[input_shape.size() - 2]);
    int shape_w = input_shape.size() == 0 ? 1 : SizeToInt(input_shape[input_shape.size() - 1]);
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensor4dDescriptor(inputA_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_,
                                                           shape_n, shape_c, shape_h, shape_w),
                                "cudnnSetTensor4dDescriptor failed");
    int shapeB_n = input_shapeB.size() < 4 ? 1 : SizeToInt(input_shapeB[input_shapeB.size() - 4]);
    int shapeB_c = input_shapeB.size() < 3 ? 1 : SizeToInt(input_shapeB[input_shapeB.size() - 3]);
    int shapeB_h = input_shapeB.size() < 2 ? 1 : SizeToInt(input_shapeB[input_shapeB.size() - 2]);
    int shapeB_w = input_shapeB.size() == 0 ? 1 : SizeToInt(input_shapeB[input_shapeB.size() - 1]);
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensor4dDescriptor(inputB_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_,
                                                           shapeB_n, shapeB_c, shapeB_h, shapeB_w),
                                "cudnnSetTensor4dDescriptor failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetOpTensorDescriptor(opTensor_descriptor_, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN),
      "cudnnSetOpTensorDescriptor failed");

    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&inputA_descriptor_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&inputB_descriptor_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateOpTensorDescriptor(&opTensor_descriptor_),
                                "cudnnCreateOpTensorDescriptor failed");
  }
  void InitSizeLists() {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(inputA_descriptor_, &input_size_),
                                  "cudnnGetTensorSizeInBytes failed");
      input_size_list_.push_back(input_size_);
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(inputB_descriptor_, &output_size_),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(output_size_);

    if (output_size_ > input_size_) {
      output_size_list_.push_back(output_size_);
    } else {
      output_size_list_.push_back(input_size_);
    }
    workspace_size_list_.push_back(workspace_size_);

    return;
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(inputA_descriptor_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(inputB_descriptor_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyOpTensorDescriptor(opTensor_descriptor_),
                               "cudnnDestroyOpTensorDescriptor failed");
  }
  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t inputA_descriptor_;
  cudnnTensorDescriptor_t inputB_descriptor_;
  cudnnOpTensorDescriptor_t opTensor_descriptor_;
  cudnnDataType_t cudnn_data_type_;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  bool is_null_input_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_TENSORADD_GPU_KERNEL_H_
