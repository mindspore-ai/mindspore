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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_BINARYOP_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_BINARYOP_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/unary_op_impl.cuh"
#include "kernel/gpu/kernel_constants.h"
namespace mindspore {
namespace kernel {
enum BinaryOpType { BINARY_OP_ADD = 0, BINARY_OP_SUB, BINARY_OP_MUL, BINARY_OP_DIV, BINARY_OP_INVALID_TYPE = 255 };
const std::map<std::string, BinaryOpType> kBinaryOpTypeMap = {
  {"Sub", BINARY_OP_SUB},
  {"Mul", BINARY_OP_MUL},
  {"RealDiv", BINARY_OP_DIV},
};
template <typename T>
class BinaryOpGpuKernel : public GpuKernel {
 public:
  BinaryOpGpuKernel()
      : cudnn_handle_(nullptr),
        binary_op_type_(BINARY_OP_INVALID_TYPE),
        tensor_op_(CUDNN_OP_TENSOR_MUL),
        inputA_descriptor_(nullptr),
        inputB_descriptor_(nullptr),
        opTensor_descriptor_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        is_null_input_(false),
        input_size_(0),
        output_size_(0),
        workspace_size_(0) {}
  ~BinaryOpGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *input_addr2 = GetDeviceAddress<T>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    const float alpha = 1;
    const float beta = 0;

    T *inputB_addr = nullptr;
    switch (binary_op_type_) {
      case BINARY_OP_SUB: {
        T *workspace_addr = GetDeviceAddress<T>(workspace, 0);
        Negative(input_addr2, workspace_addr, inputs[1]->size / sizeof(T), reinterpret_cast<cudaStream_t>(stream_ptr));
        inputB_addr = workspace_addr;
        break;
      }
      case BINARY_OP_MUL: {
        inputB_addr = input_addr2;
        break;
      }
      case BINARY_OP_DIV: {
        T *workspace_addr = GetDeviceAddress<T>(workspace, 0);
        Reciprocal(input_addr2, workspace_addr, inputs[1]->size / sizeof(T),
                   reinterpret_cast<cudaStream_t>(stream_ptr));
        inputB_addr = workspace_addr;
        break;
      }
      default: {
        MS_LOG(EXCEPTION) << "Binary operation " << binary_op_type_ << " is not supported.";
      }
    }
    if (inputs[0]->size > inputs[1]->size) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnOpTensor(cudnn_handle_, opTensor_descriptor_, &alpha, inputA_descriptor_, input_addr, &alpha,
                      inputB_descriptor_, inputB_addr, &beta, inputA_descriptor_, output_addr),
        "cudnnOpTensor failed");
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnOpTensor(cudnn_handle_, opTensor_descriptor_, &alpha, inputB_descriptor_, inputB_addr, &alpha,
                      inputA_descriptor_, input_addr, &beta, inputB_descriptor_, output_addr),
        "cudnnOpTensor failed");
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    cudnn_data_type_ = kCudnnDtypeMap[TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0))];
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but binary operation needs 2 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but binary operation needs 1 output.";
      return false;
    }
    InferBinaryType(kernel_node);
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto input_shapeB = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    is_null_input_ = CHECK_NULL_INPUT(input_shape) || CHECK_NULL_INPUT(input_shapeB);
    if (is_null_input_) {
      MS_LOG(WARNING) << "BinaryOpGpuKernel input is null";
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
    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&inputA_descriptor_),
                                "cudnnCreateTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&inputB_descriptor_),
                                "cudnnCreateTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateOpTensorDescriptor(&opTensor_descriptor_),
                                "cudnnCreateOpTensorDescriptor failed.");
  }
  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(inputA_descriptor_, &input_size_),
                                  "cudnnGetTensorSizeInBytes failed.");
      input_size_list_.push_back(input_size_);
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(inputB_descriptor_, &output_size_),
                                  "cudnnGetTensorSizeInBytes failed.");
    }
    input_size_list_.push_back(output_size_);
    if (binary_op_type_ == BINARY_OP_DIV || binary_op_type_ == BINARY_OP_SUB) {
      workspace_size_ = output_size_;
    }
    workspace_size_list_.push_back(workspace_size_);

    if (output_size_ > input_size_) {
      output_size_list_.push_back(output_size_);
    } else {
      output_size_list_.push_back(input_size_);
    }
    return;
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(inputA_descriptor_),
                               "cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(inputB_descriptor_),
                               "cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyOpTensorDescriptor(opTensor_descriptor_),
                               "cudnnDestroyOpTensorDescriptor failed.");
  }
  void InferBinaryType(const CNodePtr &kernel_node) {
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    auto iter = kBinaryOpTypeMap.find(kernel_name);
    if (iter == kBinaryOpTypeMap.end()) {
      MS_LOG(EXCEPTION) << "Binary operation " << kernel_name << " is not supported.";
    } else {
      binary_op_type_ = iter->second;
    }

    switch (binary_op_type_) {
      case BINARY_OP_DIV:
      case BINARY_OP_MUL: {
        tensor_op_ = CUDNN_OP_TENSOR_MUL;
        break;
      }
      case BINARY_OP_SUB: {
        tensor_op_ = CUDNN_OP_TENSOR_ADD;
        break;
      }
      default: {
        MS_LOG(EXCEPTION) << "Binary operation " << binary_op_type_ << " is not supported.";
      }
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetOpTensorDescriptor(opTensor_descriptor_, tensor_op_, cudnn_data_type_, CUDNN_NOT_PROPAGATE_NAN),
      "cudnnSetOpTensorDescriptor failed");
    return;
  }

  cudnnHandle_t cudnn_handle_;
  BinaryOpType binary_op_type_;
  cudnnOpTensorOp_t tensor_op_;
  cudnnTensorDescriptor_t inputA_descriptor_;
  cudnnTensorDescriptor_t inputB_descriptor_;
  cudnnOpTensorDescriptor_t opTensor_descriptor_;
  cudnnDataType_t cudnn_data_type_;
  bool is_null_input_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_BINARYOP_GPU_KERNEL_H_
