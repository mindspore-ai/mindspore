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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARRAY_REDUCE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARRAY_REDUCE_GPU_KERNEL_H_

#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
namespace mindspore {
namespace kernel {
const std::map<std::string, cudnnReduceTensorOp_t> kReduceTypeMap = {
  {"ReduceMax", CUDNN_REDUCE_TENSOR_MAX}, {"ReduceMean", CUDNN_REDUCE_TENSOR_AVG},
  {"ReduceSum", CUDNN_REDUCE_TENSOR_ADD}, {"ReduceMin", CUDNN_REDUCE_TENSOR_MIN},
  {"ReduceAny", CUDNN_REDUCE_TENSOR_MAX}, {"ReduceAll", CUDNN_REDUCE_TENSOR_MUL},
};
template <typename T>
class ArrayReduceGpuKernel : public GpuKernel {
 public:
  ArrayReduceGpuKernel() { ResetResource(); }
  ~ArrayReduceGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    T *workspace_addr = GetDeviceAddress<T>(workspace, 0);

    T alpha = static_cast<T>(1.0f);
    T beta = static_cast<T>(0.0f);
    if (all_match_) {
      MS_LOG(DEBUG)
        << "The corresponding dimensions of the input and output tensors all match. No need to call cuDNN kernel.";
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(output_addr, input_addr, inputs[0]->size, cudaMemcpyDeviceToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync failed in ArrayReduceGpuKernel::Launch.");
    } else {
      if (data_type_ == CUDNN_DATA_DOUBLE) {
        CHECK_CUDNN_RET_WITH_EXCEPT(
          kernel_node_,
          cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, workspace_addr, workspace_size_,
                            &alpha, inputA_descriptor_, input_addr, &beta, outputC_descriptor_, output_addr),
          "cudnnReduceTensor failed.");
      } else {
        const float alphaf = static_cast<float>(alpha);
        const float betaf = static_cast<float>(beta);
        CHECK_CUDNN_RET_WITH_EXCEPT(
          kernel_node_,
          cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, workspace_addr, workspace_size_,
                            &alphaf, inputA_descriptor_, input_addr, &betaf, outputC_descriptor_, output_addr),
          "cudnnReduceTensor failed.");
      }
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    InitResource();
    auto type_id = TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0));
    auto node_name = AnfAlgo::GetCNodeName(kernel_node);
    if ((node_name == kReduceAnyOpName || node_name == kReduceAllOpName) &&
        std::strncmp(type_id, "kNumberTypeBool", std::strlen(type_id)) != 0) {
      MS_LOG(ERROR) << "Input data type of ReduceAny or ReduceAll should be bool, but got " << type_id;
      return false;
    }
    data_type_ = GetCudnnDataType(type_id);
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but reduce op needs 1 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but reduce op needs 1 output.";
      return false;
    }
    int input_dim_length = SizeToInt(AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0).size());

    if (AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("axis")->isa<ValueTuple>() ||
        AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("axis")->isa<ValueList>()) {
      std::vector<int> attr_axis;
      std::vector<int64_t> attr_axis_me = GetAttr<std::vector<int64_t>>(kernel_node, "axis");
      (void)std::transform(attr_axis_me.begin(), attr_axis_me.end(), std::back_inserter(attr_axis),
                           [](const int64_t &value) { return static_cast<int>(value); });
      if (attr_axis.empty()) {
        axis_.push_back(-1);
      } else {
        for (auto axis : attr_axis) {
          axis < 0 ? axis_.push_back(axis + input_dim_length) : axis_.push_back(axis);
        }
        std::sort(axis_.begin(), axis_.end());
      }
    } else if (AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("axis")->isa<Int64Imm>()) {
      int axis = static_cast<int>(GetAttr<int64_t>(kernel_node, "axis"));
      axis < 0 ? axis_.push_back(axis + input_dim_length) : axis_.push_back(axis);
    } else {
      MS_LOG(EXCEPTION) << "Attribute axis type is invalid.";
    }
    keep_dims_ = GetAttr<bool>(kernel_node, "keep_dims");

    auto inputA_shape = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0);
    auto outputC_shape = AnfAlgo::GetOutputRealDeviceShapeIfExist(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(inputA_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "ArrayReduceGpuKernel input is null";
      InitSizeLists();
      return true;
    }
    InferInAndOutDesc(inputA_shape, outputC_shape);
    InferArrayReduceType(kernel_node);

    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    cudnn_handle_ = nullptr;
    reduce_tensor_op_ = CUDNN_REDUCE_TENSOR_ADD;
    data_type_ = CUDNN_DATA_FLOAT;
    nan_prop_ = CUDNN_NOT_PROPAGATE_NAN;
    reduce_indices_ = CUDNN_REDUCE_TENSOR_NO_INDICES;
    reduce_tensor_descriptor_ = nullptr;
    inputA_descriptor_ = nullptr;
    outputC_descriptor_ = nullptr;
    keep_dims_ = false;
    all_match_ = false;
    is_null_input_ = false;
    input_size_ = 0;
    output_size_ = 0;
    workspace_size_ = 0;
    axis_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyReduceTensorDescriptor(reduce_tensor_descriptor_),
                               "cudnnDestroyReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(inputA_descriptor_),
                               "cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(outputC_descriptor_),
                               "cudnnDestroyTensorDescriptor failed.");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateReduceTensorDescriptor(&reduce_tensor_descriptor_),
                                "cudnnCreateReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&inputA_descriptor_),
                                "cudnnCreateTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&outputC_descriptor_),
                                "cudnnCreateTensorDescriptor failed.");
  }
  void InitSizeLists() override {
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(inputA_descriptor_, &input_size_),
                                "cudnnGetTensorSizeInBytes failed.");
    input_size_list_.push_back(input_size_);

    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(outputC_descriptor_, &output_size_),
                                "cudnnGetTensorSizeInBytes failed.");
    output_size_list_.push_back(output_size_);

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnGetReductionWorkspaceSize(cudnn_handle_, reduce_tensor_descriptor_, inputA_descriptor_, outputC_descriptor_,
                                     &workspace_size_),
      "cudnnGetReductionWorkspaceSize failed.");
    workspace_size_list_.push_back(workspace_size_);
    return;
  }

 private:
  void InferArrayReduceType(const CNodePtr &kernel_node) {
    std::string kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    auto iter = kReduceTypeMap.find(kernel_name);
    if (iter == kReduceTypeMap.end()) {
      MS_LOG(EXCEPTION) << "Array reduce kernel type " << kernel_name << " is not supported.";
    }
    reduce_tensor_op_ = iter->second;
    // add check for float64
    cudnnDataType_t comp_type = (data_type_ == CUDNN_DATA_DOUBLE) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetReduceTensorDescriptor(reduce_tensor_descriptor_, reduce_tensor_op_, comp_type,
                                                               nan_prop_, reduce_indices_, CUDNN_32BIT_INDICES),
                                "cudnnSetReduceTensorDescriptor failed");
    return;
  }
  void InferInAndOutDesc(const std::vector<size_t> &input_shape, const std::vector<size_t> &output_shape) {
    std::vector<size_t> inputA;
    std::vector<size_t> outputC_shape = output_shape;
    const int split_dim = 4;

    if (input_shape.size() <= split_dim) {
      ShapeNdTo4d(input_shape, &inputA);
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSetTensor4dDescriptor(inputA_descriptor_, CUDNN_TENSOR_NCHW, data_type_, SizeToInt(inputA[0]),
                                   SizeToInt(inputA[1]), SizeToInt(inputA[2]), SizeToInt(inputA[3])),
        "cudnnSetTensor4dDescriptor failed");
    } else {
      CudnnSetTensorNdDescriptor(input_shape, inputA_descriptor_, data_type_, kernel_node_);
      for (auto dim : input_shape) {
        inputA.emplace_back(SizeToInt(dim));
      }
    }

    if (axis_[0] == -1) {
      outputC_shape.resize(input_shape.size(), 1);
      if (outputC_shape.size() <= split_dim) {
        CHECK_CUDNN_RET_WITH_EXCEPT(
          kernel_node_, cudnnSetTensor4dDescriptor(outputC_descriptor_, CUDNN_TENSOR_NCHW, data_type_, 1, 1, 1, 1),
          "cudnnSetTensor4dDescriptor failed");
      } else {
        CudnnSetTensorNdDescriptor(outputC_shape, outputC_descriptor_, data_type_, kernel_node_);
      }

      for (auto dim : inputA) {
        if (dim != 1) {
          return;
        }
      }

      all_match_ = true;
      return;
    }

    std::vector<size_t> outputC;
    if (!keep_dims_) {
      for (auto i : axis_) {
        (void)(outputC_shape.insert(outputC_shape.begin() + i, 1));
      }
    }

    if (outputC_shape.size() <= split_dim) {
      ShapeNdTo4d(outputC_shape, &outputC);
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSetTensor4dDescriptor(outputC_descriptor_, CUDNN_TENSOR_NCHW, data_type_, SizeToInt(outputC[0]),
                                   SizeToInt(outputC[1]), SizeToInt(outputC[2]), SizeToInt(outputC[3])),
        "cudnnSetTensor4dDescriptor failed");
    } else {
      CudnnSetTensorNdDescriptor(outputC_shape, outputC_descriptor_, data_type_, kernel_node_);
      for (auto dim : outputC_shape) {
        outputC.emplace_back(SizeToInt(dim));
      }
    }

    if (inputA == outputC) {
      all_match_ = true;
    }
    return;
  }

  cudnnHandle_t cudnn_handle_;
  cudnnReduceTensorOp_t reduce_tensor_op_;
  cudnnDataType_t data_type_;
  cudnnNanPropagation_t nan_prop_;
  cudnnReduceTensorIndices_t reduce_indices_;
  cudnnReduceTensorDescriptor_t reduce_tensor_descriptor_;
  cudnnTensorDescriptor_t inputA_descriptor_;
  cudnnTensorDescriptor_t outputC_descriptor_;

  std::vector<int> axis_;
  bool keep_dims_;
  bool all_match_;
  bool is_null_input_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARRAY_REDUCE_GPU_KERNEL_H_
