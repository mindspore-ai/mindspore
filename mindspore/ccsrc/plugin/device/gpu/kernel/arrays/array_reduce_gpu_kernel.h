/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <functional>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "utils/check_convert_utils.h"
namespace mindspore {
namespace kernel {
const std::map<std::string, cudnnReduceTensorOp_t> kReduceTypeMap = {
  {"ReduceMax", CUDNN_REDUCE_TENSOR_MAX},  {"ReduceMean", CUDNN_REDUCE_TENSOR_AVG},
  {"ReduceSum", CUDNN_REDUCE_TENSOR_ADD},  {"ReduceMin", CUDNN_REDUCE_TENSOR_MIN},
  {"ReduceAny", CUDNN_REDUCE_TENSOR_MAX},  {"ReduceAll", CUDNN_REDUCE_TENSOR_MUL},
  {"ReduceProd", CUDNN_REDUCE_TENSOR_MUL},
};
template <typename T, typename S = int64_t>
class ArrayReduceGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  ArrayReduceGpuKernelMod() { ResetResource(); }
  ~ArrayReduceGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    if (is_dynamic_axis_ && !get_dynamic_axis_value_ && !need_skip_execute_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', fail to get value of the axis when axis is dynamic!";
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);

    T alpha = static_cast<T>(1.0f);
    T beta = static_cast<T>(0.0f);
    if (all_match_ || need_skip_execute_) {
      MS_LOG(DEBUG)
        << "The corresponding dimensions of the input and output tensors all match. No need to call cuDNN kernel.";
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(output_addr, input_addr, input_size_, cudaMemcpyDeviceToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync failed in ArrayReduceGpuKernelMod::Launch.");
    } else {
      T *workspace_addr = GetPossiblyNullDeviceAddress<T>(workspace, 0);
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
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();
    auto type_id = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
    auto type_name = TypeIdLabel(type_id);
    auto node_name = common::AnfAlgo::GetCNodeName(kernel_node);
    if ((node_name == kReduceAnyOpName || node_name == kReduceAllOpName) && type_id != kNumberTypeBool) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input data type must be bool, but got " << type_name;
    }
    data_type_ = GetCudnnDataType(type_name);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    constexpr size_t kDynamicAxisInputNum = 2;
    if (input_num != 1 && input_num != kDynamicAxisInputNum) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1 or " << kDynamicAxisInputNum
                        << ", but got " << input_num;
    }
    if (input_num == kDynamicAxisInputNum) {
      is_dynamic_axis_ = true;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
    }
    auto inputA_shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
    if (is_dynamic_axis_ && AnfAlgo::IsDynamicShapeSkipExecute(kernel_node)) {
      need_skip_execute_ = true;
      input_size_ = std::accumulate(inputA_shape.begin(), inputA_shape.end(), sizeof(T), std::multiplies<size_t>());
      InitSizeLists();
      return true;
    }

    int input_dim_length = SizeToInt(AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0).size());
    std::vector<int64_t> attr_axis;
    if (is_dynamic_axis_) {
      get_dynamic_axis_value_ =
        GetDynamicAttrIntValue(kernel_node, kAxisIndex_, &attr_axis, kernel::GetKernelDepends(kernel_node));
      if (!get_dynamic_axis_value_) {
        InitSizeLists();
        return true;
      }
      dynamic_axis_size_ = attr_axis.size();
    } else {
      attr_axis = GetAxisValue(kernel_node);
    }
    if (attr_axis.empty()) {
      axis_.push_back(-1);
    } else {
      for (auto axis : attr_axis) {
        axis < 0 ? axis_.push_back(axis + input_dim_length) : axis_.push_back(axis);
      }
      std::sort(axis_.begin(), axis_.end());
      auto multiple_pos = std::unique(axis_.begin(), axis_.end());
      axis_.erase(multiple_pos, axis_.end());
    }

    keep_dims_ = GetAttr<bool>(kernel_node, "keep_dims");

    auto outputC_shape = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, 0);
    is_null_input_ =
      CHECK_SHAPE_NULL(inputA_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(outputC_shape, kernel_name_, "output");
    if (is_null_input_) {
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
    kernel_name_ = "ArrayReduce";
    axis_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    dynamic_axis_size_ = 0;
    is_dynamic_axis_ = false;
    get_dynamic_axis_value_ = false;
    need_skip_execute_ = false;
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
    if (need_skip_execute_) {
      input_size_list_.push_back(input_size_);
      input_size_list_.push_back(dynamic_axis_size_ * sizeof(S));
      output_size_ = input_size_;
      output_size_list_.push_back(output_size_);
      return;
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(inputA_descriptor_, &input_size_),
                                "cudnnGetTensorSizeInBytes failed.");
    input_size_list_.push_back(input_size_);

    if (is_dynamic_axis_) {
      input_size_list_.push_back(dynamic_axis_size_ * sizeof(S));
    }

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
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    auto iter = kReduceTypeMap.find(kernel_name);
    if (iter == kReduceTypeMap.end()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "Only support these array reduce kernel types: "
                        << "ReduceMax, ReduceMean, ReduceSum, ReduceMin, ReduceAny, ReduceAll, ReduceProd currently"
                        << ", but got " << kernel_name;
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
  void InferInAndOutDesc(const ShapeVector &input_shape, const ShapeVector &output_shape) {
    ShapeVector inputA;
    ShapeVector outputC_shape = output_shape;
    const int split_dim = 4;
    CheckTensorSize({input_shape, output_shape});
    if (input_shape.size() <= split_dim) {
      ShapeNdTo4d(input_shape, &inputA);
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSetTensor4dDescriptor(inputA_descriptor_, CUDNN_TENSOR_NCHW, data_type_, LongToInt(inputA[0]),
                                   LongToInt(inputA[1]), LongToInt(inputA[2]), LongToInt(inputA[3])),
        "cudnnSetTensor4dDescriptor failed");
    } else {
      CudnnSetTensorNdDescriptor(input_shape, inputA_descriptor_, data_type_, kernel_node_);
      for (auto dim : input_shape) {
        inputA.emplace_back(dim);
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

    ShapeVector outputC;
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
        outputC.emplace_back(dim);
      }
    }

    if (inputA == outputC) {
      all_match_ = true;
    }
    return;
  }

  std::vector<int64_t> GetAxisValue(const CNodePtr &kernel_node) {
    auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    std::vector<int64_t> attr_axis_me;
    auto value_ptr = prim->GetAttr("axis");
    if (value_ptr->isa<tensor::Tensor>()) {
      attr_axis_me = CheckAndConvertUtils::CheckTensorIntValue("axis", value_ptr, kernel_name_);
    } else {
      attr_axis_me = CheckAndConvertUtils::CheckIntOrTupleInt("axis", value_ptr, kernel_name_);
    }
    return attr_axis_me;
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
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  size_t dynamic_axis_size_;
  bool is_dynamic_axis_;
  bool get_dynamic_axis_value_;
  bool need_skip_execute_;
  static constexpr size_t kAxisIndex_{1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_ARRAY_REDUCE_GPU_KERNEL_H_
