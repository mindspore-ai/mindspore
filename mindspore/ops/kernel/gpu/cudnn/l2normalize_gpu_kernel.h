/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_L2NORMALIZE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_L2NORMALIZE_GPU_KERNEL_H_

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "kernel/gpu/gpu_kernel.h"
#include "primitive/math_ops.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/binary_ops_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/l2normalize_impl.cuh"
#include "kernel/gpu/cuda/math/broadcast_public.h"
#include "kernel/gpu/kernel_constants.h"
#include "mindspore/ops/infer/l2_normalize.h"

namespace mindspore {
namespace kernel {
constexpr int MAX_DIMS = 7;
template <typename T>
class L2NormalizeGpuKernelMod : public NativeGpuKernelMod {
 public:
  L2NormalizeGpuKernelMod()
      : cudnn_handle_(nullptr),
        data_type_(CUDNN_DATA_FLOAT),
        nan_prop_(CUDNN_NOT_PROPAGATE_NAN),
        reduce_indices_(CUDNN_REDUCE_TENSOR_NO_INDICES),
        reduce_tensor_descriptor_(nullptr),
        inputA_descriptor_(nullptr),
        outputC_descriptor_(nullptr),
        all_match_(false),
        is_null_input_(false),
        kernel_name_("L2Normalize"),
        input_size_(0),
        output_size_(0),
        workspace_size_(0),
        epsilon_(0.0),
        axis_(0) {}
  ~L2NormalizeGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    T *reduce_workspace_addr = GetDeviceAddress<T>(workspace, 0);
    T *workspace_addr = GetPossiblyNullDeviceAddress<T>(workspace, 1);

    T alpha = static_cast<T>(1.0f);
    T beta = static_cast<T>(0.0f);

    if (all_match_) {
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(reduce_workspace_addr, input_addr, inputs[0]->size(), cudaMemcpyDeviceToDevice,
                        reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cudaMemcpyAsync failed in L2Normalize::Launch.");
    } else {
      if (data_type_ == CUDNN_DATA_DOUBLE) {
        CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
          cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, workspace_addr, workspace_size_,
                            &alpha, inputA_descriptor_, input_addr, &beta, outputC_descriptor_, reduce_workspace_addr),
          "cudnnReduceTensor failed.");
      } else {
        const float alphaf = static_cast<float>(alpha);
        const float betaf = static_cast<float>(beta);
        CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
          cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, workspace_addr, workspace_size_,
                            &alphaf, inputA_descriptor_, input_addr, &betaf, outputC_descriptor_,
                            reduce_workspace_addr),
          "cudnnReduceTensor failed.");
      }
    }
    auto status = GetMaxWithEpsAndValue(workspace_size_list_[0] / sizeof(T), epsilon_, reduce_workspace_addr,
                                        reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    std::vector<int64_t> simplified_in0_shape;
    std::vector<int64_t> simplified_in1_shape;
    std::vector<int64_t> simplified_out_shape;
    SimplifyBinaryBroadcastShape(lhs_shape_, rhs_shape_, output_shape_, &simplified_in0_shape, &simplified_in1_shape,
                                 &simplified_out_shape);
    bool is_broadcast = IsBinaryBroadcast(simplified_in0_shape, simplified_in1_shape);
    status = BinaryOpWithBroadcastCudaFunc<BinaryOpType::kRealDiv, T, T, T>(
      is_broadcast, simplified_in0_shape, simplified_in1_shape, simplified_out_shape, input_addr, reduce_workspace_addr,
      output_addr, GET_CTX_DEVICE_ID, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    return true;
  }
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    int ret = KernelMod::Resize(inputs, outputs);
    if (ret != KRET_OK) {
      return ret;
    }

    auto inputA_shape = inputs[0]->GetShapeVector();

    int input_dim_length = SizeToInt(inputA_shape.size());
    // failed to get vector<int64_t> axis from infer
    int axis = GetValue<int64_t>(primitive_->GetAttr("axis"));
    axis_ = axis < 0 ? (axis + input_dim_length) : axis;

    auto output_shape = outputs[0]->GetShapeVector();
    output_size_ = sizeof(T) * SizeOf(output_shape);
    CheckTensorSize({inputA_shape, output_shape});
    if (inputA_shape.size() > MAX_DIMS) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than " << MAX_DIMS
                        << ", but got " << inputA_shape.size();
    }

    ShapeVector outputC_shape = output_shape;
    if (static_cast<size_t>(axis_) >= output_shape.size()) {
      MS_LOG(EXCEPTION) << "For 'L2NormalizeGpuKernelMod', axis_ must be less than the rank of output "
                        << "but got axis_: " << axis_ << ", rank of output: " << output_shape.size();
    }
    outputC_shape[axis_] = 1;

    if (inputA_shape.size() != output_shape.size() || inputA_shape.size() != outputC_shape.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input and output must be the same, but "
                        << "got the dimension of input: " << inputA_shape.size()
                        << ", the dimension of output: " << output_shape.size();
    }

    lhs_shape_.resize(MAX_DIMS, 1);
    rhs_shape_.resize(MAX_DIMS, 1);
    output_shape_.resize(MAX_DIMS, 1);
    all_match_ = true;
    for (size_t i = 0; i < output_shape.size(); i++) {
      output_shape_[i] = output_shape[i];
      lhs_shape_[i] = inputA_shape[i];
      rhs_shape_[i] = outputC_shape[i];
      if (lhs_shape_[i] != rhs_shape_[i]) {
        all_match_ = false;
      }
    }
    InferInAndOutDesc(inputA_shape, outputC_shape);
    InferArrayReduceType();
    InitSizeLists();
    return KRET_OK;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    InitResource();
    data_type_ = GetCudnnDataType(TypeIdLabel(inputs[kIndex0]->dtype_id()));
    (void)CheckIONumber(inputs, outputs);
    epsilon_ = GetValue<float>(primitive_->GetAttr("epsilon"));
    return true;
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateReduceTensorDescriptor(&reduce_tensor_descriptor_),
                                        "cudnnCreateReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&inputA_descriptor_),
                                        "cudnnCreateTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&outputC_descriptor_),
                                        "cudnnCreateTensorDescriptor failed.");
  }
  void InitSizeLists() {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(inputA_descriptor_, &input_size_),
                                        "cudnnGetTensorSizeInBytes failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(outputC_descriptor_, &workspace_size_),
                                        "cudnnGetTensorSizeInBytes failed.");
    workspace_size_list_.push_back(workspace_size_);

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetReductionWorkspaceSize(cudnn_handle_, reduce_tensor_descriptor_, inputA_descriptor_, outputC_descriptor_,
                                     &workspace_size_),
      "cudnnGetReductionWorkspaceSize failed.");
    workspace_size_list_.push_back(workspace_size_);
  }

 private:
  void CheckIONumber(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    size_t input_num = inputs.size();
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1, but got " << input_num;
    }
    size_t output_num = outputs.size();
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
    }
  }
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyReduceTensorDescriptor(reduce_tensor_descriptor_),
                                       "cudnnDestroyReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(inputA_descriptor_),
                                       "cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(outputC_descriptor_),
                                       "cudnnDestroyTensorDescriptor failed.");
  }
  void InferArrayReduceType() {
    cudnnDataType_t comp_type = (data_type_ == CUDNN_DATA_DOUBLE) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetReduceTensorDescriptor(reduce_tensor_descriptor_, CUDNN_REDUCE_TENSOR_NORM2, comp_type, nan_prop_,
                                     reduce_indices_, CUDNN_32BIT_INDICES),
      "cudnnSetReduceTensorDescriptor failed");
  }
  void InferInAndOutDesc(const ShapeVector &input_shape, const ShapeVector &output_shape) {
    ShapeVector inputA;
    ShapeVector outputC_shape = output_shape;
    const int split_dim = 4;

    if (input_shape.size() <= split_dim) {
      ShapeNdTo4d(input_shape, &inputA);
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetTensor4dDescriptor(inputA_descriptor_, CUDNN_TENSOR_NCHW, data_type_,
                                                                     inputA[0], inputA[1], inputA[2], inputA[3]),
                                          "cudnnSetTensor4dDescriptor failed");
    } else {
      CudnnSetTensorNdDescriptor(input_shape, inputA_descriptor_, data_type_, kernel_name_);
      for (auto dim : input_shape) {
        inputA.emplace_back(dim);
      }
    }

    ShapeVector outputC;

    if (outputC_shape.size() <= split_dim) {
      ShapeNdTo4d(outputC_shape, &outputC);
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetTensor4dDescriptor(outputC_descriptor_, CUDNN_TENSOR_NCHW, data_type_,
                                                                     outputC[0], outputC[1], outputC[2], outputC[3]),
                                          "cudnnSetTensor4dDescriptor failed");
    } else {
      CudnnSetTensorNdDescriptor(outputC_shape, outputC_descriptor_, data_type_, kernel_name_);
      for (auto dim : outputC_shape) {
        outputC.emplace_back(dim);
      }
    }
  }

  cudnnHandle_t cudnn_handle_;
  cudnnDataType_t data_type_;
  cudnnNanPropagation_t nan_prop_;
  cudnnReduceTensorIndices_t reduce_indices_;
  cudnnReduceTensorDescriptor_t reduce_tensor_descriptor_;
  cudnnTensorDescriptor_t inputA_descriptor_;
  cudnnTensorDescriptor_t outputC_descriptor_;

  bool all_match_;
  bool is_null_input_;
  std::string kernel_name_;

  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  float epsilon_;
  int axis_;
  ShapeVector lhs_shape_;
  ShapeVector rhs_shape_;
  ShapeVector output_shape_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_L2NORMALIZE_GPU_KERNEL_H_
