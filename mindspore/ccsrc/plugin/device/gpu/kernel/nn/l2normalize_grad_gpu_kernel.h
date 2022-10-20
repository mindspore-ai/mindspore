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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_L2NORMALIZE_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_L2NORMALIZE_GRAD_GPU_KERNEL_H_

#include <map>
#include <string>
#include <vector>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/l2normalize_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"
namespace mindspore {
namespace kernel {
constexpr int MAX_DIMS = 7;
class L2NormalizeGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  L2NormalizeGradGpuKernelMod() = default;
  ~L2NormalizeGradGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using L2NormalizeGradGpuLaunchFunc =
    std::function<bool(L2NormalizeGradGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;

  bool CheckInputShape(const ShapeVector &output_shape) {
    for (auto &shape : input_shape_list_) {
      if (output_shape != shape) {
        MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of input and output must be the same, but "
                      << "got the shape of input: " << CONVERT_VECTOR_TO_STRING(shape)
                      << ", the shape of output: " << CONVERT_VECTOR_TO_STRING(output_shape);
        return false;
      }
    }
    return true;
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateReduceTensorDescriptor(&reduce_tensor_descriptor_),
                                        kernel_name_ + " cudnnCreateReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateReduceTensorDescriptor(&reduce_sum_tensor_descriptor_),
                                        kernel_name_ + " cudnnCreateReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&inputA_descriptor_),
                                        kernel_name_ + " cudnnCreateTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&outputC_descriptor_),
                                        kernel_name_ + " cudnnCreateTensorDescriptor failed.");
  }
  void InitWorkSpaceSizeLists() {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(outputC_descriptor_, &workspace_size_),
                                        kernel_name_ + " cudnnGetTensorSizeInBytes failed.");
    workspace_size_list_.push_back(workspace_size_);
    workspace_size_list_.push_back(workspace_size_);

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetReductionWorkspaceSize(cudnn_handle_, reduce_tensor_descriptor_, inputA_descriptor_, outputC_descriptor_,
                                     &workspace_size_),
      kernel_name_ + " cudnnGetReductionWorkspaceSize failed.");
    workspace_size_list_.push_back(workspace_size_);

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetReductionWorkspaceSize(cudnn_handle_, reduce_sum_tensor_descriptor_, inputA_descriptor_,
                                     outputC_descriptor_, &workspace_size_),
      kernel_name_ + " cudnnGetReductionWorkspaceSize failed.");
    workspace_size_list_.push_back(workspace_size_);
    return;
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyReduceTensorDescriptor(reduce_tensor_descriptor_),
                                       kernel_name_ + " cudnnDestroyReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyReduceTensorDescriptor(reduce_sum_tensor_descriptor_),
                                       kernel_name_ + " cudnnDestroyReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(inputA_descriptor_),
                                       kernel_name_ + " cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(outputC_descriptor_),
                                       kernel_name_ + " cudnnDestroyTensorDescriptor failed.");
  }
  void InferArrayReduceType() {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetReduceTensorDescriptor(reduce_tensor_descriptor_, CUDNN_REDUCE_TENSOR_NORM2, CUDNN_DATA_FLOAT, nan_prop_,
                                     reduce_indices_, CUDNN_32BIT_INDICES),
      kernel_name_ + " cudnnSetReduceTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetReduceTensorDescriptor(reduce_sum_tensor_descriptor_, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT,
                                     nan_prop_, reduce_indices_, CUDNN_32BIT_INDICES),
      kernel_name_ + " cudnnSetReduceTensorDescriptor failed");
    return;
  }
  void InferInAndOutDesc(const ShapeVector &input_shape, const ShapeVector &output_shape) {
    ShapeVector inputA;
    ShapeVector outputC_shape = output_shape;
    constexpr int split_dim = 4;
    CheckTensorSize({input_shape, output_shape});
    if (input_shape.size() <= split_dim) {
      ShapeNdTo4d(input_shape, &inputA);
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetTensor4dDescriptor(inputA_descriptor_, CUDNN_TENSOR_NCHW, data_type_,
                                                                     inputA[0], inputA[1], inputA[2], inputA[3]),
                                          kernel_name_ + " cudnnSetTensor4dDescriptor failed");
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
                                          kernel_name_ + " cudnnSetTensor4dDescriptor failed");
    } else {
      CudnnSetTensorNdDescriptor(outputC_shape, outputC_descriptor_, data_type_, kernel_name_);
      for (auto dim : outputC_shape) {
        outputC.emplace_back(dim);
      }
    }
    return;
  }

  cudnnHandle_t cudnn_handle_{nullptr};
  cudnnDataType_t data_type_{CUDNN_DATA_FLOAT};
  cudnnNanPropagation_t nan_prop_{CUDNN_NOT_PROPAGATE_NAN};
  cudnnReduceTensorIndices_t reduce_indices_{CUDNN_REDUCE_TENSOR_NO_INDICES};
  cudnnReduceTensorDescriptor_t reduce_tensor_descriptor_{nullptr};
  cudnnReduceTensorDescriptor_t reduce_sum_tensor_descriptor_{nullptr};
  cudnnTensorDescriptor_t inputA_descriptor_{nullptr};
  cudnnTensorDescriptor_t outputC_descriptor_{nullptr};

  bool all_match_{false};
  bool is_null_input_{false};
  std::vector<ShapeVector> input_shape_list_{};
  size_t workspace_size_{0};
  float epsilon_{0.0};
  int axis_origin_{0};
  int axis_{0};
  std::vector<size_t> lhs_shape_{};
  std::vector<size_t> rhs_shape_{};
  std::vector<size_t> output_shape_{};

  L2NormalizeGradGpuLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, L2NormalizeGradGpuLaunchFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_L2NORMALIZE_GRAD_GPU_KERNEL_H_
