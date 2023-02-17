/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LAMB_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LAMB_GPU_KERNEL_H_

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/lamb_impl.cuh"
#include "ops/lamb.h"

namespace mindspore {
namespace kernel {
constexpr size_t INPUT_NUM = 10;
constexpr size_t kArgMaxDim = 7;
constexpr float ten = 10;

// input param index
constexpr size_t kVarIndex = 0;
constexpr size_t kMIndex = 1;
constexpr size_t kVIndex = 2;
constexpr size_t kLearningRateIndex = 3;
constexpr size_t kBeta1Index = 4;
constexpr size_t kBeta2Index = 5;
constexpr size_t kEpsilonIndex = 6;
constexpr size_t kWeightDecayIndex = 7;
constexpr size_t kGlobalStepIndex = 8;
constexpr size_t kGradIndex = 9;

// workspaces param index
constexpr size_t kUpdateIndex = 0;
constexpr size_t kVarFloatIndex = 1;
constexpr size_t kGradFloatIndex = 2;
constexpr size_t kGHatValIndex = 3;
constexpr size_t kTrustRatioIndex = 4;
constexpr size_t kReduceWorkspaceIndex = 5;
constexpr size_t kWNormIndex = 6;
constexpr size_t kGNormIndex = 7;
constexpr size_t kGHatNormIndex = 8;

template <typename T>
class LambGpuKernelMod : public NativeGpuKernelMod {
 public:
  LambGpuKernelMod() = default;

  ~LambGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }

    T *variable = GetDeviceAddress<T>(inputs, kVarIndex);
    T *m = GetDeviceAddress<T>(inputs, kMIndex);
    T *v = GetDeviceAddress<T>(inputs, kVIndex);
    float *learning_rate = GetDeviceAddress<float>(inputs, kLearningRateIndex);
    float *beta1 = GetDeviceAddress<float>(inputs, kBeta1Index);
    float *beta2 = GetDeviceAddress<float>(inputs, kBeta2Index);
    float *epsilon = GetDeviceAddress<float>(inputs, kEpsilonIndex);
    float *decay = GetDeviceAddress<float>(inputs, kWeightDecayIndex);
    int32_t *global_step = GetDeviceAddress<int32_t>(inputs, kGlobalStepIndex);
    T *gradient = GetDeviceAddress<T>(inputs, kGradIndex);
    float *update = GetDeviceAddress<float>(workspaces, kUpdateIndex);
    float *var_float = GetDeviceAddress<float>(workspaces, kVarFloatIndex);
    float *grad_float = GetDeviceAddress<float>(workspaces, kGradFloatIndex);
    float *g_hat_var = GetDeviceAddress<float>(workspaces, kGHatValIndex);

    ApplyLambEraly(inputs[0]->size / sizeof(T), variable, m, v, beta1, beta2, epsilon, decay, global_step, gradient,
                   update, var_float, grad_float, g_hat_var, reinterpret_cast<cudaStream_t>(stream_ptr));

    float trust_ratio{0};
    CalcTrustRatio(workspaces, var_float, grad_float, g_hat_var, stream_ptr, &trust_ratio);

    float *trust_ratio_ptr = GetDeviceAddress<float>(workspaces, kTrustRatioIndex);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(trust_ratio_ptr, &trust_ratio, sizeof(float), cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For " + kernel_name_ + " cudaMemcpyAsync trust_ratio failed.");

    ApplyLambLater(inputs[0]->size / sizeof(T), variable, learning_rate, update, trust_ratio_ptr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    auto kernel_ptr = std::dynamic_pointer_cast<ops::Lamb>(base_operator);
    kernel_name_ = kernel_ptr->name();
    InitResource();
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override {
    if (inputs.size() != INPUT_NUM) {
      MS_LOG(EXCEPTION) << "For 'Lamb', the number of inputs should be " << INPUT_NUM << ", but got " << inputs.size();
    }

    InitParamSizeByType();

    auto variable_int64_shape = inputs[kVarIndex]->GetShapeVector();
    auto m_int64_shape = inputs[kMIndex]->GetShapeVector();
    auto v_int64_shape = inputs[kVIndex]->GetShapeVector();
    auto gradient_int64_shape = inputs[kGradIndex]->GetShapeVector();
    if (AnfAlgo::IsShapesDynamic({variable_int64_shape, m_int64_shape, v_int64_shape, gradient_int64_shape})) {
      return true;
    }

    is_null_input_ = CHECK_SHAPE_NULL(variable_int64_shape, kernel_name_, "var") ||
                     CHECK_SHAPE_NULL(m_int64_shape, kernel_name_, "m") ||
                     CHECK_SHAPE_NULL(v_int64_shape, kernel_name_, "v") ||
                     CHECK_SHAPE_NULL(gradient_int64_shape, kernel_name_, "gradient");
    if (is_null_input_) {
      InitSizeLists();
      return 0;
    }

    InitParamSizeByShape(variable_int64_shape, m_int64_shape, v_int64_shape, gradient_int64_shape);

    auto output_int64_shape = outputs[0]->GetShapeVector();
    size_t input_dim = variable_int64_shape.size();
    if (!CheckValidShape(variable_int64_shape, output_int64_shape, input_dim)) {
      return 0;
    }

    InitShapeInfo(variable_int64_shape, output_int64_shape);
    // Determine the reduce operation.
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetReduceTensorDescriptor(reduce_tensor_descriptor_, CUDNN_REDUCE_TENSOR_NORM2, CUDNN_DATA_FLOAT, nan_prop_,
                                     reduce_indices_, CUDNN_32BIT_INDICES),
      "For " + kernel_name_ + " cudnnSetReduceTensorDescriptor failed");

    InitSizeLists();
    return 0;
  }

 protected:
  void InitSizeLists() {
    input_size_list_.clear();
    workspace_size_list_.clear();
    output_size_list_.clear();

    input_size_list_.push_back(variable_size_);
    input_size_list_.push_back(m_size_);
    input_size_list_.push_back(v_size_);
    input_size_list_.push_back(learning_rate_size_);
    input_size_list_.push_back(beta1_size_);
    input_size_list_.push_back(beta2_size_);
    input_size_list_.push_back(epsilon_size_);
    input_size_list_.push_back(decay_size_);
    input_size_list_.push_back(global_step_size_);
    input_size_list_.push_back(gradient_size_);
    workspace_size_list_.push_back(update_size_);
    workspace_size_list_.push_back(var_float_size_);
    workspace_size_list_.push_back(grad_float_size_);
    workspace_size_list_.push_back(g_hat_val_size_);
    workspace_size_list_.push_back(trust_ratio_size_);
    size_t workspace_size{0};
    // Init workspace size for gradient tensor reduce sum calculate.
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetReductionWorkspaceSize(cudnn_handle_, reduce_tensor_descriptor_, input_descriptor_, output_descriptor_,
                                     &workspace_size),
      "For " + kernel_name_ + " cudnnGetReductionWorkspaceSize failed.");
    workspace_size_list_.emplace_back(workspace_size);

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(output_descriptor_, &reduce_output_size_),
                                        "For " + kernel_name_ + " cudnnGetTensorSizeInBytes failed.");
    workspace_size_list_.emplace_back(reduce_output_size_);
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(output_descriptor_, &reduce_output_size_),
                                        "For " + kernel_name_ + " cudnnGetTensorSizeInBytes failed.");
    workspace_size_list_.emplace_back(reduce_output_size_);
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(output_descriptor_, &reduce_output_size_),
                                        "For " + kernel_name_ + " cudnnGetTensorSizeInBytes failed.");
    workspace_size_list_.emplace_back(reduce_output_size_);

    output_size_list_.push_back(0);
  }

  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateReduceTensorDescriptor(&reduce_tensor_descriptor_),
                                        "For " + kernel_name_ + " cudnnCreateReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&input_descriptor_),
                                        "For " + kernel_name_ + " cudnnCreateTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&output_descriptor_),
                                        "For " + kernel_name_ + " cudnnCreateTensorDescriptor failed.");
  }

 private:
  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyReduceTensorDescriptor(reduce_tensor_descriptor_),
                                        "For " + kernel_name_ + " cudnnDestroyReduceTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(input_descriptor_),
                                        "For " + kernel_name_ + " cudnnDestroyTensorDescriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(output_descriptor_),
                                        "For " + kernel_name_ + " cudnnDestroyTensorDescriptor failed.");
  }

  bool CheckValidShape(const ShapeVector &input_shape, const ShapeVector &output_shape, size_t input_dim) {
    is_null_input_ = CHECK_NULL_INPUT(input_shape) || CHECK_NULL_INPUT(output_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "For 'LambGpuKernelMod', input or output is null.";
      InitSizeLists();
      return false;
    }
    if (input_shape.size() != output_shape.size()) {
      MS_LOG(EXCEPTION) << "The size of input shape: " << input_shape.size()
                        << " and the size of output shape: " << output_shape.size() << " are different.";
    }
    if (input_dim > kArgMaxDim) {
      MS_LOG(EXCEPTION) << "Broadcast operation is not supported when dim exceeds than " << kArgMaxDim;
    }
    CheckTensorSize({input_shape, output_shape});
    return true;
  }

  void InitParamSizeByType() {
    variable_size_ = sizeof(T);
    m_size_ = sizeof(T);
    v_size_ = sizeof(T);
    learning_rate_size_ = sizeof(T);
    beta1_size_ = sizeof(T);
    beta2_size_ = sizeof(T);
    epsilon_size_ = sizeof(T);
    decay_size_ = sizeof(T);
    global_step_size_ = sizeof(int32_t);
    gradient_size_ = sizeof(T);
    update_size_ = sizeof(float);
    var_float_size_ = sizeof(float);
    grad_float_size_ = sizeof(float);
    g_hat_val_size_ = sizeof(float);
    trust_ratio_size_ = sizeof(float);
  }

  void InitParamSizeByShape(const ShapeVector &variable_shape, const ShapeVector &m_shape, const ShapeVector &v_shape,
                            const ShapeVector &gradient_shape) {
    for (size_t i = 0; i < variable_shape.size(); i++) {
      variable_size_ *= variable_shape[i];
      // save intermediate value
      update_size_ *= variable_shape[i];
      var_float_size_ *= variable_shape[i];
      grad_float_size_ *= variable_shape[i];
      g_hat_val_size_ *= variable_shape[i];
    }

    for (size_t i = 0; i < m_shape.size(); i++) {
      m_size_ *= m_shape[i];
    }

    for (size_t i = 0; i < v_shape.size(); i++) {
      v_size_ *= v_shape[i];
    }

    for (size_t i = 0; i < gradient_shape.size(); i++) {
      gradient_size_ *= gradient_shape[i];
    }
  }

  void CalcTrustRatio(const std::vector<AddressPtr> &workspaces, float *var_float, float *grad_float, float *g_hat_var,
                      void *stream_ptr, float *trust_ratio) {
    if (var_float == nullptr || grad_float == nullptr || g_hat_var == nullptr) {
      MS_LOG(EXCEPTION) << "var_float or grad_float or g_hat_var is null";
    }

    float *w_norm_ptr = nullptr;
    float *g_norm_ptr = nullptr;
    float *g_hat_norm_ptr = nullptr;

    if (is_all_match_) {
      w_norm_ptr = var_float;
      g_norm_ptr = grad_float;
      g_hat_norm_ptr = g_hat_var;
    } else {
      float *reduce_workspace_addr = GetPossiblyNullDeviceAddress<float>(workspaces, kReduceWorkspaceIndex);
      w_norm_ptr = GetDeviceAddress<float>(workspaces, kWNormIndex);
      g_norm_ptr = GetDeviceAddress<float>(workspaces, kGNormIndex);
      g_hat_norm_ptr = GetDeviceAddress<float>(workspaces, kGHatNormIndex);

      // Calc sum of square
      constexpr float alpha = 1;
      constexpr float beta = 0;
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, reduce_workspace_addr,
                          workspace_size_list_[kReduceWorkspaceIndex], &alpha, input_descriptor_, var_float, &beta,
                          output_descriptor_, w_norm_ptr),
        "For " + kernel_name_ + " cudnnReduceTensor for 'var_float' failed");

      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, reduce_workspace_addr,
                          workspace_size_list_[kReduceWorkspaceIndex], &alpha, input_descriptor_, grad_float, &beta,
                          output_descriptor_, g_norm_ptr),
        "For " + kernel_name_ + " cudnnReduceTensor for 'grad_float' failed");

      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnReduceTensor(cudnn_handle_, reduce_tensor_descriptor_, nullptr, 0, reduce_workspace_addr,
                          workspace_size_list_[kReduceWorkspaceIndex], &alpha, input_descriptor_, g_hat_var, &beta,
                          output_descriptor_, g_hat_norm_ptr),
        "For " + kernel_name_ + " cudnnReduceTensor for 'g_hat_var' failed");
    }

    float w_norm = 0;
    float g_norm = 0;
    float g_norm_hat = 0;
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&w_norm, w_norm_ptr, reduce_output_size_, cudaMemcpyDeviceToHost,
                                                       reinterpret_cast<cudaStream_t>(stream_ptr)),
                                       "For " + kernel_name_ + " cudaMemcpyAsync w_square_sum failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&g_norm, g_norm_ptr, reduce_output_size_, cudaMemcpyDeviceToHost,
                                                       reinterpret_cast<cudaStream_t>(stream_ptr)),
                                       "For " + kernel_name_ + " cudaMemcpyAsync g_square_sum failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(&g_norm_hat, g_hat_norm_ptr, reduce_output_size_, cudaMemcpyDeviceToHost,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For " + kernel_name_ + " cudaMemcpyAsync g_hat_square_sum failed.");

    *trust_ratio = w_norm > 0 ? (g_norm_hat > 0 ? (w_norm / g_norm_hat) : 1) : 1;
    if (*trust_ratio < 0 || std::isnan(*trust_ratio)) {
      *trust_ratio = 0;
    } else if (*trust_ratio > ten) {
      *trust_ratio = ten;
    }
  }

  void InitShapeInfo(const ShapeVector &input_shape, const ShapeVector &output_shape) {
    // Determine which dimension will be reduced.
    is_all_match_ = false;
    ShapeVector reduce_output_shape = output_shape;
    std::fill(reduce_output_shape.begin(), reduce_output_shape.end(), 1);

    if (input_shape == reduce_output_shape) {
      is_all_match_ = true;
    }

    // Infer input and output descriptor.
    InferInAndOutDesc(input_shape, reduce_output_shape);
  }

  void InferInAndOutDesc(const ShapeVector &input_shape, const ShapeVector &reduce_output_shape) {
    constexpr size_t split_dim = 4;
    constexpr size_t dim_idx_two = 2;
    constexpr size_t dim_idx_three = 3;
    if (input_shape.size() <= split_dim) {
      ShapeVector new_input_shape;
      ShapeNdTo4d(input_shape, &new_input_shape);
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnSetTensor4dDescriptor(input_descriptor_, CUDNN_TENSOR_NCHW, data_type_, new_input_shape[0],
                                   new_input_shape[1], new_input_shape[dim_idx_two], new_input_shape[dim_idx_three]),
        "For " + kernel_name_ + " cudnnSetTensor4dDescriptor failed");
    } else {
      CudnnSetTensorNdDescriptor(input_shape, input_descriptor_, data_type_, kernel_name_);
    }
    if (reduce_output_shape.size() <= split_dim) {
      ShapeVector new_reduce_output_shape;
      ShapeNdTo4d(reduce_output_shape, &new_reduce_output_shape);
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(

        cudnnSetTensor4dDescriptor(output_descriptor_, CUDNN_TENSOR_NCHW, data_type_, new_reduce_output_shape[0],
                                   new_reduce_output_shape[1], new_reduce_output_shape[dim_idx_two],
                                   new_reduce_output_shape[dim_idx_three]),
        "For " + kernel_name_ + " cudnnSetTensor4dDescriptor failed");
    } else {
      CudnnSetTensorNdDescriptor(reduce_output_shape, output_descriptor_, data_type_, kernel_name_);
    }
  }

  size_t variable_size_{0};
  size_t m_size_{0};
  size_t v_size_{0};
  size_t learning_rate_size_{0};
  size_t beta1_size_{0};
  size_t beta2_size_{0};
  size_t epsilon_size_{0};
  size_t decay_size_{0};
  size_t global_step_size_{0};
  size_t gradient_size_{0};
  size_t update_size_{0};
  size_t var_float_size_{0};
  size_t grad_float_size_{0};
  size_t g_hat_val_size_{0};
  size_t trust_ratio_size_{0};
  size_t reduce_output_size_{0};
  bool is_null_input_{false};
  bool is_all_match_{false};

  cudnnHandle_t cudnn_handle_{nullptr};
  cudnnDataType_t data_type_{CUDNN_DATA_FLOAT};
  cudnnNanPropagation_t nan_prop_{CUDNN_NOT_PROPAGATE_NAN};
  cudnnReduceTensorIndices_t reduce_indices_{CUDNN_REDUCE_TENSOR_NO_INDICES};
  cudnnReduceTensorDescriptor_t reduce_tensor_descriptor_{nullptr};
  cudnnTensorDescriptor_t input_descriptor_{nullptr};
  cudnnTensorDescriptor_t output_descriptor_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LAMB_GPU_KERNEL_H_
