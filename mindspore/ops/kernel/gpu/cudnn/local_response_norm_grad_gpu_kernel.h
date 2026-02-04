/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LOCAL_RESPONSE_NORM_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LOCAL_RESPONSE_NORM_GRAD_GPU_KERNEL_H_

#include <map>
#include <string>
#include <vector>
#include "include/utils/utils.h"
#include "mindspore/ops/infer/grad/lrn_grad.h"
#include "kernel/gpu/cuda_impl/cuda_ops/local_response_norm_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/transpose_impl.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
constexpr size_t k3DSize = 3;
constexpr size_t k4DSize = 4;
const unsigned int kCoef = 2;

template <typename T>
class LocalResponseNormGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  LocalResponseNormGradGpuKernelMod() { ResetResource(); }
  ~LocalResponseNormGradGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    auto dy = GetDeviceAddress<T>(inputs, 0);
    auto x = GetDeviceAddress<T>(inputs, 1);
    auto y = GetDeviceAddress<T>(inputs, kIndex2);
    auto dx = GetDeviceAddress<T>(outputs, 0);
    const float alpha = 1;
    const float beta = 0;

    if (use_native_) {
      MS_LOG(WARNING) << "TOM: grad use native";
      MS_LOG(WARNING) << "TOM: num_elements_ " << num_elements_;
      T *ws_dy = GetDeviceAddress<T>(workspace, kIndex0);
      T *ws_x = GetDeviceAddress<T>(workspace, kIndex1);
      T *ws_y = GetDeviceAddress<T>(workspace, kIndex2);
      T *ws_dx = GetDeviceAddress<T>(workspace, kIndex3);
      float *ws_scale = GetDeviceAddress<float>(workspace, kIndex4);

      TransposeInfo InInfo;
      TransposeInfo OutInfo;
      InInfo.input_shape = input_shape_;
      InInfo.perm = std::vector<int32_t>{0, 2, 3, 1};
      OutInfo.input_shape = transpose_shape_;
      OutInfo.perm = std::vector<int32_t>{0, 3, 1, 2};

      auto status = CalTranspose<T, true>(num_elements_, dy, InInfo, ws_dy, reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, "Transpose called by " + kernel_name_);

      status = CalTranspose<T, true>(num_elements_, x, InInfo, ws_x, reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, "Transpose called by " + kernel_name_);

      status = CalTranspose<T, true>(num_elements_, y, InInfo, ws_y, reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, "Transpose called by " + kernel_name_);

      status =
        CalLocalResponseNormGradNHWC(ws_dy, ws_x, ws_y, depth_radius_, bias_, alpha_, beta_, transpose_shape_[kIndex3],
                                     num_elements_, ws_scale, ws_dx, reinterpret_cast<cudaStream_t>(stream_ptr));

      CHECK_CUDA_STATUS(status, kernel_name_);

      status = CalTranspose<T, true>(num_elements_, ws_dx, OutInfo, dx, reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, "Transpose called by " + kernel_name_);
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnLRNCrossChannelBackward(handle_, norm_desc_, lrn_mode_, &alpha, y_desc_, y, dy_desc_, dy, x_desc_, x,
                                     &beta, dx_desc_, dx),
        "cudnnLRNCrossChannelBackward failed");
    }
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    size_t input_num = inputs.size();
    if (input_num != k3DSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 3, but got " << input_num;
    }
    size_t output_num = outputs.size();
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
    }
    depth_radius_ = GetValue<int64_t>(primitive_->GetAttr("depth_radius"));
    bias_ = GetValue<float>(primitive_->GetAttr("bias"));
    alpha_ = GetValue<float>(primitive_->GetAttr("alpha"));
    beta_ = GetValue<float>(primitive_->GetAttr("beta"));
    use_native_ = false;
    int lrnN = kCoef * depth_radius_ + 1;
    if (lrnN < CUDNN_LRN_MIN_N || lrnN > CUDNN_LRN_MAX_N || bias_ < CUDNN_LRN_MIN_K || beta_ < CUDNN_LRN_MIN_BETA) {
      use_native_ = true;
    }
    InitResource();
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    auto ret = KernelMod::Resize(inputs, outputs);
    if (ret != KRET_OK) {
      return ret;
    }
    int lrnN = kCoef * depth_radius_ + 1;
    double lrnAlpha = lrnN * alpha_;
    input_shape_ = inputs[kIndex0]->GetShapeVector();
    is_null_input_ = CHECK_SHAPE_NULL(input_shape_, kernel_name_, "input");
    if (is_null_input_) {
      InitWorkspaceSizeLists();
      return true;
    }
    const size_t kInputNum = 4;
    if (input_shape_.size() != kInputNum) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input must be 4, but got "
                        << input_shape_.size();
    }
    transpose_shape_.clear();
    if (use_native_) {
      num_elements_ = SizeOf(input_shape_);
      transpose_shape_.push_back(input_shape_[0]);
      transpose_shape_.push_back(input_shape_[kIndex2]);
      transpose_shape_.push_back(input_shape_[kIndex3]);
      transpose_shape_.push_back(input_shape_[1]);
    } else {
      lrn_mode_ = CUDNN_LRN_CROSS_CHANNEL_DIM1;
      cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs[kIndex0]->dtype_id()));
      SetCUDNNDescriptors(input_shape_, lrnN, lrnAlpha);
    }

    InitWorkspaceSizeLists();
    return static_cast<int>(KRET_OK);
  }

  void ResetResource() noexcept {
    input_size_ = 0;
    output_size_ = 0;
    is_null_input_ = false;
    kernel_name_ = "LocalResponseNormGrad";
    dy_desc_ = nullptr;
    x_desc_ = nullptr;
    y_desc_ = nullptr;
    dx_desc_ = nullptr;
    norm_desc_ = nullptr;
    lrn_mode_ = CUDNN_LRN_CROSS_CHANNEL_DIM1;
    handle_ = nullptr;
    cudnn_data_type_ = CUDNN_DATA_FLOAT;
    depth_radius_ = 0;
    bias_ = 0;
    alpha_ = 0;
    beta_ = 0;
    use_native_ = false;
    num_elements_ = 0;
  }

  void DestroyResource() noexcept {
    if (!use_native_) {
      CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(dy_desc_), "Destroy dy desc failed");
      CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(x_desc_), "Destroy x desc failed");
      CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(y_desc_), "Destroy y desc failed");
      CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(dx_desc_), "Destroy dx desc failed");
      CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyLRNDescriptor(norm_desc_), "Destroy LRN norm desc failed");
    }
  }

 protected:
  void InitResource() override {
    if (!use_native_) {
      handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dy_desc_), "Create dy desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&x_desc_), "Create x desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&y_desc_), "Create y desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dx_desc_), "Create dx desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateLRNDescriptor(&norm_desc_), "Create LRN norm desc failed");
    }
  }

  void InitWorkspaceSizeLists() {
    if (!is_null_input_) {
      if (use_native_) {
        input_size_ = num_elements_ * sizeof(T);
        output_size_ = num_elements_ * sizeof(T);
        workspace_size_list_.push_back(input_size_);
        workspace_size_list_.push_back(input_size_);
        workspace_size_list_.push_back(input_size_);
        workspace_size_list_.push_back(input_size_);
        workspace_size_list_.push_back(num_elements_ * sizeof(float));
      } else {
        CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(dy_desc_, &input_size_),
                                            "Get input dy size failed");
        CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(x_desc_, &input_size_),
                                            "Get input x size failed");
        CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(y_desc_, &input_size_),
                                            "Get input y size failed");
        CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(dx_desc_, &output_size_),
                                            "Get output dx size failed");
      }
    }
  }

 private:
  void SetCUDNNDescriptors(const std::vector<int64_t> &shape, int lrnN, double lrnAlpha) {
    int batch = static_cast<int>(shape[0]);
    int channel = static_cast<int>(shape[1]);
    int height = static_cast<int>(shape[kIndex2]);
    int width = static_cast<int>(shape[kIndex3]);

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(dy_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch, channel, height, width),
      "Set dy desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch, channel, height, width),
      "Set x desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(y_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch, channel, height, width),
      "Set y desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(dx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch, channel, height, width),
      "Set dx desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetLRNDescriptor(norm_desc_, lrnN, lrnAlpha, beta_, bias_),
                                        "cudnnSetLRNDescriptor failed");
  }

  size_t input_size_;
  size_t output_size_;
  bool is_null_input_;
  cudnnTensorDescriptor_t dy_desc_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  cudnnTensorDescriptor_t dx_desc_;
  cudnnLRNDescriptor_t norm_desc_;
  cudnnLRNMode_t lrn_mode_;
  cudnnHandle_t handle_;
  cudnnDataType_t cudnn_data_type_;
  int depth_radius_;
  double bias_;
  double alpha_;
  double beta_;
  bool use_native_;

  size_t num_elements_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> transpose_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LOCAL_RESPONSE_NORM_GRAD_GPU_KERNEL_H_
