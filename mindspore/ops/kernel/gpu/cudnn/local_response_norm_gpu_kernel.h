/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LOCAL_RESPONSE_NORM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LOCAL_RESPONSE_NORM_GPU_KERNEL_H_

#include <map>
#include <string>
#include <vector>
#include "include/utils/utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/local_response_norm_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/transpose_impl.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class LocalResponseNormGpuKernelMod : public NativeGpuKernelMod {
 public:
  LocalResponseNormGpuKernelMod() {
    ResetResource();
    depth_radius_ = 0;
    bias_ = 0;
    alpha_ = 0;
    beta_ = 0;
    input_size_ = 0;
    output_size_ = 0;
    is_null_input_ = false;
    kernel_name_ = "LocalResponseNorm";
    x_desc_ = nullptr;
    y_desc_ = nullptr;
    norm_desc_ = nullptr;
    lrn_mode_ = CUDNN_LRN_CROSS_CHANNEL_DIM1;
    handle_ = nullptr;
    cudnn_data_type_ = CUDNN_DATA_FLOAT;
    use_native_ = false;
    num_elements_ = 0;
  }
  ~LocalResponseNormGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    auto x = GetDeviceAddress<T>(inputs, 0);
    auto y = GetDeviceAddress<T>(outputs, 0);
    const float alpha = 1;
    const float beta = 0;
    if (use_native_) {
      T *ws_x = GetDeviceAddress<T>(workspace, 0);
      T *ws_y = GetDeviceAddress<T>(workspace, 1);
      float *ws_scale = GetDeviceAddress<float>(workspace, 2);

      TransposeInfo InInfo;
      TransposeInfo OutInfo;
      InInfo.input_shape = input_shape_;
      InInfo.perm = std::vector<int32_t>{0, 2, 3, 1};
      OutInfo.input_shape = transpose_shape_;
      OutInfo.perm = std::vector<int32_t>{0, 3, 1, 2};

      auto status = CalTranspose<T, true>(num_elements_, x, InInfo, ws_x, reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, "Transpose called by " + kernel_name_);
      status = CalLocalResponseNormNHWC(ws_x, depth_radius_, bias_, alpha_, beta_, transpose_shape_[kDim3],
                                        num_elements_, ws_scale, ws_y, reinterpret_cast<cudaStream_t>(stream_ptr));

      CHECK_CUDA_STATUS(status, kernel_name_);
      CalTranspose<T, true>(num_elements_, ws_y, OutInfo, y, reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, "Transpose called by " + kernel_name_);
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnLRNCrossChannelForward(handle_, norm_desc_, lrn_mode_, &alpha, x_desc_, x, &beta, y_desc_, y),
        "cudnnLRNCrossChannelForward failed");
    }
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    PrimitivePtr prim = primitive_;
    MS_EXCEPTION_IF_NULL(prim);

    if (inputs.size() != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1, but got " << inputs.size();
    }
    if (outputs.size() != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << outputs.size();
    }

    depth_radius_ = static_cast<int>(GetValue<int64_t>(prim->GetAttr("depth_radius")));
    bias_ = GetValue<float>(prim->GetAttr("bias"));
    alpha_ = GetValue<float>(prim->GetAttr("alpha"));
    beta_ = GetValue<float>(prim->GetAttr("beta"));
    use_native_ = false;
    const unsigned int lrnN = 2 * depth_radius_ + 1;
    if (lrnN < CUDNN_LRN_MIN_N || lrnN > CUDNN_LRN_MAX_N || bias_ < CUDNN_LRN_MIN_K || beta_ < CUDNN_LRN_MIN_BETA) {
      use_native_ = true;
    }
    InitResource();
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
      return ret;
    }
    ResetResource();
    input_shape_ = inputs[0]->GetShapeVector();
    is_null_input_ = CHECK_SHAPE_NULL(input_shape_, kernel_name_, "input");
    if (is_null_input_) {
      InitSizeLists();
      return KRET_OK;
    }
    const int dimension = 4;
    if (input_shape_.size() != dimension) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input must be 4, but got "
                        << input_shape_.size();
    }

    const int THIRD_ELEMENT_INDEX = 2;
    const int FOURTH_ELEMENT_INDEX = 3;
    if (use_native_) {
      num_elements_ = static_cast<size_t>(SizeOf(input_shape_));
      transpose_shape_.push_back(input_shape_[0]);
      transpose_shape_.push_back(input_shape_[THIRD_ELEMENT_INDEX]);
      transpose_shape_.push_back(input_shape_[FOURTH_ELEMENT_INDEX]);
      transpose_shape_.push_back(input_shape_[1]);
    } else {
      const unsigned int lrnN = 2 * depth_radius_ + 1;
      double lrnAlpha = lrnN * alpha_;
      lrn_mode_ = CUDNN_LRN_CROSS_CHANNEL_DIM1;
      cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs[0]->dtype_id()));
      SetCUDNNDescriptors(input_shape_, lrnN, lrnAlpha);
    }

    InitSizeLists();
    return KRET_OK;
  }

  void ResetResource() noexcept {
    input_shape_.clear();
    transpose_shape_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  void DestroyResource() noexcept override {
    if (!use_native_) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(x_desc_), "Destroy x desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(y_desc_), "Destroy y desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyLRNDescriptor(norm_desc_), "Destroy LRN norm desc failed");
    }
  }

 protected:
  void InitResource() override {
    if (!use_native_) {
      handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&x_desc_), "Create x desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&y_desc_), "Create y desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateLRNDescriptor(&norm_desc_), "Create LRN norm desc failed");
    }
  }

  void InitSizeLists() {
    if (!is_null_input_) {
      if (use_native_) {
        input_size_ = num_elements_ * sizeof(T);
        output_size_ = num_elements_ * sizeof(T);
        workspace_size_list_.push_back(input_size_);
        workspace_size_list_.push_back(input_size_);
        workspace_size_list_.push_back(num_elements_ * sizeof(float));
      } else {
        CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(x_desc_, &input_size_),
                                            "Get input x size failed");
        CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(y_desc_, &output_size_),
                                            "Get output y size failed");
      }
    }
    output_size_list_.push_back(output_size_);
  }

 private:
  void SetCUDNNDescriptors(const std::vector<int64_t> &shape, int lrnN, double lrnAlpha) {
    cudnnTensorFormat_t cudnn_format;
    int batch, channel, height, width;
    batch = static_cast<int>(shape[kIndex0]);
    channel = static_cast<int>(shape[kIndex1]);
    height = static_cast<int>(shape[kIndex2]);
    width = static_cast<int>(shape[kIndex3]);
    cudnn_format = CUDNN_TENSOR_NCHW;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(x_desc_, cudnn_format, cudnn_data_type_, batch, channel, height, width),
      "Set x desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensor4dDescriptor(y_desc_, cudnn_format, cudnn_data_type_, batch, channel, height, width),
      "Set y desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetLRNDescriptor(norm_desc_, lrnN, lrnAlpha, beta_, bias_),
                                        "cudnnSetLRNDescriptor failed");
  }

  size_t input_size_;
  size_t output_size_;
  bool is_null_input_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  cudnnLRNDescriptor_t norm_desc_;
  cudnnLRNMode_t lrn_mode_;
  cudnnHandle_t handle_;
  cudnnDataType_t cudnn_data_type_;
  int depth_radius_;
  float bias_;
  float alpha_;
  float beta_;
  bool use_native_;
  size_t num_elements_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> transpose_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LOCAL_RESPONSE_NORM_GPU_KERNEL_H_
