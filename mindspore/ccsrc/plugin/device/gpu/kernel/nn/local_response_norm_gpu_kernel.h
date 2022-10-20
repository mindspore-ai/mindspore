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

#include <string>
#include <vector>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/local_response_norm_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl_opt.cuh"
#include "include/common/utils/utils.h"

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

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    auto x = GetDeviceAddress<T>(inputs, 0);
    auto y = GetDeviceAddress<T>(outputs, 0);
    const float alpha = 1;
    const float beta = 0;

    if (use_native_) {
      std::vector<size_t> to_nhwc_axis = {0, 2, 3, 1};
      std::vector<size_t> to_nchw_axis = {0, 3, 1, 2};
      const size_t shape_size = 4 * sizeof(size_t);
      size_t *ws_input_shape = GetDeviceAddress<size_t>(workspace, 0);
      size_t *ws_transpose_shape = GetDeviceAddress<size_t>(workspace, 1);
      size_t *ws_to_nhwc_axis = GetDeviceAddress<size_t>(workspace, 2);
      size_t *ws_to_nchw_axis = GetDeviceAddress<size_t>(workspace, 3);
      T *ws_x = GetDeviceAddress<T>(workspace, 4);
      T *ws_y = GetDeviceAddress<T>(workspace, 5);
      float *ws_scale = GetDeviceAddress<float>(workspace, 6);

      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(ws_input_shape, &input_shape_[0], shape_size, cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cudaMemcpyAsync input_shape_ failed");
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(ws_transpose_shape, &transpose_shape_[0], shape_size, cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cudaMemcpyAsync transpose_shape_ failed");
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(ws_to_nhwc_axis, &to_nhwc_axis[0], shape_size, cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cudaMemcpyAsync to_nhwc_axis failed");
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(ws_to_nchw_axis, &to_nchw_axis[0], shape_size, cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cudaMemcpyAsync to_nchw_axis failed");

      CalNCHW2NHWCInterface(num_elements_, 4, x, &input_shape_[0], &to_nhwc_axis[0], ws_input_shape, ws_to_nhwc_axis,
                            ws_x, reinterpret_cast<cudaStream_t>(stream_ptr));

      CalLocalResponseNormNHWC(ws_x, depth_radius_, bias_, alpha_, beta_, transpose_shape_[3], num_elements_, ws_scale,
                               ws_y, reinterpret_cast<cudaStream_t>(stream_ptr));

      CalNHWC2NCHWInterface(num_elements_, 4, ws_y, &transpose_shape_[0], &to_nchw_axis[0], ws_transpose_shape,
                            ws_to_nchw_axis, y, reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnLRNCrossChannelForward(handle_, norm_desc_, lrn_mode_, &alpha, x_desc_, x, &beta, y_desc_, y),
        "cudnnLRNCrossChannelForward failed");
    }
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) {
    MS_EXCEPTION_IF_NULL(base_operator);
    PrimitivePtr prim = base_operator->GetPrim();
    MS_EXCEPTION_IF_NULL(prim);
    kernel_name_ = prim->name();
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

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) {
    if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
      return ret;
    }
    ResetResource();
    auto shape_signed = inputs[0]->GetShapeVector();
    auto input_shape = Convert2SizeTClipNeg(shape_signed);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    if (is_null_input_) {
      InitSizeLists();
      return KRET_OK;
    }
    if (input_shape.size() != 4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input must be 4, but got "
                        << input_shape.size();
    }

    if (use_native_) {
      num_elements_ = 1;
      for (auto x : input_shape) {
        input_shape_.push_back(x);
        num_elements_ *= x;
      }
      transpose_shape_.push_back(input_shape_[0]);
      transpose_shape_.push_back(input_shape_[2]);
      transpose_shape_.push_back(input_shape_[3]);
      transpose_shape_.push_back(input_shape_[1]);
    } else {
      const unsigned int lrnN = 2 * depth_radius_ + 1;
      double lrnAlpha = lrnN * alpha_;
      lrn_mode_ = CUDNN_LRN_CROSS_CHANNEL_DIM1;
      cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs[0]->GetDtype()));
      SetCUDNNDescriptors(input_shape, lrnN, lrnAlpha);
    }

    InitSizeLists();
    return KRET_OK;
  }

  void ResetResource() noexcept {
    input_shape_.clear();
    transpose_shape_.clear();
    input_size_list_.clear();
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
        const size_t shape_size = 4 * sizeof(size_t);
        workspace_size_list_.push_back(shape_size);
        workspace_size_list_.push_back(shape_size);
        workspace_size_list_.push_back(shape_size);
        workspace_size_list_.push_back(shape_size);
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
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  void SetCUDNNDescriptors(const std::vector<size_t> &shape, int lrnN, double lrnAlpha) {
    cudnnTensorFormat_t cudnn_format;
    int batch, channel, height, width;
    batch = SizeToInt(shape[0]);
    channel = SizeToInt(shape[1]);
    height = SizeToInt(shape[2]);
    width = SizeToInt(shape[3]);
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
  std::vector<size_t> input_shape_;
  std::vector<size_t> transpose_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LOCAL_RESPONSE_NORM_GPU_KERNEL_H_
