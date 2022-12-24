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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LOCAL_RESPONSE_NORM_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LOCAL_RESPONSE_NORM_GRAD_GPU_KERNEL_H_

#include <string>
#include <vector>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/local_response_norm_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl_opt.cuh"
#include "include/common/utils/utils.h"
#include "mindspore/core/ops/grad/lrn_grad.h"

namespace mindspore {
namespace kernel {
constexpr size_t k4DSize = 4;

constexpr size_t kIdx2 = 2;
constexpr size_t kIdx3 = 3;
constexpr size_t kIdx4 = 4;
constexpr size_t kIdx5 = 5;
constexpr size_t kIdx6 = 6;
constexpr size_t kIdx7 = 7;
constexpr size_t kIdx8 = 8;
const unsigned int kCoef = 2;

template <typename T>
class LocalResponseNormGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  LocalResponseNormGradGpuKernelMod() { ResetResource(); }
  ~LocalResponseNormGradGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    auto dy = GetDeviceAddress<T>(inputs, 0);
    auto x = GetDeviceAddress<T>(inputs, 1);
    auto y = GetDeviceAddress<T>(inputs, kIdx2);
    auto dx = GetDeviceAddress<T>(outputs, 0);
    const float alpha = 1;
    const float beta = 0;

    if (use_native_) {
      MS_LOG(WARNING) << "TOM: grad use native";
      MS_LOG(WARNING) << "TOM: num_elements_ " << num_elements_;
      std::vector<size_t> to_nhwc_axis = {0, kIdx2, kIdx3, 1};
      std::vector<size_t> to_nchw_axis = {0, kIdx3, 1, kIdx2};
      const size_t shape_size = k4DSize * sizeof(size_t);
      size_t *ws_input_shape = GetDeviceAddress<size_t>(workspace, 0);
      size_t *ws_transpose_shape = GetDeviceAddress<size_t>(workspace, 1);
      size_t *ws_to_nhwc_axis = GetDeviceAddress<size_t>(workspace, kIdx2);
      size_t *ws_to_nchw_axis = GetDeviceAddress<size_t>(workspace, kIdx3);
      T *ws_dy = GetDeviceAddress<T>(workspace, kIdx4);
      T *ws_x = GetDeviceAddress<T>(workspace, kIdx5);
      T *ws_y = GetDeviceAddress<T>(workspace, kIdx6);
      T *ws_dx = GetDeviceAddress<T>(workspace, kIdx7);
      float *ws_scale = GetDeviceAddress<float>(workspace, kIdx8);

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

      CalNCHW2NHWCInterface(num_elements_, k4DSize, dy, &input_shape_[0], &to_nhwc_axis[0], ws_input_shape,
                            ws_to_nhwc_axis, ws_dy, reinterpret_cast<cudaStream_t>(stream_ptr));
      CalNCHW2NHWCInterface(num_elements_, k4DSize, x, &input_shape_[0], &to_nhwc_axis[0], ws_input_shape,
                            ws_to_nhwc_axis, ws_x, reinterpret_cast<cudaStream_t>(stream_ptr));
      CalNCHW2NHWCInterface(num_elements_, k4DSize, y, &input_shape_[0], &to_nhwc_axis[0], ws_input_shape,
                            ws_to_nhwc_axis, ws_y, reinterpret_cast<cudaStream_t>(stream_ptr));

      CalLocalResponseNormGradNHWC(ws_dy, ws_x, ws_y, depth_radius_, bias_, alpha_, beta_, transpose_shape_[3],
                                   num_elements_, ws_scale, ws_dx, reinterpret_cast<cudaStream_t>(stream_ptr));

      CalNHWC2NCHWInterface(num_elements_, k4DSize, ws_dx, &transpose_shape_[0], &to_nchw_axis[0], ws_transpose_shape,
                            ws_to_nchw_axis, dx, reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnLRNCrossChannelBackward(handle_, norm_desc_, lrn_mode_, &alpha, y_desc_, y, dy_desc_, dy, x_desc_, x,
                                     &beta, dx_desc_, dx),
        "cudnnLRNCrossChannelBackward failed");
    }
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override {
    auto kernel_ptr = std::dynamic_pointer_cast<ops::LRNGrad>(base_operator);
    MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
    kernel_name_ = kernel_ptr->name();
    size_t input_num = inputs.size();
    if (input_num != 3) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 3, but got " << input_num;
    }
    size_t output_num = outputs.size();
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
    }
    depth_radius_ = kernel_ptr->get_depth_radius();
    bias_ = kernel_ptr->get_bias();
    alpha_ = kernel_ptr->get_alpha();
    beta_ = kernel_ptr->get_beta();
    use_native_ = false;
    int lrnN = kCoef * depth_radius_ + 1;
    if (lrnN < CUDNN_LRN_MIN_N || lrnN > CUDNN_LRN_MAX_N || bias_ < CUDNN_LRN_MIN_K || beta_ < CUDNN_LRN_MIN_BETA) {
      use_native_ = true;
    }
    InitResource();
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override {
    auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
    if (ret != KRET_OK) {
      return ret;
    }
    transpose_shape_.clear();
    int lrnN = kCoef * depth_radius_ + 1;
    double lrnAlpha = lrnN * alpha_;
    auto in_shape = inputs[kIndex0]->GetShapeVector();
    auto input_shape = Convert2SizeTClipNeg(in_shape);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    if (is_null_input_) {
      InitWorkspaceSizeLists();
      return true;
    }
    const size_t kInputNum = 4;
    if (input_shape.size() != kInputNum) {
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
      transpose_shape_.push_back(input_shape_[kIdx2]);
      transpose_shape_.push_back(input_shape_[kIdx3]);
      transpose_shape_.push_back(input_shape_[1]);
    } else {
      lrn_mode_ = CUDNN_LRN_CROSS_CHANNEL_DIM1;
      cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs.at(kIndex0)->GetDtype()));
      SetCUDNNDescriptors(input_shape, lrnN, lrnAlpha);
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
    input_shape_.clear();
    transpose_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
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
        const size_t shape_size = k4DSize * sizeof(size_t);
        workspace_size_list_.push_back(shape_size);
        workspace_size_list_.push_back(shape_size);
        workspace_size_list_.push_back(shape_size);
        workspace_size_list_.push_back(shape_size);
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
  void SetCUDNNDescriptors(const std::vector<size_t> &shape, int lrnN, double lrnAlpha) {
    int batch = SizeToInt(shape[0]);
    int channel = SizeToInt(shape[1]);
    int height = SizeToInt(shape[kIdx2]);
    int width = SizeToInt(shape[kIdx3]);

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
  std::vector<size_t> input_shape_;
  std::vector<size_t> transpose_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LOCAL_RESPONSE_NORM_GRAD_GPU_KERNEL_H_
