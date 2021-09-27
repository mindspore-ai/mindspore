/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "backend/kernel_compiler/gpu/cuda_impl/local_response_norm_impl.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/transpose_impl_opt.cuh"
#include "utils/utils.h"

namespace mindspore {
namespace kernel {
template <typename T>
class LocalResponseNormGradGpuKernel : public GpuKernel {
 public:
  LocalResponseNormGradGpuKernel() { ResetResource(); }
  ~LocalResponseNormGradGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    auto dy = GetDeviceAddress<T>(inputs, 0);
    auto x = GetDeviceAddress<T>(inputs, 1);
    auto y = GetDeviceAddress<T>(inputs, 2);
    auto dx = GetDeviceAddress<T>(outputs, 0);
    const float alpha = 1;
    const float beta = 0;

    if (use_native_) {
      MS_LOG(WARNING) << "TOM: grad use native";
      MS_LOG(WARNING) << "TOM: num_elements_ " << num_elements_;
      std::vector<size_t> to_nhwc_axis = {0, 2, 3, 1};
      std::vector<size_t> to_nchw_axis = {0, 3, 1, 2};
      const size_t shape_size = 4 * sizeof(size_t);
      size_t *ws_input_shape = GetDeviceAddress<size_t>(workspace, 0);
      size_t *ws_transpose_shape = GetDeviceAddress<size_t>(workspace, 1);
      size_t *ws_to_nhwc_axis = GetDeviceAddress<size_t>(workspace, 2);
      size_t *ws_to_nchw_axis = GetDeviceAddress<size_t>(workspace, 3);
      T *ws_dy = GetDeviceAddress<T>(workspace, 4);
      T *ws_x = GetDeviceAddress<T>(workspace, 5);
      T *ws_y = GetDeviceAddress<T>(workspace, 6);
      T *ws_dx = GetDeviceAddress<T>(workspace, 7);
      float *ws_scale = GetDeviceAddress<float>(workspace, 8);

      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(ws_input_shape, &input_shape_[0], shape_size, cudaMemcpyHostToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync input_shape_ failed");
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(ws_transpose_shape, &transpose_shape_[0], shape_size,
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync transpose_shape_ failed");
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(ws_to_nhwc_axis, &to_nhwc_axis[0], shape_size, cudaMemcpyHostToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync to_nhwc_axis failed");
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(ws_to_nchw_axis, &to_nchw_axis[0], shape_size, cudaMemcpyHostToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync to_nchw_axis failed");

      CalNCHW2NHWCInterface(num_elements_, 4, dy, &input_shape_[0], &to_nhwc_axis[0], ws_input_shape, ws_to_nhwc_axis,
                            ws_dy, reinterpret_cast<cudaStream_t>(stream_ptr));
      CalNCHW2NHWCInterface(num_elements_, 4, x, &input_shape_[0], &to_nhwc_axis[0], ws_input_shape, ws_to_nhwc_axis,
                            ws_x, reinterpret_cast<cudaStream_t>(stream_ptr));
      CalNCHW2NHWCInterface(num_elements_, 4, y, &input_shape_[0], &to_nhwc_axis[0], ws_input_shape, ws_to_nhwc_axis,
                            ws_y, reinterpret_cast<cudaStream_t>(stream_ptr));

      CalLocalResponseNormGradNHWC(ws_dy, ws_x, ws_y, depth_radius_, bias_, alpha_, beta_, transpose_shape_[3],
                                   num_elements_, ws_scale, ws_dx, reinterpret_cast<cudaStream_t>(stream_ptr));

      CalNHWC2NCHWInterface(num_elements_, 4, ws_dx, &transpose_shape_[0], &to_nchw_axis[0], ws_transpose_shape,
                            ws_to_nchw_axis, dx, reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnLRNCrossChannelBackward(handle_, norm_desc_, lrn_mode_, &alpha, y_desc_, y,
                                                               dy_desc_, dy, x_desc_, x, &beta, dx_desc_, dx),
                                  "cudnnLRNCrossChannelBackward failed");
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    MS_EXCEPTION_IF_NULL(kernel_node);
    if (!CheckParam(kernel_node)) {
      return false;
    }

    depth_radius_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "depth_radius"));
    bias_ = GetAttr<float>(kernel_node, "bias");
    alpha_ = GetAttr<float>(kernel_node, "alpha");
    beta_ = GetAttr<float>(kernel_node, "beta");

    use_native_ = false;
    const unsigned int lrnN = 2 * depth_radius_ + 1;
    double lrnAlpha = lrnN * alpha_;
    if (lrnN < CUDNN_LRN_MIN_N || lrnN > CUDNN_LRN_MAX_N || bias_ < CUDNN_LRN_MIN_K || beta_ < CUDNN_LRN_MIN_BETA) {
      use_native_ = true;
    }
    InitResource();

    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "LocalResponseNormGradGpuKernel input is null";
      InitSizeLists();
      return true;
    }
    if (input_shape.size() != 4) {
      MS_LOG(EXCEPTION) << "tensor shape is " << input_shape.size() << ", LocalResponseNormGradGpuKernel should be 4D";
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
      lrn_mode_ = CUDNN_LRN_CROSS_CHANNEL_DIM1;
      cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
      SetCUDNNDescriptors(input_shape, lrnN, lrnAlpha);
    }

    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_ = 0;
    output_size_ = 0;
    is_null_input_ = false;
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

  void DestroyResource() noexcept override {
    if (!use_native_) {
      CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dy_desc_), "Destroy dy desc failed");
      CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(x_desc_), "Destroy x desc failed");
      CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(y_desc_), "Destroy y desc failed");
      CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dx_desc_), "Destroy dx desc failed");
      CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyLRNDescriptor(norm_desc_), "Destroy LRN norm desc failed");
    }
  }

 protected:
  void InitResource() override {
    if (!use_native_) {
      handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dy_desc_), "Create dy desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&x_desc_), "Create x desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&y_desc_), "Create y desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dx_desc_), "Create dx desc failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateLRNDescriptor(&norm_desc_), "Create LRN norm desc failed");
    }
  }

  void InitSizeLists() override {
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
        workspace_size_list_.push_back(input_size_);
        workspace_size_list_.push_back(input_size_);
        workspace_size_list_.push_back(num_elements_ * sizeof(float));
      } else {
        CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(dy_desc_, &input_size_),
                                    "Get input dy size failed");
        CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(x_desc_, &input_size_),
                                    "Get input x size failed");
        CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(y_desc_, &input_size_),
                                    "Get input y size failed");
        CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(dx_desc_, &output_size_),
                                    "Get output dx size failed");
      }
    }
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 3) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but LocalResponseNormGradGpuKernel needs 3 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but LocalResponseNormGradGpuKernel needs 1 output.";
      return false;
    }
    return true;
  }

  void SetCUDNNDescriptors(const std::vector<size_t> &shape, int lrnN, double lrnAlpha) {
    int batch = SizeToInt(shape[0]);
    int channel = SizeToInt(shape[1]);
    int height = SizeToInt(shape[2]);
    int width = SizeToInt(shape[3]);

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensor4dDescriptor(dy_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch, channel, height, width),
      "Set dy desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensor4dDescriptor(x_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch, channel, height, width),
      "Set x desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensor4dDescriptor(y_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch, channel, height, width),
      "Set y desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensor4dDescriptor(dx_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, batch, channel, height, width),
      "Set dx desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetLRNDescriptor(norm_desc_, lrnN, lrnAlpha, beta_, bias_),
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
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_LOCAL_RESPONSE_NORM_GRAD_GPU_KERNEL_H_
