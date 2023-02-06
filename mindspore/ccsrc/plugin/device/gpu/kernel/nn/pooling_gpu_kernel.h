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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <functional>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/avg_pool3d_helper_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
constexpr auto kNumberFive = 5;
constexpr auto kAvgPool = "AvgPool";
constexpr auto kAvgPool3D = "AvgPool3D";

template <typename T>
class PoolingFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  PoolingFwdGpuKernelMod()
      : cudnn_handle_(nullptr),
        input_descriptor_(nullptr),
        output_descriptor_(nullptr),
        pooling_descriptor_(nullptr),
        pooling_mode_(CUDNN_POOLING_MAX),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        compute_format_(CUDNN_TENSOR_NCHW),
        old_depth_(0),
        old_height_(0),
        old_width_(0),
        pad_depth_(0),
        pad_height_(0),
        pad_width_(0),
        pad_front_(0),
        pad_top_(0),
        pad_left_(0),
        n_(0),
        c_(0),
        divisor_override_(0),
        pad_value_(0),
        is_null_input_(false),
        ceil_mode_(false),
        kernel_name_("Pooling"),
        input_size_(0),
        output_size_(0),
        workspace_size_(0) {}
  ~PoolingFwdGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    T alpha = static_cast<T>(1.0f);
    T beta = static_cast<T>(0.0f);
    if (cudnn_data_type_ == CUDNN_DATA_DOUBLE) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnPoolingForward(cudnn_handle_, pooling_descriptor_, &alpha, input_descriptor_, input_addr, &beta,
                            output_descriptor_, output_addr),
        "cudnnPoolingForward failed");
    } else {
      const float alphaf = static_cast<float>(alpha);
      const float betaf = static_cast<float>(beta);
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnPoolingForward(cudnn_handle_, pooling_descriptor_, &alphaf, input_descriptor_, input_addr, &betaf,
                            output_descriptor_, output_addr),
        "cudnnPoolingForward failed");
    }
    if (divisor_override_ != 0) {
      T *work_addr = GetDeviceAddress<T>(workspace, 0);
      size_t output_num = output_size_ / sizeof(T);
      int64_t size = std::accumulate(kernel_size_.begin(), kernel_size_.end(), int64_t(1), std::multiplies<int64_t>());
      T divisor = static_cast<T>(LongToFloat(size) / LongToFloat(divisor_override_));
      std::vector<T> divisor_value(output_num, divisor);
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
        cudaMemcpyAsync(work_addr, divisor_value.data(), output_size_, cudaMemcpyHostToDevice,
                        reinterpret_cast<cudaStream_t>(stream_ptr)),
        "cudaMemcpyAsync failed.");
      if (ceil_mode_) {
        CalRealKernelSize(output_shape_exclude_nc_, kernel_size_, edge_kernel_, work_addr, 0,
                          reinterpret_cast<cudaStream_t>(stream_ptr));
      }
      ElewiseArith(output_num, BinaryOpType::kMul, output_addr, work_addr, output_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) {
    auto prim = base_operator->GetPrim();
    MS_EXCEPTION_IF_NULL(prim);
    kernel_name_ = prim->name();
    InitResource();
    if (inputs.size() != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1, but got " << inputs.size();
    }

    if (kernel_name_ == kAvgPool3D) {
      divisor_override_ = GetValue<int64_t>(prim->GetAttr("divisor_override"));
      ceil_mode_ = GetValue<bool>(prim->GetAttr("ceil_mode"));
    }
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs[0]->GetDtype()));
    data_format_ = mindspore::FormatEnumToString(inputs[0]->GetFormat());
    auto format_attr = GetValue<std::string>(prim->GetAttr("format"));
    if (Anyone(format_attr, kOpFormat_NHWC, kOpFormat_NDHWC)) {
      data_format_ = format_attr;
    }
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) {
    if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
      return ret;
    }
    ResetResource();
    auto input_shape = inputs[0]->GetDeviceShapeAdaptively();
    auto output_shape = outputs[0]->GetDeviceShapeAdaptively();
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    CheckTensorSize({input_shape, output_shape});
    auto dim = input_shape.size();
    if (dim == kDim2DShapeSize) {
      SetNCHW(input_shape, &n_, &c_, &old_height_, &old_width_, data_format_);
    } else if (dim == kDim3DShapeSize) {
      SetNCDHW(input_shape, &n_, &c_, &old_depth_, &old_height_, &old_width_, data_format_);
    }

    int dimA[kPoolingNbDims];
    int strideAin[kPoolingNbDims];
    int dimAout[kPoolingNbDims];
    int strideAout[kPoolingNbDims];
    SetDimA(input_shape, dimA, dim, data_format_);
    SetStrideA(input_shape, strideAin, dim, data_format_);
    SetDimA(output_shape, dimAout, dim, data_format_);
    SetStrideA(output_shape, strideAout, dim, data_format_);
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensorNdDescriptor(input_descriptor_, cudnn_data_type_, dim, dimA, strideAin),
      "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensorNdDescriptor(output_descriptor_, cudnn_data_type_, dim, dimAout, strideAout),
      "cudnnSetTensorNdDescriptor failed");
    SetPoolingMode(base_operator);
    if (dim == kDim2DShapeSize) {
      SetPad(base_operator);
    } else if (dim == kDim3DShapeSize) {
      SetPad3D(base_operator);
    }
    edge_kernel_ = GetEdgeKernelSize(base_operator, inputs, outputs);
    InitSizeLists();
    return KRET_OK;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyPoolingDescriptor(pooling_descriptor_),
                                        "cudnnDestroyPoolingDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(output_descriptor_),
                                        "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(input_descriptor_),
                                        "cudnnDestroyTensorDescriptor failed");
  }

 protected:
  void InitResource() {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&input_descriptor_),
                                        "cudnnCreateTensorDescriptor failed");

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&output_descriptor_),
                                        "cudnnCreateTensorDescriptor failed");

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreatePoolingDescriptor(&pooling_descriptor_),
                                        "cudnnCreatePoolingDescriptor failed");
  }
  void InitSizeLists() {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnGetTensorSizeInBytes(input_descriptor_, reinterpret_cast<size_t *>(&input_size_)),
        "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnGetTensorSizeInBytes(output_descriptor_, reinterpret_cast<size_t *>(&output_size_)),
        "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    workspace_size_list_.push_back(output_size_);
  }
  void ResetResource() {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    stride_.clear();
  }

 private:
  void SetPoolingMode(const BaseOperatorPtr &base_operator) {
    auto prim = base_operator->GetPrim();
    MS_EXCEPTION_IF_NULL(prim);
    mode_ = kernel_name_;
    bool include = false;
    if (prim->HasAttr("count_include_pad")) {
      include = GetValue<bool>(prim->GetAttr("count_include_pad"));
    }
    if (mode_ == kAvgPool || mode_ == kAvgPool3D) {
      pooling_mode_ =
        include ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
      pad_value_ = 0.0;
    } else {
      pooling_mode_ = CUDNN_POOLING_MAX;
      pad_value_ = kSignedMinFloat;
    }
  }

  void SetPad(const BaseOperatorPtr &base_operator) {
    auto prim = base_operator->GetPrim();
    MS_EXCEPTION_IF_NULL(prim);
    pad_mode_ = GetValue<std::string>(prim->GetAttr("pad_mode"));
    std::vector<int> window;
    std::vector<int64_t> window_me = GetValue<std::vector<int64_t>>(prim->GetAttr("kernel_size"));
    (void)std::transform(window_me.begin(), window_me.end(), std::back_inserter(window),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (window.size() < 4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'kernel_size' cannot be less than 4, but got "
                        << window.size();
    }
    int window_height = window[2];
    int window_width = window[3];
    std::vector<int64_t> stride_me = GetValue<std::vector<int64_t>>(prim->GetAttr("strides"));
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    int windowDimA[2] = {window_height, window_width};
    int paddingA[2] = {0, 0};
    if (stride_.size() < 4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'strides' cannot be less than 4, but got "
                        << stride_.size();
    }
    int strideA[2] = {stride_[2], stride_[3]};
    int stride_h = stride_[2];
    int stride_w = stride_[3];
    if (pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) {
      pad_height_ = GetPad(old_height_, window_height, stride_h);
      pad_width_ = GetPad(old_width_, window_width, stride_w);
      pad_top_ = pad_height_ / 2;
      pad_left_ = pad_width_ / 2;
      paddingA[0] = pad_top_;
      paddingA[1] = pad_left_;
    } else {
      pad_height_ = 0;
      pad_width_ = 0;
    }
    const size_t k2dDim = 2;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetPoolingNdDescriptor(pooling_descriptor_, pooling_mode_, CUDNN_NOT_PROPAGATE_NAN, k2dDim, windowDimA,
                                  paddingA, strideA),
      "cudnnSetPoolingNdDescriptor failed");
  }

  void SetPad3D(const BaseOperatorPtr &base_operator) {
    const int kPadListSize = 6;
    const int kDims = 3;
    const int kPadScale = 2;
    auto prim = base_operator->GetPrim();
    MS_EXCEPTION_IF_NULL(prim);
    pad_mode_ = GetValue<std::string>(prim->GetAttr("pad_mode"));
    std::vector<int> window;
    std::vector<int64_t> window_me = GetValue<std::vector<int64_t>>(prim->GetAttr("kernel_size"));
    (void)std::transform(window_me.begin(), window_me.end(), std::back_inserter(window),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (window.size() < 5) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'kernel_size' cannot be less than 5, but got "
                        << window.size();
    }
    int window_depth = window[2];
    int window_height = window[3];
    int window_width = window[4];
    std::vector<int64_t> stride_me = GetValue<std::vector<int64_t>>(prim->GetAttr("strides"));
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    int windowDimA[3] = {window_depth, window_height, window_width};
    int paddingA[3] = {0, 0, 0};
    if (stride_.size() < kNumberFive) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'strides' cannot be less than 5, but got "
                        << stride_.size();
    }
    int strideA[3] = {stride_[2], stride_[3], stride_[4]};
    int stride_d = stride_[2];
    int stride_h = stride_[3];
    int stride_w = stride_[4];
    if (pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) {
      pad_depth_ = GetPad(old_depth_, window_depth, stride_d);
      pad_height_ = GetPad(old_height_, window_height, stride_h);
      pad_width_ = GetPad(old_width_, window_width, stride_w);
      pad_front_ = pad_depth_ / 2;
      pad_top_ = pad_height_ / 2;
      pad_left_ = pad_width_ / 2;
      paddingA[0] = pad_front_;
      paddingA[1] = pad_top_;
      paddingA[2] = pad_left_;
    } else if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
      pad_depth_ = 0;
      pad_height_ = 0;
      pad_width_ = 0;
    } else {
      const std::vector<int64_t> &pad_list = GetValue<std::vector<int64_t>>(prim->GetAttr("pad_list"));
      if (pad_list.size() != kPadListSize) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'pad_list' must be " << kPadListSize
                          << ", but got " << pad_list.size();
      }
      for (size_t idx = 0; idx < kDims; idx++) {
        paddingA[idx] = pad_list[idx * kPadScale];
      }
    }
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetPoolingNdDescriptor(pooling_descriptor_, pooling_mode_, CUDNN_NOT_PROPAGATE_NAN, kDims, windowDimA,
                                  paddingA, strideA),
      "cudnnSetPoolingNdDescriptor failed");
  }

  std::vector<int64_t> GetEdgeKernelSize(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
    if (!ceil_mode_ && divisor_override_ == 0) {
      return {};
    }

    const size_t k3dSizeLowerLimit = 5;
    const size_t kIdxD = 2;
    const size_t kIdxH = 3;
    const size_t kIdxW = 4;
    const size_t kScale = 2;
    std::vector<int64_t> edge_kernel;
    auto prim = base_operator->GetPrim();
    MS_EXCEPTION_IF_NULL(prim);
    std::vector<int64_t> kernel_size = GetValue<std::vector<int64_t>>(prim->GetAttr("kernel_size"));
    std::vector<int64_t> strides = GetValue<std::vector<int64_t>>(prim->GetAttr("strides"));
    std::vector<int64_t> pad = GetValue<std::vector<int64_t>>(prim->GetAttr("pad_list"));
    if (kernel_size.size() != k3dSizeLowerLimit) {
      MS_LOG(EXCEPTION) << "kernel_size must be " << k3dSizeLowerLimit << "D, but got " << kernel_size.size();
    }
    if (strides.size() != k3dSizeLowerLimit) {
      MS_LOG(EXCEPTION) << "strides must be " << k3dSizeLowerLimit << "D, but got " << strides.size();
    }
    auto input_shape = inputs[0]->GetDeviceShapeAdaptively();
    auto output_shape = outputs[0]->GetDeviceShapeAdaptively();
    kernel_size_ = {kernel_size[kIdxD], kernel_size[kIdxH], kernel_size[kIdxW]};
    std::vector<int64_t> stride = {strides[kIdxD], strides[kIdxH], strides[kIdxW]};
    std::vector<int64_t> shape_exclude_nc = {SizeToLong(input_shape[kIdxD]), SizeToLong(input_shape[kIdxH]),
                                             SizeToLong(input_shape[kIdxW])};
    (void)std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(output_shape_exclude_nc_),
                         SizeToLong);

    const size_t dim = shape_exclude_nc.size();
    if (pad.size() != dim * kScale) {
      MS_LOG(EXCEPTION) << "pad_list must be " << (dim * kScale) << "D, but got " << pad.size() << "D!";
    }

    for (size_t i = 0; i < dim; ++i) {
      size_t l_index = kScale * i;
      size_t r_index = kScale * i + 1;

      int64_t len = shape_exclude_nc[i] + pad[l_index] + pad[r_index] - kernel_size_[i];
      int64_t padding_iv = FloatToLong(std::ceil(LongToFloat(len) / LongToFloat(stride[i]))) * stride[i] - len;
      int64_t padding_r = pad[r_index] + padding_iv;
      if (padding_r > pad[r_index] && padding_r < kernel_size_[i]) {
        edge_kernel.push_back(kernel_size_[i] - padding_iv);
      } else {
        edge_kernel.push_back(kernel_size_[i]);
      }
    }
    return edge_kernel;
  }

  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t input_descriptor_;
  cudnnTensorDescriptor_t output_descriptor_;
  cudnnPoolingDescriptor_t pooling_descriptor_;
  cudnnPoolingMode_t pooling_mode_ = CUDNN_POOLING_MAX;
  std::vector<int> stride_;
  std::vector<int64_t> kernel_size_;
  std::vector<int64_t> shape_exclude_nc_;
  std::vector<int64_t> edge_kernel_;
  std::vector<int64_t> output_shape_exclude_nc_;
  std::string mode_;
  std::string pad_mode_;
  std::string data_format_ = kOpFormat_NCHW;

  cudnnDataType_t cudnn_data_type_;
  cudnnTensorFormat_t compute_format_;
  int old_depth_;
  int old_height_;
  int old_width_;
  int pad_depth_;
  int pad_height_;
  int pad_width_;
  int pad_front_;
  int pad_top_;
  int pad_left_;
  int n_;
  int c_;
  int64_t divisor_override_;
  float pad_value_;
  bool is_null_input_;
  bool ceil_mode_;
  std::string kernel_name_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GPU_KERNEL_H_
