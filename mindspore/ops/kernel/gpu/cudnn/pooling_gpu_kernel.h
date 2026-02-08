/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "kernel/gpu/gpu_kernel.h"
#include "primitive/math_ops.h"
#include "primitive/conv_pool_op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "ir/format_utils.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/avg_pool3d_helper_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/binary_ops_impl.cuh"
#include "kernel/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
constexpr auto kNumberThree = 3;
constexpr auto kNumberTwo = 2;
constexpr auto kKernelSizeIdx = 1;
constexpr auto kStridesIdx = 2;
constexpr auto kPadModeIdx = 3;
constexpr auto kDataFormatIdx = 4;
constexpr auto kPadListIdx = 4;
constexpr auto kCeilModeIdx = 5;
constexpr auto kCountIncludePadIdx = 6;
constexpr auto kDivisorOverrideIdx = 7;
constexpr auto kFormatAvg3DIdx = 8;
constexpr auto kFormatMax3DIdx = 6;
size_t kKernelSize = 2;
size_t kStrideSize = 2;
size_t kNC_SIZE = 0;
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
        input_size_(0),
        output_size_(0),
        workspace_size_(0) {}
  ~PoolingFwdGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
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
        auto status = CalRealKernelSize(output_shape_exclude_nc_, kernel_size_, edge_kernel_, work_addr, 0,
                                        reinterpret_cast<cudaStream_t>(stream_ptr));
        CHECK_CUDA_STATUS(status, kernel_name_);
      }
      std::vector<int64_t> shape = {static_cast<int64_t>(output_num)};
      auto status = BinaryOpWithBroadcastCudaFunc<BinaryOpType::kMul, T, T, T>(
        false, shape, shape, shape, output_addr, work_addr, output_addr, device_id_,
        reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, kernel_name_);
    }
    return true;
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    InitResource();
    size_t format_index = kDataFormatIdx;
    if (kernel_name_ == kAvgPool3DOpName) {
      divisor_override_ = GetValue<int64_t>(primitive_->GetAttr("divisor_override"));
      ceil_mode_ = GetValue<bool>(primitive_->GetAttr("ceil_mode"));
      AvgPool3DPadListCheck(inputs);
      format_index = kFormatAvg3DIdx;
    }
    if (kernel_name_ == kMaxPool3DOpName) {
      format_index = kFormatMax3DIdx;
    }
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs[0]->dtype_id()));
    data_format_ = inputs[0]->format();
    mindspore::Format format_attr;
    if (kernel_name_ == kAvgPoolOpName) {
      format_attr = static_cast<mindspore::Format>(inputs[format_index]->GetValueWithCheck<int64_t>());
    } else {
      format_attr = GetFormatFromStrToEnum(GetValue<std::string>(primitive_->GetAttr("format")));
    }
    if (Anyone(format_attr, mindspore::Format::NHWC, mindspore::Format::NDHWC)) {
      data_format_ = format_attr;
    }
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
      return ret;
    }
    ResetResource();
    auto input_shape = inputs[0]->GetDeviceShapeVector();
    auto output_shape = outputs[0]->GetDeviceShapeVector();
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
    SetPoolingMode(inputs);
    if (dim == kDim2DShapeSize) {
      SetPad(inputs);
    } else if (dim == kDim3DShapeSize) {
      SetPad3D(inputs);
    }
    edge_kernel_ = GetEdgeKernelSize(inputs, outputs);
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
    output_size_list_.push_back(output_size_);
    workspace_size_list_.push_back(output_size_);
  }
  void ResetResource() {
    output_size_list_.clear();
    workspace_size_list_.clear();
    stride_.clear();
  }

 private:
  void SetPoolingMode(const std::vector<KernelTensor *> &inputs) {
    mode_ = kernel_name_;
    bool include = false;
    if (kernel_name_ == kAvgPool3DOpName) {
      if (primitive_->HasAttr("count_include_pad")) {
        include = GetValue<bool>(primitive_->GetAttr("count_include_pad"));
      }
    }
    if (mode_ == kAvgPoolOpName || mode_ == kAvgPool3DOpName) {
      pooling_mode_ =
        include ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
      pad_value_ = 0.0;
    } else {
      pooling_mode_ = CUDNN_POOLING_MAX;
      pad_value_ = kSignedMinFloat;
    }
  }

  void SetPad(const std::vector<KernelTensor *> &inputs) {
    mindspore::PadMode pad_mode_;
    if (kernel_name_ == kAvgPoolOpName) {
      pad_mode_ = static_cast<mindspore::PadMode>(inputs[kPadModeIdx]->GetValueWithCheck<int64_t>());
    } else {
      pad_mode_ = static_cast<mindspore::PadMode>(
        ops::PadModeStringToInt(GetValue<std::string>(primitive_->GetAttr("pad_mode"))));
      kNC_SIZE += 2;
    }
    std::vector<int> window;
    std::vector<int64_t> window_me;
    if (kernel_name_ == kAvgPoolOpName) {
      window_me = inputs[kKernelSizeIdx]->GetValueWithCheck<std::vector<int64_t>>();
    } else {
      window_me = GetValue<std::vector<int64_t>>(primitive_->GetAttr("kernel_size"));
    }

    (void)std::transform(window_me.begin(), window_me.end(), std::back_inserter(window),
                         [](const int64_t &value) { return static_cast<int>(value); });

    if (window.size() < kKernelSize + kNC_SIZE) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'kernel_size' cannot be less than "
                        << kKernelSize + kNC_SIZE << ", but got " << window.size();
    }
    int window_height = window[0 + kNC_SIZE];
    int window_width = window[1 + kNC_SIZE];
    std::vector<int64_t> stride_me;
    if (kernel_name_ == kAvgPoolOpName) {
      stride_me = inputs[kStridesIdx]->GetValueWithCheck<std::vector<int64_t>>();
    } else {
      stride_me = GetValue<std::vector<int64_t>>(primitive_->GetAttr("strides"));
    }
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    int windowDimA[2] = {window_height, window_width};
    int paddingA[2] = {0, 0};
    if (stride_.size() < kStrideSize + kNC_SIZE) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'strides' cannot be less than "
                        << kStrideSize + kNC_SIZE << ", but got " << stride_.size();
    }
    int strideA[2] = {stride_[kNC_SIZE], stride_[1 + kNC_SIZE]};
    int stride_h = stride_[kNC_SIZE];
    int stride_w = stride_[1 + kNC_SIZE];
    if (pad_mode_ == mindspore::PadMode::SAME) {
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
    if (kernel_name_ != kAvgPoolOpName) {
      kNC_SIZE -= 2;
    }
    const size_t k2dDim = 2;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetPoolingNdDescriptor(pooling_descriptor_, pooling_mode_, CUDNN_NOT_PROPAGATE_NAN, k2dDim, windowDimA,
                                  paddingA, strideA),
      "cudnnSetPoolingNdDescriptor failed");
  }

  void SetPad3D(const std::vector<KernelTensor *> &inputs) {
    const int kPadListSize = 6;
    const int kDims = 3;
    const int kPadScale = 2;
    pad_mode_ =
      static_cast<mindspore::PadMode>(ops::PadModeStringToInt(GetValue<std::string>(primitive_->GetAttr("pad_mode"))));

    std::vector<int> window;
    auto window_me = GetValue<std::vector<int64_t>>(primitive_->GetAttr("kernel_size"));
    (void)std::transform(window_me.begin(), window_me.end(), std::back_inserter(window),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (window.size() < 5) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'kernel_size' cannot be less than 5, but got "
                        << window.size();
    }
    int window_depth = window[2];
    int window_height = window[3];
    int window_width = window[4];
    auto stride_me = GetValue<std::vector<int64_t>>(primitive_->GetAttr("strides"));
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    int windowDimA[3] = {window_depth, window_height, window_width};
    int paddingA[3] = {0, 0, 0};
    if (stride_.size() < 5) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'strides' cannot be less than 5, but got "
                        << stride_.size();
    }
    int strideA[3] = {stride_[2], stride_[3], stride_[4]};
    int stride_d = stride_[2];
    int stride_h = stride_[3];
    int stride_w = stride_[4];
    if (pad_mode_ == mindspore::PadMode::SAME) {
      pad_depth_ = GetPad(old_depth_, window_depth, stride_d);
      pad_height_ = GetPad(old_height_, window_height, stride_h);
      pad_width_ = GetPad(old_width_, window_width, stride_w);
      pad_front_ = pad_depth_ / 2;
      pad_top_ = pad_height_ / 2;
      pad_left_ = pad_width_ / 2;
      paddingA[0] = pad_front_;
      paddingA[1] = pad_top_;
      paddingA[2] = pad_left_;
    } else if (pad_mode_ == mindspore::PadMode::VALID) {
      pad_depth_ = 0;
      pad_height_ = 0;
      pad_width_ = 0;
    } else {
      std::vector<int64_t> pad_list = GetValue<std::vector<int64_t>>(primitive_->GetAttr("pad_list"));
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

  std::vector<int64_t> GetEdgeKernelSize(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
    if (!ceil_mode_ && divisor_override_ == 0) {
      return {};
    }

    const size_t k3dSizeLowerLimit = 5;
    const size_t kIdxD = 2;
    const size_t kIdxH = 3;
    const size_t kIdxW = 4;
    const size_t kScale = 2;
    std::vector<int64_t> edge_kernel;
    std::vector<int64_t> kernel_size;
    std::vector<int64_t> strides;
    std::vector<int64_t> pad;
    if (kernel_name_ == kAvgPoolOpName) {
      kernel_size = inputs[kKernelSizeIdx]->GetValueWithCheck<std::vector<int64_t>>();
      strides = inputs[kStridesIdx]->GetValueWithCheck<std::vector<int64_t>>();
      pad = inputs[kPadListIdx]->GetValueWithCheck<std::vector<int64_t>>();
    } else {
      kernel_size = GetValue<std::vector<int64_t>>(primitive_->GetAttr("kernel_size"));
      strides = GetValue<std::vector<int64_t>>(primitive_->GetAttr("strides"));
      pad = GetValue<std::vector<int64_t>>(primitive_->GetAttr("pad_list"));
    }
    if (kernel_size.size() != k3dSizeLowerLimit) {
      MS_LOG(EXCEPTION) << "kernel_size must be " << k3dSizeLowerLimit << "D, but got " << kernel_size.size();
    }
    if (strides.size() != k3dSizeLowerLimit) {
      MS_LOG(EXCEPTION) << "strides must be " << k3dSizeLowerLimit << "D, but got " << strides.size();
    }
    auto input_shape = inputs[0]->GetDeviceShapeVector();
    auto output_shape = outputs[0]->GetDeviceShapeVector();
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

  void AvgPool3DPadListCheck(const std::vector<KernelTensor *> &inputs) {
    mindspore::PadMode pad_mode =
      static_cast<mindspore::PadMode>(ops::PadModeStringToInt(GetValue<std::string>(primitive_->GetAttr("pad_mode"))));
    if (pad_mode == mindspore::PadMode::SAME || pad_mode == mindspore::PadMode::VALID) {
      return;
    }
    const std::vector<int64_t> &pad_list = GetValue<std::vector<int64_t>>(primitive_->GetAttr("pad_list"));
    if (primitive_->HasAttr("count_include_pad") && !GetValue<bool>(primitive_->GetAttr("count_include_pad")) &&
        primitive_->HasAttr("divisor_override") && GetValue<int64_t>(primitive_->GetAttr("divisor_override")) != 0 &&
        std::any_of(pad_list.begin(), pad_list.end(), [](int64_t pad) { return pad > 0; })) {
      MS_LOG(EXCEPTION) << kernel_name_ << "does not support the scenes while padmode == " << pad_mode
                        << " && padding > 0 && count_include_pad == False && divisor_override != None";
    }

    const size_t kPadScale = 2;
    for (size_t idx = 0; idx < pad_list.size(); idx += kPadScale) {
      if (pad_list[idx] != pad_list[idx + 1]) {
        MS_LOG(EXCEPTION) << "For " << kernel_name_ << ", pad[" << idx << "] and pad[" << (idx + 1)
                          << "] must be equal, but got " << pad_list[idx] << " and " << pad_list[idx + 1];
      }
    }
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
  mindspore::PadMode pad_mode_;
  mindspore::Format data_format_ = mindspore::Format::NCHW;

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
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GPU_KERNEL_H_
