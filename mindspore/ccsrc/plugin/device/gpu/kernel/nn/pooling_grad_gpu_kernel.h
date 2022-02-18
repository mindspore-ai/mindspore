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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GRAD_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/pad_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
constexpr size_t INPUT_NUM = 3;
template <typename T>
class PoolingGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  PoolingGradGpuKernelMod()
      : cudnn_handle_(nullptr),
        pooling_descriptor_(nullptr),
        y_descriptor_(nullptr),
        dy_descriptor_(nullptr),
        x_descriptor_(nullptr),
        dx_descriptor_(nullptr),
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
        pad_value_(0),
        is_null_input_(false),
        kernel_name_("PoolingGrad"),
        input_size_(0),
        output_size_(0),
        workspace_size_(0) {}
  ~PoolingGradGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *x_data = GetDeviceAddress<T>(inputs, 0);
    T *y = GetDeviceAddress<T>(inputs, 1);
    T *dy = GetDeviceAddress<T>(inputs, 2);
    T *dx = GetDeviceAddress<T>(outputs, 0);

    const float alpha = 1;
    const float beta = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnPoolingBackward(cudnn_handle_, pooling_descriptor_, &alpha, y_descriptor_, y, dy_descriptor_, dy,
                           x_descriptor_, x_data, &beta, dx_descriptor_, dx),
      "cudnnPoolingBackward failed");
    return true;
  }

  bool InitShape(const CNodePtr &kernel_node, int *dimA, int *strideAin, int *dimAy, int *strideAiny, int *dimAdy,
                 int *strideAdy, int *dimAout, int *strideAout, int nbDims) {
    const size_t kDoutIdx = 2;
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    auto input_mask = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
    auto dout_shape = AnfAlgo::GetInputDeviceShape(kernel_node, kDoutIdx);
    auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
    auto data_format = AnfAlgo::GetInputFormat(kernel_node, 0);
    format_attr_ = GetAttr<std::string>(kernel_node, "format");
    if (Anyone(format_attr_, kOpFormat_NHWC, kOpFormat_NDHWC)) {
      data_format = format_attr_;
    }
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(input_mask, kernel_name_, "mask") ||
      CHECK_SHAPE_NULL(dout_shape, kernel_name_, "dout") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    CheckTensorSize({input_shape, input_mask, dout_shape, output_shape});
    if (nbDims == kDim2DShapeSize) {
      SetNCHW(input_shape, &n_, &c_, &old_height_, &old_width_, data_format);
    } else if (nbDims == kDim3DShapeSize) {
      SetNCDHW(input_shape, &n_, &c_, &old_depth_, &old_height_, &old_width_, data_format);
    }
    SetDimA(input_shape, dimA, nbDims, data_format);
    SetStrideA(input_shape, strideAin, nbDims, data_format);
    SetDimA(input_mask, dimAy, nbDims, data_format);
    SetStrideA(input_mask, strideAiny, nbDims, data_format);
    SetDimA(dout_shape, dimAdy, nbDims, data_format);
    SetStrideA(dout_shape, strideAdy, nbDims, data_format);
    SetDimA(output_shape, dimAout, nbDims, data_format);
    SetStrideA(output_shape, strideAout, nbDims, data_format);
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();
    (void)CheckParam(kernel_node);
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    int nbDims = SizeToInt(input_shape.size());

    int dimA[kPoolingNbDims];
    int strideAin[kPoolingNbDims];
    int dimAy[kPoolingNbDims];
    int strideAiny[kPoolingNbDims];
    int dimAdy[kPoolingNbDims];
    int strideAdy[kPoolingNbDims];
    int dimAout[kPoolingNbDims];
    int strideAout[kPoolingNbDims];
    if (!InitShape(kernel_node, dimA, strideAin, dimAy, strideAiny, dimAdy, strideAdy, dimAout, strideAout, nbDims)) {
      return true;
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(y_descriptor_, cudnn_data_type_, nbDims, dimAy, strideAiny),
                                "cudnnSetTensor4dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(dy_descriptor_, cudnn_data_type_, nbDims, dimAdy, strideAdy),
                                "cudnnSetTensor4dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetTensorNdDescriptor(dx_descriptor_, cudnn_data_type_, nbDims, dimAout, strideAout),
      "cudnnSetTensor4dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(x_descriptor_, cudnn_data_type_, nbDims, dimA, strideAin),
                                "cudnnSetTensor4dDescriptor failed");
    SetPoolingMode(kernel_node);
    if (nbDims == kDim2DShapeSize) {
      SetPad(kernel_node);
    } else if (nbDims == kDim3DShapeSize) {
      SetPad3D(kernel_node);
    }
    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyPoolingDescriptor(pooling_descriptor_),
                               "cudnnDestroyPoolingDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dx_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(x_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dy_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(y_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&y_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dy_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&x_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dx_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreatePoolingDescriptor(&pooling_descriptor_),
                                "cudnnCreatePoolingDescriptor failed");
  }
  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(x_descriptor_, &input_size_),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(dx_descriptor_, &output_size_),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(y_descriptor_, &input_size_),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);

    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(dy_descriptor_, &input_size_),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);
  }

 private:
  void CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != INPUT_NUM) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be " << INPUT_NUM << ", but got "
                        << input_num;
    }
  }
  void SetPad(const CNodePtr &kernel_node) {
    pad_mode_ = GetAttr<std::string>(kernel_node, "pad_mode");
    std::vector<int64_t> stride_me = GetAttr<std::vector<int64_t>>(kernel_node, "strides");
    std::vector<int> window;
    std::vector<int64_t> window_me = GetAttr<std::vector<int64_t>>(kernel_node, "kernel_size");
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    (void)std::transform(window_me.begin(), window_me.end(), std::back_inserter(window),
                         [](const int64_t &value) { return static_cast<int>(value); });
    const size_t kSizeLowerLimit = 4;
    const size_t kIdxH = 2;
    const size_t kIdxW = 3;
    if (window.size() < kSizeLowerLimit) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'kernel_size' cannot be less than 4, but got "
                        << window.size();
    }
    int window_height = window[kIdxH];
    int window_width = window[kIdxW];
    if (stride_.size() < kSizeLowerLimit) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'strides' cannot be less than 4, but got "
                        << stride_.size();
    }
    int stride_h = stride_[kIdxH];
    int stride_w = stride_[kIdxW];
    if (format_attr_ == kOpFormat_NHWC) {
      const size_t kNHWCIdxH = 1;
      const size_t kNHWCIdxW = 2;
      window_height = window[kNHWCIdxH];
      window_width = window[kNHWCIdxW];
      stride_h = stride_[kNHWCIdxH];
      stride_w = stride_[kNHWCIdxW];
    }
    const size_t k2dDim = 2;
    int windowDimA[k2dDim] = {window_height, window_width};
    int paddingA[k2dDim] = {0, 0};
    int strideA[k2dDim] = {stride_h, stride_w};
    if (kSamePadModeUpperCase == pad_mode_ || kSamePadModeLowerCase == pad_mode_) {
      pad_height_ = GetPad(old_height_, window_height, stride_h);
      pad_width_ = GetPad(old_width_, window_width, stride_w);
      const int kSymCoef = 2;
      pad_top_ = pad_height_ / kSymCoef;
      pad_left_ = pad_width_ / kSymCoef;
      paddingA[0] = pad_top_;
      paddingA[1] = pad_left_;
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_height_ = 0;
        pad_width_ = 0;
      }
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetPoolingNdDescriptor(pooling_descriptor_, pooling_mode_, CUDNN_NOT_PROPAGATE_NAN,
                                                            2, windowDimA, paddingA, strideA),
                                "cudnnSetPoolingNdDescriptor failed");
  }

  void SetPad3D(const CNodePtr &kernel_node) {
    pad_mode_ = GetAttr<std::string>(kernel_node, "pad_mode");
    std::vector<int64_t> stride_me = GetAttr<std::vector<int64_t>>(kernel_node, "strides");
    std::vector<int> window;
    std::vector<int64_t> window_me = GetAttr<std::vector<int64_t>>(kernel_node, "kernel_size");
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    (void)std::transform(window_me.begin(), window_me.end(), std::back_inserter(window),
                         [](const int64_t &value) { return static_cast<int>(value); });
    const size_t k3dSizeLowerLimit = 5;
    const size_t kIdxD = 2;
    const size_t kIdxH = 3;
    const size_t kIdxW = 4;
    if (window.size() < k3dSizeLowerLimit) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'kernel_size' cannot be less than 5, but got "
                        << window.size();
    }
    int window_depth = window[kIdxD];
    int window_height = window[kIdxH];
    int window_width = window[kIdxW];
    if (stride_.size() < k3dSizeLowerLimit) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'strides' cannot be less than 5, but got "
                        << stride_.size();
    }
    int stride_d = stride_[kIdxD];
    int stride_h = stride_[kIdxH];
    int stride_w = stride_[kIdxW];
    if (format_attr_ == kOpFormat_NDHWC) {
      const size_t kNDHWCIdxD = 1;
      const size_t kNDHWCIdxH = 2;
      const size_t kNDHWCIdxW = 3;
      window_depth = window[kNDHWCIdxD];
      window_height = window[kNDHWCIdxH];
      window_width = window[kNDHWCIdxW];
      stride_d = stride_[kNDHWCIdxD];
      stride_h = stride_[kNDHWCIdxH];
      stride_w = stride_[kNDHWCIdxW];
    }
    const size_t k3dDimSize = 3;
    int windowDimA[k3dDimSize] = {window_depth, window_height, window_width};
    int paddingA[k3dDimSize] = {0, 0, 0};
    int strideA[k3dDimSize] = {stride_d, stride_h, stride_w};
    if (kSamePadModeUpperCase == pad_mode_ || kSamePadModeLowerCase == pad_mode_) {
      pad_depth_ = GetPad(old_depth_, window_depth, stride_d);
      pad_height_ = GetPad(old_height_, window_height, stride_h);
      pad_width_ = GetPad(old_width_, window_width, stride_w);
      const int kSymCoef = 2;
      pad_front_ = pad_depth_ / kSymCoef;
      pad_top_ = pad_height_ / kSymCoef;
      pad_left_ = pad_width_ / kSymCoef;
      paddingA[0] = pad_front_;
      paddingA[1] = pad_top_;
      const size_t kPadLeftIdx = 2;
      paddingA[kPadLeftIdx] = pad_left_;
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_depth_ = 0;
        pad_height_ = 0;
        pad_width_ = 0;
      }
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetPoolingNdDescriptor(pooling_descriptor_, pooling_mode_, CUDNN_NOT_PROPAGATE_NAN,
                                                            3, windowDimA, paddingA, strideA),
                                "cudnnSetPoolingNdDescriptor failed");
  }

  void SetPoolingMode(const CNodePtr &kernel_node) {
    mode_ = AnfAlgo::GetCNodeName(kernel_node);
    if (mode_ == "AvgPoolGrad") {
      pooling_mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
      pad_value_ = 0.0;
    } else {
      pooling_mode_ = CUDNN_POOLING_MAX;
      pad_value_ = kSignedMinFloat;
    }
  }

  cudnnHandle_t cudnn_handle_;
  cudnnPoolingDescriptor_t pooling_descriptor_;
  cudnnTensorDescriptor_t y_descriptor_;
  cudnnTensorDescriptor_t dy_descriptor_;
  cudnnTensorDescriptor_t x_descriptor_;
  cudnnTensorDescriptor_t dx_descriptor_;
  cudnnPoolingMode_t pooling_mode_ = CUDNN_POOLING_MAX;
  std::vector<int> stride_;

  std::string mode_;
  std::string pad_mode_;
  std::string format_attr_ = kOpFormat_NCHW;
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
  float pad_value_;
  bool is_null_input_;
  std::string kernel_name_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GRAD_GPU_KERNEL_H_
