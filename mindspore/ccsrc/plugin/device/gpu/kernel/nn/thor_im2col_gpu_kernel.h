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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_THORIM2COLGPUKERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_THORIM2COLGPUKERNEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/pad_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
constexpr size_t kPadSize = 4;
constexpr size_t kTopPadIndex = 0;
constexpr size_t kBottomPadIndex = 1;
constexpr size_t kLeftPadIndex = 2;
constexpr size_t kRightPadIndex = 3;

constexpr size_t kStrideSize = 4;
constexpr size_t kHeightStrideIndex = 2;
constexpr size_t kWidthStrideIndex = 3;

constexpr size_t kDilationSize = 4;
constexpr size_t kHeightDilationIndex = 2;
constexpr size_t kWidthDilationIndex = 3;
template <typename T>
class ThorIm2ColFwdGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  ThorIm2ColFwdGpuKernelMod()
      : cudnn_handle_(nullptr),
        input_desc_(nullptr),
        output_desc_(nullptr),
        filter_desc_(nullptr),
        conv_algorithm_(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM),
        conv_desc_n(nullptr),
        padded_desc_n(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        old_height_(0),
        old_width_(0),
        pad_height_(0),
        pad_width_n(0),
        pad_top_(0),
        pad_left_(0),
        n_(0),
        c_(0),
        is_null_input_(false),
        kernel_name_("ThorIm2col"),
        input_size_(0),
        output_size_(0),
        padded_size_(0),
        workspace_size_(0),
        use_pad_(true) {}
  ~ThorIm2ColFwdGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    if ((pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) && use_pad_) {
      T *padded_addr = GetDeviceAddress<T>(workspace, 0);
      auto status = CalPad(padded_size_ / sizeof(T), input_addr, n_, c_, old_height_, old_width_,
                           old_height_ + pad_height_, old_width_ + pad_width_n, pad_top_, pad_left_, pad_value_,
                           padded_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDA_STATUS(status, kernel_name_);
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnIm2Col(cudnn_handle_, padded_desc_n, padded_addr, filter_desc_, conv_desc_n, output_addr),
        "cudnnThorIm2ColForward failed");
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnIm2Col(cudnn_handle_, input_desc_, input_addr, filter_desc_, conv_desc_n, output_addr),
        "cudnnThorIm2ColForward failed");
    }

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();
    (void)CheckParam(kernel_node);
    auto in_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    if (AnfAlgo::IsShapesDynamic({in_shape, output_shape})) {
      return true;
    }
    is_null_input_ =
      CHECK_SHAPE_NULL(in_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    const size_t kInputDimSize = 4;
    if (in_shape.size() != kInputDimSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input must be 4, but got "
                        << in_shape.size();
    }
    auto filter_shape = GetAttr<std::vector<int64_t>>(kernel_node, "kernel_size");
    const size_t kFilterDimSize = 2;
    if (filter_shape.size() < kFilterDimSize) {
      MS_LOG(EXCEPTION) << "For 'ThorIm2ColGpuKernel', the dimension of filter must be greater than or equal to 2, "
                        << "but got " << filter_shape.size();
    }
    const size_t kOutputDimSize = 6;
    if (output_shape.size() < kOutputDimSize) {
      MS_LOG(EXCEPTION) << "For 'ThorIm2ColGpuKernel', the dimension of output must be greater than or equal to 6, "
                        << "but got " << filter_shape.size();
    }
    CheckTensorSize({in_shape, output_shape});
    Set4DDesc(in_shape, filter_shape, output_shape);
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionGroupCount(conv_desc_n, 1),
                                "cudnnSetConvGroupCount failed");
    pad_height_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "pad"));
    pad_width_n = pad_height_;
    pad_mode_ = GetAttr<std::string>(kernel_node, "pad_mode");
    SetStrideAndDilation(kernel_node);
    if (pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) {
      SetPad(in_shape, kernel_node);
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_height_ = 0;
        pad_width_n = 0;
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSetConvolution2dDescriptor(conv_desc_n, pad_height_, pad_width_n, stride_[kHeightStrideIndex],
                                        stride_[kWidthStrideIndex], dilation_[kHeightDilationIndex],
                                        dilation_[kWidthDilationIndex], CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
        "cudnnSetConvolution2dDescriptor failed");
    }
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionMathType(conv_desc_n, CUDNN_TENSOR_OP_MATH),
                                  "cudnnSetConvolutionMathType failed.")
    }
    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyConvolutionDescriptor(conv_desc_n),
                               "cudnnDestroyConvolutionDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyFilterDescriptor(filter_desc_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(padded_desc_n),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(output_desc_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(input_desc_),
                               "cudnnDestroyTensorDescriptor failed");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&input_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&output_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&padded_desc_n),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateFilterDescriptor(&filter_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateConvolutionDescriptor(&conv_desc_n),
                                "cudnnCreateConvolutionDescriptor failed");
  }

  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetTensorSizeInBytes(input_desc_, reinterpret_cast<size_t *>(&input_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetTensorSizeInBytes(output_desc_, reinterpret_cast<size_t *>(&output_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetTensorSizeInBytes(padded_desc_n, reinterpret_cast<size_t *>(&padded_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    if ((pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) && use_pad_ && !is_null_input_) {
      workspace_size_list_.push_back(padded_size_);
    }
    return;
  }

 private:
  void CheckParam(const CNodePtr &kernel_node) {
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 1, but got " << input_num;
    }

    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
    }
  }
  void SetPad(const ShapeVector &in_shape, const CNodePtr &kernel_node) {
    std::vector<int> pad_list;
    std::vector<int64_t> pad_list_me = GetAttr<std::vector<int64_t>>(kernel_node, "pad_list");
    (void)std::transform(pad_list_me.begin(), pad_list_me.end(), std::back_inserter(pad_list),
                         [](const int64_t &value) { return static_cast<int>(value); });
    const size_t kInIdxForN = 0;
    const size_t kInIdxForC = 1;
    const size_t kInIdxForH = 2;
    const size_t kInIdxForW = 3;
    n_ = LongToInt(in_shape[kInIdxForN]);
    c_ = LongToInt(in_shape[kInIdxForC]);
    old_height_ = LongToInt(in_shape[kInIdxForH]);
    old_width_ = LongToInt(in_shape[kInIdxForW]);

    if (pad_list.size() != kPadSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'pad' must be 4, but got " << pad_list.size();
    }
    pad_height_ = pad_list[kTopPadIndex] + pad_list[kBottomPadIndex];
    pad_width_n = pad_list[kLeftPadIndex] + pad_list[kRightPadIndex];
    pad_top_ = pad_list[kTopPadIndex];
    pad_left_ = pad_list[kLeftPadIndex];

    // if use_pad_ == true, using zero padding in advance, else using the default cudnn pad.
    const int kSymmetricCoef = 2;
    if (pad_height_ % kSymmetricCoef == 0 && pad_width_n % kSymmetricCoef == 0) {
      use_pad_ = false;
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensor4dDescriptor(padded_desc_n, CUDNN_TENSOR_NCHW, cudnn_data_type_, n_, c_,
                                                           old_height_ + pad_height_, old_width_ + pad_width_n),
                                "cudnnSetTensor4dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetConvolution2dDescriptor(conv_desc_n, use_pad_ ? 0 : pad_top_, use_pad_ ? 0 : pad_left_,
                                      stride_[kHeightStrideIndex], stride_[kWidthStrideIndex],
                                      dilation_[kHeightDilationIndex], dilation_[kWidthDilationIndex],
                                      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
      "cudnnSetConvolution2dDescriptor failed");
  }

  void Set4DDesc(const ShapeVector &in_shape, const ShapeVector &filter_shape, const ShapeVector &output_shape) {
    const size_t kIdx0 = 0;
    const size_t kIdx1 = 1;
    const size_t kIdx2 = 2;
    const size_t kIdx3 = 3;
    const size_t kIdx4 = 4;
    const size_t kIdx5 = 5;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, LongToInt(in_shape[kIdx0]),
                                 LongToInt(in_shape[kIdx1]), LongToInt(in_shape[kIdx2]), LongToInt(in_shape[kIdx3])),
      "cudnnSetTensor4dDescriptor failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetFilter4dDescriptor(filter_desc_, cudnn_data_type_, CUDNN_TENSOR_NCHW, 1,
                                                           LongToInt(in_shape[1]), filter_shape[0], filter_shape[1]),
                                "cudnnSetFilter4dDescriptor failed");

    auto out_H = output_shape[kIdx0] * output_shape[kIdx1] * output_shape[kIdx2];
    auto out_W = output_shape[kIdx3] * output_shape[kIdx4] * output_shape[kIdx5];
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_,
                                                           LongToInt(out_H), LongToInt(out_W), 1, 1),
                                "cudnnSetTensor4dDescriptor failed");
  }

  void SetStrideAndDilation(const CNodePtr &kernel_node) {
    std::vector<int64_t> stride_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "stride");
    std::vector<int64_t> dilation_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "dilation");
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    (void)std::transform(dilation_me.begin(), dilation_me.end(), std::back_inserter(dilation_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (stride_.size() != kStrideSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'stride' must be 4, but got " << stride_.size();
    }
    if (stride_[0] != 1 || stride_[1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'stride' at 0 and 1 axis must be 1, but got "
                        << "stride[0]: " << stride_[0] << ", stride[1]: " << stride_[1];
    }
    if (dilation_.size() != kDilationSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'dilation' must be 4, but got "
                        << dilation_.size();
    }
    if (dilation_[0] != 1 || dilation_[1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'dilation' at 0 and 1 axis must be 1, but got "
                        << "dilation[0]: " << dilation_[0] << ", dilation[1]: " << dilation_[1];
    }
  }

  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionFwdAlgo_t conv_algorithm_;
  cudnnConvolutionDescriptor_t conv_desc_n;
  cudnnTensorDescriptor_t padded_desc_n;
  std::string pad_mode_;

  const float pad_value_ = 0.0;
  cudnnDataType_t cudnn_data_type_;
  int old_height_;
  int old_width_;
  int pad_height_;
  int pad_width_n;
  int pad_top_;
  int pad_left_;
  int n_;
  int c_;
  std::vector<int> stride_;
  std::vector<int> dilation_;
  bool is_null_input_;
  std::string kernel_name_;
  size_t input_size_;
  size_t output_size_;
  size_t padded_size_;
  size_t workspace_size_;
  bool use_pad_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_THORIM2COLGPUKERNEL_H_
