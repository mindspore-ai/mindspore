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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV2D_GRAD_FILTER_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV2D_GRAD_FILTER_GPU_KERNEL_H_

#include <algorithm>
#include <string>
#include <vector>

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/pad_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
constexpr int NBDIMS = 4;
constexpr size_t kConv2dDimSize = 2;
constexpr int kSymmetricCoef = 2;

constexpr size_t k2DPadSize = 4;
constexpr size_t kTop2DPadIndex = 0;
constexpr size_t kBottom2DPadIndex = 1;
constexpr size_t kLeft2DPadIndex = 2;
constexpr size_t kRight2DPadIndex = 3;

constexpr size_t k2DStrideSize = 4;
constexpr size_t kHeight2DStrideIndex = 2;
constexpr size_t kWidth2DStrideIndex = 3;

constexpr size_t k2DDilationSize = 4;
constexpr size_t kHeight2DDilationIndex = 2;
constexpr size_t kWidth2DDilationIndex = 3;
constexpr auto StaticInput = 2;
constexpr auto DynamicInput = 3;

constexpr auto k2DHeightIndexNCHW = 2;
constexpr auto k2DHeightIndexNHWC = 1;

template <typename T, typename S = int64_t>
class ConvGradFilterBkwGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  ConvGradFilterBkwGpuKernelMod()
      : cudnn_handle_(nullptr),
        dw_desc_(nullptr),
        conv_desc_(nullptr),
        dy_desc_(nullptr),
        x_desc_(nullptr),
        padded_descriptor_(nullptr),
        algo_(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        compute_format_(CUDNN_TENSOR_NCHW),
        old_height_(0),
        old_width_(0),
        pad_height_(0),
        pad_width_(0),
        pad_top_(0),
        pad_left_(0),
        n_(0),
        c_(0),
        group_(1),
        is_null_input_(false),
        kernel_name_("Conv2dGradFilter"),
        input_size_(0),
        dy_size_(0),
        output_size_(0),
        padded_size_(0),
        workspace_size_(0),
        use_pad_(true) {}
  ~ConvGradFilterBkwGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *dy = GetDeviceAddress<T>(inputs, 0);
    T *x = GetDeviceAddress<T>(inputs, 1);
    T *dw = GetDeviceAddress<T>(outputs, 0);
    T *work_space = GetPossiblyNullDeviceAddress<T>(workspace, 0);

    const float alpha = 1;
    const float beta = 0;

    if (use_pad_) {
      T *padded = GetDeviceAddress<T>(workspace, 1);
      if (data_format_ == kOpFormat_NHWC) {
        CalPadNHWC(padded_size_ / sizeof(T), x, n_, old_height_, old_width_, c_, old_height_ + pad_height_,
                   old_width_ + pad_width_, pad_top_, pad_left_, pad_value_, padded,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
      } else {
        CalPad(padded_size_ / sizeof(T), x, n_, c_, old_height_, old_width_, old_height_ + pad_height_,
               old_width_ + pad_width_, pad_top_, pad_left_, pad_value_, padded,
               reinterpret_cast<cudaStream_t>(stream_ptr));
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnConvolutionBackwardFilter(cudnn_handle_, &alpha, padded_descriptor_, padded, dy_desc_, dy, conv_desc_,
                                       algo_, work_space, workspace_size_, &beta, dw_desc_, dw),
        "ConvolutionBackwardFilter failed");
      return true;
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnConvolutionBackwardFilter(cudnn_handle_, &alpha, x_desc_, x, dy_desc_, dy, conv_desc_, algo_, work_space,
                                     workspace_size_, &beta, dw_desc_, dw),
      "ConvolutionBackwardFilter failed");
    return true;
  }

  void CalPadList(const std::vector<int> pad_list, const ShapeVector in_shape, const ShapeVector filter_shape,
                  int h_index, int w_index) {
    if (pad_list[kTop2DPadIndex] == -1 || pad_list[kBottom2DPadIndex] == -1) {
      int pad_needed_h = (static_cast<int>(std::ceil((in_shape[h_index] * 1.0) / stride_[2])) - 1) * stride_[2] +
                         dilation_[2] * (filter_shape[h_index] - 1) + 1 - in_shape[h_index];
      pad_height_ = std::max(0, pad_needed_h);
      pad_top_ = static_cast<int>(std::floor(pad_height_ * 1.0 / kSymmetricCoef));
    } else {
      pad_height_ = pad_list[kTop2DPadIndex] + pad_list[kBottom2DPadIndex];
      pad_top_ = pad_list[kTop2DPadIndex];
    }
    if (pad_list[kLeft2DPadIndex] == -1 || pad_list[kRight2DPadIndex] == -1) {
      int pad_needed_w = (static_cast<int>(std::ceil((in_shape[w_index] * 1.0) / stride_[3])) - 1) * stride_[3] +
                         dilation_[3] * (filter_shape[w_index] - 1) + 1 - in_shape[w_index];
      pad_width_ = std::max(0, pad_needed_w);
      pad_left_ = static_cast<int>(std::floor(pad_width_ * 1.0 / kSymmetricCoef));
    } else {
      pad_width_ = pad_list[kLeft2DPadIndex] + pad_list[kRight2DPadIndex];
      pad_left_ = pad_list[kLeft2DPadIndex];
    }
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();
    (void)CheckParam(kernel_node);
    if (is_dynamic_attr_ && !get_dynamic_attr_value_) {
      return true;
    }
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    auto dy_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    auto in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
    is_null_input_ = CHECK_SHAPE_NULL(dy_shape, kernel_name_, "dy") || CHECK_SHAPE_NULL(in_shape, kernel_name_, "x");
    if (is_null_input_ || AnfAlgo::IsShapesDynamic({in_shape, dy_shape})) {
      InitSizeLists();
      return true;
    }
    data_format_ = AnfAlgo::GetInputFormat(kernel_node, 0);
    format_attr_ = GetAttr<std::string>(kernel_node, "format");
    if (format_attr_ == kOpFormat_NHWC) {
      data_format_ = kOpFormat_NHWC;
    }
    ShapeVector filter_shape;
    GetFilterShape(kernel_node, &filter_shape);
    CheckTensorSize({in_shape, dy_shape, filter_shape});

    int h_index = k2DHeightIndexNCHW;
    int w_index = k2DHeightIndexNCHW + 1;
    if (data_format_ == kOpFormat_NHWC) {
      compute_format_ = CUDNN_TENSOR_NHWC;
      h_index = k2DHeightIndexNHWC;
      w_index = k2DHeightIndexNHWC + 1;
    }
    SetNCHW(in_shape, &n_, &c_, &old_height_, &old_width_, data_format_);
    Set4DDesc(dy_shape, filter_shape, in_shape);
    group_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "group"));
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionGroupCount(conv_desc_, group_),
                                "cudnnSetConvGroupCount failed");

    std::vector<int> pad_list;
    std::vector<int64_t> pad_list_me = GetAttr<std::vector<int64_t>>(kernel_node, "pad_list");
    (void)std::transform(pad_list_me.begin(), pad_list_me.end(), std::back_inserter(pad_list),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (pad_list.size() != k2DPadSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'pad' must be 4, but got " << pad_list.size();
    }
    SetStrideAndDilation(kernel_node);
    CalPadList(pad_list, in_shape, filter_shape, h_index, w_index);
    use_pad_ = !(pad_height_ % kSymmetricCoef == 0 && pad_width_ % kSymmetricCoef == 0);
    pad_mode_ = GetAttr<std::string>(kernel_node, "pad_mode");

    cudnnTensorDescriptor_t x_desc_real = nullptr;
    int padA[kConv2dDimSize];
    int strideA[kConv2dDimSize] = {stride_[kHeight2DStrideIndex], stride_[kWidth2DStrideIndex]};
    int dilaA[kConv2dDimSize] = {dilation_[kHeight2DDilationIndex], dilation_[kWidth2DDilationIndex]};
    if (use_pad_) {
      use_pad_ = !(pad_height_ % kSymmetricCoef == 0 && pad_width_ % kSymmetricCoef == 0);
      int dimA[NBDIMS];
      int strideApadded[NBDIMS];
      if (data_format_ == kOpFormat_NCHW || data_format_ == kOpFormat_DEFAULT) {
        ShapeVector padded_shape = {n_, c_, old_height_ + pad_height_, old_width_ + pad_width_};
        SetDimA(padded_shape, dimA, NBDIMS, data_format_);
        SetStrideA(padded_shape, strideApadded, NBDIMS, data_format_);
      } else if (data_format_ == kOpFormat_NHWC) {
        ShapeVector padded_shape = {n_, old_height_ + pad_height_, old_width_ + pad_width_, c_};
        SetDimA(padded_shape, dimA, NBDIMS, data_format_);
        SetStrideA(padded_shape, strideApadded, NBDIMS, data_format_);
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnSetTensorNdDescriptor(padded_descriptor_, cudnn_data_type_, NBDIMS, dimA, strideApadded),
        "cudnnSetTensor4dDescriptor failed");
      padA[0] = 0;
      padA[1] = 0;
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetConvolutionNdDescriptor(conv_desc_, kConv2dDimSize, padA, strideA, dilaA,
                                                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                                  "cudnnSetConvolutionNdDescriptor failed");
      x_desc_real = padded_descriptor_;
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_top_ = 0;
        pad_left_ = 0;
      }
      padA[0] = pad_top_;
      padA[1] = pad_left_;

      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetConvolutionNdDescriptor(conv_desc_, kConv2dDimSize, padA, strideA, dilaA,
                                                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                                  "cudnnSetConvolution2dDescriptor failed");
      x_desc_real = x_desc_;
    }
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH),
                                  "cudnnSetConvolutionMathType failed.")
    }
    SelectAlgorithm(x_desc_real);
    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyConvolutionDescriptor(conv_desc_),
                               "cudnnDestroyConvolutionDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyFilterDescriptor(dw_desc_),
                               "cudnnDestroyFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(padded_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dy_desc_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(x_desc_),
                               "cudnnDestroyTensorDescriptor failed");
  }

  void ResetResource() noexcept override {
    cudnn_handle_ = nullptr;
    dw_desc_ = nullptr;
    conv_desc_ = nullptr;
    dy_desc_ = nullptr;
    x_desc_ = nullptr;
    padded_descriptor_ = nullptr;
    algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    pad_mode_ = "";
    data_format_ = kOpFormat_NCHW;
    format_attr_ = kOpFormat_NCHW;
    cudnn_data_type_ = CUDNN_DATA_FLOAT;
    compute_format_ = CUDNN_TENSOR_NCHW;
    old_height_ = 0;
    old_width_ = 0;
    pad_height_ = 0;
    pad_width_ = 0;
    pad_top_ = 0;
    pad_left_ = 0;
    n_ = 0;
    c_ = 0;
    group_ = 1;
    stride_.clear();
    dilation_.clear();
    is_null_input_ = false;
    kernel_name_ = "Conv2dGradFilter";
    input_size_ = 0;
    dy_size_ = 0;
    output_size_ = 0;
    padded_size_ = 0;
    workspace_size_ = 0;
    use_pad_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&x_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dy_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&padded_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateFilterDescriptor(&dw_desc_),
                                "cudnnCreateFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateConvolutionDescriptor(&conv_desc_),
                                "cudnnCreateConvolutionDescriptor failed");
  }
  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetTensorSizeInBytes(dy_desc_, reinterpret_cast<size_t *>(&dy_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetTensorSizeInBytes(x_desc_, reinterpret_cast<size_t *>(&input_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetFilterSizeInBytes(dw_desc_, reinterpret_cast<size_t *>(&output_size_)),
                                  "cudnnGetFilterSizeInBytes failed");
    }
    input_size_list_.push_back(dy_size_);
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);

    if (use_pad_ && !is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnGetTensorSizeInBytes(padded_descriptor_, reinterpret_cast<size_t *>(&padded_size_)),
        "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle_, padded_descriptor_, dy_desc_, conv_desc_,
                                                       dw_desc_, algo_, reinterpret_cast<size_t *>(&workspace_size_)),
        "cudnnGetConvolutionBackwardFilterWorkspaceSize failed");
      workspace_size_list_.push_back(padded_size_);
    } else {
      if (!is_null_input_) {
        CHECK_CUDNN_RET_WITH_EXCEPT(
          kernel_node_,
          cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle_, x_desc_, dy_desc_, conv_desc_, dw_desc_, algo_,
                                                         reinterpret_cast<size_t *>(&workspace_size_)),
          "cudnnGetConvolutionBackwardFilterWorkspaceSize failed");
      }
    }
    (void)workspace_size_list_.insert(workspace_size_list_.begin(), workspace_size_);
  }

 private:
  void CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != StaticInput && input_num != DynamicInput) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 2 or 3, but got " << input_num;
    }
    if (input_num == DynamicInput) {
      is_dynamic_attr_ = true;
    }
    if (TryGetIntValue(kernel_node, kShapeIndex_, &filter_shape_)) {
      get_dynamic_attr_value_ = true;
    }
    if (is_dynamic_attr_ && !get_dynamic_attr_value_) {
      input_size_list_.push_back(0);
      input_size_list_.push_back(0);
      output_size_list_.push_back(0);
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
    }
  }
  void SelectAlgorithm(cudnnTensorDescriptor_t x_desc_real) {
    constexpr int requested_algo_count = 1;
    int returned_algo_count = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t perf_results;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnn_handle_, x_desc_real, dy_desc_, conv_desc_, dw_desc_,
                                                    requested_algo_count, &returned_algo_count, &perf_results),
      "GetConvolutionBackwardFilterAlgorithm failed");
    algo_ = perf_results.algo;
#if CUDNN_VERSION < 8000
    if (group_ > 1) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle_, x_desc_real, dy_desc_, conv_desc_, dw_desc_,
                                                   CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, 0, &algo_),
        "GetConvolutionBackwardFilterAlgorithm failed");
    }
#endif
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    }
  }
  void GetFilterShape(const CNodePtr &kernel_node, ShapeVector *filter_shape) {
    if (is_dynamic_attr_ && get_dynamic_attr_value_) {
      (void)std::transform(std::begin(filter_shape_), std::end(filter_shape_), std::back_inserter(*filter_shape),
                           [](const int64_t &e) -> size_t { return e; });
    } else {
      auto shp_tuple_x = GetAttrAndConvertValueTuple(kernel_node, "filter_sizes");
      (void)std::transform(std::begin(shp_tuple_x), std::end(shp_tuple_x), std::back_inserter(*filter_shape),
                           [](const ValuePtr &e) -> int64_t {
                             auto cast_value = e->cast<Int64ImmPtr>();
                             MS_EXCEPTION_IF_NULL(cast_value);
                             return static_cast<int64_t>(cast_value->value());
                           });
    }
  }
  void Set4DDesc(const ShapeVector &dy_shape, const ShapeVector &filter_shape, const ShapeVector &in_shape) {
    const int nbDims = 4;
    int dimA[NBDIMS];
    int strideAin[NBDIMS];
    int dimAdy[NBDIMS];
    int strideAdy[NBDIMS];
    SetDimA(in_shape, dimA, nbDims, data_format_);
    SetStrideA(in_shape, strideAin, nbDims, data_format_);
    SetDimA(dy_shape, dimAdy, nbDims, data_format_);
    SetStrideA(dy_shape, strideAdy, nbDims, data_format_);
    // filter shape relued by format_attr_. In native mode it's OHWI. In transpose mode it's OIHW.
    int filterDimA[NBDIMS];
    SetDimA(filter_shape, filterDimA, nbDims, format_attr_);
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(dy_desc_, cudnn_data_type_, nbDims, dimAdy, strideAdy),
                                "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetFilterNdDescriptor(dw_desc_, cudnn_data_type_, compute_format_, nbDims, filterDimA),
      "cudnnSetFilterNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(x_desc_, cudnn_data_type_, nbDims, dimA, strideAin),
                                "cudnnSetTensorNdDescriptor failed");
  }
  void SetStrideAndDilation(const CNodePtr &kernel_node) {
    std::vector<int64_t> stride_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "stride");
    std::vector<int64_t> dilation_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "dilation");
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    (void)std::transform(dilation_me.begin(), dilation_me.end(), std::back_inserter(dilation_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (stride_.size() != k2DStrideSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'stride' must be 2, but got " << stride_.size();
    }
    if (dilation_.size() != k2DDilationSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'dilation' must be 4, but got "
                        << dilation_.size();
    }
    if (dilation_[0] != 1 || dilation_[1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'dilation' at 0 and 1 axis must be 1, but got "
                        << "dilation[0]: " << dilation_[0] << ", dilation[1]: " << dilation_[1];
    }
  }
  cudnnHandle_t cudnn_handle_;
  cudnnFilterDescriptor_t dw_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorDescriptor_t dy_desc_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t padded_descriptor_;
  cudnnConvolutionBwdFilterAlgo_t algo_;
  std::string pad_mode_;
  std::string data_format_ = kOpFormat_NCHW;
  std::string format_attr_ = kOpFormat_NCHW;

  const float pad_value_ = 0.0;
  cudnnDataType_t cudnn_data_type_;
  cudnnTensorFormat_t compute_format_;
  int old_height_;
  int old_width_;
  int pad_height_;
  int pad_width_;
  int pad_top_;
  int pad_left_;
  int n_;
  int c_;
  std::vector<int> stride_;
  std::vector<int> dilation_;
  int group_;
  bool is_null_input_;
  std::string kernel_name_;
  size_t input_size_;
  size_t dy_size_;
  size_t output_size_;
  size_t padded_size_;
  size_t workspace_size_;
  bool use_pad_;
  bool is_dynamic_attr_{false};
  bool get_dynamic_attr_value_{false};
  std::vector<int64_t> filter_shape_;
  static constexpr size_t kShapeIndex_{2};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDePORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV2D_GRAD_FILTER_GPU_KERNEL_H_
