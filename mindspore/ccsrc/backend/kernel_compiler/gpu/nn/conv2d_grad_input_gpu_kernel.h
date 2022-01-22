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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV2D_GRAD_INPUT_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV2D_GRAD_INPUT_GPU_KERNEL_H_

#include <algorithm>
#include <string>
#include <vector>
#include <map>

#include "backend/kernel_compiler/gpu/cuda_impl/pad_impl.cuh"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
const std::map<std::string, size_t> kFormatIndexMap = {{"NCHW", 2}, {"HWCN", 0}, {"NHWC", 1}};

constexpr size_t kConv2dDimSize = 2;
constexpr int kSymmetricCoef = 2;

constexpr size_t k2DPadSize = 4;
constexpr size_t kTop2DPadIndex = 0;
constexpr size_t kBottom2DPadIndex = 1;
constexpr size_t kLeft2DPadIndex = 2;
constexpr size_t kRight2DPadIndex = 3;

constexpr size_t k2DStrideSize = 2;
constexpr size_t kHeight2DStrideIndex = 0;
constexpr size_t kWidth2DStrideIndex = 1;

constexpr size_t k2DDilationSize = 4;
constexpr size_t kHeight2DDilationIndex = 2;
constexpr size_t kWidth2DDilationIndex = 3;

template <typename T>
class ConvGradInputBkwGpuKernelMod : public NativeGpuKernelMod {
 public:
  ConvGradInputBkwGpuKernelMod()
      : cudnn_handle_(nullptr),
        w_desc_(nullptr),
        conv_desc_(nullptr),
        dy_desc_(nullptr),
        dx_desc_(nullptr),
        padded_descriptor_(nullptr),
        algo_(CUDNN_CONVOLUTION_BWD_DATA_ALGO_0),
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
        kernel_name_("Conv2dGradInput"),
        dy_size_(0),
        w_size_(0),
        output_size_(0),
        padded_size_(0),
        workspace_size_(0),
        use_pad_(true),
        beta_(0) {}
  ~ConvGradInputBkwGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *dy = GetDeviceAddress<T>(inputs, 0);
    T *w = GetDeviceAddress<T>(inputs, 1);
    T *dx = GetDeviceAddress<T>(outputs, 0);
    T *work_space = GetPossiblyNullDeviceAddress<T>(workspace, 0);

    const float alpha = 1;
    if (use_pad_) {
      T *padded = GetDeviceAddress<T>(workspace, 1);

      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnConvolutionBackwardData(cudnn_handle_, &alpha, w_desc_, w, dy_desc_, dy, conv_desc_, algo_, work_space,
                                     workspace_size_, &beta_, padded_descriptor_, padded),
        "ConvolutionBackwardData failed");
      if (data_format_ == kOpFormat_NHWC) {
        CalPadGradNHWC(output_size_ / sizeof(T), padded, n_, old_height_, old_width_, c_, old_height_ + pad_height_,
                       old_width_ + pad_width_, pad_top_, pad_left_, dx, reinterpret_cast<cudaStream_t>(stream_ptr));
      } else {
        CalPadGrad(output_size_ / sizeof(T), padded, n_, c_, old_height_, old_width_, old_height_ + pad_height_,
                   old_width_ + pad_width_, pad_top_, pad_left_, dx, reinterpret_cast<cudaStream_t>(stream_ptr));
      }
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnConvolutionBackwardData(cudnn_handle_, &alpha, w_desc_, w, dy_desc_, dy, conv_desc_, algo_, work_space,
                                     workspace_size_, &beta_, dx_desc_, dx),
        "ConvolutionBackwardData failed");
    }
    return true;
  }

  bool CheckNull(const std::vector<size_t> dy_shape, const std::vector<size_t> filter_shape) {
    is_null_input_ =
      CHECK_SHAPE_NULL(dy_shape, kernel_name_, "dy") || CHECK_SHAPE_NULL(filter_shape, kernel_name_, "weight");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    return false;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();
    (void)CheckParam(kernel_node);
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    data_format_ = AnfAlgo::GetInputFormat(kernel_node, 0);
    auto format_attr = GetAttr<std::string>(kernel_node, "format");
    if (format_attr == kOpFormat_NHWC) {
      data_format_ = kOpFormat_NHWC;
    }
    auto dy_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    auto filter_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
    if (CheckNull(dy_shape, filter_shape)) {
      return true;
    }

    std::vector<size_t> input_shape;
    GetInputShape(kernel_node, &input_shape);
    if (data_format_ == kOpFormat_NHWC) {
      compute_format_ = CUDNN_TENSOR_NHWC;
      if (format_attr == kOpFormat_NCHW) {
        ShapeNCHW2NHWC(&input_shape);
      }
    }
    CheckTensorSize({input_shape, dy_shape, filter_shape});
    SetNCHW(input_shape, &n_, &c_, &old_height_, &old_width_, data_format_);
    Set4DDesc(dy_shape, input_shape, filter_shape);

    group_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "group"));
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionGroupCount(conv_desc_, group_),
                                "cudnnSetConvGroupCount failed");

    std::vector<int> pad_list;
    std::vector<int64_t> pad_list_me = GetAttr<std::vector<int64_t>>(kernel_node, "pad_list");
    (void)std::transform(pad_list_me.begin(), pad_list_me.end(), std::back_inserter(pad_list),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (pad_list.size() != k2DPadSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'pad' should be 4, but got " << pad_list.size();
    }
    pad_height_ = pad_list[kTop2DPadIndex];
    pad_width_ = pad_list[kLeft2DPadIndex];
    use_pad_ = !((pad_height_ == pad_list[kBottom2DPadIndex]) && (pad_width_ == pad_list[kRight2DPadIndex]));
    pad_mode_ = GetAttr<std::string>(kernel_node, "pad_mode");
    SetStrideAndDilation(kernel_node);
    cudnnTensorDescriptor_t dx_desc_real = nullptr;
    int padA[kConv2dDimSize];
    int strideA[kConv2dDimSize] = {stride_[kHeight2DStrideIndex], stride_[kWidth2DStrideIndex]};
    int dilaA[kConv2dDimSize] = {dilation_[kHeight2DDilationIndex], dilation_[kWidth2DDilationIndex]};
    if (use_pad_) {
      pad_height_ = pad_list[kTop2DPadIndex] + pad_list[kBottom2DPadIndex];
      pad_width_ = pad_list[kLeft2DPadIndex] + pad_list[kRight2DPadIndex];
      pad_top_ = pad_list[kTop2DPadIndex];
      pad_left_ = pad_list[kLeft2DPadIndex];
      if (pad_height_ % kSymmetricCoef == 0 && pad_width_ % kSymmetricCoef == 0) {
        use_pad_ = false;
      }
      int dimA[k2DPadSize];
      int strideApadded[k2DPadSize];
      if (data_format_ == kOpFormat_NCHW || data_format_ == kOpFormat_DEFAULT) {
        auto padded_shape = {IntToSize(n_), IntToSize(c_), IntToSize(old_height_ + pad_height_),
                             IntToSize(old_width_ + pad_width_)};
        SetDimA(padded_shape, dimA, k2DPadSize, data_format_);
        SetStrideA(padded_shape, strideApadded, k2DPadSize, data_format_);
      } else if (data_format_ == kOpFormat_NHWC) {
        auto padded_shape = {IntToSize(n_), IntToSize(old_height_ + pad_height_), IntToSize(old_width_ + pad_width_),
                             IntToSize(c_)};
        SetDimA(padded_shape, dimA, k2DPadSize, data_format_);
        SetStrideA(padded_shape, strideApadded, k2DPadSize, data_format_);
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnSetTensorNdDescriptor(padded_descriptor_, cudnn_data_type_, k2DPadSize, dimA, strideApadded),
        "cudnnSetTensor4dDescriptor failed");
      padA[0] = 0;
      padA[1] = 0;
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetConvolutionNdDescriptor(conv_desc_, kConv2dDimSize, padA, strideA, dilaA,
                                                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                                  "cudnnSetConvolutionNdDescriptor failed");
      dx_desc_real = padded_descriptor_;
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_height_ = 0;
        pad_width_ = 0;
      }
      padA[0] = pad_height_;
      padA[1] = pad_width_;
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetConvolutionNdDescriptor(conv_desc_, kConv2dDimSize, padA, strideA, dilaA,
                                                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                                  "cudnnSetConvolution2dDescriptor failed");
      dx_desc_real = dx_desc_;
    }
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH),
                                  "cudnnSetConvolutionMathType failed.")
    }
    SelectAlgorithm(dx_desc_real);
    beta_ = GetAttrWithDefault(kernel_node, "inplace_algo", std::string("cover")) == "cover" ? 0 : 1;
    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyConvolutionDescriptor(conv_desc_),
                               "cudnnDestroyConvolutionDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyFilterDescriptor(w_desc_),
                               "cudnnDestroyFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(padded_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dy_desc_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dx_desc_),
                               "cudnnDestroyTensorDescriptor failed");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dx_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dy_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&padded_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateFilterDescriptor(&w_desc_),
                                "cudnnCreateFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateConvolutionDescriptor(&conv_desc_),
                                "cudnnCreateConvolutionDescriptor failed");
  }
  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(dy_desc_, &dy_size_),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetFilterSizeInBytes(w_desc_, &w_size_),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(dx_desc_, &output_size_),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(dy_size_);
    input_size_list_.push_back(w_size_);
    output_size_list_.push_back(output_size_);

    if (use_pad_ && !is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(padded_descriptor_, &padded_size_),
                                  "cudnnGetTensorSizeInBytes failed");

      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_, w_desc_, dy_desc_, conv_desc_, padded_descriptor_,
                                                     algo_, &workspace_size_),
        "cudnnGetConvolutionBackwardDataWorkspaceSize failed");
      workspace_size_list_.push_back(padded_size_);
    } else {
      if (!is_null_input_) {
        CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                    cudnnGetConvolutionBackwardDataWorkspaceSize(
                                      cudnn_handle_, w_desc_, dy_desc_, conv_desc_, dx_desc_, algo_, &workspace_size_),
                                    "cudnnGetConvolutionBackwardDataWorkspaceSize failed");
      }
    }
    (void)workspace_size_list_.insert(workspace_size_list_.begin(), workspace_size_);
  }

 private:
  void CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 2, but got " << input_num;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
  }
  void SelectAlgorithm(cudnnTensorDescriptor_t dx_desc_real) {
    constexpr int requested_algo_count = 1;
    int returned_algo_count = 0;
    cudnnConvolutionBwdDataAlgoPerf_t perf_results;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnn_handle_, w_desc_, dy_desc_, conv_desc_, dx_desc_real,
                                                  requested_algo_count, &returned_algo_count, &perf_results),
      "cudnnGetConvolutionBackwardDataAlgorithm_v7 failed");
    algo_ = perf_results.algo;
#if CUDNN_VERSION < 8000
    if (group_ > 1) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle_, w_desc_, dy_desc_, conv_desc_, dx_desc_real,
                                                 CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, 0, &algo_),
        "cudnnGetConvolutionBackwardDataAlgorithm failed");
    }
#endif
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    }
  }
  void GetInputShape(const CNodePtr &kernel_node, std::vector<size_t> *input_shape) {
    auto prim = AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    auto shp_tuple_x = prim->GetAttr("input_sizes")->cast<ValueTuplePtr>()->value();
    (void)std::transform(std::begin(shp_tuple_x), std::end(shp_tuple_x), std::back_inserter(*input_shape),
                         [](const ValuePtr &e) -> size_t { return static_cast<int>(e->cast<Int64ImmPtr>()->value()); });
  }
  void Set4DDesc(const std::vector<size_t> &dy_shape, const std::vector<size_t> &input_shape,
                 const std::vector<size_t> &filter_shape) {
    const int kNbDims = 4;
    int dimA[kNbDims];
    int strideAin[kNbDims];
    int dimAdy[kNbDims];
    int strideAdy[kNbDims];
    int filterDimA[kNbDims];
    SetDimA(input_shape, dimA, kNbDims, data_format_);
    SetStrideA(input_shape, strideAin, kNbDims, data_format_);
    SetDimA(dy_shape, dimAdy, kNbDims, data_format_);
    SetStrideA(dy_shape, strideAdy, kNbDims, data_format_);
    SetDimA(filter_shape, filterDimA, kNbDims, data_format_);

    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(dy_desc_, cudnn_data_type_, kNbDims, dimAdy, strideAdy),
                                "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetFilterNdDescriptor(w_desc_, cudnn_data_type_, compute_format_, kNbDims, filterDimA),
      "cudnnSetFilterNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(dx_desc_, cudnn_data_type_, kNbDims, dimA, strideAin),
                                "cudnnSetTensorNdDescriptor failed");
  }
  void SetStrideAndDilation(const CNodePtr &kernel_node) {
    std::vector<int64_t> stride_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "stride");
    std::vector<int64_t> dilation_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "dilation");
    std::string format_me = AnfAlgo::GetNodeAttr<std::string>(kernel_node, "format");
    auto iter = kFormatIndexMap.find(format_me);
    if (iter == kFormatIndexMap.end()) {
      MS_LOG(EXCEPTION) << "OriFormat is " << format_me << ", Please confirm that in {NCHW, HWCN, NHWC}.";
    }
    size_t h_index = iter->second;
    const size_t h_index_offset = 2;
    if (stride_me.size() < h_index + h_index_offset) {
      MS_LOG(EXCEPTION) << "Strides should be greater than " << h_index + 1 << ", but got " << stride_me.size();
    }
    (void)std::transform(stride_me.begin() + h_index, stride_me.begin() + h_index + h_index_offset,
                         std::back_inserter(stride_), [](const int64_t &value) { return static_cast<int>(value); });
    (void)std::transform(dilation_me.begin(), dilation_me.end(), std::back_inserter(dilation_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (stride_.size() != k2DStrideSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'stride' should be 2, but got "
                        << stride_.size();
    }
    if (dilation_.size() != k2DDilationSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'dilation' should be 4, but got "
                        << dilation_.size();
    }
    if (dilation_[0] != 1 || dilation_[1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'dilation' at 0 and 1 axis should be 1, but got "
                        << "dilation[0]: " << dilation_[0] << ", dilation[1]: " << dilation_[1];
    }
  }
  cudnnHandle_t cudnn_handle_;
  cudnnFilterDescriptor_t w_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorDescriptor_t dy_desc_;
  cudnnTensorDescriptor_t dx_desc_;
  cudnnTensorDescriptor_t padded_descriptor_;
  cudnnConvolutionBwdDataAlgo_t algo_;
  std::string pad_mode_;
  std::string data_format_ = kOpFormat_NCHW;

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
  size_t dy_size_;
  size_t w_size_;
  size_t output_size_;
  size_t padded_size_;
  size_t workspace_size_;
  bool use_pad_;
  float beta_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV2D_GRAD_INPUT_GPU_KERNEL_H_
