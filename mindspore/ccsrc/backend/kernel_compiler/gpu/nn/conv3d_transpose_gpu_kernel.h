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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3D_TRANSPOSE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3D_TRANSPOSE_GPU_KERNEL_H_

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "backend/kernel_compiler/gpu/cuda_impl/pad_impl.cuh"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
constexpr size_t kConv3dDimSize = 3;
constexpr int kSymmetricCoef = 2;

constexpr size_t k3DPadSize = 6;
constexpr size_t kHead3DPadIdx = 0;
constexpr size_t kTail3DPadIdx = 1;
constexpr size_t kTop3DPadIdx = 2;
constexpr size_t kBottom3DPadIdx = 3;
constexpr size_t kLeft3DPadIdx = 4;
constexpr size_t kRight3DPadIdx = 5;

constexpr size_t kPadDepthIdx = 0;
constexpr size_t kPadHeightIdx = 1;
constexpr size_t kPadWidthIdx = 2;

constexpr size_t k3DStrideSize = 5;
constexpr size_t kDepth3DStrideIdx = 2;
constexpr size_t kHeight3DStrideIdx = 3;
constexpr size_t kWidth3DStrideIdx = 4;

constexpr size_t k3DDilationSize = 5;
constexpr size_t kDepth3DDilationIdx = 2;
constexpr size_t kHeight3DDilationIdx = 3;
constexpr size_t kWidth3DDilationIdx = 4;
template <typename T>
class Conv3dTransposeFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  Conv3dTransposeFwdGpuKernelMod() { ResetResource(); }
  ~Conv3dTransposeFwdGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *filter_addr = GetDeviceAddress<T>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    T *work_space = GetPossiblyNullDeviceAddress<T>(workspace, 0);

    const float alpha = 1;
    if (use_pad_) {
      T *input_padded = GetDeviceAddress<T>(workspace, 1);
      const size_t kWsOutPadIdx = 2;
      T *output_padded = GetDeviceAddress<T>(workspace, kWsOutPadIdx);
      CalPad3d(input_padded_size_ / sizeof(T), input_addr, input_n_, input_c_, input_old_depth_, input_old_height_,
               input_old_width_, input_old_depth_ + pad_depth_, input_old_height_ + pad_height_,
               input_old_width_ + pad_width_, input_pad_head_, input_pad_top_, input_pad_left_, pad_value_,
               input_padded, reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnConvolutionBackwardData(cudnn_handle_, &alpha, filter_desc_, filter_addr, input_padded_descriptor_,
                                     input_padded, conv_desc_, algo_, work_space, workspace_size_, &beta_,
                                     padded_descriptor_, output_padded),
        "ConvolutionBackwardData failed");
      if (data_format_ == kOpFormat_NCDHW || data_format_ == kOpFormat_DEFAULT) {
        CalPadGrad3d(output_size_ / sizeof(T), output_padded, n_, c_, old_depth_, old_height_, old_width_,
                     old_depth_ + (1 + stride_[kDepth3DStrideIdx]) * pad_depth_,
                     old_height_ + (1 + stride_[kHeight3DStrideIdx]) * pad_height_,
                     old_width_ + (1 + stride_[kWidth3DStrideIdx]) * pad_width_, pad_head_, pad_top_, pad_left_,
                     output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
      } else {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'data_format' only support 'NCDHW' right now "
                          << ", but got " << data_format_;
      }
    } else {
      if (greater_stride_) {
        T *stride_padded = GetDeviceAddress<T>(workspace, 1);
        CHECK_CUDNN_RET_WITH_EXCEPT(
          kernel_node_,
          cudnnConvolutionBackwardData(cudnn_handle_, &alpha, filter_desc_, filter_addr, input_desc_, input_addr,
                                       conv_desc_, algo_, work_space, workspace_size_, &beta_,
                                       stride_padded_descriptor_, stride_padded),
          "ConvolutionBackwardData failed");
        CalPad3d(output_size_ / sizeof(T), stride_padded, input_n_, input_c_, stride_pad_depth_, stride_pad_height_,
                 stride_pad_width_, old_depth_, old_height_, old_width_, stride_pad_head_, stride_pad_top_,
                 stride_pad_left_, pad_value_, output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
      } else {
        CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                    cudnnConvolutionBackwardData(cudnn_handle_, &alpha, filter_desc_, filter_addr,
                                                                 input_desc_, input_addr, conv_desc_, algo_, work_space,
                                                                 workspace_size_, &beta_, output_desc_, output_addr),
                                    "ConvolutionBackwardData failed");
      }
    }
    return true;
  }

  bool CheckNull(const std::vector<size_t> filter_shape, const std::vector<size_t> input_shape) {
    is_null_input_ =
      CHECK_SHAPE_NULL(filter_shape, kernel_name_, "weight") || CHECK_SHAPE_NULL(input_shape, kernel_name_, "dout");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    return false;
  }

  void CheckSize(const size_t value, const size_t expect_value, const string arg_name) {
    if (value != expect_value) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of " << arg_name << " must be " << expect_value
                        << ", but got " << value;
    }
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();
    (void)CheckParam(kernel_node);
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    data_format_ = AnfAlgo::GetInputFormat(kernel_node, 0);
    auto format_attr = GetAttr<std::string>(kernel_node, "format");
    if (format_attr == kOpFormat_NDHWC) {
      data_format_ = kOpFormat_NDHWC;
    }
    auto filter_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    if (CheckNull(filter_shape, input_shape)) {
      return true;
    }
    std::vector<size_t> output_shape;
    GetInputShape(kernel_node, &output_shape);
    if (data_format_ == kOpFormat_NDHWC) {
      compute_format_ = CUDNN_TENSOR_NHWC;
      if (format_attr == kOpFormat_NCDHW) {
        ShapeNCDHW2NDHWC(&output_shape);
      }
    }
    SetNCDHW(output_shape, &n_, &c_, &old_depth_, &old_height_, &old_width_, data_format_);
    SetNCDHW(input_shape, &input_n_, &input_c_, &input_old_depth_, &input_old_height_, &input_old_width_, data_format_);
    Set5DDesc(input_shape, output_shape, filter_shape);
    group_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "groups"));
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionGroupCount(conv_desc_, group_),
                                "cudnnSetConvGroupCount failed");
    std::vector<int> pad_list;
    std::vector<int64_t> pad_list_me = GetAttr<std::vector<int64_t>>(kernel_node, "pad_list");
    (void)std::transform(pad_list_me.begin(), pad_list_me.end(), std::back_inserter(pad_list),
                         [](const int64_t &value) { return static_cast<int>(value); });
    std::vector<int> stride_pad_list(k3DPadSize, 0);
    SetStrideAndDilation(kernel_node);
    SetPad(kernel_node, input_shape, filter_shape, &pad_list, &stride_pad_list);
    auto [input_desc_real, output_desc_real] = GetInputAndOutputDescReal(pad_list, stride_pad_list);
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH),
                                  "cudnnSetConvolutionMathType failed.")
    }
    SelectAlgorithm(input_desc_real, output_desc_real);
    beta_ = GetAttrWithDefault(kernel_node, "inplace_algo", std::string("cover")) == "cover" ? 0 : 1;
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    cudnn_handle_ = nullptr;
    input_desc_ = nullptr;
    output_desc_ = nullptr;
    filter_desc_ = nullptr;
    conv_desc_ = nullptr;
    algo_selected_ = false;
    padded_descriptor_ = nullptr;
    stride_padded_descriptor_ = nullptr;
    algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    input_padded_descriptor_ = nullptr;
    cudnn_data_type_ = CUDNN_DATA_FLOAT;
    compute_format_ = CUDNN_TENSOR_NCHW;
    old_height_ = 0;
    old_width_ = 0;
    old_depth_ = 0;
    pad_depth_ = 0;
    pad_height_ = 0;
    pad_width_ = 0;
    pad_head_ = 0;
    pad_tail_ = 0;
    pad_top_ = 0;
    pad_left_ = 0;
    input_pad_head_ = 0;
    input_pad_top_ = 0;
    input_pad_left_ = 0;
    input_old_height_ = 0;
    input_old_width_ = 0;
    input_old_depth_ = 0;
    stride_pad_head_ = 0;
    stride_pad_top_ = 0;
    stride_pad_left_ = 0;
    stride_pad_depth_ = 0;
    stride_pad_height_ = 0;
    stride_pad_width_ = 0;
    n_ = 0;
    c_ = 0;
    input_n_ = 0;
    input_c_ = 0;
    beta_ = 0;
    stride_.clear();
    dilation_.clear();
    group_ = 1;
    is_null_input_ = false;
    kernel_name_ = "Conv3dTranspose";
    input_size_ = 0;
    filter_size_ = 0;
    output_size_ = 0;
    padded_size_ = 0;
    input_padded_size_ = 0;
    workspace_size_ = 0;
    use_pad_ = true;
    greater_stride_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyConvolutionDescriptor(conv_desc_),
                               "cudnnDestroyConvolutionDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyFilterDescriptor(filter_desc_),
                               "cudnnDestroyFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(padded_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(input_padded_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(stride_padded_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(input_desc_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(output_desc_),
                               "cudnnDestroyTensorDescriptor failed");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&output_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&input_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&padded_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&input_padded_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&stride_padded_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateFilterDescriptor(&filter_desc_),
                                "cudnnCreateFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateConvolutionDescriptor(&conv_desc_),
                                "cudnnCreateConvolutionDescriptor failed");
  }
  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(input_desc_, &input_size_),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetFilterSizeInBytes(filter_desc_, &filter_size_),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(output_desc_, &output_size_),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(filter_size_);
    output_size_list_.push_back(output_size_);

    if (use_pad_ && !is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(padded_descriptor_, &padded_size_),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetTensorSizeInBytes(input_padded_descriptor_, &input_padded_size_),
                                  "cudnnGetTensorSizeInBytes failed");

      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_, filter_desc_, input_padded_descriptor_, conv_desc_,
                                                     padded_descriptor_, algo_, &workspace_size_),
        "cudnnGetConvolutionBackwardDataWorkspaceSize failed");
      workspace_size_list_.push_back(input_padded_size_);  // 1
      workspace_size_list_.push_back(padded_size_);        // 2
    } else {
      if (!is_null_input_) {
        if (greater_stride_) {
          CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                      cudnnGetTensorSizeInBytes(stride_padded_descriptor_, &stride_padded_size_),
                                      "cudnnGetTensorSizeInBytes failed");
          CHECK_CUDNN_RET_WITH_EXCEPT(
            kernel_node_,
            cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_, filter_desc_, input_desc_, conv_desc_,
                                                         stride_padded_descriptor_, algo_, &workspace_size_),
            "cudnnGetConvolutionBackwardDataWorkspaceSize failed");
          workspace_size_list_.push_back(stride_padded_size_);  // 1
        } else {
          CHECK_CUDNN_RET_WITH_EXCEPT(
            kernel_node_,
            cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_, filter_desc_, input_desc_, conv_desc_,
                                                         output_desc_, algo_, &workspace_size_),
            "cudnnGetConvolutionBackwardDataWorkspaceSize failed");
        }
      }
    }
    (void)workspace_size_list_.insert(workspace_size_list_.begin(), workspace_size_);  // 0
  }

 private:
  void CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    const size_t kInputNum = 2;
    if (input_num != kInputNum) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 2, but got " << input_num;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
  }

  void SelectAlgorithm(cudnnTensorDescriptor_t input_desc_real, cudnnTensorDescriptor_t output_desc_real) {
    constexpr int requested_algo_count = 1;
    int returned_algo_count;
    cudnnConvolutionBwdDataAlgoPerf_t perf_results;
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnGetConvolutionBackwardDataAlgorithm_v7(
                                  cudnn_handle_, filter_desc_, input_desc_real, conv_desc_, output_desc_real,
                                  requested_algo_count, &returned_algo_count, &perf_results),
                                "cudnnGetConvolutionBackwardDataAlgorithm_v7 failed");
    algo_ = perf_results.algo;
    if (compute_format_ == CUDNN_TENSOR_NHWC && cudnn_data_type_ == CUDNN_DATA_HALF && CUDNN_MAJOR < 8) {
      MS_LOG(ERROR) << "Conv3dTransposeFwdGpuKernelMod does not support float16 data with NDHWC format.";
    }
  }

  void GetInputShape(const CNodePtr &kernel_node, std::vector<size_t> *input_shape) {
    auto shp_tuple_x = AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("input_size")->cast<ValueTuplePtr>()->value();
    (void)std::transform(std::begin(shp_tuple_x), std::end(shp_tuple_x), std::back_inserter(*input_shape),
                         [](const ValuePtr &e) -> size_t { return static_cast<int>(e->cast<Int64ImmPtr>()->value()); });
  }

  void Set5DDesc(const std::vector<size_t> &input_shape, const std::vector<size_t> &output_shape,
                 const std::vector<size_t> &filter_shape) {
    const int kNbDims = 5;
    int dim_a[kNbDims];
    int stride_a_in[kNbDims];
    int dim_a_dy[kNbDims];
    int stride_a_dy[kNbDims];
    int filter_dim_a[kNbDims];
    SetDimA(output_shape, dim_a, kNbDims, data_format_);
    SetStrideA(output_shape, stride_a_in, kNbDims, data_format_);
    SetDimA(input_shape, dim_a_dy, kNbDims, data_format_);
    SetStrideA(input_shape, stride_a_dy, kNbDims, data_format_);
    SetDimA(filter_shape, filter_dim_a, kNbDims, data_format_);

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetTensorNdDescriptor(input_desc_, cudnn_data_type_, kNbDims, dim_a_dy, stride_a_dy),
      "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetFilterNdDescriptor(filter_desc_, cudnn_data_type_, compute_format_, kNbDims, filter_dim_a),
      "cudnnSetFilterNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(output_desc_, cudnn_data_type_, kNbDims, dim_a, stride_a_in),
                                "cudnnSetTensorNdDescriptor failed");
  }

  void SetStrideAndDilation(const CNodePtr &kernel_node) {
    std::vector<int64_t> stride_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "strides");
    std::vector<int64_t> dilation_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "dilations");
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    (void)std::transform(dilation_me.begin(), dilation_me.end(), std::back_inserter(dilation_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (stride_.size() != k3DStrideSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'stride' should be 5, but got "
                        << stride_.size();
    }
    if (stride_[0] != 1 || stride_[1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'stride' at 0 and 1 axis should be 1, but got "
                        << "stride[0]: " << stride_[0] << ", stride[1]: " << stride_[1];
    }
    if (dilation_.size() != k3DDilationSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'dilation' should be 5, but got "
                        << dilation_.size();
    }
    if (dilation_[0] != 1 || dilation_[1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'dilation' at 0 and 1 axis should be 1, but got "
                        << "dilation[0]: " << dilation_[0] << ", dilation[1]: " << dilation_[1];
    }
  }
  void UpdatePaddingAndDilation(const std::vector<size_t> &input_shape, const std::vector<size_t> &filter_shape,
                                int *pad_list, int *stride_pad_list) {  // pad_mode_ = same
    const size_t kIdxOffset = 2;
    for (size_t i = 0; i < kConv3dDimSize; i++) {
      int pad_sum = SizeToInt(filter_shape[i + kIdxOffset]) * dilation_[i + kIdxOffset] - stride_[i + kIdxOffset] -
                    dilation_[i + kIdxOffset] + 1;
      if (pad_sum >= 0) {
        int pad_0 = pad_sum / kSymmetricCoef;
        int pad_1 = pad_sum - pad_0;
        pad_list[i * kSymmetricCoef] = pad_0;
        pad_list[i * kSymmetricCoef + 1] = pad_1;
        stride_pad_list[i * kSymmetricCoef] = 0;
        stride_pad_list[i * kSymmetricCoef + 1] = 0;
      } else {  // pad_sum < 0, stride greater, need pad zero at end.
        pad_list[i * kSymmetricCoef] = 0;
        pad_list[i * kSymmetricCoef + 1] = 0;
        int pad_0 = (-pad_sum) / kSymmetricCoef;
        int pad_1 = (-pad_sum) - pad_0;
        stride_pad_list[i * kSymmetricCoef] = pad_0;
        stride_pad_list[i * kSymmetricCoef + 1] = pad_1;
        greater_stride_ = true;
      }
    }
  }
  void UsePadProcess(const std::vector<int> &pad_list, const int *strideA, const int *dilaA) {
    std::vector<int> padding_diff(kConv3dDimSize);
    std::vector<int> padding_common(kConv3dDimSize, 0);
    for (int i = 0; i < SizeToInt(kConv3dDimSize); i++) {
      padding_diff[i] = std::abs(pad_list[kSymmetricCoef * i + 1] - pad_list[kSymmetricCoef * i]);
      padding_common[i] = std::min(pad_list[kSymmetricCoef * i], pad_list[kSymmetricCoef * i + 1]);
    }
    pad_depth_ = padding_diff[kPadDepthIdx];
    pad_height_ = padding_diff[kPadHeightIdx];
    pad_width_ = padding_diff[kPadWidthIdx];
    pad_head_ = (pad_list[kHead3DPadIdx] - padding_common[kHead3DPadIdx]) * (stride_[kDepth3DStrideIdx] + 1);
    pad_top_ = (pad_list[kTop3DPadIdx] - padding_common[kTail3DPadIdx]) * (stride_[kHeight3DStrideIdx] + 1);
    pad_left_ = (pad_list[kLeft3DPadIdx] - padding_common[kTop3DPadIdx]) * (stride_[kWidth3DStrideIdx] + 1);
    input_pad_head_ = pad_list[kHead3DPadIdx] - padding_common[kHead3DPadIdx];
    input_pad_top_ = pad_list[kTop3DPadIdx] - padding_common[kTail3DPadIdx];
    input_pad_left_ = pad_list[kLeft3DPadIdx] - padding_common[kTop3DPadIdx];
    const size_t kDataSize = 5;
    int dim_a[kDataSize];
    int strideApadded[kDataSize];
    int input_dim_a[kDataSize];
    int input_strideApadded[kDataSize];
    if (data_format_ == kOpFormat_NCDHW || data_format_ == kOpFormat_DEFAULT) {
      auto padded_shape = {IntToSize(n_), IntToSize(c_),
                           IntToSize(old_depth_ + (1 + stride_[kDepth3DStrideIdx]) * padding_diff[kHead3DPadIdx]),
                           IntToSize(old_height_ + (1 + stride_[kHeight3DStrideIdx]) * padding_diff[kTail3DPadIdx]),
                           IntToSize(old_width_ + (1 + stride_[kWidth3DStrideIdx]) * padding_diff[kTop3DPadIdx])};
      SetDimA(padded_shape, dim_a, kDataSize, data_format_);
      SetStrideA(padded_shape, strideApadded, kDataSize, data_format_);
      std::vector<size_t> input_padded_shape = {IntToSize(input_n_), IntToSize(input_c_),
                                                IntToSize(input_old_depth_ + padding_diff[0]),
                                                IntToSize(input_old_height_ + padding_diff[kTail3DPadIdx]),
                                                IntToSize(input_old_width_ + padding_diff[kTop3DPadIdx])};
      SetDimA(input_padded_shape, input_dim_a, kDataSize, data_format_);
      SetStrideA(input_padded_shape, input_strideApadded, kDataSize, data_format_);
    } else if (data_format_ == kOpFormat_NDHWC) {
      auto padded_shape = {IntToSize(n_), IntToSize(old_depth_ + pad_depth_), IntToSize(old_height_ + pad_height_),
                           IntToSize(old_width_ + pad_width_), IntToSize(c_)};
      SetDimA(padded_shape, dim_a, kDataSize, data_format_);
      SetStrideA(padded_shape, strideApadded, kDataSize, data_format_);
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetTensorNdDescriptor(padded_descriptor_, cudnn_data_type_, kDataSize, dim_a, strideApadded),
      "cudnnSetTensor5dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(input_padded_descriptor_, cudnn_data_type_, kDataSize,
                                                           input_dim_a, input_strideApadded),
                                "cudnnSetTensor5dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetConvolutionNdDescriptor(conv_desc_, kConv3dDimSize, padding_common.data(), strideA, dilaA,
                                      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
      "cudnnSetConvolutionNdDescriptor failed");
  }

  void SetPad(const CNodePtr &kernel_node, const std::vector<size_t> &input_shape,
              const std::vector<size_t> &filter_shape, std::vector<int> *pad_list, std::vector<int> *stride_pad_list) {
    pad_mode_ = GetAttr<std::string>(kernel_node, "pad_mode");
    const size_t kFilterSize = 5;
    (void)CheckSize(filter_shape.size(), kFilterSize, "weight shape");
    (void)CheckSize(pad_list->size(), k3DPadSize, "pad");
    if (pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) {  // pad_mode_ = same
      UpdatePaddingAndDilation(input_shape, filter_shape, pad_list->data(), stride_pad_list->data());
    }
    pad_depth_ = (*pad_list)[kHead3DPadIdx];
    pad_height_ = (*pad_list)[kTop3DPadIdx];
    pad_width_ = (*pad_list)[kLeft3DPadIdx];
    use_pad_ = !((pad_depth_ == (*pad_list)[kTail3DPadIdx]) && (pad_height_ == (*pad_list)[kBottom3DPadIdx]) &&
                 (pad_width_ == (*pad_list)[kRight3DPadIdx]));
  }

  std::pair<cudnnTensorDescriptor_t, cudnnTensorDescriptor_t> GetInputAndOutputDescReal(
    const std::vector<int> &pad_list, const std::vector<int> &stride_pad_list) {
    cudnnTensorDescriptor_t output_desc_real = nullptr;
    cudnnTensorDescriptor_t input_desc_real = nullptr;
    int strideA[kConv3dDimSize] = {stride_[kDepth3DStrideIdx], stride_[kHeight3DStrideIdx], stride_[kWidth3DStrideIdx]};
    int dilaA[kConv3dDimSize] = {dilation_[kDepth3DDilationIdx], dilation_[kHeight3DDilationIdx],
                                 dilation_[kWidth3DDilationIdx]};
    if (use_pad_) {
      UsePadProcess(pad_list, strideA, dilaA);
      output_desc_real = padded_descriptor_;
      input_desc_real = input_padded_descriptor_;
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_depth_ = 0;
        pad_height_ = 0;
        pad_width_ = 0;
      }
      int padA[kConv3dDimSize];
      padA[kPadDepthIdx] = pad_depth_;
      padA[kPadHeightIdx] = pad_height_;
      padA[kPadWidthIdx] = pad_width_;
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetConvolutionNdDescriptor(conv_desc_, kConv3dDimSize, padA, strideA, dilaA,
                                                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                                  "cudnnSetConvolution3dDescriptor failed");
      if (greater_stride_) {
        stride_pad_head_ = stride_pad_list[kHead3DPadIdx];
        stride_pad_top_ = stride_pad_list[kTop3DPadIdx];
        stride_pad_left_ = stride_pad_list[kLeft3DPadIdx];
        stride_pad_depth_ = old_depth_ - stride_pad_list[kHead3DPadIdx] - stride_pad_list[kTail3DPadIdx];
        stride_pad_height_ = old_height_ - stride_pad_list[kTop3DPadIdx] - stride_pad_list[kBottom3DPadIdx];
        stride_pad_width_ = old_width_ - stride_pad_list[kLeft3DPadIdx] - stride_pad_list[kRight3DPadIdx];
        const size_t kDataLen = 5;
        int dim_a[kDataLen];
        int strideApadded[kDataLen];
        if (data_format_ == kOpFormat_NCDHW || data_format_ == kOpFormat_DEFAULT) {
          auto padded_shape = {IntToSize(n_), IntToSize(c_), IntToSize(stride_pad_depth_),
                               IntToSize(stride_pad_height_), IntToSize(stride_pad_width_)};
          SetDimA(padded_shape, dim_a, kDataLen, data_format_);
          SetStrideA(padded_shape, strideApadded, kDataLen, data_format_);
        }
        CHECK_CUDNN_RET_WITH_EXCEPT(
          kernel_node_,
          cudnnSetTensorNdDescriptor(stride_padded_descriptor_, cudnn_data_type_, kDataLen, dim_a, strideApadded),
          "cudnnSetTensor5dDescriptor failed");
      }
      output_desc_real = greater_stride_ ? stride_padded_descriptor_ : output_desc_;
      input_desc_real = input_desc_;

    return std::make_pair(input_desc_real, output_desc_real);
  }

  cudnnHandle_t cudnn_handle_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnTensorDescriptor_t padded_descriptor_;
  cudnnTensorDescriptor_t input_padded_descriptor_;
  cudnnTensorDescriptor_t stride_padded_descriptor_;
  cudnnConvolutionBwdDataAlgo_t algo_;
  bool algo_selected_;
  std::string pad_mode_;
  std::string data_format_ = kOpFormat_NCDHW;
  cudnnDataType_t cudnn_data_type_;
  cudnnTensorFormat_t compute_format_;
  int old_depth_;
  int old_height_;
  int old_width_;
  int pad_depth_;
  int pad_height_;
  int pad_width_;
  int pad_head_;
  int pad_tail_;
  int pad_top_;
  int pad_left_;
  int input_pad_head_;
  int input_pad_top_;
  int input_pad_left_;
  int input_old_height_;
  int input_old_width_;
  int input_old_depth_;
  int stride_pad_head_;
  int stride_pad_top_;
  int stride_pad_left_;
  int stride_pad_depth_;
  int stride_pad_height_;
  int stride_pad_width_;
  int n_;
  int c_;
  int input_n_;
  int input_c_;
  const float pad_value_ = 0.0;
  std::vector<int> stride_;
  std::vector<int> dilation_;
  int group_;
  bool is_null_input_;
  std::string kernel_name_;
  size_t input_size_;
  size_t filter_size_;
  size_t output_size_;
  size_t padded_size_;
  size_t input_padded_size_;
  size_t stride_padded_size_;
  size_t workspace_size_;
  bool use_pad_;
  bool greater_stride_;
  float beta_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3D_TRANSPOSE_GPU_KERNEL_H_
