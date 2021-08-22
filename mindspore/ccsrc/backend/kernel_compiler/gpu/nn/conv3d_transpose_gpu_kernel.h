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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3D_TRANSPOSE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3D_TRANSPOSE_GPU_KERNEL_H_

#include <algorithm>
#include <string>
#include <vector>

#include "backend/kernel_compiler/gpu/cuda_impl/pad_impl.cuh"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class Conv3dTransposeGpuFwdKernel : public GpuKernel {
 public:
  Conv3dTransposeGpuFwdKernel() { ResetResource(); }
  ~Conv3dTransposeGpuFwdKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *filter_addr = GetDeviceAddress<T>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    T *work_space = nullptr;
    if (workspace_size_ != 0) {
      work_space = GetDeviceAddress<T>(workspace, 0);
    }

    const float alpha = 1;
    if (use_pad_) {
      T *padded = GetDeviceAddress<T>(workspace, 1);
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnConvolutionBackwardData(cudnn_handle_, &alpha, filter_desc_, filter_addr,
                                                               input_desc_, input_addr, conv_desc_, algo_, work_space,
                                                               workspace_size_, &beta_, padded_descriptor_, padded),
                                  "ConvolutionBackwardData failed");
      if (data_format_ == kOpFormat_NCDHW) {
        CalPadGrad3d(output_size_ / sizeof(T), padded, n_, c_, old_depth_, old_height_, old_width_,
                     old_depth_ + pad_depth_, old_height_ + pad_height_, old_width_ + pad_width_, pad_head_, pad_top_,
                     pad_left_, output_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
      } else {
        MS_LOG(EXCEPTION) << "ConvTranspose3d only support NCDHW format right now.";
      }

    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnConvolutionBackwardData(cudnn_handle_, &alpha, filter_desc_, filter_addr, input_desc_, input_addr,
                                     conv_desc_, algo_, work_space, workspace_size_, &beta_, output_desc_, output_addr),
        "ConvolutionBackwardData failed");
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    InitResource();
    if (!CheckParam(kernel_node)) return false;
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    data_format_ = AnfAlgo::GetInputFormat(kernel_node, 0);
    auto format_attr = GetAttr<std::string>(kernel_node, "format");
    if (format_attr == kOpFormat_NDHWC) {
      data_format_ = kOpFormat_NDHWC;
    }
    auto filter_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "Conv3dTransposeGpuBkwKernel input is null.";
      InitSizeLists();
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
    Set5DDesc(input_shape, output_shape, filter_shape);
    group_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "groups"));
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionGroupCount(conv_desc_, group_),
                                "cudnnSetConvGroupCount failed");
    std::vector<int> pad_list;
    std::vector<int64_t> pad_list_me = GetAttr<std::vector<int64_t>>(kernel_node, "pad_list");
    (void)std::transform(pad_list_me.begin(), pad_list_me.end(), std::back_inserter(pad_list),
                         [](const int64_t &value) { return static_cast<int>(value); });
    pad_depth_ = pad_list[0];
    pad_height_ = pad_list[2];
    pad_width_ = pad_list[4];
    use_pad_ = !((pad_depth_ == pad_list[1]) && (pad_height_ == pad_list[3]) && (pad_width_ == pad_list[5]));
    pad_mode_ = GetAttr<std::string>(kernel_node, "pad_mode");
    SetStrideAndDilation(kernel_node);
    cudnnTensorDescriptor_t output_desc_real = nullptr;
    int padA[3];
    int strideA[3] = {stride_[2], stride_[3], stride_[4]};
    int dilaA[3] = {dilation_[2], dilation_[3], dilation_[4]};
    if (use_pad_) {
      pad_depth_ = pad_list[0] + pad_list[1];
      pad_height_ = pad_list[2] + pad_list[3];
      pad_width_ = pad_list[4] + pad_list[5];
      pad_head_ = pad_list[0];
      pad_top_ = pad_list[2];
      pad_left_ = pad_list[4];
      int dim_a[5];
      int strideApadded[5];
      if (data_format_ == kOpFormat_NCDHW) {
        auto padded_shape = {IntToSize(n_), IntToSize(c_), IntToSize(old_depth_ + pad_depth_),
                             IntToSize(old_height_ + pad_height_), IntToSize(old_width_ + pad_width_)};
        SetDimA(padded_shape, dim_a, 5, data_format_);
        SetStrideA(padded_shape, strideApadded, 5, data_format_);
      } else if (data_format_ == kOpFormat_NDHWC) {
        auto padded_shape = {IntToSize(n_), IntToSize(old_depth_ + pad_depth_), IntToSize(old_height_ + pad_height_),
                             IntToSize(old_width_ + pad_width_), IntToSize(c_)};
        SetDimA(padded_shape, dim_a, 5, data_format_);
        SetStrideA(padded_shape, strideApadded, 5, data_format_);
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnSetTensorNdDescriptor(padded_descriptor_, cudnn_data_type_, 5, dim_a, strideApadded),
        "cudnnSetTensor5dDescriptor failed");
      padA[0] = 0;
      padA[1] = 0;
      padA[2] = 0;
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSetConvolutionNdDescriptor(conv_desc_, 3, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
        "cudnnSetConvolutionNdDescriptor failed");
      output_desc_real = padded_descriptor_;
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_depth_ = 0;
        pad_height_ = 0;
        pad_width_ = 0;
      }
      padA[0] = pad_depth_;
      padA[1] = pad_height_;
      padA[2] = pad_width_;
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSetConvolutionNdDescriptor(conv_desc_, 3, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
        "cudnnSetConvolution3dDescriptor failed");
      output_desc_real = output_desc_;
    }
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH),
                                  "cudnnSetConvolutionMathType failed.")
    }
    SelectAlgorithm(output_desc_real);
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
    n_ = 0;
    c_ = 0;
    beta_ = 0;
    stride_.clear();
    dilation_.clear();
    group_ = 1;
    is_null_input_ = false;
    input_size_ = 0;
    filter_size_ = 0;
    output_size_ = 0;
    padded_size_ = 0;
    workspace_size_ = 0;
    use_pad_ = true;
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

      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_, filter_desc_, input_desc_, conv_desc_,
                                                     padded_descriptor_, algo_, &workspace_size_),
        "cudnnGetConvolutionBackwardDataWorkspaceSize failed");
      workspace_size_list_.push_back(padded_size_);
    } else {
      if (!is_null_input_) {
        CHECK_CUDNN_RET_WITH_EXCEPT(
          kernel_node_,
          cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_, filter_desc_, input_desc_, conv_desc_,
                                                       output_desc_, algo_, &workspace_size_),
          "cudnnGetConvolutionBackwardDataWorkspaceSize failed");
      }
    }
    (void)workspace_size_list_.insert(workspace_size_list_.begin(), workspace_size_);
  }

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but Conv3dTranspose needs 2 inputs.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but Conv3dTranspose needs 1 output.";
      return false;
    }
    return true;
  }

  void SetPad(const std::vector<int> &output_shape, const CNodePtr &kernel_node) {
    std::vector<int> pad_list;
    std::vector<int64_t> pad_list_me = GetAttr<std::vector<int64_t>>(kernel_node, "pad_list");
    (void)std::transform(pad_list_me.begin(), pad_list_me.end(), std::back_inserter(pad_list),
                         [](const int64_t &value) { return static_cast<int>(value); });
  }

  void SelectAlgorithm(cudnnTensorDescriptor_t output_desc_real) {
    constexpr int requested_algo_count = 1;
    int returned_algo_count;
    cudnnConvolutionBwdDataAlgoPerf_t perf_results;
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnGetConvolutionBackwardDataAlgorithm_v7(
                                  cudnn_handle_, filter_desc_, input_desc_, conv_desc_, output_desc_real,
                                  requested_algo_count, &returned_algo_count, &perf_results),
                                "cudnnGetConvolutionBackwardDataAlgorithm_v7 failed");
    algo_ = perf_results.algo;
    if (compute_format_ == CUDNN_TENSOR_NHWC && cudnn_data_type_ == CUDNN_DATA_HALF && CUDNN_MAJOR < 8) {
      MS_LOG(ERROR) << "Conv3dTransposeGpuFwdKernel does not support float16 data with NDHWC format.";
    }
  }

  void GetInputShape(const CNodePtr &kernel_node, std::vector<size_t> *input_shape) {
    auto shp_tuple_x = AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("input_size")->cast<ValueTuplePtr>()->value();
    (void)std::transform(std::begin(shp_tuple_x), std::end(shp_tuple_x), std::back_inserter(*input_shape),
                         [](const ValuePtr &e) -> size_t { return static_cast<int>(e->cast<Int64ImmPtr>()->value()); });
  }

  void Set5DDesc(const std::vector<size_t> &input_shape, const std::vector<size_t> &output_shape,
                 const std::vector<size_t> &filter_shape) {
    const int nbDims = 5;
    int dim_a[5];
    int stride_a_in[5];
    int dim_a_dy[5];
    int stride_a_dy[5];
    int filter_dim_a[5];
    SetDimA(output_shape, dim_a, 5, data_format_);
    SetStrideA(output_shape, stride_a_in, 5, data_format_);
    SetDimA(input_shape, dim_a_dy, 5, data_format_);
    SetStrideA(input_shape, stride_a_dy, 5, data_format_);
    SetDimA(filter_shape, filter_dim_a, 5, data_format_);

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetTensorNdDescriptor(input_desc_, cudnn_data_type_, nbDims, dim_a_dy, stride_a_dy),
      "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetFilterNdDescriptor(filter_desc_, cudnn_data_type_, compute_format_, nbDims, filter_dim_a),
      "cudnnSetFilterNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(output_desc_, cudnn_data_type_, nbDims, dim_a, stride_a_in),
                                "cudnnSetTensorNdDescriptor failed");
  }

  void SetStrideAndDilation(const CNodePtr &kernel_node) {
    std::vector<int64_t> stride_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "strides");
    std::vector<int64_t> dilation_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "dilations");
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    (void)std::transform(dilation_me.begin(), dilation_me.end(), std::back_inserter(dilation_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (stride_.size() != 5) {
      MS_LOG(EXCEPTION) << "Conv3dTransposeGpuFwdKernel's stride must be 5d!";
    }
    if (stride_[0] != 1 || stride_[1] != 1) {
      MS_LOG(EXCEPTION) << "Conv3dTransposeGpuFwdKernel stride only support 1 in N axis and C axis!";
    }
    if (dilation_.size() != 5) {
      MS_LOG(EXCEPTION) << "Conv3dTransposeGpuFwdKernel's dilation must be 5d!";
    }
    if (dilation_[0] != 1 || dilation_[1] != 1) {
      MS_LOG(EXCEPTION) << "Conv3dTransposeGpuFwdKernel dilation only support 1 in N axis and C axis!";
    }
  }
  cudnnHandle_t cudnn_handle_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnTensorDescriptor_t padded_descriptor_;
  cudnnConvolutionBwdDataAlgo_t algo_;
  bool algo_selected_;
  std::string pad_mode_;
  std::string data_format_ = kOpFormat_NCDHW;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
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
  int n_;
  int c_;
  std::vector<int> stride_;
  std::vector<int> dilation_;
  int group_;
  bool is_null_input_;
  size_t input_size_;
  size_t filter_size_;
  size_t output_size_;
  size_t padded_size_;
  size_t workspace_size_;
  bool use_pad_;
  float beta_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3D_TRANSPOSE_GPU_KERNEL_H_
