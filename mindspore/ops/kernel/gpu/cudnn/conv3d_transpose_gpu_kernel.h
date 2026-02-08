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
#include <map>

#include "kernel/gpu/cuda_impl/cuda_ops/pad_impl.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "kernel/gpu/cudnn/convolution/conv_gpu_common.h"

namespace mindspore {
namespace kernel {
constexpr size_t kInputNum = 2;
constexpr size_t kOutputNum = 1;
constexpr size_t kOutputShapeSize = 2;
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

class Conv3dTransposeFwdGpuKernelMod : public NativeGpuKernelMod,
                                       public MatchKernelHelper<Conv3dTransposeFwdGpuKernelMod> {
 public:
  Conv3dTransposeFwdGpuKernelMod() { ResetResource(); }
  ~Conv3dTransposeFwdGpuKernelMod() override { DestroyResource(); }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  bool CheckNull(const ShapeVector filter_shape, const ShapeVector input_shape) {
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

  void ResetResource() noexcept {
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
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyConvolutionDescriptor(conv_desc_),
                                       "cudnnDestroyConvolutionDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyFilterDescriptor(filter_desc_),
                                       "cudnnDestroyFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(padded_descriptor_),
                                       "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(input_padded_descriptor_),
                                       "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(stride_padded_descriptor_),
                                       "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(input_desc_),
                                       "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(output_desc_),
                                       "cudnnDestroyTensorDescriptor failed");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&output_desc_),
                                        "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&input_desc_),
                                        "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&padded_descriptor_),
                                        "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&input_padded_descriptor_),
                                        "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&stride_padded_descriptor_),
                                        "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateFilterDescriptor(&filter_desc_),
                                        "cudnnCreateFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateConvolutionDescriptor(&conv_desc_),
                                        "cudnnCreateConvolutionDescriptor failed");
  }
  void InitSizeLists() {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(input_desc_, &input_size_),
                                          "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetFilterSizeInBytes(filter_desc_, &filter_size_),
                                          "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(output_desc_, &output_size_),
                                          "cudnnGetTensorSizeInBytes failed");
    }

    if (use_pad_ && !is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(padded_descriptor_, &padded_size_),
                                          "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(input_padded_descriptor_, &input_padded_size_),
                                          "cudnnGetTensorSizeInBytes failed");

      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_, filter_desc_, input_padded_descriptor_, conv_desc_,
                                                     padded_descriptor_, algo_, &workspace_size_),
        "cudnnGetConvolutionBackwardDataWorkspaceSize failed");
      workspace_size_list_.push_back(input_padded_size_);  // 1
      workspace_size_list_.push_back(padded_size_);        // 2
    } else {
      if (!is_null_input_) {
        if (greater_stride_) {
          CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
            cudnnGetTensorSizeInBytes(stride_padded_descriptor_, &stride_padded_size_),
            "cudnnGetTensorSizeInBytes failed");
          CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
            cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_, filter_desc_, input_desc_, conv_desc_,
                                                         stride_padded_descriptor_, algo_, &workspace_size_),
            "cudnnGetConvolutionBackwardDataWorkspaceSize failed");
          workspace_size_list_.push_back(stride_padded_size_);  // 1
        } else {
          CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
            cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_, filter_desc_, input_desc_, conv_desc_,
                                                         output_desc_, algo_, &workspace_size_),
            "cudnnGetConvolutionBackwardDataWorkspaceSize failed");
        }
      }
    }
    (void)workspace_size_list_.insert(workspace_size_list_.begin(), workspace_size_);  // 0
  }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);

  void Set5DDesc(const ShapeVector &input_shape, const ShapeVector &output_shape, const ShapeVector &filter_shape);
  void SetStrideAndDilation(std::vector<int64_t> stride_me, std::vector<int64_t> dilation_me);
  void UpdatePaddingAndDilation(const ShapeVector &input_shape, const ShapeVector &filter_shape, int *pad_list,
                                int *stride_pad_list);
  void UsePadProcess(const std::vector<int> &pad_list, const int *strideA, const int *dilaA);
  void SetPad(const ShapeVector &input_shape, const ShapeVector &filter_shape, std::vector<int> *pad_list,
              std::vector<int> *stride_pad_list);
  std::pair<cudnnTensorDescriptor_t, cudnnTensorDescriptor_t> GetInputAndOutputDescReal(
    const std::vector<int> &pad_list, const std::vector<int> &stride_pad_list);

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
  std::string format_attr_;
  void *cuda_stream_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3D_TRANSPOSE_GPU_KERNEL_H_
