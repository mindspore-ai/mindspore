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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV2D_GRAD_INPUT_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV2D_GRAD_INPUT_GPU_KERNEL_H_

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <utility>

#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace kernel {
class ConvGradInputBkwGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<ConvGradInputBkwGpuKernelMod> {
 public:
  ConvGradInputBkwGpuKernelMod() { ResetResource(); }
  ~ConvGradInputBkwGpuKernelMod() override { DestroyResource(); }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyConvolutionDescriptor(conv_desc_),
                                       "cudnnDestroyConvolutionDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyFilterDescriptor(w_desc_), "cudnnDestroyFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(padded_descriptor_),
                                       "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(dy_desc_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(dx_desc_), "cudnnDestroyTensorDescriptor failed");
  }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnCreateTensorDescriptor(&dx_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnCreateTensorDescriptor(&dy_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnCreateTensorDescriptor(&padded_descriptor_),
                                       "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnCreateFilterDescriptor(&w_desc_), "cudnnCreateFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnCreateConvolutionDescriptor(&conv_desc_),
                                       "cudnnCreateConvolutionDescriptor failed");
  }

  void CalPadList(const ShapeVector input_shape, const ShapeVector filter_shape, int h_index, int w_index,
                  std::vector<int> *pad_list);
  void CheckParam(const std::vector<KernelTensorPtr> &inputs);
  void Set4DDesc(const ShapeVector &dy_shape, const ShapeVector &input_shape, const ShapeVector &filter_shape);
  void SetStrideAndDilation(const std::vector<int64_t> &stride_me, const std::vector<int64_t> &dilation_me,
                            const std::string &format_me);
  void SelectAlgorithm(cudnnTensorDescriptor_t x_desc_real);
  void ResetResource() noexcept;
  void InitSizeLists();
  void SetDilaA(int *dilaA);

  cudnnHandle_t cudnn_handle_{nullptr};
  cudnnFilterDescriptor_t w_desc_{nullptr};
  cudnnConvolutionDescriptor_t conv_desc_{nullptr};
  cudnnTensorDescriptor_t dy_desc_{nullptr};
  cudnnTensorDescriptor_t dx_desc_{nullptr};
  cudnnTensorDescriptor_t padded_descriptor_{nullptr};
  cudnnConvolutionBwdDataAlgo_t algo_;
  std::string pad_mode_;
  std::string data_format_{kOpFormat_NCHW};
  cudnnDataType_t cudnn_data_type_{CUDNN_DATA_FLOAT};
  cudnnTensorFormat_t compute_format_{CUDNN_TENSOR_NCHW};
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
  int group_{1};
  bool is_null_input_;
  std::string kernel_name_{"Conv2dGradInput"};
  size_t dy_size_;
  size_t w_size_;
  size_t output_size_;
  size_t padded_size_;
  size_t workspace_size_;
  bool use_pad_;
  float beta_;
  bool is_dynamic_attr_{false};
  bool get_dynamic_attr_value_{false};
  std::vector<int64_t> input_shape_;
  static constexpr size_t kShapeIndex_{2};
  void *stream_ptr_{nullptr};
  std::string format_attr_;
  std::vector<int> pad_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV2D_GRAD_INPUT_GPU_KERNEL_H_
