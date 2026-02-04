/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONVOLUTION_DEPTH_WISE_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONVOLUTION_DEPTH_WISE_KERNEL_H_

#include <cuda.h>
#include <unordered_map>
#include <string>
#include <vector>
#include "kernel/gpu/cudnn/convolution/abstract_conv_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/convolution/convolution_ops_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/transpose_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class ConvolutionDepthWiseGpuKernel : public AbstractConvolutionGpuKernel {
 public:
  explicit ConvolutionDepthWiseGpuKernel(enum ConvType conv_type) : AbstractConvolutionGpuKernel(conv_type) {}
  ~ConvolutionDepthWiseGpuKernel() {}

  void InitResource() override{};
  void DestroyResource() noexcept override{};
  void ResetResource(ConvolutionArgs *conv_args, std::vector<size_t> *input_size_list,
                     std::vector<size_t> *output_size_list,
                     std::vector<size_t> *workspace_size_list) noexcept override {
    conv_args->in_height = 0;
    conv_args->in_width = 0;
    conv_args->pad_height = 0;
    conv_args->pad_width = 0;
    conv_args->pad_top = 0;
    conv_args->pad_left = 0;
    conv_args->use_pad = false;
    conv_args->batch_size = 0;
    conv_args->in_channel = 0;
    workspace_size_list->clear();
  }

 private:
  bool LaunchForward(const ConvolutionArgs &conv_args, const void *input0_addr, const void *input1_addr,
                     void *output_addr, const std::vector<KernelTensor *> &workspace, void *stream_ptr) override;
  bool LaunchInputGrad(const ConvolutionArgs &conv_args, const void *input0_addr, const void *input1_addr,
                       void *output_addr, const std::vector<KernelTensor *> &workspace, void *stream_ptr) override;
  bool LaunchFilterGrad(const ConvolutionArgs &conv_args, const void *input0_addr, const void *input1_addr,
                        void *output_addr, const std::vector<KernelTensor *> &workspace, void *stream_ptr) override;

  void CallTransposeToNHWC(const size_t output_size, const void *input_addr, TransposeInfo info, void *output_addr,
                           void *stream_ptr);

  void CallTransposeToNCHW(const size_t output_size, const void *input_addr, TransposeInfo info, void *output_addr,
                           void *stream_ptr);

  template <enum ConvolutionOpType op>
  bool CallDepthWiseKernel(const ConvolutionArgs &conv_args, const void *input0_addr, const void *input1_addr,
                           void *output_addr, void *stream_ptr) {
    PrintConvolutionArgs(conv_args);
    ConvolutionCudaArgs cuda_args;
    TransformConvolutionCudaArgs(conv_args, &cuda_args);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto status = cudaSuccess;
    status = ConvolutionOpCudaFunc<op, T>(cuda_args, static_cast<T *>(const_cast<void *>(input0_addr)),
                                          static_cast<T *>(const_cast<void *>(input1_addr)),
                                          static_cast<T *>(output_addr), cuda_stream);
    CHECK_CUDA_STATUS(status, "CallDepthWiseKernel");
    return true;
  }

  int InitialForward(ConvolutionArgs *conv_args, const std::vector<int64_t> &input_shape,
                     const std::vector<int64_t> &filter_shape, const std::vector<int64_t> &output_shape,
                     std::vector<size_t> *input_size_list, std::vector<size_t> *output_size_list,
                     std::vector<size_t> *workspace_size_list) override {
    SetNCHW(input_shape, &conv_args->batch_size, &conv_args->in_channel, &conv_args->in_height, &conv_args->in_width,
            conv_args->data_format);
    SetConvolutionCudaArgs(conv_args, input_shape, output_shape, filter_shape, workspace_size_list);
    conv_args->output_size = SizeOf(output_shape);
    return KRET_OK;
  }

  int InitialInputGrad(ConvolutionArgs *conv_args, const std::vector<int64_t> &dy_shape,
                       const std::vector<int64_t> &input_shape, const std::vector<int64_t> &filter_shape,
                       std::vector<size_t> *input_size_list, std::vector<size_t> *output_size_list,
                       std::vector<size_t> *workspace_size_list) override {
    SetNCHW(input_shape, &conv_args->batch_size, &conv_args->in_channel, &conv_args->in_height, &conv_args->in_width,
            conv_args->data_format);
    SetConvolutionCudaArgs(conv_args, dy_shape, input_shape, filter_shape, workspace_size_list);
    conv_args->output_size = SizeOf(input_shape);
    return KRET_OK;
  }

  int InitialFilterGrad(ConvolutionArgs *conv_args, const std::vector<int64_t> &dy_shape,
                        const std::vector<int64_t> &input_shape, const std::vector<int64_t> &filter_shape,
                        std::vector<size_t> *input_size_list, std::vector<size_t> *output_size_list,
                        std::vector<size_t> *workspace_size_list) override {
    SetNCHW(input_shape, &conv_args->batch_size, &conv_args->in_channel, &conv_args->in_height, &conv_args->in_width,
            conv_args->data_format);
    SetConvolutionCudaArgs(conv_args, dy_shape, input_shape, filter_shape, workspace_size_list);
    conv_args->output_size = SizeOf(filter_shape);
    if (conv_args->data_format == kOpFormat_NHWC) {
      auto type_id_size = conv_args->type_id_size;

      conv_args->tensor0_shape = {dy_shape.begin(), dy_shape.end()};
      conv_args->tensor1_shape = {input_shape.begin(), input_shape.end()};
      conv_args->tensor2_shape = {filter_shape.begin(), filter_shape.end()};

      workspace_size_list->push_back(SizeOf(dy_shape) * type_id_size);
      workspace_size_list->push_back(SizeOf(input_shape) * type_id_size);
      workspace_size_list->push_back(SizeOf(filter_shape) * type_id_size);
    }
    return KRET_OK;
  }

  void TransformConvolutionCudaArgs(const ConvolutionArgs &conv_args, ConvolutionCudaArgs *cuda_args) {
    cuda_args->output_size = conv_args.output_size;
    cuda_args->batch_size = conv_args.batch_size;
    cuda_args->in_height = conv_args.in_height;
    cuda_args->in_width = conv_args.in_width;
    cuda_args->in_channel = conv_args.in_channel;
    cuda_args->out_channel = conv_args.out_channel;
    cuda_args->filter_height = conv_args.filter_height;
    cuda_args->filter_width = conv_args.filter_width;
    cuda_args->pad_height = conv_args.pad_height;
    cuda_args->pad_width = conv_args.pad_width;
    cuda_args->pad_top = conv_args.pad_top;
    cuda_args->pad_left = conv_args.pad_left;
    cuda_args->out_height = conv_args.out_height;
    cuda_args->out_width = conv_args.out_width;
    cuda_args->group = conv_args.group;
    cuda_args->stride_height = conv_args.stride[kIndex2];
    cuda_args->stride_width = conv_args.stride[kIndex3];
    cuda_args->dilation_height = conv_args.dilation[kIndex2];
    cuda_args->dilation_width = conv_args.dilation[kIndex3];
  }

  void SetConvolutionCudaArgs(ConvolutionArgs *conv_args, const std::vector<int64_t> &tensor0_shape,
                              const std::vector<int64_t> &tensor1_shape, const std::vector<int64_t> &filter_shape,
                              std::vector<size_t> *workspace_size_list) {
    auto pad_list = conv_args->pad_list;

    conv_args->pad_top = pad_list[kTop2DPadIndex];
    conv_args->pad_height = pad_list[kBottom2DPadIndex];
    conv_args->pad_left = pad_list[kLeft2DPadIndex];
    conv_args->pad_width = pad_list[kRight2DPadIndex];

    if (conv_args->data_format == kOpFormat_NCHW) {
      conv_args->filter_height = filter_shape[kIndex2];
      conv_args->filter_width = filter_shape[kIndex3];
    } else if (conv_args->data_format == kOpFormat_NHWC) {
      conv_args->filter_height = filter_shape[kIndex1];
      conv_args->filter_width = filter_shape[kIndex2];
    } else {
      MS_LOG(EXCEPTION) << "Data format: " << conv_args->data_format << " is invalid.";
    }

    conv_args->out_height =
      (conv_args->in_height + conv_args->pad_top + conv_args->pad_height - conv_args->filter_height) /
        conv_args->stride[kIndex2] +
      1;
    conv_args->out_width =
      (conv_args->in_width + conv_args->pad_left + conv_args->pad_width - conv_args->filter_width) /
        conv_args->stride[kIndex3] +
      1;
  }
};

template <typename T>
void ConvolutionDepthWiseGpuKernel<T>::CallTransposeToNHWC(const size_t output_size, const void *input_addr,
                                                           TransposeInfo info, void *output_addr, void *stream_ptr) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  CalTranspose<T, true>(output_size, static_cast<T *>(const_cast<void *>(input_addr)), info,
                        static_cast<T *>(output_addr), cuda_stream);
}

template <typename T>
void ConvolutionDepthWiseGpuKernel<T>::CallTransposeToNCHW(const size_t output_size, const void *input_addr,
                                                           TransposeInfo info, void *output_addr, void *stream_ptr) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  CalTranspose<T, true>(output_size, static_cast<T *>(const_cast<void *>(input_addr)), info,
                        static_cast<T *>(output_addr), cuda_stream);
}

template <typename T>
bool ConvolutionDepthWiseGpuKernel<T>::LaunchForward(const ConvolutionArgs &conv_args, const void *input_addr,
                                                     const void *filter_addr, void *output_addr,
                                                     const std::vector<KernelTensor *> &workspace, void *stream_ptr) {
  auto data_format = conv_args.data_format;
  if (data_format == kOpFormat_NCHW) {
    return CallDepthWiseKernel<ConvolutionOpType::kConv2dDepthWiseForwardNCHW>(conv_args, input_addr, filter_addr,
                                                                               output_addr, stream_ptr);
  } else {
    return CallDepthWiseKernel<ConvolutionOpType::kConv2dDepthWiseForwardNHWC>(conv_args, input_addr, filter_addr,
                                                                               output_addr, stream_ptr);
  }
}

template <typename T>
bool ConvolutionDepthWiseGpuKernel<T>::LaunchInputGrad(const ConvolutionArgs &conv_args, const void *dy_addr,
                                                       const void *filter_addr, void *dx_addr,
                                                       const std::vector<KernelTensor *> &workspace, void *stream_ptr) {
  auto data_format = conv_args.data_format;
  if (data_format == kOpFormat_NCHW) {
    return CallDepthWiseKernel<ConvolutionOpType::kConv2dDepthWiseInputGradNCHW>(conv_args, dy_addr, filter_addr,
                                                                                 dx_addr, stream_ptr);
  } else {
    return CallDepthWiseKernel<ConvolutionOpType::kConv2dDepthWiseInputGradNHWC>(conv_args, dy_addr, filter_addr,
                                                                                 dx_addr, stream_ptr);
  }
}

template <typename T>
bool ConvolutionDepthWiseGpuKernel<T>::LaunchFilterGrad(const ConvolutionArgs &conv_args, const void *dy_addr,
                                                        const void *x_addr, void *dw_addr,
                                                        const std::vector<KernelTensor *> &workspace,
                                                        void *stream_ptr) {
  void *workspace0_addr = nullptr;
  void *workspace1_addr = nullptr;
  void *workspace2_addr = nullptr;
  if (workspace.size() >= kIndex3) {
    workspace0_addr = GetPossiblyNullDeviceAddress<void>(workspace, kIndex0);
    workspace1_addr = GetPossiblyNullDeviceAddress<void>(workspace, kIndex1);
    workspace2_addr = GetPossiblyNullDeviceAddress<void>(workspace, kIndex2);
  }
  if (conv_args.data_format == kOpFormat_NCHW) {
    return CallDepthWiseKernel<ConvolutionOpType::kConv2dDepthWiseFilterGradNCHW>(conv_args, dy_addr, x_addr, dw_addr,
                                                                                  stream_ptr);
  } else {
    std::vector<int64_t> dy_shape = conv_args.tensor0_shape;
    std::vector<int64_t> tx_shape = conv_args.tensor1_shape;
    std::vector<int64_t> t_dw_shape = {conv_args.tensor2_shape[kIndex0], conv_args.tensor2_shape[kIndex3],
                                       conv_args.tensor2_shape[kIndex1], conv_args.tensor2_shape[kIndex2]};
    TransposeInfo dy_info, x_info, dw_info;
    dy_info.input_shape = dy_shape;
    dy_info.perm = to_nchw_axis;

    x_info.input_shape = tx_shape;
    x_info.perm = to_nchw_axis;

    dw_info.input_shape = t_dw_shape;
    dw_info.perm = to_nhwc_axis;

    CallTransposeToNCHW(SizeOf(dy_shape), dy_addr, dy_info, workspace0_addr, stream_ptr);
    CallTransposeToNCHW(SizeOf(tx_shape), x_addr, x_info, workspace1_addr, stream_ptr);
    CallDepthWiseKernel<ConvolutionOpType::kConv2dDepthWiseFilterGradNCHW>(conv_args, workspace0_addr, workspace1_addr,
                                                                           workspace2_addr, stream_ptr);
    CallTransposeToNHWC(SizeOf(t_dw_shape), workspace2_addr, dw_info, dw_addr, stream_ptr);

    return true;
  }
}
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONVOLUTION_DEPTH_WISE_KERNEL_H_
