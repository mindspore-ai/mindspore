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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONVOLUTION_ABSTRACT_CONV_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONVOLUTION_ABSTRACT_CONV_GPU_KERNEL_H_

#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "include/utils/utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/convolution/convolution_ops_impl.cuh"
#include "kernel/gpu/cudnn/convolution/conv_gpu_common.h"

namespace mindspore {
namespace kernel {
class AbstractConvolutionGpuKernel {
 public:
  AbstractConvolutionGpuKernel() {}
  virtual ~AbstractConvolutionGpuKernel() {}
  explicit AbstractConvolutionGpuKernel(enum ConvType conv_type) : conv_type_(conv_type) {}

  ConvType get_conv_type() const { return conv_type_; }
  virtual void DestroyResource() noexcept = 0;
  virtual void ResetResource(ConvolutionArgs *conv_args, std::vector<size_t> *input_size_list,
                             std::vector<size_t> *output_size_list,
                             std::vector<size_t> *workspace_size_list) noexcept = 0;
  virtual void InitResource() = 0;

  int InitialKernel(ConvolutionArgs *conv_args, const std::vector<int64_t> &tensor0_shape,
                    const std::vector<int64_t> &tensor1_shape, const std::vector<int64_t> &tensor2_shape,
                    std::vector<size_t> *input_size_list, std::vector<size_t> *output_size_list,
                    std::vector<size_t> *workspace_size_list) {
    switch (conv_type_) {
      case ConvType::kForward:
        return InitialForward(conv_args, tensor0_shape, tensor1_shape, tensor2_shape, input_size_list, output_size_list,
                              workspace_size_list);
      case ConvType::kInputGrad:
        return InitialInputGrad(conv_args, tensor0_shape, tensor1_shape, tensor2_shape, input_size_list,
                                output_size_list, workspace_size_list);
      case ConvType::kFilterGrad:
        return InitialFilterGrad(conv_args, tensor0_shape, tensor1_shape, tensor2_shape, input_size_list,
                                 output_size_list, workspace_size_list);
      default:
        MS_LOG(EXCEPTION) << "Conv type:" << conv_type_ << " is invalid.";
    }
  }
  template <typename T>
  bool LaunchKernel(const ConvolutionArgs &conv_args, const T *input0_addr, const T *input1_addr, T *output_addr,
                    const std::vector<KernelTensor *> &workspace, void *stream_ptr) {
    switch (conv_type_) {
      case ConvType::kForward:
        return LaunchForward(conv_args, input0_addr, input1_addr, output_addr, workspace, stream_ptr);
      case ConvType::kInputGrad:
        return LaunchInputGrad(conv_args, input0_addr, input1_addr, output_addr, workspace, stream_ptr);
      case ConvType::kFilterGrad:
        return LaunchFilterGrad(conv_args, input0_addr, input1_addr, output_addr, workspace, stream_ptr);
      default:
        MS_LOG(EXCEPTION) << "Conv type:" << conv_type_ << " is invalid.";
    }
    return true;
  }

 private:
  virtual int InitialForward(ConvolutionArgs *conv_args, const std::vector<int64_t> &tensor0_shape,
                             const std::vector<int64_t> &tensor1_shape, const std::vector<int64_t> &tensor2_shape,
                             std::vector<size_t> *input_size_list, std::vector<size_t> *output_size_list,
                             std::vector<size_t> *workspace_size_list) = 0;

  virtual int InitialInputGrad(ConvolutionArgs *conv_args, const std::vector<int64_t> &tensor0_shape,
                               const std::vector<int64_t> &tensor1_shape, const std::vector<int64_t> &tensor2_shape,
                               std::vector<size_t> *input_size_list, std::vector<size_t> *output_size_list,
                               std::vector<size_t> *workspace_size_list) = 0;

  virtual int InitialFilterGrad(ConvolutionArgs *conv_args, const std::vector<int64_t> &tensor0_shape,
                                const std::vector<int64_t> &tensor1_shape, const std::vector<int64_t> &tensor2_shape,
                                std::vector<size_t> *input_size_list, std::vector<size_t> *output_size_list,
                                std::vector<size_t> *workspace_size_list) = 0;

  virtual bool LaunchForward(const ConvolutionArgs &conv_args, const void *input0_addr, const void *input1_addr,
                             void *output_addr, const std::vector<KernelTensor *> &workspace, void *stream_ptr) = 0;
  virtual bool LaunchInputGrad(const ConvolutionArgs &conv_args, const void *input0_addr, const void *input1_addr,
                               void *output_addr, const std::vector<KernelTensor *> &workspace, void *stream_ptr) = 0;
  virtual bool LaunchFilterGrad(const ConvolutionArgs &conv_args, const void *input0_addr, const void *input1_addr,
                                void *output_addr, const std::vector<KernelTensor *> &workspace, void *stream_ptr) = 0;
  ConvType conv_type_;
  std::string kernel_name_;
};

static void PrintConvolutionArgs(const ConvolutionArgs &conv_args) {
  MS_LOG(DEBUG) << "Convolution args: \n"
                << "kernel_name: " << conv_args.kernel_name << "\n"
                << "output_size: " << conv_args.output_size << "\n"
                << "batch_size: " << conv_args.batch_size << "\n"
                << "in_height: " << conv_args.in_height << "\n"
                << "in_width: " << conv_args.in_width << "\n"
                << "in_channel: " << conv_args.in_channel << "\n"
                << "out_channel: " << conv_args.out_channel << "\n"
                << "filter_height: " << conv_args.filter_height << "\n"
                << "filter_width: " << conv_args.filter_width << "\n"
                << "pad_height: " << conv_args.pad_height << "\n"
                << "pad_width: " << conv_args.pad_width << "\n"
                << "pad_top: " << conv_args.pad_top << "\n"
                << "pad_left: " << conv_args.pad_left << "\n"
                << "out_height: " << conv_args.out_height << "\n"
                << "out_width: " << conv_args.out_width << "\n"
                << "group: " << conv_args.group << "\n"
                << "stride: " << conv_args.stride << "\n"
                << "dilation: " << conv_args.dilation << "\n"
                << "pad_list: " << conv_args.pad_list << "\n"
                << "data_type: " << conv_args.data_type << "\n"
                << "data_format: " << conv_args.data_format << "\n"
                << "pad_mode: " << conv_args.pad_mode << "\n"
                << "use_pad: " << conv_args.use_pad << "\n"
                << "alpha: " << conv_args.alpha << "\n"
                << "beta: " << conv_args.beta;
}

static bool InitialAttributes(const PrimitivePtr &prim, ConvolutionArgs *conv_args,
                              const std::vector<KernelTensor *> &inputs) {
  MS_EXCEPTION_IF_NULL(prim);
  auto kernel_name_ = prim->name();
  auto out_channel = static_cast<int>(GetValue<int64_t>(prim->GetAttr("out_channel")));
  auto group = static_cast<int>(GetValue<int64_t>(prim->GetAttr("group")));
  auto pad_mode = GetValue<std::string>(prim->GetAttr("pad_mode"));

  auto data_type = inputs[0]->dtype_id();
  auto data_format_attr = GetValue<std::string>(prim->GetAttr("format"));
  auto data_format = mindspore::FormatEnumToString(inputs[0]->format());
  if (data_format == kOpFormat_DEFAULT) {
    data_format = kOpFormat_NCHW;
  }
  if (data_format_attr == kOpFormat_NHWC) {
    data_format = kOpFormat_NHWC;
  }
  auto stride_attr = GetValue<std::vector<int64_t>>(prim->GetAttr("stride"));
  if (stride_attr.size() != kConv2dInputDimSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'stride' must be 4, but got "
                      << stride_attr.size();
  }

  auto dilation_attr = GetValue<std::vector<int64_t>>(prim->GetAttr("dilation"));
  if (dilation_attr.size() != kConv2dInputDimSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'dilation' must be 4, but got "
                      << dilation_attr.size();
  }
  auto pad_list_attr = GetValue<std::vector<int64_t>>(prim->GetAttr("pad_list"));
  if (pad_list_attr.size() != kConv2dInputDimSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'pad' must be 4, but got "
                      << pad_list_attr.size();
  }

  conv_args->kernel_name = kernel_name_;
  conv_args->out_channel = out_channel;
  conv_args->group = group;
  conv_args->pad_mode = pad_mode;
  conv_args->data_type = TypeIdLabel(data_type);
  conv_args->type_id_size = abstract::TypeIdSize(data_type);
  conv_args->data_format_attr = data_format_attr;
  conv_args->data_format = data_format;

  std::transform(stride_attr.begin(), stride_attr.end(), std::back_inserter(conv_args->stride),
                 [](const int64_t &value) { return static_cast<int>(value); });
  std::transform(dilation_attr.begin(), dilation_attr.end(), std::back_inserter(conv_args->dilation),
                 [](const int64_t &value) { return static_cast<int>(value); });
  std::transform(pad_list_attr.begin(), pad_list_attr.end(), std::back_inserter(conv_args->pad_list),
                 [](const int64_t &value) { return static_cast<int>(value); });
  return true;
}

static bool MatchDepthWiseGpuKernel(const ConvolutionArgs &conv_args) {
  auto in_channel = conv_args.in_channel;
  auto out_channel = conv_args.out_channel;
  auto group = conv_args.group;
  if (in_channel != out_channel || in_channel != group) {
    return false;
  }
  const int marjor_sm = GET_MAJOR_SM;
  if (marjor_sm < AMPER_ARCH_SM || conv_args.kernel_name == "Conv2DBackpropFilter") {
    return false;
  }
  if (conv_args.pad_mode != kPadPadModeUpperCase && conv_args.pad_mode != kPadPadModeLowerCase) {
    return false;
  }
  auto &dilation = conv_args.dilation;
  auto &pad_list = conv_args.pad_list;
  auto &stride = conv_args.stride;
  if (dilation[kIndex0] != dilation[kIndex1] || dilation[kIndex0] != dilation[kIndex2] ||
      dilation[kIndex0] != dilation[kIndex3]) {
    return false;
  }
  if (pad_list[kIndex0] != pad_list[kIndex1] || pad_list[kIndex0] != pad_list[kIndex2] ||
      pad_list[kIndex0] != pad_list[kIndex3]) {
    return false;
  }
  if (stride[kIndex0] != 1 || stride[kIndex1] != 1) {
    return false;
  }
  if (conv_args.data_format == kOpFormat_NCHW && conv_args.data_type == "Float32") {
    return false;
  }
  return true;
}

static void SetConvolutionInChannel(ConvolutionArgs *conv_args, const std::vector<int64_t> &input_shape) {
  if (conv_args->data_format_attr == kOpFormat_NHWC) {
    conv_args->data_format = kOpFormat_NHWC;
  }
  if (conv_args->data_format == kOpFormat_NHWC) {
    conv_args->in_channel = input_shape[kIndex3];
  } else if (conv_args->data_format == kOpFormat_NCHW) {
    conv_args->in_channel = input_shape[kIndex1];
  } else {
    MS_LOG(EXCEPTION) << "Data format:" << conv_args->data_format << " is invalid.";
  }
}

static ConvKernelType SelectConvolutionGpuKernel(const ConvolutionArgs &conv_args) {
  if (MatchDepthWiseGpuKernel(conv_args)) {
    return ConvKernelType::kDepthWise;
  } else {
    return ConvKernelType::kCudnn;
  }
}
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONVOLUTION_ABSTRACT_CONV_GPU_KERNEL_H_
