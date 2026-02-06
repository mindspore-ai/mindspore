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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONVOLUTION_GPU_KERNEL_FACTORY_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONVOLUTION_GPU_KERNEL_FACTORY_H_

#include <cuda.h>
#include <cudnn.h>
#include <unordered_map>
#include <string>
#include <memory>
#include "utils/ms_context.h"
#include "kernel/gpu/cudnn/convolution/conv2d_cudnn_gpu_kernel.h"
#include "kernel/gpu/cudnn/convolution/depth_wise_gpu_kernel.h"

namespace mindspore {
namespace kernel {
struct ConvolutionGpuKernelFactory {
 public:
  static std::shared_ptr<AbstractConvolutionGpuKernel> CreateConvCudnnGpuKernel(const std::string &data_type,
                                                                                enum ConvType conv_type) {
    if (data_type == "Float32") {
      return std::make_shared<ConvolutionCudnnGpuKernel<float>>(conv_type);
    } else if (data_type == "Float16") {
      return std::make_shared<ConvolutionCudnnGpuKernel<half>>(conv_type);
    } else {
      MS_LOG(EXCEPTION) << "Create cudnn kernel failed.";
    }
  }
  static std::shared_ptr<AbstractConvolutionGpuKernel> CreateConvDepthWiseGpuKernel(const std::string &data_type,
                                                                                    enum ConvType conv_type) {
    if (data_type == "Float32") {
      return std::make_shared<ConvolutionDepthWiseGpuKernel<float>>(conv_type);
    } else if (data_type == "Float16") {
      return std::make_shared<ConvolutionDepthWiseGpuKernel<half>>(conv_type);
    } else {
      MS_LOG(EXCEPTION) << "Create cudnn kernel failed.";
    }
  }

  static std::shared_ptr<AbstractConvolutionGpuKernel> CreateConvolutionGpuKernel(const ConvolutionArgs &conv_args,
                                                                                  enum ConvKernelType conv_kernel_type,
                                                                                  enum ConvType conv_type) {
    switch (conv_kernel_type) {
      case ConvKernelType::kCudnn:
        return CreateConvCudnnGpuKernel(conv_args.data_type, conv_type);
      case ConvKernelType::kDepthWise:
        return CreateConvDepthWiseGpuKernel(conv_args.data_type, conv_type);
      default:
        MS_LOG(EXCEPTION) << "Convolution kernel type: " << conv_type << " is invalid.";
    }
  }
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONVOLUTION_GPU_KERNEL_FACTORY_H_
