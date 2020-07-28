/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/opencl/kernel/convolution.h"
#include <vector>
#include <string>
#include <set>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/opencl/opencl_runtime.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/fp32/convolution.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Conv2D;

namespace mindspore::kernel {

int ConvolutionOpenCLKernel::Init() {
  MS_LOG(INFO) << "ConvolutionOpenCLKernel::Init()";

  if (inputs_[0]->Batch() != 1 || outputs_[0]->Batch() != 1) {
    MS_LOG(ERROR) << "ConvolutionOpenCLKernel only support Batch=1!";
  }

  auto io_NHWC = inputs_[0]->GetFormat() == schema::Format_NHWC && outputs_[0]->GetFormat() == schema::Format_NHWC;
  auto io_NHWC4 = inputs_[0]->GetFormat() == schema::Format_NHWC4 && outputs_[0]->GetFormat() == schema::Format_NHWC4;
  if (!io_NHWC && !io_NHWC4) {
    MS_LOG(ERROR) << "input and output data_format is invalid!";
  }
  io_dataformat_ = inputs_[0]->GetFormat();

  if (inputs_[1]->GetFormat() != schema::Format_KHWC) {
    MS_LOG(ERROR) << "weight data_format is invalid!";
  }

  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  std::string kernel_name = "convolution_NHWC_OHWI";
#ifdef PROGRAM_WITH_IL
  ocl_runtime->CreateKernelFromIL(kernel_(), kernel_name);
#else
  std::set<std::string> build_options;
  std::string source = convolution_source_fp32;
  std::string program_name = "convolution";
  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif

  this->InitBuffer();
  return 0;
}
int ConvolutionOpenCLKernel::InitBuffer() {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto allocator = ocl_runtime->GetAllocator();

  auto weight_tensor = inputs_[1];
  auto bias_tensor = inputs_[2];
  if (io_dataformat_ == schema::Format_NHWC) {
    packed_weight_ = reinterpret_cast<float *>(allocator->Malloc(weight_tensor->Size()));
    packed_weight_ = reinterpret_cast<float *>(allocator->MapBuffer(packed_weight_, CL_MAP_WRITE, nullptr, true));
    memcpy_s(packed_weight_, weight_tensor->Size(), weight_tensor->Data(), weight_tensor->Size());
    allocator->UnmapBuffer(packed_weight_);

    packed_bias_ = reinterpret_cast<float *>(allocator->Malloc(bias_tensor->Size()));
    packed_bias_ = reinterpret_cast<float *>(allocator->MapBuffer(packed_bias_, CL_MAP_WRITE, nullptr, true));
    memcpy_s(packed_bias_, bias_tensor->Size(), bias_tensor->Data(), bias_tensor->Size());
    allocator->UnmapBuffer(packed_bias_);
  } else if (io_dataformat_ == schema::Format_NHWC4) {
    auto weight_shape = weight_tensor->shape();
    size_t CO = weight_shape[0];
    size_t KH = weight_shape[1];
    size_t KW = weight_shape[2];
    size_t CI = weight_shape[3];
    size_t CI_ALIGN = UP_DIV(CI, C4NUM) * C4NUM;
    size_t CO_ALIGN = UP_DIV(CO, C4NUM) * C4NUM;
    size_t weight_size_tiled = CO_ALIGN * KH * KW * CI_ALIGN * sizeof(float);

    packed_weight_ = reinterpret_cast<float *>(allocator->Malloc(weight_size_tiled));
    packed_weight_ = reinterpret_cast<float *>(allocator->MapBuffer(packed_weight_, CL_MAP_WRITE, nullptr, true));
    memset_s(packed_weight_, weight_size_tiled, 0x00, weight_size_tiled);
    auto weight_data = reinterpret_cast<float *>(weight_tensor->Data());
    for (int co = 0; co < CO; ++co) {
      for (int kh = 0; kh < KH; ++kh) {
        for (int kw = 0; kw < KW; ++kw) {
          for (int ci = 0; ci < CI; ++ci) {
            packed_weight_[co * KH * KW * CI_ALIGN + kh * KW * CI_ALIGN + kw * CI_ALIGN + ci] =
              weight_data[co * KH * KW * CI + kh * KW * CI + kw * CI + ci];
          }
        }
      }
    }
    allocator->UnmapBuffer(packed_weight_);

    size_t bias_size_tiled = CO_ALIGN * sizeof(float);
    packed_bias_ = reinterpret_cast<float *>(allocator->Malloc(bias_size_tiled));
    packed_bias_ = reinterpret_cast<float *>(allocator->MapBuffer(packed_bias_, CL_MAP_WRITE, nullptr, true));
    memset_s(packed_bias_, bias_size_tiled, 0x00, bias_size_tiled);
    auto bias_data = reinterpret_cast<float *>(bias_tensor->Data());
    for (int co = 0; co < CO; ++co) {
      packed_bias_[co] = bias_data[co];
    }
    allocator->UnmapBuffer(packed_bias_);
  }

  return 0;
}

int ConvolutionOpenCLKernel::ReSize() { return 0; }

int ConvolutionOpenCLKernel::Run() {
  MS_LOG(INFO) << "ConvolutionOpenCLKernel::Run()";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();

  auto param = reinterpret_cast<ConvParameter *>(opParameter);
  auto input0_shape = inputs_[0]->shape();   // NHWC
  auto input1_shape = inputs_[1]->shape();   // OHWI
  auto outpu0_shape = outputs_[0]->shape();  // NHWC
  cl_uint N = input0_shape[0];
  cl_uint CI = input0_shape[3];
  cl_uint IH = input0_shape[1];
  cl_uint IW = input0_shape[2];
  cl_uint CO = outpu0_shape[3];
  cl_uint OH = outpu0_shape[1];
  cl_uint OW = outpu0_shape[2];
  cl_uint KH = input1_shape[1];
  cl_uint KW = input1_shape[2];
  cl_uint CI_TILE_NUM = UP_DIV(CI, C4NUM);
  cl_uint CO_TILE_NUM = UP_DIV(CO, C4NUM);
  cl_uint CI_ALIGN = CI_TILE_NUM * C4NUM;
  cl_uint CO_ALIGN = CO_TILE_NUM * C4NUM;

  cl_uint4 input_shape;
  cl_uint4 weight_shape;
  cl_uint4 output_shape;
  if (io_dataformat_ == schema::Format_NHWC) {
    input_shape = {N, IH, IW, CI};
    weight_shape = {CO, KH, KW, CI};
    output_shape = {N, OH, OW, CO};
  } else if (io_dataformat_ == schema::Format_NHWC4) {
    input_shape = {N, IH, IW, CI_ALIGN};
    weight_shape = {CO_ALIGN, KH, KW, CI_ALIGN};
    output_shape = {N, OH, OW, CO_ALIGN};
  }
  cl_uint2 stride = {static_cast<cl_uint>(param->stride_h_), static_cast<cl_uint>(param->stride_w_)};
  cl_uint4 pad = {static_cast<cl_uint>(param->pad_u_), static_cast<cl_uint>(param->pad_d_),
                  static_cast<cl_uint>(param->pad_l_), static_cast<cl_uint>(param->pad_r_)};

  int arg_cn = 0;
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, inputs_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, packed_weight_);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, packed_bias_);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, outputs_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, input_shape);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, weight_shape);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, output_shape);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, stride);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, pad);

  std::vector<size_t> global = {OW, OH, CO_TILE_NUM};
  std::vector<size_t> local = {1, 1, CO_TILE_NUM};

  ocl_runtime->RunKernel(kernel_, global, local, nullptr);

  return 0;
}

kernel::LiteKernel *OpenCLConvolutionKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                   const std::vector<lite::tensor::Tensor *> &outputs,
                                                   OpParameter *opParameter, const lite::Context *ctx,
                                                   const kernel::KernelKey &desc) {
  auto *kernel = new ConvolutionOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Create OpenCL Convolution kernel failed!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (0 != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: Convolution";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, PrimitiveType_Conv2D, OpenCLConvolutionKernelCreator)
}  // namespace mindspore::kernel

