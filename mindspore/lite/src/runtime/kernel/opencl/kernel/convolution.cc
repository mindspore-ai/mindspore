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

#include <string>
#include <set>
#include <algorithm>
#include "src/runtime/kernel/opencl/kernel/convolution.h"
#include "src/runtime/kernel/opencl/cl/fp32/convolution.cl.inc"
#include "src/kernel_registry.h"

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

  std::set<std::string> build_options;
  std::string source = convolution_source_fp32;
  std::string program_name = "convolution";
  std::string kernel_name = io_NHWC4 ? "convolution_NHWC4_OHWIIO_float8" : "convolution_NHWC_OHWI";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();

  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
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
    // OHWI -> OHWIIO
    auto weight_shape = weight_tensor->shape();
    size_t CO = weight_shape[0];
    size_t KH = weight_shape[1];
    size_t KW = weight_shape[2];
    size_t CI = weight_shape[3];
    size_t CI_SLICES = UP_DIV(CI, C4NUM);
    size_t CO_SLICES = UP_DIV(CO, C4NUM);
    constexpr size_t CI_TILE = C4NUM;
    constexpr size_t CO_TILE = C4NUM;
    size_t packed_weight_size = CO_SLICES * KH * KW * CI_SLICES * CI_TILE * CO_TILE * sizeof(float);

    packed_weight_ = reinterpret_cast<float *>(allocator->Malloc(packed_weight_size));
    packed_weight_ = reinterpret_cast<float *>(allocator->MapBuffer(packed_weight_, CL_MAP_WRITE, nullptr, true));
    memset_s(packed_weight_, packed_weight_size, 0x00, packed_weight_size);
    auto weight_data = reinterpret_cast<float *>(weight_tensor->Data());
    for (int co = 0; co < CO; ++co) {
      for (int kh = 0; kh < KH; ++kh) {
        for (int kw = 0; kw < KW; ++kw) {
          for (int ci = 0; ci < CI; ++ci) {
            auto co_outer = co / CO_TILE;
            auto co_inner = co % CO_TILE;
            auto ci_outer = ci / CI_TILE;
            auto ci_inner = ci % CI_TILE;
            packed_weight_[((((co_outer * KH + kh) * KW + kw) * CI_SLICES + ci_outer) * CI_TILE + ci_inner) * CO_TILE +
                           co_inner] = *(weight_data++);
          }
        }
      }
    }
    allocator->UnmapBuffer(packed_weight_);
    size_t packed_bias_size = CO_SLICES * CO_TILE * sizeof(float);
    packed_bias_ = reinterpret_cast<float *>(allocator->Malloc(packed_bias_size));
    packed_bias_ = reinterpret_cast<float *>(allocator->MapBuffer(packed_bias_, CL_MAP_WRITE, nullptr, true));
    memset_s(packed_bias_, packed_bias_size, 0x00, packed_bias_size);
    auto bias_data = reinterpret_cast<float *>(bias_tensor->Data());
    for (int co = 0; co < CO; ++co) {
      packed_bias_[co] = bias_data[co];
    }
    allocator->UnmapBuffer(packed_bias_);
  }

  return 0;
}  // namespace mindspore::kernel

int ConvolutionOpenCLKernel::ReSize() { return 0; }

static int GetBiggestDivider(int x, int y) {
  for (int i = y; i != 0; i--) {
    if (x % i == 0) {
      return i;
    }
  }
  return 1;
}

static void GetLocalSize(const ConvParameter *param, std::vector<size_t> *global, std::vector<size_t> *local) {
  constexpr size_t work_group_size[] = {4, 4, 1};
  constexpr size_t max_work_item_sizes[] = {512, 512, 512};
  constexpr size_t max_work_group_size = 512;
  const size_t max_z_size = std::min<size_t>(16, max_work_item_sizes[2]);

  // 先用OH OW CO_SLICES初始化global，并且441对齐
  size_t global_h = UP_DIV(param->output_h_, work_group_size[0]) * work_group_size[0];
  size_t global_w = UP_DIV(param->output_w_, work_group_size[1]) * work_group_size[1];
  size_t global_c = UP_DIV(UP_DIV(param->output_channel_, C4NUM), work_group_size[2]) * work_group_size[2];

  // 使用策略计算local
  size_t local_c = GetBiggestDivider(global_c, max_z_size);
  size_t local_hw_size = std::min<size_t>(256, max_work_group_size) / local_c;
  size_t local_w = std::min(global_w, local_hw_size);
  size_t local_h = std::min(local_hw_size / local_w, global_h);
  if (local_h == global_h && global_h % 2 == 0) {
    local_h = global_h / 2;
  }

  global->clear();
  global->push_back(UP_DIV(param->output_h_, local_h) * local_h);
  global->push_back(UP_DIV(param->output_w_, local_w) * local_w);
  global->push_back(UP_DIV(UP_DIV(param->output_channel_, C4NUM), local_c) * local_c);
  local->clear();
  local->push_back(local_h);
  local->push_back(local_w);
  local->push_back(local_c);
}

int ConvolutionOpenCLKernel::Run() {
  MS_LOG(INFO) << "ConvolutionOpenCLKernel::Run()";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();

  auto param = reinterpret_cast<ConvParameter *>(opParameter);
  auto input0_shape = inputs_[0]->shape();   // NHWC
  auto input1_shape = inputs_[1]->shape();   // OHWI
  auto outpu0_shape = outputs_[0]->shape();  // NHWC
  cl_int N = input0_shape[0];
  cl_int CI = input0_shape[3];
  cl_int IH = input0_shape[1];
  cl_int IW = input0_shape[2];
  cl_int CO = outpu0_shape[3];
  cl_int OH = outpu0_shape[1];
  cl_int OW = outpu0_shape[2];
  cl_int KH = input1_shape[1];
  cl_int KW = input1_shape[2];
  cl_int CI_ALIGN = UP_DIV(CI, C4NUM) * C4NUM;
  cl_int CO_ALIGN = UP_DIV(CO, C4NUM) * C4NUM;

  cl_int4 input_shape;
  cl_int4 output_shape;
  if (io_dataformat_ == schema::Format_NHWC) {
    input_shape = {N, IH, IW, CI};
    output_shape = {N, OH, OW, CO};
  } else if (io_dataformat_ == schema::Format_NHWC4) {
    input_shape = {N, IH, IW, CI_ALIGN};
    output_shape = {N, OH, OW, CO_ALIGN};
  }
  cl_int4 kernel_stride = {KH, KW, param->stride_h_, param->stride_w_};
  cl_int4 pad = {param->pad_u_, param->pad_d_, param->pad_l_, param->pad_r_};

  int arg_cn = 0;
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, inputs_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, packed_weight_);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, packed_bias_);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, outputs_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, input_shape);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, output_shape);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, kernel_stride);
  ocl_runtime->SetKernelArg(kernel_, arg_cn++, pad);

  std::vector<size_t> global;
  std::vector<size_t> local;
  GetLocalSize(reinterpret_cast<ConvParameter *>(this->opParameter), &global, &local);
  // float8 per thread
  if (io_dataformat_ == schema::Format_NHWC4) {
    local[2] = UP_DIV(local[2], 2);
    global[2] = UP_DIV(global[2], 2);
    global[2] = UP_DIV(global[2], global[2]) * global[2];
  }
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
