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

#include "src/runtime/kernel/opencl/kernel/conv2d_transpose.h"
#include <string>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/opencl/opencl_runtime.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/fp16/conv2d_transpose2x2.cl.inc"
#include "src/runtime/kernel/opencl/cl/fp32/conv2d_transpose2x2.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_DeConv2D;

namespace mindspore::kernel {

int Conv2dTransposeOpenCLKernel::Init() {
  ConvParameter *param = reinterpret_cast<ConvParameter *>(op_parameter_);
  if (param->kernel_h_ != 2 || param->kernel_w_ != 2 || param->stride_h_ != 2 || param->stride_w_ != 2) {
    MS_LOG(ERROR) << "only support kh=kw=2 and stride_h=stride_w=2.";
    return 1;
  }
  if (param->pad_h_ >= 2 || param->pad_w_ >= 2) {
    MS_LOG(ERROR) << "only support pad in {0,1}.";
    return 1;
  }
  std::string kernel_name = "conv2d_transpose2x2";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
#ifdef PROGRAM_WITH_IL
  ocl_runtime->CreateKernelFromIL(kernel_(), kernel_name);
#else
#ifdef ENABLE_FP16
  std::string source = conv2d_transpose2x2_source_fp16;
#else
  std::string source = conv2d_transpose2x2_source_fp32;
#endif
  std::set<std::string> build_options;
  std::string program_name = "conv2d_transpose2x2";
  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  int ci = param->input_channel_;
  int co = param->output_channel_;
  int kh = param->kernel_h_;
  int kw = param->kernel_w_;
  int div_ci = UP_DIV(ci, 4);
  int div_co = UP_DIV(co, 4);
  auto allocator = ocl_runtime->GetAllocator();
  padWeight_ = reinterpret_cast<FLOAT_T *>(allocator->Malloc(div_ci * div_co * 16 * kh * kw * sizeof(FLOAT_T)));
  padWeight_ = reinterpret_cast<FLOAT_T *>(allocator->MapBuffer(padWeight_, CL_MAP_WRITE, nullptr, true));
  PadWeight();
  allocator->UnmapBuffer(padWeight_);
  out_tensors_[0]->SetFormat(schema::Format_NHWC4);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return 0;
}

int Conv2dTransposeOpenCLKernel::ReSize() { return 0; }

void Conv2dTransposeOpenCLKernel::PadWeight() {
  // OHWI to OHWI4(I)4(O)
  ConvParameter *param = reinterpret_cast<ConvParameter *>(op_parameter_);
  int ci = param->input_channel_;
  int co = param->output_channel_;
  int kh = param->kernel_h_;
  int kw = param->kernel_w_;
  int div_ci = UP_DIV(ci, 4);
  int div_co = UP_DIV(co, 4);
  auto origin_weight = reinterpret_cast<FLOAT_T *>(in_tensors_.at(kWeightIndex)->Data());
  int index = 0;
  for (int co_i = 0; co_i < div_co; co_i++) {
    for (int kw_i = 0; kw_i < kw; kw_i++) {
      for (int kh_i = 0; kh_i < kh; kh_i++) {
        for (int ci_i = 0; ci_i < div_ci; ci_i++) {
          for (int ci4_i = 0; ci4_i < 4; ci4_i++) {
            for (int co4_i = 0; co4_i < 4; co4_i++) {
              int co_offset = co_i * 4 + co4_i;
              int ci_offset = ci_i * 4 + ci4_i;
              if (co_offset < co && ci_offset < ci) {
                int ori_index = ((co_offset * kh + kh_i) * kw + kw_i) * ci + ci_offset;
                padWeight_[index++] = origin_weight[ori_index];
              } else {
                padWeight_[index++] = 0.;
              }
            }
          }
        }
      }
    }
  }
}

int Conv2dTransposeOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  std::vector<int> shapex = in_tensors_[0]->shape();
  int n = shapex[0];
  if (n > 1) {
    MS_LOG(ERROR) << "Conv2dTranspose n > 1 not supported!";
    return 1;
  }
  ConvParameter *param = reinterpret_cast<ConvParameter *>(op_parameter_);
  int ci = param->input_channel_;
  int co = param->output_channel_;
  int kh = param->kernel_h_;
  int kw = param->kernel_w_;
  int pad = param->pad_h_;
  int oh = out_tensors_[0]->shape()[1];
  int ow = out_tensors_[0]->shape()[2];
  int h = in_tensors_[0]->shape()[1];
  int w = in_tensors_[0]->shape()[2];
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();

  cl::ImageFormat image_format;
  {
    image_format.image_channel_order = CL_RGBA;
#ifdef ENABLE_FP16
    image_format.image_channel_data_type = CL_HALF_FLOAT;
#else
    image_format.image_channel_data_type = CL_FLOAT;
#endif
  }
  cl_int in_error_code, in_error_code_weight, in_error_code_bias, out_error_code;
  cl::Image2D img_x(*ocl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image_format, w * ci / 4, h, 0,
                    in_tensors_[0]->Data(), &in_error_code);
  cl::Image2D img_bias(*ocl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image_format, co / 4, 1, 0,
                       in_tensors_[2]->Data(), &in_error_code_bias);
  cl::Image2D out_mem(*ocl_runtime->Context(), CL_MEM_WRITE_ONLY, image_format, ow * co / 4, oh, 0, nullptr,
                      &out_error_code);
  // local size should less than MAX_GROUP_SIZE
  std::vector<size_t> local = {16, 1, 16};
  std::vector<size_t> global = {UP_ROUND((size_t)UP_ROUND(oh / 2, 2), local[0]),
                                UP_ROUND((size_t)UP_ROUND(ow / 2, 2), local[1]), UP_ROUND((size_t)co / 4, local[2])};

  cl_int2 kernel_size = {kh, kw};
  cl_int2 stride = {2, 2};
  cl_int2 padding = {pad, pad};
  cl_int4 src_size = {h, w, UP_DIV(ci, 4), 1};
  cl_int4 dst_size = {oh, ow, UP_DIV(co, 4), 1};
  ocl_runtime->SetKernelArg(kernel_, 0, img_x);
  ocl_runtime->SetKernelArg(kernel_, 1, padWeight_);
  ocl_runtime->SetKernelArg(kernel_, 2, img_bias);
  ocl_runtime->SetKernelArg(kernel_, 3, out_mem);
  ocl_runtime->SetKernelArg(kernel_, 4, kernel_size);
  ocl_runtime->SetKernelArg(kernel_, 5, stride);
  ocl_runtime->SetKernelArg(kernel_, 6, padding);
  ocl_runtime->SetKernelArg(kernel_, 7, src_size);
  ocl_runtime->SetKernelArg(kernel_, 8, dst_size);
  ocl_runtime->RunKernel(kernel_, global, local, nullptr);
  auto origin = cl::array<cl::size_type, 3U>{0, 0, 0};
  auto region = cl::array<cl::size_type, 3U>{(size_t)(ow * co / 4), (size_t)(oh), 1};
  ocl_runtime->GetDefaultCommandQueue()->enqueueReadImage(out_mem, CL_TRUE, origin, region, 0, 0,
                                                          out_tensors_[0]->Data());
  return 0;
}

kernel::LiteKernel *OpenCLConv2dTransposeKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                       const std::vector<lite::tensor::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::Context *ctx,
                                                       const kernel::KernelKey &desc,
                                                       const lite::Primitive *primitive) {
  auto *kernel = new Conv2dTransposeOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << "is nullptr.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (0 != ret) {
    // MS_LOG(ERROR) << "Init kernel failed, name: " << opDef.name()->str()
    //               << ", type: " << lite::EnumNameOpT(opDef.attr_type());
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_DeConv2D, OpenCLConv2dTransposeKernelCreator)
}  // namespace mindspore::kernel
