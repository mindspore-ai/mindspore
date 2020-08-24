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

#include "src/runtime/kernel/opencl/kernel/depthwise_conv2d.h"
#include <string>
#include <set>
#include <utility>
#include "src/kernel_registry.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "nnacl/fp32/common_func.h"
#include "nnacl/op_base.h"
#include "include/errorcode.h"

#ifndef PROGRAM_WITH_IL

#include "src/runtime/kernel/opencl/cl/depthwise_conv2d.cl.inc"

#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DepthwiseConv2D;

namespace mindspore::kernel {

int DepthwiseConv2dOpenCLKernel::Init() {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  std::string kernel_name = "DepthwiseConv2d";
  auto in_format = in_tensors_[0]->GetFormat();
  in_ori_format_ = in_format;
  out_ori_format_ = out_tensors_[0]->GetFormat();
  out_tensors_[0]->SetFormat(in_format);
  if (in_format != schema::Format_NHWC4 && in_format != schema::Format_NC4HW4) {
    MS_LOG(ERROR) << "input format(" << in_format << ") "
                  << "format not support!";
  }
  if (out_mem_type_ == OpenCLMemType::BUF) {
    kernel_name += "_BUF";
  } else {
    kernel_name += "_IMG";
  }
  if (in_format == schema::Format_NC4HW4) {
    kernel_name += "_NC4HW4";
  } else if (in_format == schema::Format_NHWC4) {
    kernel_name += "_NHWC4";
  }
  auto parameter = reinterpret_cast<ConvParameter *>(op_parameter_);
  if (parameter->kernel_h_ == 1) {
    kernel_name += "_1x1";
  }
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime->GetKernelFromBinary(kernel_name);
#else
  std::string program_name = "DepthwiseConv2d";
  std::set<std::string> build_options;
  std::string source = depthwise_conv2d_source;
  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  this->InitBuffer();
  MS_LOG(DEBUG) << kernel_name << " Init Done! mem type=" << static_cast<int>(out_mem_type_);
  return RET_OK;
}

int DepthwiseConv2dOpenCLKernel::InitBuffer() {
  auto parameter = reinterpret_cast<ConvParameter *>(op_parameter_);
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto allocator = ocl_runtime->GetAllocator();
  bool is_fp16 = ocl_runtime->GetFp16Enable();

  // weight: o, h, w, i; o == group, i == 1
  void *origin_weight = in_tensors_.at(kWeightIndex)->Data();
  int CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  int pack_weight_size = C4NUM * CO4 * parameter->kernel_h_ * parameter->kernel_w_;

  int plane = parameter->kernel_h_ * parameter->kernel_w_;
  if (is_fp16) {
    packed_weight_ = allocator->Malloc(pack_weight_size * sizeof(int16_t));
    packed_weight_ = allocator->MapBuffer(packed_weight_, CL_MAP_WRITE, nullptr, true);
    if (in_tensors_.at(kWeightIndex)->data_type() == kNumberTypeFloat16) {
      std::function<int16_t(int16_t)> to_dtype = [](int16_t x) -> int16_t { return x; };
      PackNCHWToNC4HW4<int16_t, int16_t>(origin_weight, packed_weight_, 1, plane, out_tensors_[0]->Channel(), to_dtype);
    } else if (in_tensors_.at(kWeightIndex)->data_type() == kNumberTypeFloat32) {
      std::function<int16_t(float)> to_dtype = Float32ToShort;
      PackNCHWToNC4HW4<float, int16_t>(origin_weight, packed_weight_, 1, plane, out_tensors_[0]->Channel(), to_dtype);
    } else {
      MS_LOG(ERROR) << "Only support float16/float32, actual data type " << in_tensors_.at(kWeightIndex)->data_type();
    }
  } else {
    packed_weight_ = allocator->Malloc(pack_weight_size * sizeof(float));
    packed_weight_ = allocator->MapBuffer(packed_weight_, CL_MAP_WRITE, nullptr, true);
    if (in_tensors_.at(kWeightIndex)->data_type() == kNumberTypeFloat32) {
      std::function<float(float)> to_dtype = [](float x) -> float { return (float)x; };
      PackNCHWToNC4HW4<float, float>(origin_weight, packed_weight_, 1, plane, out_tensors_[0]->Channel(), to_dtype);
    } else {
      MS_LOG(ERROR) << "Only support float16/float32, actual data type " << in_tensors_.at(kWeightIndex)->data_type();
    }
  }

  allocator->UnmapBuffer(packed_weight_);

  if (in_tensors_.size() == kInputSize2) {
    size_t dtype_size = sizeof(float);
    if (is_fp16 && in_tensors_.at(kBiasIndex)->data_type() == kNumberTypeFloat16) {
      dtype_size = sizeof(int16_t);
    }
    bias_data_ = allocator->Malloc(C4NUM * CO4 * dtype_size);
    bias_data_ = allocator->MapBuffer(bias_data_, CL_MAP_WRITE, nullptr, true);
    size_t up_co_size = C4NUM * CO4 * dtype_size;
    memset(bias_data_, 0, up_co_size);
    auto ori_bias = in_tensors_.at(kBiasIndex)->Data();
    memcpy(bias_data_, ori_bias, out_tensors_[0]->Channel() * dtype_size);
    allocator->UnmapBuffer(bias_data_);
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  return RET_OK;
}

int DepthwiseConv2dOpenCLKernel::ReSize() { return RET_OK; }

int DepthwiseConv2dOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  size_t im_dst_x, im_dst_y;
  if (in_tensors_[0]->GetFormat() == schema::Format_NHWC4) {
    im_dst_x = out_tensors_[0]->Width() * CO4;
    im_dst_y = out_tensors_[0]->Height();
  } else {
    im_dst_y = out_tensors_[0]->Height() * CO4;
    im_dst_x = out_tensors_[0]->Width();
  }
  size_t img_dtype = CL_FLOAT;
  if (lite::opencl::OpenCLRuntime::GetInstance()->GetFp16Enable()) {
    img_dtype = CL_HALF_FLOAT;
  }
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return RET_OK;
}

int DepthwiseConv2dOpenCLKernel::GetGlobalSize(size_t idx, std::vector<size_t> *global_size) {
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  std::vector<size_t> global = {(size_t)out_tensors_[0]->Width(), (size_t)out_tensors_[0]->Height(), CO4};
  *global_size = std::move(global);
  return RET_OK;
}

int DepthwiseConv2dOpenCLKernel::GetLocalSize(size_t idx, const std::vector<size_t> &global_size,
                                              std::vector<size_t> *local_size) {
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  std::vector<size_t> local = {1, 1, CO4};
  *local_size = std::move(local);
  return RET_OK;
}

int DepthwiseConv2dOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto parameter = reinterpret_cast<ConvParameter *>(op_parameter_);
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  size_t CI4 = UP_DIV(in_tensors_[0]->Channel(), C4NUM);
  std::vector<size_t> global = {(size_t)out_tensors_[0]->Width(), (size_t)out_tensors_[0]->Height(), CO4};
  std::vector<size_t> local;
  GetLocalSize(0, global, &local);

  float relu_clip1 = 6.0;
  cl_int2 kernel_size = {parameter->kernel_h_, parameter->kernel_w_};
  cl_int2 stride = {parameter->stride_h_, parameter->stride_w_};
  cl_int2 padding = {-parameter->pad_u_, -parameter->pad_l_};
  cl_int2 dilation = {parameter->dilation_h_, parameter->dilation_w_};
  cl_int4 src_size = {in_tensors_[0]->Width(), in_tensors_[0]->Height(), (cl_int)CI4, in_tensors_[0]->Batch()};
  cl_int4 dst_size = {(cl_int)out_tensors_[0]->Width(), (cl_int)out_tensors_[0]->Height(), (cl_int)CO4,
                      (cl_int)out_tensors_[0]->Batch()};

  ocl_runtime->SetKernelArg(kernel_, 1, packed_weight_);
  ocl_runtime->SetKernelArg(kernel_, 2, bias_data_);
  ocl_runtime->SetKernelArg(kernel_, 3, relu_clip1);
  ocl_runtime->SetKernelArg(kernel_, 5, kernel_size);
  ocl_runtime->SetKernelArg(kernel_, 6, stride);
  ocl_runtime->SetKernelArg(kernel_, 7, padding);
  ocl_runtime->SetKernelArg(kernel_, 8, dilation);
  ocl_runtime->SetKernelArg(kernel_, 9, src_size);
  ocl_runtime->SetKernelArg(kernel_, 10, dst_size);
  ocl_runtime->SetKernelArg(kernel_, 0, in_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, 4, out_tensors_[0]->Data());
  ocl_runtime->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLDepthwiseConv2dKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                       const std::vector<lite::tensor::Tensor *> &outputs,
                                                       OpParameter *opParameter, const lite::Context *ctx,
                                                       const kernel::KernelKey &desc,
                                                       const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel =
    new (std::nothrow) DepthwiseConv2dOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << "is nullptr.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init DepthwiseConv2dOpenCLKernel failed!";
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_DepthwiseConv2D, OpenCLDepthwiseConv2dKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_DepthwiseConv2D, OpenCLDepthwiseConv2dKernelCreator)
}  // namespace mindspore::kernel
