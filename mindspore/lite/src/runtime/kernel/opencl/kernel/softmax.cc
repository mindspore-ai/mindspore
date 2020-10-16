/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/opencl/kernel/softmax.h"
#include <string>
#include <set>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "nnacl/softmax_parameter.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/softmax.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_SoftMax;

namespace mindspore::kernel {

std::vector<float> SoftmaxOpenCLKernel::GetMaskForLastChannel(int channels) {
  std::vector<float> mask{0.0f, 0.0f, 0.0f, 0.0f};
  const int reminder = channels % 4 == 0 ? 4 : channels % 4;
  for (int i = 0; i < reminder; ++i) {
    mask[i] = 1.0f;
  }
  return mask;
}

int SoftmaxOpenCLKernel::InitGlobalSize() {
  size_t global_x, global_y, global_z;
  global_z = 1;
  if (axis_ == 1) {
    global_x = UP_DIV(nhwc_shape_[3], C4NUM);
    global_y = nhwc_shape_[2];
  } else if (axis_ == 2) {
    global_x = UP_DIV(nhwc_shape_[3], C4NUM);
    global_y = nhwc_shape_[1];
  } else if (axis_ == 3) {
    global_x = nhwc_shape_[2];
    global_y = nhwc_shape_[1];
  } else {
    global_x = 1;
    global_y = 1;
  }
  global_size_ = {global_x, global_y, global_z};
  return lite::RET_OK;
}

int SoftmaxOpenCLKernel::SetWorkGroupSize() {
  // set work group size
  InitGlobalSize();
  int max_work_group_size = ocl_runtime_->GetKernelMaxWorkGroupSize(kernel_(), (*ocl_runtime_->Device())());
  local_size_ = GetCommonLocalSize(global_size_, max_work_group_size);
  global_size_ = GetCommonGlobalSize(local_size_, global_size_);
  return lite::RET_OK;
}

int SoftmaxOpenCLKernel::SetWorkGroupSize1x1() {
  local_size_ = {32, 1, 1};
  global_size_ = {32, 1, 1};
  return lite::RET_OK;
}

int SoftmaxOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t im_dst_x, im_dst_y;
  auto out_shape = out_tensors_[0]->shape();
  int n = nhwc_shape_[0], h = nhwc_shape_[1], w = nhwc_shape_[2], c = nhwc_shape_[3];
  if (op_format_ == schema::Format_NHWC4) {
    im_dst_x = w * UP_DIV(c, C4NUM);
    im_dst_y = n * h;
  } else if (op_format_ == schema::Format_NC4HW4) {
    im_dst_x = w;
    im_dst_y = n * UP_DIV(c, C4NUM) * h;
  } else {
    MS_LOG(ERROR) << "not support op format:" << EnumNameFormat(op_format_);
    return RET_ERROR;
  }
  size_t img_dtype = CL_FLOAT;
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return RET_OK;
}

int SoftmaxOpenCLKernel::Init() {
  std::string kernel_name = "SoftMax";
  std::string program_name = "SoftMax";
  auto softmax_param = reinterpret_cast<SoftmaxParameter *>(op_parameter_);
  axis_ = softmax_param->axis_;
  auto in_shape = in_tensors_[0]->shape();
  if (in_shape.size() > 4) {
    MS_LOG(ERROR) << "Init `Softmax` kernel failed: Unsupported shape size: " << in_shape.size();
    return RET_ERROR;
  }
  if (axis_ < 0) {
    axis_ = in_shape.size() + axis_;
  }
  axis_ += 4 - in_shape.size();
  if (axis_ != 1 && axis_ != 2 && axis_ != 3) {
    MS_LOG(ERROR) << "Init `Softmax` kernel failed: softmax axis should be H W or C";
    return RET_ERROR;
  }
  nhwc_shape_ = GetNHWCShape(in_shape);
  std::string source = softmax_source;
  enable_fp16_ = ocl_runtime_->GetFp16Enable();
  // framework not set this param yet! just use default.
  if (nhwc_shape_[1] == 1 && nhwc_shape_[2] == 1 && axis_ == 3) {
    // support 4d tensor
    onexone_flag_ = true;
    kernel_name += "1x1";
    program_name += "1x1";
  } else {
    onexone_flag_ = false;
    kernel_name += "Axis" + std::to_string(axis_);
    program_name += "Axis" + std::to_string(axis_);
  }
  kernel_name += "_" + std::string(EnumNameFormat(op_format_));
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime->GetKernelFromBinary(kernel_name);
#else
  std::set<std::string> build_options;
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  in_ori_format_ = in_tensors_[0]->GetFormat();
  out_ori_format_ = out_tensors_[0]->GetFormat();
  in_tensors_[0]->SetFormat(op_format_);
  out_tensors_[0]->SetFormat(op_format_);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return lite::RET_OK;
}

int SoftmaxOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";

  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  int channel = nhwc_shape_[3];
  int c4 = UP_DIV(channel, C4NUM);
  auto mask_ = GetMaskForLastChannel(channel);
  cl_float4 mask = {mask_[0], mask_[1], mask_[2], mask_[3]};
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, mask);
  cl_int4 input_shape = {nhwc_shape_[0], nhwc_shape_[1], nhwc_shape_[2], c4};
  ocl_runtime_->SetKernelArg(kernel_, arg_idx, input_shape);
  if (onexone_flag_) {
    SetWorkGroupSize1x1();
  } else {
    SetWorkGroupSize();
  }

  // run opengl kernel
  ocl_runtime_->RunKernel(kernel_, global_size_, local_size_, nullptr);
  return lite::RET_OK;
}

kernel::LiteKernel *OpenCLSoftMaxKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                               const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) SoftmaxOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << "is nullptr.";
    free(opParameter);
    delete kernel;
    return nullptr;
  }
  if (inputs[0]->shape()[0] > 1) {
    MS_LOG(ERROR) << "Init `Softmax` kernel failed: Unsupported multi-batch.";
    delete kernel;
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init `Softmax` kernel failed!";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_SoftMax, OpenCLSoftMaxKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_SoftMax, OpenCLSoftMaxKernelCreator)
}  // namespace mindspore::kernel
