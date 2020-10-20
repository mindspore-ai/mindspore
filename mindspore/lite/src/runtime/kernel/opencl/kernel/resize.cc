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

#include "src/runtime/kernel/opencl/kernel/resize.h"
#include <map>
#include <set>
#include <string>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/cl/resize.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_Resize;

namespace mindspore::kernel {

int ResizeOpenCLKernel::Init() {
  auto resize_param = reinterpret_cast<ResizeParameter *>(op_parameter_);
  if (resize_param == nullptr) {
    return RET_NULL_PTR;
  }
  alignCorner = resize_param->align_corners_;
  preserveAspectRatio = resize_param->preserve_aspect_ratio_;
  auto in_shape = in_tensors_[0]->shape();
  auto out_shape = out_tensors_[0]->shape();
  if (in_shape.size() != 4 || out_shape.size() != 4 || in_shape[0] != out_shape[0] || in_shape[3] != out_shape[3]) {
    MS_LOG(ERROR) << "resize op only support 4D and axes HW";
    return RET_PARAM_INVALID;
  }
  std::string kernel_name = "resize";
  if (resize_param->method_ == schema::ResizeMethod_LINEAR) {
    kernel_name += "_bilinear";
  } else if (resize_param->method_ == schema::ResizeMethod_NEAREST) {
    kernel_name += "_nearest_neighbor";
  } else {
    MS_LOG(ERROR) << "unsupported resize method:" << resize_param->method_;
    return RET_PARAM_INVALID;
  }
  kernel_name += "_" + std::string(EnumNameFormat(op_format_));
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::set<std::string> build_options;
  std::string source = resize_source;
  std::string program_name = "Resize";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif
  in_ori_format_ = in_tensors_[0]->GetFormat();
  out_ori_format_ = out_tensors_[0]->GetFormat();
  in_tensors_[0]->SetFormat(op_format_);
  out_tensors_[0]->SetFormat(op_format_);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int ResizeOpenCLKernel::ReSize() { return RET_OK; }

int ResizeOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t im_dst_x, im_dst_y;
  auto nhwc_shape_ = out_tensors_[0]->shape();
  if (op_format_ == schema::Format_NHWC4) {
    im_dst_x = nhwc_shape_[2] * UP_DIV(nhwc_shape_[3], C4NUM);
    im_dst_y = nhwc_shape_[0] * nhwc_shape_[1];
  } else if (op_format_ == schema::Format_NC4HW4) {
    im_dst_x = nhwc_shape_[2];
    im_dst_y = nhwc_shape_[0] * UP_DIV(nhwc_shape_[3], C4NUM) * nhwc_shape_[1];
  } else {
    MS_LOG(ERROR) << "not support op format:" << EnumNameFormat(op_format_);
    return RET_ERROR;
  }
  size_t img_dtype = CL_FLOAT;
  if (ocl_runtime_->GetFp16Enable()) {
    img_dtype = CL_HALF_FLOAT;
  }
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return RET_OK;
}

float ResizeOpenCLKernel::getResizeScaleFactor(int input_size, int output_size) {
  return input_size > 1 && output_size > 1 && alignCorner
           ? static_cast<float>(input_size - 1) / static_cast<float>(output_size - 1)
           : static_cast<float>(input_size) / static_cast<float>(output_size);
}

int ResizeOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto in_shape = in_tensors_[0]->shape();
  auto out_shape = out_tensors_[0]->shape();
  int n = out_shape[0];
  int h = out_shape[1];
  int w = out_shape[2];
  int c = out_shape[3];
  int c4 = UP_DIV(c, C4NUM);
  float scale_h = getResizeScaleFactor(in_tensors_[0]->shape()[1], out_tensors_[0]->shape()[1]);
  float scale_w = getResizeScaleFactor(in_tensors_[0]->shape()[2], out_tensors_[0]->shape()[2]);
  std::vector<size_t> local = {};
  std::vector<size_t> global = {static_cast<size_t>(c4), static_cast<size_t>(w), static_cast<size_t>(h)};
  cl_int4 in_size = {in_shape[0], in_shape[1], in_shape[2], UP_DIV(in_shape[3], C4NUM)};
  cl_int4 out_size = {n, h, w, c4};
  cl_float2 scale = {scale_h, scale_w};
  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, scale);
  ocl_runtime_->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLResizeKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                              const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                              const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) ResizeOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << " create failed.";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Resize, OpenCLResizeKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Resize, OpenCLResizeKernelCreator)
}  // namespace mindspore::kernel
