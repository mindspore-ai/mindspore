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

#include <set>
#include <string>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/transpose.h"
#ifndef PROGRAM_WITH_IL
#include "src/runtime/kernel/opencl/cl/transpose.cl.inc"
#endif

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Transpose;

namespace mindspore::kernel {

int TransposeOpenCLKernel::Init() {
  std::string kernel_name = "transpose";
  enable_fp16_ = ocl_runtime_->GetFp16Enable();
  auto param = reinterpret_cast<TransposeParameter *>(op_parameter_);
  if (in_tensors_[0]->shape().size() != 4 || in_tensors_[0]->shape()[0] > 1) {
    MS_LOG(ERROR) << "Transpose only support 4d tensor and n = 1 yet.";
    return RET_ERROR;
  }
  if (param->num_axes_ == 4 && param->perm_[0] == 0 && param->perm_[1] == 3 && param->perm_[2] == 1 &&
      param->perm_[3] == 2) {
    kernel_name += "_0312";
    type = TransposeType::AXIS0312;
  } else if (param->num_axes_ == 4 && param->perm_[0] == 0 && param->perm_[1] == 2 && param->perm_[2] == 3 &&
             param->perm_[3] == 1) {
    kernel_name += "_0231";
    type = TransposeType::AXIS0231;
  } else {
    MS_LOG(ERROR) << "unsupported transpose axes.";
    return RET_ERROR;
  }
  if (in_tensors_[0]->shape()[2] * UP_DIV(in_tensors_[0]->shape()[3], C4NUM) > MAX_IMAGE2D_SIZE) {
    // just for input
    kernel_name += "_oversize";
  }
  kernel_name += "_" + std::string(EnumNameFormat(op_format_));
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::set<std::string> build_options;
  std::string source = transpose_source;
  std::string program_name = "transpose";
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

int TransposeOpenCLKernel::ReSize() { return RET_OK; }

int TransposeOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t im_dst_x = 1, im_dst_y = 1;
  auto out_shape = out_tensors_[0]->shape();
  if (op_format_ == schema::Format_NHWC4) {
    im_dst_x = out_shape[2] * UP_DIV(out_shape[3], C4NUM);  // W * C4
    im_dst_y = out_shape[0] * out_shape[1];                 // N * H
  } else if (op_format_ == schema::Format_NC4HW4) {
    im_dst_x = out_shape[2];                                               // W
    im_dst_y = out_shape[0] * UP_DIV(out_shape[3], C4NUM) * out_shape[1];  // N * C4 * H
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

int TransposeOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  std::vector<int> shapex = out_tensors_[0]->shape();
  size_t n = shapex[0];  // n=1
  size_t h = shapex[1];
  size_t w = shapex[2];
  size_t c = shapex[3];
  size_t c4 = UP_DIV(c, 4);
  std::vector<size_t> local = {};
  std::vector<size_t> global;
  if (type == TransposeType::AXIS0312) {
    global = {UP_DIV(h, C4NUM), w, c4};
  } else if (type == TransposeType::AXIS0231) {
    global = {h, UP_DIV(w, C4NUM), c4};
  }

  cl_int4 shape = {static_cast<int>(n), static_cast<int>(h), static_cast<int>(w), static_cast<int>(c)};
  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, shape);
  ocl_runtime_->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLTransposeKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                 const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                 const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                 const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel =
    new (std::nothrow) TransposeOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel " << opParameter->name_ << "is nullptr.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Transpose, OpenCLTransposeKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Transpose, OpenCLTransposeKernelCreator)
}  // namespace mindspore::kernel
