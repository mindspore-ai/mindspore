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

#include "src/runtime/kernel/opencl/kernel/stack.h"
#include <cstring>
#include <string>
#include <algorithm>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/stack.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Stack;

namespace mindspore::kernel {

int StackOpenCLKernel::RunAxis0() {
  auto allocator_ = ocl_runtime_->GetAllocator();
  std::vector<size_t> img_size;
  auto dst_data = out_tensors_[0]->data_c();
  auto dst_origin = cl::array<cl::size_type, 3U>{0, 0, 0};
  cl::Image2D *out_image = reinterpret_cast<cl::Image2D *>(allocator_->GetImage(dst_data));
  for (int i = 0; i < in_tensors_.size(); i++) {
    auto src_data = in_tensors_[i]->data_c();
    allocator_->GetImageSize(src_data, &img_size);
    auto src_origin = cl::array<cl::size_type, 3U>{0, 0, 0};
    auto region = cl::array<cl::size_type, 3U>{img_size[0], img_size[1], 1};
    cl::Image2D *input_image = reinterpret_cast<cl::Image2D *>(allocator_->GetImage(src_data));
    ocl_runtime_->GetDefaultCommandQueue()->enqueueCopyImage(*input_image, *out_image, src_origin, dst_origin, region);
    dst_origin[1] += region[1];
  }
  return RET_OK;
}

int StackOpenCLKernel::Init() {
  if (in_tensors_[0]->shape().size() > 4 || in_tensors_[0]->shape().size() <= 0) {
    MS_LOG(ERROR) << " only support dim <= 4 ";
    return RET_ERROR;
  }
  auto param = reinterpret_cast<StackParameter *>(this->op_parameter_);
  axis_ = param->axis_;
  axis_ = axis_ < 0 ? axis_ + in_tensors_[0]->shape().size() + 1 : axis_;
  if (in_tensors_[0]->shape().size() != 4) {
    if (in_tensors_[0]->shape().size() == 2) {
      axis_ = axis_ + 2;
    }
  }
  if (param->axis_ < -3 || param->axis_ > 3) {
    MS_LOG(ERROR) << " only support axis >= -3 and axis <= 3 ";
    return RET_ERROR;
  }

  std::string kernel_name = "stack";
  if (in_tensors_.size() == 8) {
    kernel_name += "8inputaxis" + std::to_string(axis_);
  } else {
    MS_LOG(ERROR) << " input must be 8";
    return RET_ERROR;
  }
  MS_LOG(DEBUG) << "kernel_name=: " << kernel_name;
  std::set<std::string> build_options;
  std::string source = stack_source;
  std::string program_name = "stack";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);

  return RET_OK;
}

int StackOpenCLKernel::ReSize() { return RET_OK; }

void StackGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
  const int max_divider = 8;
  const int max_x = 4, max_y = 8;
  int x = std::min(GetMaxDivisorStrategy1(global[0], max_divider), max_x);
  int yz = max_size / x;
  int y = std::min(std::min(GetMaxDivisorStrategy1(global[1], max_divider), yz), max_y);
  int z = std::min(yz / y, static_cast<int>(UP_DIV(global[2], 2)));

  local->clear();
  local->push_back(x);
  local->push_back(y);
  local->push_back(z);
}

int StackOpenCLKernel::InferInTensorShapeTo4D(int *arg_cn) {
  if (in_tensors_.size() == 8) {
    int size = in_tensors_[0]->shape().size();
    switch (size) {
      case 1:
        for (int i = 0; i < in_tensors_.size(); ++i) {
          ocl_runtime_->SetKernelArg(kernel_, (*arg_cn)++, in_tensors_[i]->data_c());
        }
        ocl_runtime_->SetKernelArg(kernel_, (*arg_cn)++, out_tensors_[0]->data_c());
        for (int i = 0; i < in_tensors_.size(); ++i) {
          cl_int4 temp = {in_tensors_[i]->shape()[0], 1, 1, 1};
          ocl_runtime_->SetKernelArg(kernel_, (*arg_cn)++, temp);
        }
        break;
      case 2:
        for (int i = 0; i < in_tensors_.size(); ++i) {
          ocl_runtime_->SetKernelArg(kernel_, (*arg_cn)++, in_tensors_[i]->data_c());
        }
        ocl_runtime_->SetKernelArg(kernel_, (*arg_cn)++, out_tensors_[0]->data_c());
        for (int i = 0; i < in_tensors_.size(); ++i) {
          cl_int4 temp = {in_tensors_[i]->shape()[0], 1, 1, UP_DIV(in_tensors_[i]->shape()[1], C4NUM)};
          ocl_runtime_->SetKernelArg(kernel_, (*arg_cn)++, temp);
        }
        break;
      case 3:
        for (int i = 0; i < in_tensors_.size(); ++i) {
          ocl_runtime_->SetKernelArg(kernel_, (*arg_cn)++, in_tensors_[i]->data_c());
        }
        ocl_runtime_->SetKernelArg(kernel_, (*arg_cn)++, out_tensors_[0]->data_c());
        for (int i = 0; i < in_tensors_.size(); ++i) {
          cl_int4 temp = {in_tensors_[i]->shape()[0], 1, in_tensors_[i]->shape()[1],
                          UP_DIV(in_tensors_[i]->shape()[2], C4NUM)};
          ocl_runtime_->SetKernelArg(kernel_, (*arg_cn)++, temp);
        }
        break;
      default:
        MS_LOG(ERROR) << "unsupported input size > 3 or size <= 0 :" << in_tensors_.size();
        return RET_ERROR;
    }
  } else {
    MS_LOG(ERROR) << "unsupported input size :" << in_tensors_.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int StackOpenCLKernel::InferOutTensorShapeTo4D(cl_int4 *output_shape) {
  std::vector<int> out_shape = out_tensors_[0]->shape();
  if (out_shape.size() == 3) {
    N_ = out_shape[0];
    C_ = out_shape[1] * UP_DIV(out_shape[2], C4NUM);
  } else if (out_shape.size() == 4) {
    if (axis_ == 1) {
      N_ = out_shape[0];
      H_ = out_shape[1];
      W_ = out_shape[2];
      C_ = UP_DIV(out_shape[3], C4NUM);
    } else {
      MS_LOG(ERROR) << "Unsupported out_shape.size=: " << out_shape.size() << " axis=: " << axis_;
      return RET_ERROR;
    }
  }
  OH_ = N_ * H_;
  OW_ = W_;
  OC_ = C_;
  output_shape->s[0] = N_;
  output_shape->s[1] = H_;
  output_shape->s[2] = W_;
  output_shape->s[3] = C_;
  return RET_OK;
}

int StackOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  if (axis_ == 0) {
    return RunAxis0();
  }
  cl_int4 output_shape = {1, 1, 1, 1};
  const std::vector<size_t> &max_global = ocl_runtime_->GetWorkItemSize();
  std::vector<size_t> local = {1, 1, 1};
  int arg_cn = 0;
  InferInTensorShapeTo4D(&arg_cn);
  InferOutTensorShapeTo4D(&output_shape);
  std::vector<size_t> global = {OH_, OW_, OC_};
  StackGetWorkGroup(global, &local, max_global[0]);
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, output_shape);
  ocl_runtime_->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLStackKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                             const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                             const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) StackOpenCLKernel(opParameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << " new StackOpenCLKernel failed ";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << " Init kernel failed, name: Stack ";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Stack, OpenCLStackKernelCreator);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Stack, OpenCLStackKernelCreator);
}  // namespace mindspore::kernel
