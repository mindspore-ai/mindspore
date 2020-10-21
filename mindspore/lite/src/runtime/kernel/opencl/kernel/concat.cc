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

#include "src/runtime/kernel/opencl/kernel/concat.h"
#include <cstring>
#include <string>
#include <algorithm>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/concat.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Concat;

namespace mindspore::kernel {

int ConcatOpenCLKernel::RunAxis0() {
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

int ConcatOpenCLKernel::Init() {
  if (in_tensors_[0]->shape().size() != 4) {
    MS_LOG(ERROR) << " only support dim = 4 ";
    return RET_ERROR;
  }

  auto param = reinterpret_cast<ConcatParameter *>(this->op_parameter_);
  MS_LOG(DEBUG) << " concat at axis=:  " << param->axis_;
  if (param->axis_ < 0) {
    param->axis_ += in_tensors_.front()->shape().size();
  }
  if (param->axis_ < 0 || param->axis_ > 3) {
    MS_LOG(ERROR) << " only support axis >= 0 and axis <= 3 ";
    return RET_ERROR;
  }

  std::string kernel_name = "Concat";
  if (in_tensors_.size() == 2 || in_tensors_.size() == 3 || in_tensors_.size() == 4 || in_tensors_.size() == 6) {
    kernel_name += std::to_string(in_tensors_.size()) + "inputaxis" + std::to_string(param->axis_);
  } else {
    MS_LOG(ERROR) << " input must be 2 , 3 , 4 or 6";
    return RET_ERROR;
  }
  kernel_name += "_NHWC4";
  MS_LOG(DEBUG) << "kernel_name=: " << kernel_name;
  std::set<std::string> build_options;
  std::string source = concat_source;
  std::string program_name = "Concat";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

void ConcatGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
  const int max_divider = 8;
  const int max_x = 2, max_y = 8;
  int x = std::min(GetMaxDivisorStrategy1(global[0], max_divider), max_x);
  int yz = max_size / x;
  int y = std::min(std::min(GetMaxDivisorStrategy1(global[1], max_divider), yz), max_y);
  int z = std::min(yz / y, static_cast<int>(UP_DIV(global[2], 2)));

  local->clear();
  local->push_back(x);
  local->push_back(y);
  local->push_back(z);
}

int ConcatOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  auto param = reinterpret_cast<ConcatParameter *>(this->op_parameter_);
  if (param->axis_ == 0) {
    return RunAxis0();
  }
  auto output_shape = out_tensors_[0]->shape();
  cl_int4 output_shape_ = {output_shape[0], output_shape[1], output_shape[2], UP_DIV(output_shape[3], C4NUM)};
  const std::vector<size_t> &max_global = ocl_runtime_->GetWorkItemSize();
  std::vector<size_t> local = {1, 1, 1};
  uint32_t OH = output_shape_.s[0] * output_shape_.s[1];
  uint32_t OW = output_shape_.s[2];
  uint32_t OC = output_shape_.s[3];
  std::vector<size_t> global = {OH, OW, OC};
  ConcatGetWorkGroup(global, &local, max_global[0]);
  if (in_tensors_.size() == 2 || in_tensors_.size() == 3 || in_tensors_.size() == 4 || in_tensors_.size() == 6) {
    int arg_cn = 0;
    for (int i = 0; i < in_tensors_.size(); ++i) {
      ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_[i]->data_c());
    }
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->data_c());
    for (int i = 0; i < in_tensors_.size(); ++i) {
      cl_int4 temp = {in_tensors_[i]->shape()[0], in_tensors_[i]->shape()[1], in_tensors_[i]->shape()[2],
                      UP_DIV(in_tensors_[i]->shape()[3], C4NUM)};
      ocl_runtime_->SetKernelArg(kernel_, arg_cn++, temp);
    }
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, output_shape_);
    ocl_runtime_->SetKernelArg(kernel_, arg_cn++, param->axis_);
  } else {
    MS_LOG(ERROR) << "unsupported input size :" << in_tensors_.size();
    return RET_ERROR;
  }
  ocl_runtime_->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *OpenCLConcatKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                              const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                              const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) ConcatOpenCLKernel(opParameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << " new ConcatOpenCLKernel failed ";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << " Init kernel failed, name: Concat ";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Concat, OpenCLConcatKernelCreator);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Concat, OpenCLConcatKernelCreator);
}  // namespace mindspore::kernel
