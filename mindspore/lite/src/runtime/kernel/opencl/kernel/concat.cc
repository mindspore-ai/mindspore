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
#include <cstring>
#include <string>
#include <algorithm>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/kernel/concat.h"
#include "src/runtime/kernel/opencl/cl/concat.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Concat;

namespace mindspore::kernel {

int ConcatOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  size_t im_dst_x, im_dst_y;
  if (in_tensors_[0]->GetFormat() == schema::Format_NHWC4) {
    im_dst_x = out_tensors_[0]->Width() * CO4;
    im_dst_y = out_tensors_[0]->Height() * out_tensors_[0]->Batch();
  } else {
    im_dst_y = out_tensors_[0]->Height() * CO4;
    im_dst_x = out_tensors_[0]->Width() * out_tensors_[0]->Batch();
  }
  size_t img_dtype = CL_FLOAT;
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto enable_fp16_ = ocl_runtime->GetFp16Enable();
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  img_size->clear();
  std::vector<size_t> vec{im_dst_x, im_dst_y, img_dtype};
  *img_size = vec;
  return RET_OK;
}
int ConcatOpenCLKernel::Init() {
  if (in_tensors_[0]->shape().size() != 4) {
    MS_LOG(ERROR) << " only support dim = 4 ";
    return RET_ERROR;
  }

  auto param = reinterpret_cast<ConcatParameter *>(this->op_parameter_);
  MS_LOG(DEBUG) << " concat at axis=:  " << param->axis_;
  if (param->axis_ < 0 || param->axis_ > 3) {
    MS_LOG(ERROR) << " only support axis >= 0 and axis <= 3 ";
    return RET_ERROR;
  }

  if (in_tensors_.size() == 2) {
    std::set<std::string> build_options;
    std::string source = concat_source;
    std::string program_name = "Concat";
    std::string kernel_name = "Concat";
    auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
    ocl_runtime->LoadSource(program_name, source);
    ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
  }

  if (in_tensors_.size() == 3) {
    std::set<std::string> build_options;
    std::string source = concat_source;
    std::string program_name = "Concat3input";
    std::string kernel_name = "Concat3input";
    auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
    ocl_runtime->LoadSource(program_name, source);
    ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);
  }
  in_ori_format_ = in_tensors_[0]->GetFormat();
  in_tensors_[0]->SetFormat(schema::Format_NHWC4);
  out_ori_format_ = out_tensors_[0]->GetFormat();
  out_tensors_[0]->SetFormat(schema::Format_NHWC4);

  return RET_OK;
}

int ConcatOpenCLKernel::ReSize() { return RET_OK; }

int ConcatGetBiggestDividerWithPriority(int number, int max_divider) {
  if (number % 8 == 0 && max_divider >= 8) {
    return number / 8;
  }
  if (number % 4 == 0 && 4 <= max_divider) {
    return number / 4;
  }
  if (number % 2 == 0 && 2 <= max_divider) {
    return number / 2;
  }

  for (int i = max_divider; i != 0; i--) {
    if (number % i == 0) {
      return i;
    }
  }
  return RET_OK;
}

void ConcatGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
  const int max_divider = 8;
  const int max_x = 4, max_y = 8;
  int x = std::min(ConcatGetBiggestDividerWithPriority(global[0], max_divider), max_x);
  int yz = max_size / x;
  int y = std::min(std::min(ConcatGetBiggestDividerWithPriority(global[1], max_divider), yz), max_y);
  int z = std::min(yz / y, static_cast<int>(UP_DIV(global[2], 2)));

  local->clear();
  local->push_back(x);
  local->push_back(y);
  local->push_back(z);
}
int ConcatOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  auto param = reinterpret_cast<ConcatParameter *>(this->op_parameter_);

  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  auto input1_shape = in_tensors_[0]->shape();
  auto input2_shape = in_tensors_[1]->shape();
  auto output_shape = out_tensors_[0]->shape();

  cl_int4 input_shape1_ = {input1_shape[0], input1_shape[1], input1_shape[2], UP_DIV(input1_shape[3], C4NUM)};
  cl_int4 input_shape2_ = {input2_shape[0], input2_shape[1], input2_shape[2], UP_DIV(input2_shape[3], C4NUM)};
  cl_int4 output_shape_ = {output_shape[0], output_shape[1], output_shape[2], UP_DIV(output_shape[3], C4NUM)};

  uint32_t OH = output_shape[0] * output_shape[1];  // N*H
  uint32_t OW = output_shape[2];
  uint32_t OC = UP_DIV(output_shape[3], C4NUM);

  const std::vector<size_t> &max_global = ocl_runtime->GetWorkItemSize();
  std::vector<size_t> local = {1, 1, 1};  // init local
  std::vector<size_t> global = {OH, OW, OC};
  ConcatGetWorkGroup(global, &local, max_global[0]);

  int arg_cn = 0;
  if (in_tensors_.size() == 2) {
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, in_tensors_[0]->Data());
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, in_tensors_[1]->Data());
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->Data());
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, input_shape1_);
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, input_shape2_);
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, output_shape_);
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, param->axis_);
  } else if (in_tensors_.size() == 3) {
    auto input3_shape = in_tensors_[2]->shape();
    cl_int4 input_shape3_ = {input3_shape[0], input3_shape[1], input3_shape[2], UP_DIV(input3_shape[3], C4NUM)};

    ocl_runtime->SetKernelArg(kernel_, arg_cn++, in_tensors_[0]->Data());
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, in_tensors_[1]->Data());
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, in_tensors_[2]->Data());
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->Data());
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, input_shape1_);
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, input_shape2_);
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, input_shape3_);
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, output_shape_);
    ocl_runtime->SetKernelArg(kernel_, arg_cn++, param->axis_);
  } else if (in_tensors_.size() < 2) {
    MS_LOG(ERROR) << " input sizes must >= 2 ";
    return RET_ERROR;
  } else {
    MS_LOG(ERROR) << " only support inputs <= 3 ";
    return RET_ERROR;
  }
  ocl_runtime->RunKernel(kernel_, global, local, nullptr);

  return RET_OK;
}

kernel::LiteKernel *OpenCLConcatKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc,
                                              const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) ConcatOpenCLKernel(opParameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << " new ConcatOpenCLKernel failed ";
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
