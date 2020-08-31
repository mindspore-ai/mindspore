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

#include <vector>
#include <map>
#include <string>
#include <set>

#include "src/runtime/kernel/opencl/kernel/activation.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/opencl/cl/activation.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_SIGMOID;
using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::kernel {

void ActivationOpenClKernel::InitBuffer() {
  auto allocator = lite::opencl::OpenCLRuntime::GetInstance()->GetAllocator();
  alpha_buff_ = allocator->Malloc(fp_size);
  alpha_buff_ = allocator->MapBuffer(alpha_buff_, CL_MAP_WRITE, nullptr, true);
  memset(alpha_buff_, 0x00, fp_size);
  if (enable_fp16_) {
    auto fp16 = (float16_t)alpha_;
    memcpy(alpha_buff_, &fp16, fp_size);
  } else {
    memcpy(alpha_buff_, &alpha_, fp_size);
  }
  allocator->UnmapBuffer(alpha_buff_);
}

int ActivationOpenClKernel::Init() {
  in_size_ = in_tensors_[0]->shape().size();
  out_size_ = out_tensors_[0]->shape().size();
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  enable_fp16_ = ocl_runtime->GetFp16Enable();
  fp_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  if (in_size_ != 2 && in_size_ != 4) {
    MS_LOG(ERROR) << "Activate fun only support dim=4 or 2, but your dim=" << in_size_;
    return RET_ERROR;
  }
  InitBuffer();
  std::map<int, std::vector<std::string>> Program_Kernel{
    {ActivationType_LEAKY_RELU, std::vector<std::string>{"LEAKY_RELU", "LeakyRelu"}},
    {ActivationType_RELU, std::vector<std::string>{"RELU", "Relu"}},
    {ActivationType_SIGMOID, std::vector<std::string>{"SIGMOID", "Sigmoid"}},
    {ActivationType_RELU6, std::vector<std::string>{"RELU6", "Relu6"}}};
  if (Program_Kernel.count(type_) == 0) {
    MS_LOG(ERROR) << "schema::ActivationType:" << type_ << "not found";
    return RET_ERROR;
  }

  std::string source = activation_source;
  std::set<std::string> build_options;
  ocl_runtime->LoadSource(Program_Kernel[type_][0], source);
  ocl_runtime->BuildKernel(kernel_, Program_Kernel[type_][0], Program_Kernel[type_][1], build_options);

  std::map<int, schema::Format> format{{4, schema::Format_NHWC4}, {2, schema::Format_NC4}};
  if (format.count(out_size_) == 0) {
    MS_LOG(ERROR) << "Not found output tensor format";
    return RET_ERROR;
  }
  in_ori_format_ = in_tensors_[0]->GetFormat();
  out_ori_format_ = out_tensors_[0]->GetFormat();
  in_tensors_[0]->SetFormat(format[in_size_]);
  out_tensors_[0]->SetFormat(format[out_size_]);
  if (in_size_ == 2) {
    in_ori_format_ = schema::Format_NC4;
    out_ori_format_ = schema::Format_NC4;
  }
  MS_LOG(DEBUG) << op_parameter_->name_ << " init Done!";
  return RET_OK;
}

int ActivationOpenClKernel::Run() {
  MS_LOG(DEBUG) << op_parameter_->name_ << " begin running!";
  cl_int4 img2d_shape = GetImg2dShape();
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  int arg_idx = 0;
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, img2d_shape);
  if (type_ == ActivationType_LEAKY_RELU) {
    ocl_runtime->SetKernelArg(kernel_, arg_idx++, alpha_buff_, lite::opencl::MemType::BUF);
  }
  std::vector<size_t> local = {1, 1};
  std::vector<size_t> global = {static_cast<size_t>(img2d_shape.s[1]), static_cast<size_t>(img2d_shape.s[2])};
  auto ret = ocl_runtime->RunKernel(kernel_, global, local, nullptr);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run kernel:" << op_parameter_->name_ << " fail.";
    return RET_ERROR;
  }
  return RET_OK;
}

cl_int4 ActivationOpenClKernel::GetImg2dShape() {
  cl_int4 img2d_shape = {0, 0, 0, 0};
  for (int i = 0; i < in_size_; ++i) {
    img2d_shape.s[i + 4 - in_size_] = in_tensors_[0]->shape()[i];
  }
  if (in_size_ == 2) {
    img2d_shape.s[1] = img2d_shape.s[2];
    img2d_shape.s[2] = UP_DIV(img2d_shape.s[3], C4NUM);
    img2d_shape.s[3] = C4NUM;
  }
  return img2d_shape;
}

int ActivationOpenClKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  cl_int4 img_shape = GetImg2dShape();
  size_t img_dtype = CL_FLOAT;
  if (enable_fp16_) {
    img_dtype = CL_HALF_FLOAT;
  }
  img_size->clear();
  img_size->push_back(img_shape.s[2] * UP_DIV(img_shape.s[3], C4NUM));
  img_size->push_back(img_shape.s[1]);
  img_size->push_back(img_dtype);
  return RET_OK;
}

kernel::LiteKernel *OpenClActivationFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                      const std::vector<lite::tensor::Tensor *> &outputs,
                                                      OpParameter *opParameter, const lite::Context *ctx,
                                                      const kernel::KernelKey &desc,
                                                      const mindspore::lite::PrimitiveC *primitive) {
  if (inputs.empty()) {
    MS_LOG(ERROR) << "Input data size must be greater than 0, but your size is " << inputs.size();
    return nullptr;
  }
  if (inputs[0]->shape().size() > 2 && inputs[0]->shape()[0] > 1) {
    MS_LOG(ERROR) << "Activation kernel:" << opParameter->name_ << " failed: Unsupported multi-batch.";
    return nullptr;
  }
  auto *kernel =
    new (std::nothrow) ActivationOpenClKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New kernel:" << opParameter->name_ << "is nullptr.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init activation kernel:" << opParameter->name_ << " failed!";
    delete kernel;
    return nullptr;
  }
  return kernel;
}
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Activation, OpenClActivationFp32KernelCreator)
}  // namespace mindspore::kernel
