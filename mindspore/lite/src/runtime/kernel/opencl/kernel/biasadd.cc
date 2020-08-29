/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *
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
#include <map>
#include <set>
#include <vector>

#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/opencl/kernel/biasadd.h"
#include "src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/opencl/cl/biasadd.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BiasAdd;

namespace mindspore::kernel {

void BiasAddOpenCLKernel::InitBuffer() {
  int C = in_tensors_[1]->shape()[0];
  int div_ci = UP_DIV(C, C4NUM);
  auto allocator = lite::opencl::OpenCLRuntime::GetInstance()->GetAllocator();
  BiasAdd_ = reinterpret_cast<FLOAT_t *>(allocator->Malloc(div_ci * C4NUM * sizeof(FLOAT_t)));
  BiasAdd_ = reinterpret_cast<FLOAT_t *>(allocator->MapBuffer(BiasAdd_, CL_MAP_WRITE, nullptr, true));
  memset(BiasAdd_, 0x00, div_ci * C4NUM * sizeof(FLOAT_t));
  auto origin_weight = reinterpret_cast<FLOAT_t *>(in_tensors_[1]->Data());
  for (int i = 0; i < in_tensors_[1]->ElementsNum(); ++i) {
    BiasAdd_[i] = origin_weight[i];
  }
  allocator->UnmapBuffer(BiasAdd_);
}

int BiasAddOpenCLKernel::Init() {
  in_size_ = in_tensors_[0]->shape().size();
  out_size_ = out_tensors_[0]->shape().size();
  if (in_size_ != 4 && in_size_ != 2) {
    MS_LOG(ERROR) << "BiasAdd only support dim=4 or 2, but your dim=" << in_size_;
    return RET_ERROR;
  }
  int C = in_tensors_[0]->shape()[3];
  int Bias_Size = in_tensors_[1]->shape()[0];
  if (UP_DIV(Bias_Size, C4NUM) != UP_DIV(C, C4NUM)) {
    MS_LOG(ERROR) << "BiasAdd weight channel size:" << Bias_Size << " must be equal with in_teneors channel size:" << C;
    return RET_ERROR;
  }
  InitBuffer();
  std::set<std::string> build_options;
  std::string source = biasadd_source;
  std::string program_name = "BiasAdd";
  std::string kernel_name = "BiasAdd";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->LoadSource(program_name, source);
  ocl_runtime->BuildKernel(kernel_, program_name, kernel_name, build_options);

  in_ori_format_ = in_tensors_[0]->GetFormat();
  out_ori_format_ = out_tensors_[0]->GetFormat();
  std::map<int, schema::Format> format{{4, schema::Format_NHWC4}, {2, schema::Format_NC4}};
  if (format.count(out_size_) == 0) {
    MS_LOG(ERROR) << "Not found output tensor format";
    return RET_ERROR;
  }
  in_tensors_[0]->SetFormat(format[in_size_]);
  out_tensors_[0]->SetFormat(format[out_size_]);
  if (in_size_ == 2) {
    in_ori_format_ = format[in_size_];
    out_ori_format_ = format[out_size_];
  }
  MS_LOG(DEBUG) << program_name << " Init Done!";
  return RET_OK;
}

int BiasAddOpenCLKernel::Run() {
  cl_int4 input_shape = GetImg2dShape();
  MS_LOG(DEBUG) << op_parameter_->name_ << " Running!";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  int arg_idx = 0;
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->Data());
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, input_shape);
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, BiasAdd_);
  ocl_runtime->SetKernelArg(kernel_, arg_idx++, in_size_);
  std::vector<size_t> local = {1, 1};
  std::vector<size_t> global = {static_cast<size_t>(input_shape.s[1]), static_cast<size_t>(input_shape.s[2])};
  auto ret = ocl_runtime->RunKernel(kernel_, global, local, nullptr);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Run kernel " << op_parameter_->name_ << " error.";
    return RET_ERROR;
  }
  return RET_OK;
}

cl_int4 BiasAddOpenCLKernel::GetImg2dShape() {
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

int BiasAddOpenCLKernel::GetImageSize(size_t idx, std::vector<size_t> *img_size) {
  cl_int4 img_shape = GetImg2dShape();
#ifdef ENABLE_FP16
  size_t img_dtype = CL_HALF_FLOAT;
#else
  size_t img_dtype = CL_FLOAT;
#endif

  img_size->clear();
  img_size->push_back(img_shape.s[2] * UP_DIV(img_shape.s[3], C4NUM));
  img_size->push_back(img_shape.s[1]);
  img_size->push_back(img_dtype);
  return RET_OK;
}

kernel::LiteKernel *OpenCLBiasAddKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx,
                                               const kernel::KernelKey &desc, const lite::PrimitiveC *primitive) {
  if (inputs.size() == 0) {
    MS_LOG(ERROR) << "Input data size must be greater than 0, but your size is " << inputs.size();
    return nullptr;
  }
  if (inputs[0]->shape()[0] > 1) {
    MS_LOG(ERROR) << "Input data size unsupported multi-batch.";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) BiasAddOpenCLKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "Kernel " << opParameter->name_ << "is nullptr.";
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init BiasAdd kernel failed!";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_BiasAdd, OpenCLBiasAddKernelCreator)
}  // namespace mindspore::kernel
