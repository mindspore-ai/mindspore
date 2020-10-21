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
#include "nnacl/fp32/common_func.h"
#include "src/runtime/kernel/opencl/cl/activation.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_SIGMOID;
using mindspore::schema::ActivationType_TANH;
using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::kernel {

int ActivationOpenClKernel::Init() {
  in_size_ = in_tensors_[0]->shape().size();
  out_size_ = out_tensors_[0]->shape().size();
  size_t n, h, w, c;
  if (in_size_ == 2) {
    n = in_tensors_[0]->shape()[0];
    c = in_tensors_[0]->shape()[1];
    h = w = 1;
  } else {
    n = in_tensors_[0]->shape()[0];
    h = in_tensors_[0]->shape()[1];
    w = in_tensors_[0]->shape()[2];
    c = in_tensors_[0]->shape()[3];
  }
  nhwc_shape_ = {n, h, w, c};
  enable_fp16_ = ocl_runtime_->GetFp16Enable();
  fp_size = enable_fp16_ ? sizeof(uint16_t) : sizeof(float);
  if (in_size_ != 2 && in_size_ != 4) {
    MS_LOG(ERROR) << "Activate fun only support dim=4 or 2, but your dim=" << in_size_;
    return mindspore::lite::RET_ERROR;
  }
  std::map<int, std::string> kernel_names{{ActivationType_LEAKY_RELU, "LeakyRelu"},
                                          {ActivationType_RELU, "Relu"},
                                          {ActivationType_SIGMOID, "Sigmoid"},
                                          {ActivationType_RELU6, "Relu6"},
                                          {ActivationType_TANH, "Tanh"}};
  if (kernel_names.count(type_) == 0) {
    MS_LOG(ERROR) << "schema::ActivationType:" << type_ << "not found";
    return mindspore::lite::RET_ERROR;
  }

  std::string source = activation_source;
  std::set<std::string> build_options;
  std::string program_name = "Activation";
  ocl_runtime_->LoadSource(program_name, source);
  std::string kernel_name = kernel_names[type_];
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
  MS_LOG(DEBUG) << op_parameter_->name_ << " init Done!";
  return mindspore::lite::RET_OK;
}

int ActivationOpenClKernel::Run() {
  MS_LOG(DEBUG) << op_parameter_->name_ << " begin running!";
  cl_int4 img2d_shape = GetImg2dShape();
  int arg_idx = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_idx++, img2d_shape);
  if (type_ == ActivationType_LEAKY_RELU) {
    ocl_runtime_->SetKernelArg(kernel_, arg_idx++, alpha_);
  }
  std::vector<size_t> local = {};
  std::vector<size_t> global = {static_cast<size_t>(img2d_shape.s[1]), static_cast<size_t>(img2d_shape.s[2])};
  auto ret = ocl_runtime_->RunKernel(kernel_, global, local, nullptr);
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Run kernel:" << op_parameter_->name_ << " fail.";
    return mindspore::lite::RET_ERROR;
  }
  return mindspore::lite::RET_OK;
}

cl_int4 ActivationOpenClKernel::GetImg2dShape() {
  cl_int4 img2d_shape = {1, 1, 1, 1};
  img2d_shape.s[1] = nhwc_shape_[1];
  img2d_shape.s[2] = nhwc_shape_[2] * UP_DIV(nhwc_shape_[3], C4NUM);
  img2d_shape.s[3] = C4NUM;
  return img2d_shape;
}

kernel::LiteKernel *OpenClActivationKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                  const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                  const mindspore::lite::PrimitiveC *primitive) {
  if (inputs.empty()) {
    MS_LOG(ERROR) << "Input data size must be greater than 0, but your size is " << inputs.size();
    return nullptr;
  }
  if (inputs[0]->shape().size() > 2 && inputs[0]->shape()[0] > 1) {
    MS_LOG(ERROR) << "Activation kernel:" << opParameter->name_ << " failed: Unsupported multi-batch.";
    free(opParameter);
    return nullptr;
  }
  auto *kernel =
    new (std::nothrow) ActivationOpenClKernel(reinterpret_cast<OpParameter *>(opParameter), inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New kernel:" << opParameter->name_ << "is nullptr.";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Init activation kernel:" << opParameter->name_ << " failed!";
    delete kernel;
    return nullptr;
  }
  return kernel;
}
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Activation, OpenClActivationKernelCreator)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Activation, OpenClActivationKernelCreator)
}  // namespace mindspore::kernel
