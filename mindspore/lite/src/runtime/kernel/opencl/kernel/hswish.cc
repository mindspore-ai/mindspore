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

#include "src/runtime/kernel/opencl/kernel/hswish.h"
#include <cstring>
#include <string>
#include <algorithm>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/hswish.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::kernel {

int HswishOpenCLKernel::Init() {
  if (out_tensors_[0]->shape().size() > 4) {
    MS_LOG(ERROR) << " only support dim <= 4";
    return RET_ERROR;
  }

  std::string kernel_name = "hswish";
  std::set<std::string> build_options;
  std::string source = hswish_source;
  std::string program_name = "hswish";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

void HswishGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
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

int HswishOpenCLKernel::InferShapeTo4D() {
  if (in_tensors_[0]->shape().size() <= 4) {
    if (in_tensors_[0]->shape().size() == 1) {
      N_ = in_tensors_[0]->shape()[0];
    } else if (in_tensors_[0]->shape().size() == 2) {
      N_ = in_tensors_[0]->shape()[0];
      C_ = in_tensors_[0]->shape()[1];
    } else if (in_tensors_[0]->shape().size() == 3) {
      N_ = in_tensors_[0]->shape()[0];
      W_ = in_tensors_[0]->shape()[1];
      C_ = in_tensors_[0]->shape()[2];
    } else {
      N_ = in_tensors_[0]->shape()[0];
      H_ = in_tensors_[0]->shape()[1];
      W_ = in_tensors_[0]->shape()[2];
      C_ = in_tensors_[0]->shape()[3];
    }
  } else {
    MS_LOG(ERROR) << "Unsupported inputdim: " << in_tensors_[0]->shape().size();
    return RET_ERROR;
  }
  return RET_OK;
}

int HswishOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  auto output_shape = out_tensors_[0]->shape();
  InferShapeTo4D();
  cl_int4 output_shape_ = {static_cast<cl_int>(N_), static_cast<cl_int>(H_), static_cast<cl_int>(W_),
                           static_cast<cl_int>(UP_DIV(C_, C4NUM))};
  const std::vector<size_t> &max_global = ocl_runtime_->GetWorkItemSize();
  std::vector<size_t> local = {1, 1, 1};
  uint32_t OH = N_ * H_;
  uint32_t OW = W_;
  uint32_t OC = UP_DIV(C_, C4NUM);
  std::vector<size_t> global = {OH, OW, OC};
  HswishGetWorkGroup(global, &local, max_global[0]);
  int arg_cn = 0;
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, arg_cn++, output_shape_);
  ocl_runtime_->RunKernel(kernel_, global, local, nullptr);
  return RET_OK;
}

kernel::LiteKernel *HswishOpenCLKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                              const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                              const mindspore::lite::PrimitiveC *primitive) {
  auto *kernel = new (std::nothrow) HswishOpenCLKernel(opParameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << " new HswishOpenCLKernel failed ";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << " Init kernel failed, name: hswish ";
    delete kernel;
    return nullptr;
  }
  return kernel;
}

}  // namespace mindspore::kernel
