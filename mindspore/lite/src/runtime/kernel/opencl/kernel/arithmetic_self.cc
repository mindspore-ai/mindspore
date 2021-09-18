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
#include <algorithm>
#include <set>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/arithmetic_self.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/arithmeticself.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ArithmeticSelfOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_1 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (!IsArithmeticSelf(type())) {
    MS_LOG(WARNING) << "UnSupported Operator: " << schema::EnumNamePrimitiveType(type());
    return RET_ERROR;
  }
  if (in_tensors_[0]->shape().size() != DIMENSION_4D && in_tensors_[0]->shape().size() != DIMENSION_2D) {
    MS_LOG(WARNING) << " only support dim = 4 or 2 but your dim = " << in_tensors_[0]->shape().size();
    return RET_ERROR;
  }
  return RET_OK;
}

void ArithmeticSelfGetWorkGroup(const std::vector<size_t> &global, std::vector<size_t> *local, int max_size) {
  const int max_divider = 8;
  const int max_x = 4, max_y = 8;
  int x = std::min(GetMaxDivisorStrategy1(global[0], max_divider), max_x);
  if (x == 0) {
    MS_LOG(ERROR) << "div num shouldn't be 0";
    return;
  }
  int yz = max_size / x;
  int y = std::min(std::min(GetMaxDivisorStrategy1(global[1], max_divider), yz), max_y);
  if (y == 0) {
    MS_LOG(ERROR) << "div num shouldn't be 0";
    return;
  }
  int z = std::min(yz / y, static_cast<int>(UP_DIV(global[2], 2)));

  local->clear();
  local->push_back(x);
  local->push_back(y);
  local->push_back(z);
}

void ArithmeticSelfOpenCLKernel::SetGlobalLocal() {
  auto output_shape = out_tensors_[0]->shape();
  uint32_t OH = 1, OW = 1, OC = 1;
  if (output_shape.size() == DIMENSION_4D) {
    output_shape_ = {output_shape[0], output_shape[1], output_shape[2], UP_DIV(output_shape[3], C4NUM)};
    OH = output_shape[0] * output_shape[1];
    OW = output_shape[2];
    OC = UP_DIV(output_shape[3], C4NUM);
  } else if (output_shape.size() == DIMENSION_2D) {
    output_shape_ = {output_shape[0], 1, 1, UP_DIV(output_shape[1], C4NUM)};
    OH = output_shape[0];
    OW = 1;
    OC = UP_DIV(output_shape[1], C4NUM);
  }
  const std::vector<size_t> &max_global = ocl_runtime_->GetWorkItemSize();
  local_size_ = {1, 1, 1};  // init local
  global_size_ = {OH, OW, OC};
  ArithmeticSelfGetWorkGroup(global_size_, &local_size_, max_global[0]);
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
}

int ArithmeticSelfOpenCLKernel::Prepare() {
  std::string kernel_name = "ArithmeticSelf_Element";
  if (type() == schema::PrimitiveType_ExpFusion) {
    kernel_name += "Exp_NHWC4";
  } else {
    kernel_name += std::string(schema::EnumNamePrimitiveType(type())) + "_NHWC4";
  }
  MS_LOG(DEBUG) << "execute kernel name : " << kernel_name;
  const std::string program_name = "ArithmeticSelf";
  if (!ocl_runtime_->LoadSource(program_name, arithmeticself_source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto build_options_ext = CreateBuildOptionsExtByDType(this->registry_data_type_);
  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  SetGlobalLocal();
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticSelfOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  if (ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_.front()->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_.front()->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Abs, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Ceil, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Cos, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_ExpFusion, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Floor, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Log, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_LogicalNot, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Round, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Rsqrt, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Sin, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Neg, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Sqrt, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Square, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Abs, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Ceil, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Cos, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_ExpFusion, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Floor, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Log, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_LogicalNot, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Round, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Rsqrt, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Sin, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Neg, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Sqrt, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Square, OpenCLKernelCreator<ArithmeticSelfOpenCLKernel>)
}  // namespace mindspore::kernel
