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
#include <utility>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/batch_to_space_nd.h"
#include "src/runtime/kernel/opencl/cl/batch_to_space_nd.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchToSpace;
using mindspore::schema::PrimitiveType_BatchToSpaceND;

namespace mindspore::kernel {

int BatchToSpaceNDOpenCLKernel::CheckSpecs() {
  if (in_tensors_.size() != 1 || out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_[0]->data_type() != kNumberTypeFloat32 && in_tensors_[0]->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Unsupported data type " << in_tensors_[0]->data_type();
    return RET_ERROR;
  }
  if (in_tensors_[0]->shape().size() != 4 && out_tensors_[0]->shape().size() != 4) {
    MS_LOG(ERROR) << "input/output shape size must be 4, actual: " << in_tensors_[0]->shape().size() << ", "
                  << out_tensors_[0]->shape().size();
    return RET_ERROR;
  }
  auto *param = reinterpret_cast<BatchToSpaceParameter *>(this->op_parameter_);
  if (param->block_shape_[0] < 1 || param->block_shape_[1] < 1) {
    MS_LOG(ERROR) << "block_sizes_ must > 1, actual " << param->block_shape_[0] << ", " << param->block_shape_[1];
    return RET_ERROR;
  }
  if (in_tensors_[0]->shape()[kNHWC_H] * param->block_shape_[0] <= (param->crops_[0] + param->crops_[1]) ||
      in_tensors_[0]->shape()[kNHWC_W] * param->block_shape_[1] <= (param->crops_[2] + param->crops_[3])) {
    MS_LOG(ERROR) << "crop shape error!";
    return RET_ERROR;
  }
  return RET_OK;
}

void BatchToSpaceNDOpenCLKernel::SetConstArgs() {
  auto param = reinterpret_cast<BatchToSpaceParameter *>(this->op_parameter_);
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  size_t CI4 = UP_DIV(in_tensors_[0]->Channel(), C4NUM);
  cl_int4 src_size = {(cl_int)CI4, in_tensors_[0]->Width(), in_tensors_[0]->Height() * out_tensors_[0]->Batch(), 1};
  std::vector<int> out_shape = out_tensors_[0]->shape();
  cl_int4 dst_size = {(cl_int)CO4, out_shape[2], out_shape[1], out_shape[0]};
  cl_int2 block_size = {param->block_shape_[0], param->block_shape_[1]};
  cl_int4 paddings = {param->crops_[0], param->crops_[1], param->crops_[2], param->crops_[3]};

  int arg_cnt = 2;
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, src_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, dst_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, block_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, paddings);
}

void BatchToSpaceNDOpenCLKernel::SetGlobalLocal() {
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  std::vector<int> out_shape = out_tensors_[0]->shape();
  cl_int4 dst_size = {(cl_int)CO4, out_shape[2], out_shape[1], out_shape[0]};
  local_size_ = {1, 1, 1};
  global_size_ = {(size_t)dst_size.s[0], (size_t)dst_size.s[1], (size_t)dst_size.s[2]};
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
}

int BatchToSpaceNDOpenCLKernel::Prepare() {
  std::string kernel_name = "batch_to_space_nd_NHWC4";

#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else

  std::string source = batch_to_space_nd_source;
  std::string program_name = "batch_to_space_nd";
  ocl_runtime_->LoadSource(program_name, source);
  auto build_options_ext = CreateBuildOptionsExtByDType(desc_.data_type);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options_ext);
#endif

  SetGlobalLocal();
  SetConstArgs();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int BatchToSpaceNDOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";
  ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_[0]->data_c());
  ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_[0]->data_c());
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_);

  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_BatchToSpaceND, OpenCLKernelCreator<BatchToSpaceNDOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_BatchToSpaceND, OpenCLKernelCreator<BatchToSpaceNDOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_BatchToSpace, OpenCLKernelCreator<BatchToSpaceNDOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_BatchToSpace, OpenCLKernelCreator<BatchToSpaceNDOpenCLKernel>);

}  // namespace mindspore::kernel
