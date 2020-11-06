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
#include "src/runtime/kernel/opencl/kernel/gather.h"
#include "src/runtime/kernel/opencl/cl/gather.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Gather;

namespace mindspore::kernel {

int GatherOpenCLKernel::CheckSpecs() { return RET_OK; }

void GatherOpenCLKernel::SetConstArgs() {
  auto param = reinterpret_cast<GatherParameter *>(this->op_parameter_);
  param->axis_ = (param->axis_ + in_tensors_[0]->shape().size()) % in_tensors_[0]->shape().size();
  auto input_shape = in_tensors_[0]->shape();
  auto output_shape = out_tensors_[0]->shape();
  int indices_num = in_tensors_[1]->ElementsNum();
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  size_t CI4 = UP_DIV(in_tensors_[0]->Channel(), C4NUM);
  cl_int4 src_size = {in_tensors_[0]->Width(), in_tensors_[0]->Height(), (cl_int)CI4, in_tensors_[0]->Batch()};
  cl_int4 dst_size = {(cl_int)out_tensors_[0]->Width(), (cl_int)out_tensors_[0]->Height(), (cl_int)CO4,
                      (cl_int)out_tensors_[0]->Batch()};
  int arg_cnt = 3;
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, src_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, dst_size);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, indices_num);
  ocl_runtime_->SetKernelArg(kernel_, arg_cnt++, param->axis_);
}

void GatherOpenCLKernel::SetGlobalLocal() {
  size_t CO4 = UP_DIV(out_tensors_[0]->Channel(), C4NUM);
  std::vector<size_t> local = {1, 1, 1};
  std::vector<size_t> global = {(size_t)out_tensors_[0]->Width(),
                                (size_t)out_tensors_[0]->Batch() * (size_t)out_tensors_[0]->Height(), CO4};
  OpenCLKernel::AlignGlobalLocal(global, local);
}

int GatherOpenCLKernel::Prepare() {
  std::string kernel_name = "gather_NHWC4";
#ifdef PROGRAM_WITH_IL
  kernel_ = ocl_runtime_->GetKernelFromBinary(kernel_name);
#else
  std::set<std::string> build_options;
  std::string source = gather_source;
  std::string program_name = "gather";
  ocl_runtime_->LoadSource(program_name, source);
  ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options);
#endif

  InitWeights();
  SetGlobalLocal();
  SetConstArgs();
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int GatherOpenCLKernel::InitWeights() {
  auto indices_tensor = in_tensors_.at(1);
  int indices_num = indices_tensor->ElementsNum();
  bool isIndicesInt32 = indices_tensor->data_type() == kNumberTypeInt32;
  auto allocator = ocl_runtime_->GetAllocator();
  if (!isIndicesInt32) {
    indices_data_ = reinterpret_cast<int32_t *>(allocator->Malloc(sizeof(int32_t) * indices_num));
    if (indices_data_ == nullptr) {
      MS_LOG(ERROR) << "Memory allocation failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int GatherOpenCLKernel::UpdateWeights() {
  auto indices_tensor = in_tensors_.at(1);
  int indices_num = indices_tensor->ElementsNum();
  bool isIndicesInt32 = indices_tensor->data_type() == kNumberTypeInt32;
  if (!isIndicesInt32) {
    if (indices_tensor->data_type() == kNumberTypeInt64) {
      for (int i = 0; i < indices_num; i++) {
        indices_data_[i] = reinterpret_cast<int64_t *>(indices_tensor->data_c())[i];
      }
    } else if (indices_tensor->data_type() == kNumberTypeFloat32) {
      for (int i = 0; i < indices_num; i++) {
        indices_data_[i] = reinterpret_cast<float *>(indices_tensor->data_c())[i];
      }
    } else if (indices_tensor->data_type() == kNumberTypeFloat16) {
      for (int i = 0; i < indices_num; i++) {
        indices_data_[i] = reinterpret_cast<float16_t *>(indices_tensor->data_c())[i];
      }
    } else {
      MS_LOG(ERROR) << "Unsupported data type: " << indices_tensor->data_type();
      return RET_ERROR;
    }
  } else {
    indices_data_ = reinterpret_cast<int32_t *>(indices_tensor->data_c());
  }
  return RET_OK;
}

int GatherOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running! ";

  if (UpdateWeights() != RET_OK) {
    return RET_ERROR;
  }

  ocl_runtime_->SetKernelArg(kernel_, 0, out_tensors_[0]->data_c(), lite::opencl::MemType::IMG);
  ocl_runtime_->SetKernelArg(kernel_, 1, in_tensors_[0]->data_c(), lite::opencl::MemType::IMG);
  ocl_runtime_->SetKernelArg(kernel_, 2, indices_data_, lite::opencl::MemType::BUF);
  ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr);

  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Gather, OpenCLKernelCreator<GatherOpenCLKernel>);
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Gather, OpenCLKernelCreator<GatherOpenCLKernel>);

}  // namespace mindspore::kernel
