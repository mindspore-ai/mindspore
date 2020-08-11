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
#include "src/runtime/kernel/arm/fp32/addn.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32/arithmetic.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_AddN;

namespace mindspore::kernel {
namespace {
int AddNLaunch(int thread_id, LiteParallelGroupEnv *penv, void *cdata) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "Input cdata is nullptr!";
    return RET_NULL_PTR;
  }
  auto kernel = reinterpret_cast<AddNCPUKernel *>(cdata);
  return kernel->AddNParallelRun(thread_id);
}
}  // namespace

int AddNCPUKernel::Init() { return RET_OK; }

int AddNCPUKernel::ReSize() { return RET_OK; }

int AddNCPUKernel::AddNParallelRun(int thread_id) {
  int count_per_thread = UP_DIV(elements_num_, op_parameter_->thread_num_);
  int count = MSMIN(count_per_thread, elements_num_ - thread_id * count_per_thread);
  auto stride = count_per_thread * thread_id;
  auto ret = ElementAdd(in1_addr_ + stride, in2_addr_ + stride, out_addr_ + stride, count);
  if (ret != NNACL_OK) {
    MS_LOG(ERROR) << "ElementAdd fail! ret: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

int AddNCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  elements_num_ = in_tensors_[0]->ElementsNum();
  auto input0_data = reinterpret_cast<float *>(in_tensors_[0]->Data());
  auto input1_data = reinterpret_cast<float *>(in_tensors_[1]->Data());
  auto output_data = reinterpret_cast<float *>(out_tensors_[0]->Data());
  if (elements_num_ < op_parameter_->thread_num_) {
    ElementAdd(input0_data, input1_data, output_data, elements_num_);
    for (int i = 2; i < in_tensors_.size(); ++i) {
      ElementAdd(reinterpret_cast<float *>(in_tensors_[i]->Data()), output_data, output_data, elements_num_);
    }
    return RET_OK;
  }
  in1_addr_ = input0_data;
  in2_addr_ = input1_data;
  out_addr_ = output_data;
  ret = LiteBackendParallelLaunch(AddNLaunch, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "addn launch fail!ret: " << ret;
    return RET_ERROR;
  }
  for (size_t i = 2; i < in_tensors_.size(); ++i) {
    in1_addr_ = reinterpret_cast<float *>(in_tensors_[i]->Data());
    in2_addr_ = output_data;
    ret = LiteBackendParallelLaunch(AddNLaunch, this, op_parameter_->thread_num_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "addn launch fail!ret: " << ret << ", input index: " << i;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

kernel::LiteKernel *CpuAddNFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                             const std::vector<lite::tensor::Tensor *> &outputs,
                                             OpParameter *op_parameter, const lite::Context *ctx,
                                             const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Input op_parameter is nullptr!";
    return nullptr;
  }
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "Input context is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_AddN);
  op_parameter->thread_num_ = ctx->thread_num_;
  auto *kernel = new (std::nothrow) AddNCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new AddNCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed! name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_AddN, CpuAddNFp32KernelCreator)
}  // namespace mindspore::kernel
