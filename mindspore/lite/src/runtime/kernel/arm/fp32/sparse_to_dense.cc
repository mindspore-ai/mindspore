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
#include "src/runtime/kernel/arm/fp32/sparse_to_dense.h"
#include <vector>
#include "schema/model_generated.h"
#include "schema/ops_generated.h"
#include "src/runtime/kernel/arm/nnacl/sparse_to_dense.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SparseToDense;

namespace mindspore::kernel {
int SparseToDenseCPUKernel::Init() {
  s2d_param_->op_parameter_.thread_num_ = thread_count_;
  return RET_OK;
}

int SparseToDenseCPUKernel::DoExcute(int task_id) {
  SparseToDense(input_data_, output_shape_, snum_, dnum_, sp_num_, output_data, s2d_param_, task_id);
  return RET_OK;
}

int SparseToDenseRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto s2ddata = reinterpret_cast<SparseToDenseCPUKernel *>(cdata);
  auto ret = s2ddata->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SparseToDenseRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
int SparseToDenseCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  auto input = in_tensors_.at(0);
  auto input1 = in_tensors_.at(1);
  auto input2 = in_tensors_.at(2);
  auto input3 = in_tensors_.at(3);
  auto output0 = out_tensors_.at(0);

  input_data_ = reinterpret_cast<int *>(input->Data());
  total_number_ = reinterpret_cast<int *>(input1->Data());
  snum_ = reinterpret_cast<float *>(input2->Data());
  dnum_ = reinterpret_cast<float *>(input3->Data());
  sp_num_ = static_cast<int>(input->ElementsNum() / 2);

  output_data = reinterpret_cast<float *>(out_tensors_.at(0)->Data());
  std::vector<int> temp_shape = output0->shape();
  output_shape_ = reinterpret_cast<int *>(temp_shape.data());

  ret = LiteBackendParallelLaunch(SparseToDenseRun, this, s2d_param_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SparseToDenseRun error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuSparseToDenseFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                      const std::vector<lite::tensor::Tensor *> &outputs,
                                                      OpParameter *opParameter, const lite::Context *ctx,
                                                      const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "input opParameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_SparseToDense);
  auto *kernel = new (std::nothrow) SparseToDenseCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SparseToDenseCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SparseToDense, CpuSparseToDenseFp32KernelCreator)
}  // namespace mindspore::kernel
