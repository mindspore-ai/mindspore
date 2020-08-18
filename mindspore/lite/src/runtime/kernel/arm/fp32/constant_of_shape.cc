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

#include "src/runtime/kernel/arm/fp32/constant_of_shape.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ConstantOfShape;

namespace mindspore::kernel {

namespace {
constexpr int kInputNum = 1;
constexpr int kOutputNum = 1;
}  // namespace

int ConstantOfShapeCPUKernel::Init() { return RET_OK; }

int ConstantOfShapeCPUKernel::ReSize() { return RET_OK; }

int ConstantOfShapeCPUKernel::DoExecute(int task_id) {
  int ret = ConstantOfShape(out_ptr_, task_id, param_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConstantOfShapeRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int ConstantOfShapeRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto g_kernel = reinterpret_cast<ConstantOfShapeCPUKernel *>(cdata);
  auto ret = g_kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConstantOfShapeRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int ConstantOfShapeCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  param_->element_sz_ = out_tensors_.front()->ElementsNum();
  int thread_num = MSMIN(param_->op_parameter_.thread_num_, param_->element_sz_);
  param_->unit_ = UP_DIV(param_->element_sz_, thread_num);
  param_->op_parameter_.thread_num_ = thread_num;
  out_ptr_ = reinterpret_cast<float *>(out_tensors_.front()->Data());
  auto ret = LiteBackendParallelLaunch(ConstantOfShapeRun, this, thread_num);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConstantOfShapeRun error error_code[" << ret << "]";
    return ret;
  }
  return ret;
}

kernel::LiteKernel *CpuConstantOfShapeFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                        const std::vector<lite::tensor::Tensor *> &outputs,
                                                        OpParameter *opParameter, const lite::Context *ctx,
                                                        const kernel::KernelKey &desc,
                                                        const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed, opParameter is nullptr, type: PrimitiveType_ConstantOfShape. ";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_ConstantOfShape);
  auto *kernel = new (std::nothrow) ConstantOfShapeCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ConstantOfShapeCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ConstantOfShape, CpuConstantOfShapeFp32KernelCreator)
}  // namespace mindspore::kernel
