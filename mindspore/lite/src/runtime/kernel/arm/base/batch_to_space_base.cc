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
#include "src/runtime/kernel/arm/base/batch_to_space_base.h"
#include "src/runtime/kernel/arm/nnacl/batch_to_space.h"
#include "src/runtime/kernel/arm/fp32/batch_to_space.h"
#include "src/runtime/kernel/arm/int8/batch_to_space_int8.h"
#include "schema/model_generated.h"
#include "src/kernel_factory.h"
#include "include/errorcode.h"
#include "include/context.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchToSpace;

namespace mindspore::kernel {
int BatchToSpaceBaseCPUKernel::Init() {
  BatchToSpaceParameter *param = reinterpret_cast<BatchToSpaceParameter *>(this->op_parameter_);
  for (int i = 0; i < BATCH_TO_SPACE_CROPS_SIZE; ++i) {
    if (param->crops_[i] != 0) {
      no_crop_ = false;
    }
  }
  return RET_OK;
}

int BatchToSpaceBaseCPUKernel::ReSize() {
  if (in_tensors_[0]->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "batch_to_space only support NHWC now!";
    return RET_FORMAT_ERR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuBatchToSpaceInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                     const std::vector<lite::tensor::Tensor *> &outputs,
                                                     OpParameter *op_parameter, const lite::Context *ctx,
                                                     const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_BatchToSpace);
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Input op_parameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) BatchToSpaceInt8CPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new BatchToSpaceInt8CPUKernel fail!";
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    return nullptr;
  }
  return kernel;
}

kernel::LiteKernel *CpuBatchToSpaceFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                     const std::vector<lite::tensor::Tensor *> &outputs,
                                                     OpParameter *op_parameter, const lite::Context *ctx,
                                                     const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_BatchToSpace);
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Input op_parameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) BatchToSpaceCPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new BatchToSpaceCPUKernel fail!";
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_BatchToSpace, CpuBatchToSpaceInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BatchToSpace, CpuBatchToSpaceFp32KernelCreator)
}  // namespace mindspore::kernel
