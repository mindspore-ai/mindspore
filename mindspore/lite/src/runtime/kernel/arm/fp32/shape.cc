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

#include "src/runtime/kernel/arm/fp32/shape.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "utils/log_adapter.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Shape;

namespace mindspore::kernel {
namespace {
constexpr int kShapeInputNum = 1;
constexpr int kShapeOutputNum = 1;
}  // namespace
int ShapeCPUKernel::Init() { return RET_OK; }

int ShapeCPUKernel::ReSize() { return RET_OK; }

int ShapeCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return RET_ERROR;
  }
  auto out_tensor = out_tensors_.front();
  auto in_tensor = in_tensors_.front();
  if (in_tensor == nullptr || out_tensor == nullptr) {
    MS_LOG(ERROR) << "null pointer dereferencing.";
    return RET_ERROR;
  }
  if (in_tensor->Data() == nullptr || out_tensor->Data() == nullptr) {
    MS_LOG(ERROR) << "null pointer dereferencing.";
    return RET_ERROR;
  }

  for (int i = 0; i < in_tensor->shape().size(); i++) {
    reinterpret_cast<int *>(out_tensor->Data())[i] = in_tensor->shape()[i];
  }

  return RET_OK;
}

kernel::LiteKernel *CpuShapeFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_Shape);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "desc type is not Shape";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) ShapeCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "New kernel fails.";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Shape, CpuShapeFp32KernelCreator)
}  // namespace mindspore::kernel
