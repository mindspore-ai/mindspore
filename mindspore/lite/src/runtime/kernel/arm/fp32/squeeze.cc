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

#include "src/runtime/kernel/arm/fp32/squeeze.h"
#include <vector>
#include "src/runtime/kernel/arm/nnacl/squeeze.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Squeeze;

namespace mindspore::kernel {
namespace {
constexpr int kSqueezeInputNum = 1;
constexpr int kSqueezeOutputNum = 1;
}  // namespace

int SqueezeCPUKernel::Init() { return RET_OK; }

int SqueezeCPUKernel::ReSize() { return RET_OK; }

int SqueezeCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  auto input_ptr = reinterpret_cast<float *>(in_tensors_.front()->Data());
  auto output_ptr = reinterpret_cast<float *>(out_tensors_.front()->Data());
  size_t data_size = in_tensors_.front()->Size();
  ret = DoSqueeze(input_ptr, output_ptr, data_size);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Do squeeze failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuSqueezeFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                const std::vector<lite::tensor::Tensor *> &outputs,
                                                OpParameter *opParameter, const lite::Context *ctx,
                                                const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_Squeeze);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "desc type is not Squeeze";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) SqueezeCPUKernel(opParameter, inputs, outputs, ctx, primitive);
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Squeeze, CpuSqueezeFp32KernelCreator)
}  // namespace mindspore::kernel
