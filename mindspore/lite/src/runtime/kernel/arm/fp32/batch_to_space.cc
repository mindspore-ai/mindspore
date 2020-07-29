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
#include "src/runtime/kernel/arm/fp32/batch_to_space.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/opclib/fp32/batch_to_space.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchToSpace;

namespace mindspore::kernel {

int BatchToSpaceCPUKernel::Init() {
  if (inputs_[0]->GetFormat() != schema::Format_NHWC) {
    MS_LOG(ERROR) << "batch_to_space only support NHWC now!";
    return RET_FORMAT_ERR;
  }
  BatchToSpaceParameter *param = reinterpret_cast<BatchToSpaceParameter *>(this->opParameter);
  for (int i = 0; i < BATCH_TO_SPACE_CROPS_SIZE; ++i) {
    if (param->crops_[i] != 0) {
      no_crop_ = false;
    }
  }
  return RET_OK;
}

int BatchToSpaceCPUKernel::Run() {
  auto input = inputs_[0];
  auto output = outputs_[0];
  const float *input_data = reinterpret_cast<const float *>(input->Data());
  float *output_data = reinterpret_cast<float *>(output->Data());
  auto in_shape = input->shape();
  auto out_shape = output->shape();
  BatchToSpaceParameter *param = reinterpret_cast<BatchToSpaceParameter *>(this->opParameter);

  if (no_crop_) {
    BatchToSpaceNoCropForNHWC(input_data, output_data, in_shape.data(), out_shape[0], param->block_shape_);
  } else {
    BatchToSpaceForNHWC(input_data, output_data, in_shape.data(), out_shape[0], param->block_shape_, param->crops_);
  }

  return RET_OK;
}

kernel::LiteKernel *CpuBatchToSpaceFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                     const std::vector<lite::tensor::Tensor *> &outputs,
                                                     OpParameter *opParameter, const lite::Context *ctx,
                                                     const kernel::KernelKey &desc) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) BatchToSpaceCPUKernel(opParameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new BatchToSpaceCPUKernel fail!";
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, PrimitiveType_BatchToSpace, CpuBatchToSpaceFp32KernelCreator)
}  // namespace mindspore::kernel


