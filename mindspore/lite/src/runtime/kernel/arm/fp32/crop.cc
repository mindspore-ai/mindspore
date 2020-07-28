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
#include "src/runtime/kernel/arm/fp32/crop.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/opclib/fp32/crop.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Crop;

namespace mindspore::kernel {
int CropCPUKernel::Init() {
  schema::Format input0_format = inputs_[0]->GetFormat();
  if (input0_format != schema::Format_NC4HW4) {
    outputs_[0]->SetFormat(input0_format);
    return RET_OK;
  }
  convert_function_ = LayoutTransform(inputs_[0]->data_type(), inputs_[0]->GetFormat(), schema::Format_NHWC);
  if (convert_function_ == nullptr) {
    MS_LOG(ERROR) << "Can not convert format " << inputs_[0]->GetFormat() << " to " << schema::Format_NHWC;
    return RET_ERROR;
  }
  auto packed_input_size = inputs_[0]->Channel() * inputs_[0]->Batch() * inputs_[0]->Height() * inputs_[0]->Width();
  packed_input_ = reinterpret_cast<float *>(malloc(packed_input_size * sizeof(float)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc memory fail!";
    return RET_ERROR;
  }
  memset(packed_input_, 0, packed_input_size * sizeof(float));
  return RET_OK;
}

int CropCPUKernel::Run() {
  auto input = inputs_[0];
  auto output = outputs_[0];
  float *input_data = reinterpret_cast<float *>(input->Data());
  if (convert_function_ != nullptr) {
    convert_function_(input_data, packed_input_, inputs_[0]->Batch(), inputs_[0]->Height() * inputs_[0]->Width(),
                      inputs_[0]->Channel());
  } else {
    packed_input_ = input_data;
  }
  float *output_data = reinterpret_cast<float *>(output->Data());
  Crop4D(input_data, output_data, input->shape().data(), output->shape().data(),
         reinterpret_cast<CropParameter *>(opParameter));
  return RET_OK;
}

kernel::LiteKernel *CpuCropFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                             const std::vector<lite::tensor::Tensor *> &outputs,
                                             OpParameter *opParameter, const lite::Context *ctx,
                                             const kernel::KernelKey &desc) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  auto *kernel = new (std::nothrow) CropCPUKernel(opParameter, inputs, outputs);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new CropCPUKernel fail!";
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

REG_KERNEL(kCPU, PrimitiveType_Crop, CpuCropFp32KernelCreator)
}  // namespace mindspore::kernel

