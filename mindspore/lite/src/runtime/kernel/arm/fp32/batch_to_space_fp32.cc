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
#include "src/runtime/kernel/arm/fp32/batch_to_space_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_BatchToSpace;
using mindspore::schema::PrimitiveType_BatchToSpaceND;

namespace mindspore::kernel {
int BatchToSpaceCPUKernel::Init() {
  MS_ASSERT(in_tensors_.at(0)->format() == schema::Format::Format_NHWC);
  if (!InferShapeDone()) {
    return lite::RET_OK;
  }
  return ReSize();
}

int BatchToSpaceCPUKernel::ReSize() {
  MS_ASSERT(in_tensors_.at(0)->shape().size() == 4);
  return lite::RET_OK;
}

int BatchToSpaceCPUKernel::Run() {
  auto input = in_tensors_[0];
  auto output = out_tensors_[0];
  const float *input_data = reinterpret_cast<const float *>(input->data_c());
  float *output_data = reinterpret_cast<float *>(output->MutableData());
  auto in_shape = input->shape();
  auto out_shape = output->shape();
  BatchToSpaceParameter *param = reinterpret_cast<BatchToSpaceParameter *>(this->op_parameter_);

  if (param->no_crop_) {
    BatchToSpaceNoCropForNHWC(input_data, output_data, in_shape.data(), out_shape[0], param->block_shape_,
                              sizeof(float));
  } else {
    BatchToSpaceForNHWC(input_data, output_data, in_shape.data(), out_shape[0], param->block_shape_, param->crops_,
                        sizeof(float));
  }

  return lite::RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BatchToSpace, LiteKernelCreator<BatchToSpaceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BatchToSpaceND, LiteKernelCreator<BatchToSpaceCPUKernel>)
}  // namespace mindspore::kernel
