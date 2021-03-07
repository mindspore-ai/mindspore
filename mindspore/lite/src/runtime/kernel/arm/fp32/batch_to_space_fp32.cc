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
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchToSpace;
using mindspore::schema::PrimitiveType_BatchToSpaceND;

namespace mindspore::kernel {
int BatchToSpaceCPUKernel::Processinput() {
  MS_ASSERT(in_tensors_[1]->data_c() != nullptr);
  MS_ASSERT(in_tensors_[2]->data_c() != nullptr);
  auto block_shape_data = in_tensors_[1]->data_c();
  auto crops_data = in_tensors_[2]->data_c();
  auto block_shape = static_cast<int *>(block_shape_data);
  auto crops = static_cast<int *>(crops_data);
  for (int i = 0; i < BATCH_TO_SPACE_BLOCK_SHAPE_SIZE; ++i) {
    block_shape_[i] = block_shape[i];
  }
  no_crop_ = true;
  for (int i = 0; i < COMM_SHAPE_SIZE; ++i) {
    crops_[i] = crops[i];
    if (crops_[i] != 0) {
      no_crop_ = false;
    }
  }
  return RET_OK;
}

int BatchToSpaceCPUKernel::Init() {
  MS_ASSERT(in_tensors_.at(0)->format() == schema::Format::Format_NHWC);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int BatchToSpaceCPUKernel::ReSize() {
  MS_ASSERT(in_tensors_.at(0)->shape().size() == COMM_SHAPE_SIZE);
  return RET_OK;
}

int BatchToSpaceCPUKernel::Run() {
  auto input = in_tensors_[0];
  auto output = out_tensors_[0];
  const float *input_data = reinterpret_cast<const float *>(input->data_c());
  float *output_data = reinterpret_cast<float *>(output->MutableData());
  auto in_shape = input->shape();
  auto out_shape = output->shape();
  if (in_tensors_.size() == 1) {
    BatchToSpaceParameter *param = reinterpret_cast<BatchToSpaceParameter *>(this->op_parameter_);
    if (param->no_crop_) {
      BatchToSpaceNoCropForNHWC(input_data, output_data, in_shape.data(), out_shape[0], param->block_shape_,
                                sizeof(float));
    } else {
      BatchToSpaceForNHWC(input_data, output_data, in_shape.data(), out_shape[0], param->block_shape_, param->crops_,
                          sizeof(float));
    }
  }
  if (in_tensors_.size() == 3) {
    auto ret = Processinput();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Processinput failed in BatchToSpace.";
      return ret;
    }
    if (no_crop_) {
      BatchToSpaceNoCropForNHWC(input_data, output_data, in_shape.data(), out_shape[0], block_shape_, sizeof(float));
    } else {
      BatchToSpaceForNHWC(input_data, output_data, in_shape.data(), out_shape[0], block_shape_, crops_, sizeof(float));
    }
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BatchToSpace, LiteKernelCreator<BatchToSpaceCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BatchToSpaceND, LiteKernelCreator<BatchToSpaceCPUKernel>)
}  // namespace mindspore::kernel
