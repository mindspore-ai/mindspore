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
#include "src/runtime/kernel/arm/fp32/depth_to_space_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_DepthToSpace;

namespace mindspore::kernel {
int DepthToSpaceCPUKernel::Init() {
  param_->data_type_size_ = sizeof(float);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int DepthToSpaceCPUKernel::ReSize() { return DepthToSpaceBaseCPUKernel::ReSize(); }

int DepthToSpaceCPUKernel::Run() {
  auto input = in_tensors_[0];
  auto output = out_tensors_[0];
  const float *input_data = reinterpret_cast<const float *>(input->data_c());
  float *output_data = reinterpret_cast<float *>(output->data_c());
  auto in_shape = input->shape();
  if (input->format() == schema::Format::Format_NHWC) {
    DepthToSpaceForNHWC(input_data, output_data, in_shape.data(), param_);
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Depth_to_space only support NHWC now!";
    return RET_ERROR;
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_DepthToSpace, LiteKernelCreator<DepthToSpaceCPUKernel>)
}  // namespace mindspore::kernel
