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
#include "src/runtime/kernel/arm/fp32/depth_to_space.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/opclib/arithmetic_common.h"
#include "src/runtime/kernel/arm/opclib/depth_to_space.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DepthToSpace;

namespace mindspore::kernel {

int DepthToSpaceCPUKernel::Init() {
  auto ret = DepthToSpaceBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  DepthToSpaceParameter *param = reinterpret_cast<DepthToSpaceParameter *>(opParameter);
  param->data_type_size_ = sizeof(float);
  return RET_OK;
}

int DepthToSpaceCPUKernel::Run() {
  auto input = inputs_[0];
  auto output = outputs_[0];
  const float *input_data = reinterpret_cast<const float *>(input->Data());
  float *output_data = reinterpret_cast<float *>(output->Data());
  auto in_shape = input->shape();
  DepthToSpaceParameter *param = reinterpret_cast<DepthToSpaceParameter *>(opParameter);
  if (input->GetFormat() == schema::Format_NHWC) {
    DepthToSpaceForNHWC(input_data, output_data, in_shape.data(), param);
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Depth_to_space only support NHWC now!";
    return RET_ERROR;
  }
}
}  // namespace mindspore::kernel
