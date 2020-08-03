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
#include "src/runtime/kernel/arm/int8/depth_to_space_int8.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/opclib/depth_to_space.h"
#include "include/errorcode.h"

using mindspore::lite::RET_OK;
using mindspore::lite::RET_ERROR;

namespace mindspore::kernel {
int DepthToSpaceInt8CPUKernel::Init() {
  auto ret = DepthToSpaceBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  DepthToSpaceParameter *param = reinterpret_cast<DepthToSpaceParameter *>(opParameter);
  param->data_type_size_ = sizeof(int8_t);
  return RET_OK;
}

int DepthToSpaceInt8CPUKernel::Run() {
  auto input = inputs_[0];
  auto output = outputs_[0];
  const int8_t *input_data = reinterpret_cast<const int8_t *>(input->Data());
  int8_t *output_data = reinterpret_cast<int8_t *>(output->Data());
  auto in_shape = input->shape();
  DepthToSpaceParameter *param = reinterpret_cast<DepthToSpaceParameter *>(opParameter);
  if (input->GetFormat() == schema::Format_NHWC) {
    DepthToSpaceForNHWC(input_data, output_data, in_shape.data(), param);
    return RET_OK;
  } else {
    MS_LOG(ERROR) << "Depth_to_space only support NHWC now!";
    return RET_ERROR;
  }

  return RET_OK;
}
}  // namespace mindspore::kernel
