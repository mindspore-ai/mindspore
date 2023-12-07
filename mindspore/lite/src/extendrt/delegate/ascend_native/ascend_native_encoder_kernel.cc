/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "extendrt/delegate/ascend_native/ascend_native_encoder_kernel.h"
#include "extendrt/delegate/ascend_native/ascend_native_kernel_registry.h"
#include "ops/encoder_layer.h"

namespace mindspore::kernel {
using mindspore::ops::kNameEncoderLayer;

std::vector<int32_t> AscendNativeEncoderKernel::getOutputDimensions() {
  std::vector<int32_t> dims;
  return dims;
}

int AscendNativeEncoderKernel::InferShape() {
  out_tensors_[0]->set_shape(getOutputDimensions());
  out_tensors_[0]->set_data_type(TypeId::kNumberTypeFloat16);
  return kSuccess;
}

int AscendNativeEncoderKernel::InitEncoderParam() { return kSuccess; }

int AscendNativeEncoderKernel::Prepare() { return kSuccess; }

int AscendNativeEncoderKernel::Run() { return kSuccess; }

size_t AscendNativeEncoderKernel::get_workspace_size() const { return 0; }

REGISTER_ASCEND_NATIVE_CREATOR(kNameEncoderLayer, AscendNativeEncoderKernel)
}  // namespace mindspore::kernel
