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

#include "extendrt/delegate/ascend_native/stub_kernel.h"
#include "extendrt/delegate/ascend_native/ascend_native_kernel_registry.h"
#include "extendrt/delegate/ascend_native/ops/ascend_native_stub.h"

namespace mindspore::kernel {
using mindspore::ops::kNameAscendNativeStub;
int AscendNativeStubKernel::Prepare() {
  for (size_t i = 0; i < out_tensors_.size(); i++) {
    if (out_tensors_[i]->shape().size() == 0) {
      if (in_tensors_[i] != nullptr) {
        std::vector<int> shape;
        for (size_t j = 0; j < in_tensors_[i]->shape().size(); j++) {
          shape.push_back(in_tensors_[i]->shape()[j]);
        }
        out_tensors_[i]->set_shape(shape);
      }
    }
  }
  return kSuccess;
}

int AscendNativeStubKernel::Execute() {
  MS_LOG(INFO) << "AscendNativeStubKernel::Execute - " << this->get_name();
  return kSuccess;
}
REGISTER_ASCEND_NATIVE_CREATOR(kNameAscendNativeStub, AscendNativeStubKernel)
}  // namespace mindspore::kernel
