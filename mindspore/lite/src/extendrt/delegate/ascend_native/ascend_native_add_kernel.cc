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

#include "extendrt/delegate/ascend_native/ascend_native_add_kernel.h"
#include "extendrt/delegate/ascend_native/ascend_native_kernel_registry.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/add.h"
#include "ops/fusion/add_fusion.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore::kernel {
using mindspore::ops::kNameAddFusion;

int AscendNativeAddKernel::InferShape() {
  if (out_tensors_[0]->shape().size() == 0) {
    if (in_tensors_[0] != nullptr) {
      out_tensors_[0]->set_shape(in_tensors_[0]->shape());
    }
  }
  return kSuccess;
}

int AscendNativeAddKernel::Prepare() { return kSuccess; }

int AscendNativeAddKernel::Run() {
  MS_LOG(INFO) << "AscendNativeAddKernel::Execute";
  const std::vector<InferTensor *> &in_tensors = this->in_tensors();
  auto aBufSize = in_tensors[0]->ElementsNum();
  ascend_native::AddFp16(in_tensors[0]->device_data(), in_tensors[1]->device_data(), out_tensors()[0]->device_data(),
                         aBufSize, const_cast<void *>(get_stream()));
  return kSuccess;
}

int AscendNativeAddKernel::ReSize() { return kSuccess; }
REGISTER_ASCEND_NATIVE_CREATOR(kNameAddFusion, AscendNativeAddKernel)
}  // namespace mindspore::kernel
