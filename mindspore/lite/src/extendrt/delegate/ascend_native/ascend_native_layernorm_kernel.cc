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

#include "extendrt/delegate/ascend_native/ascend_native_layernorm_kernel.h"
#include "extendrt/delegate/ascend_native/ascend_native_kernel_registry.h"
#include "extendrt/delegate/ascend_native/ascend_native_impl/layernorm.h"
#include "ops/fusion/layer_norm_fusion.h"
namespace mindspore::kernel {
using mindspore::ops::kNameLayerNormFusion;

int AscendNativeLayernormKernel::InferShape() {
  for (size_t i = 0; i < out_tensors_.size(); i++) {
    if (out_tensors_[i]->shape().size() == 0) {
      if (in_tensors_[i] != nullptr) {
        out_tensors_[i]->set_shape(in_tensors_[i]->shape());
      }
    }
  }
  return kSuccess;
}

int AscendNativeLayernormKernel::Prepare() { return kSuccess; }

int AscendNativeLayernormKernel::Run() {
  MS_LOG(INFO) << "AscendNativeLayernormKernel::Execute";
  const std::vector<InferTensor *> &in_tensors = this->in_tensors();
  if (in_tensors.size() != THREE_TENSOR) {
    MS_LOG(ERROR) << "AscendNativeGatherKernel inputs number should be 3, instead got " << in_tensors.size();
    return kLiteError;
  }
  auto shape = in_tensors[0]->shape();
  uint64_t m = 1;
  for (unsigned int i = 0; i < shape.size() - 1; i++) {
    m *= shape.at(i);
  }
  uint64_t n = shape.at(shape.size() - 1);
  if (out_tensors_[0]->device_data() == nullptr) {
    out_tensors_[0]->set_device_data(ascend_native::MallocDevice(out_tensors_[0]->Size(), const_cast<void *>(stream_)));
  }
  ascend_native::LayerNormFp32(out_tensors()[0]->device_data(), in_tensors.at(FIRST_INPUT)->device_data(),
                               in_tensors.at(SECOND_INPUT)->device_data(), in_tensors_.at(THIRD_INPUT)->device_data(),
                               m, n, 1e-5f, const_cast<void *>(get_stream()));

  return kSuccess;
}

int AscendNativeLayernormKernel::ReSize() { return kSuccess; }

REGISTER_ASCEND_NATIVE_CREATOR(kNameLayerNormFusion, AscendNativeLayernormKernel)
}  // namespace mindspore::kernel
