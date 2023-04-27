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

#include <algorithm>
#include "src/extendrt/kernel/base_kernel.h"
#include "src/litert/cxx_api/tensor/tensor_impl.h"

namespace mindspore::kernel {
const std::vector<mindspore::MSTensor> &BaseKernel::inputs() {
  if (inputs_.empty()) {
    std::transform(in_tensors_.begin(), in_tensors_.end(), std::back_inserter(inputs_),
                   [](lite::Tensor *tensor) { return mindspore::MSTensor(std::make_shared<LiteTensorImpl>(tensor)); });
  }
  return inputs_;
}

const std::vector<mindspore::MSTensor> &BaseKernel::outputs() {
  if (outputs_.empty()) {
    std::transform(out_tensors_.begin(), out_tensors_.end(), std::back_inserter(outputs_),
                   [](lite::Tensor *tensor) { return mindspore::MSTensor(std::make_shared<LiteTensorImpl>(tensor)); });
  }
  return outputs_;
}

void BaseKernel::set_in_tensor(lite::Tensor *in_tensor, size_t index) {
  if (index >= in_tensors_.size()) {
    MS_LOG(ERROR) << "index: " << index << " larger than in_tensors size: " << in_tensors_.size();
    return;
  }
  this->in_tensors_[index] = in_tensor;
}

void BaseKernel::set_out_tensor(lite::Tensor *out_tensor, size_t index) {
  if (index >= out_tensors_.size()) {
    MS_LOG(ERROR) << "index: " << index << " larger than out_tensors size: " << out_tensors_.size();
    return;
  }
  this->out_tensors_[index] = out_tensor;
}
}  // namespace mindspore::kernel
