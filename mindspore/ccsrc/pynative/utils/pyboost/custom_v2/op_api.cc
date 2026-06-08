/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "include/pynative/utils/pyboost/custom_v2/op_api.h"

namespace ms {

void TensorAllocate(const ms::Tensor &output) {
  if (!output.is_defined()) {
    return;
  }
  auto *dc = GetCurrentDeviceContext();
  auto stream_id = static_cast<size_t>(mindspore::CurrentStream::id());
  mindspore::pyboost::CreateOutputTensorAddress(dc, stream_id, {output.tensor()});
}

void TensorAllocate(const std::vector<ms::Tensor> &outputs) {
  auto *dc = GetCurrentDeviceContext();
  auto stream_id = static_cast<size_t>(mindspore::CurrentStream::id());
  std::vector<mindspore::tensor::TensorPtr> tensor_ptrs;
  tensor_ptrs.reserve(outputs.size());
  for (const auto &t : outputs) {
    if (t.is_defined()) {
      tensor_ptrs.push_back(t.tensor());
    }
  }
  if (!tensor_ptrs.empty()) {
    mindspore::pyboost::CreateOutputTensorAddress(dc, stream_id, tensor_ptrs);
  }
}

}  // namespace ms
