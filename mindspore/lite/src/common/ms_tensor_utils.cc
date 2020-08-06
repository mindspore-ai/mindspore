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

#include "src/common/ms_tensor_utils.h"

#include <vector>
#include "utils/log_adapter.h"

namespace mindspore {
namespace tensor {
using mindspore::lite::tensor::LiteTensor;
using mindspore::lite::tensor::Tensor;

std::vector<MSTensor *> PackToMSTensors(const std::vector<Tensor *> &in_tensors) {
  std::vector<MSTensor *> ret;
  for (auto *lite_tensor : in_tensors) {
    MS_ASSERT(lite_tensor != nullptr);
    auto *ms_tensor = new (std::nothrow) LiteTensor(lite_tensor);
    if (ms_tensor == nullptr) {
      MS_LOG(ERROR) << "new LiteTensor failed";
      return ret;
    }
    ret.emplace_back(ms_tensor);
  }
  return ret;
}
}  // namespace tensor
}  // namespace mindspore
