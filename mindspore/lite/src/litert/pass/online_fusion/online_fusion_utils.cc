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

#include "src/litert/pass/online_fusion/online_fusion_utils.h"

namespace mindspore::lite {
bool IsIntScalarValue(lite::Tensor *tensor, int value) {
  if ((tensor->shape().size() == 0 || (tensor->shape().size() == 1 && tensor->shape().at(0) == 1)) &&
      (tensor->data_type() == TypeId::kNumberTypeInt32 || tensor->data_type() == TypeId::kNumberTypeInt)) {
    auto data = static_cast<int *>(tensor->data())[0];
    if (data != value) {
      return false;
    }
    return true;
  }
  return false;
}
}  // namespace mindspore::lite
