/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ir/adapter_tensor.h"

#include <memory>
#include "abstract/utils.h"

namespace mindspore {
namespace tensor {
bool AdapterTensor::operator==(const AdapterTensor &other) const { return this == &other; }

abstract::AbstractBasePtr AdapterTensor::ToAbstract() {
  auto abs = origin_tensor_->ToAbstract();
  MS_EXCEPTION_IF_NULL(abs);
  auto tensor_abs = abs->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor_abs);
  tensor_abs->set_is_adapter(true);
  return tensor_abs;
}
}  // namespace tensor
}  // namespace mindspore
