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

#include "ir/meta_tensor.h"

#include <functional>
#include <numeric>
#include <vector>
#include <sstream>
#include <string>

#include "pipeline/static_analysis/abstract_value.h"

namespace mindspore {
namespace tensor {
abstract::AbstractBasePtr MetaTensor::ToAbstract() {
  auto tens = shared_from_base<MetaTensor>();
  auto dtype = tens->Dtype();
  if (!IsSubType(dtype, kNumber)) {
    MS_LOG(EXCEPTION) << "Expect MetaTensor type kNumber but got: " << dtype->ToString() << ".";
  }
  auto tensor_shape = tens->shape();
  auto abs_tensor = std::make_shared<abstract::AbstractTensor>(dtype, tensor_shape);
  abs_tensor->set_value(shared_from_base<MetaTensor>());
  return abs_tensor;
}
}  // namespace tensor
}  // namespace mindspore
