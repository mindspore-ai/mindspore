/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "ir/dtype/ref.h"
#include "ir/dtype/tensor_type.h"

namespace mindspore {
TypePtr RefType::DeepCopy() const {
  if (IsGeneric()) {
    return std::make_shared<RefType>();
  } else {
    auto subtype = TensorType::DeepCopy()->cast<TensorTypePtr>();
    return std::make_shared<RefType>(subtype);
  }
}

std::string RefType::ToString() const {
  std::ostringstream buffer;
  if (IsGeneric()) {
    buffer << "Ref";
  } else {
    buffer << "Ref[";
    buffer << TensorType::ToString() << "]";
  }
  return buffer.str();
}

std::string RefType::DumpText() const {
  std::ostringstream buffer;
  if (IsGeneric()) {
    buffer << "Ref";
  } else {
    buffer << "Ref[";
    buffer << TensorType::DumpText() << "]";
  }
  return buffer.str();
}
}  // namespace mindspore
