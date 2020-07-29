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

#ifndef MINDSPORE_LITE_SRC_ANF_IMPORTER_PRIMITIVE_H_
#define MINDSPORE_LITE_SRC_ANF_IMPORTER_PRIMITIVE_H_

#include "ir/value.h"
#include "src/ops/ops.h"

namespace mindspore::lite {
class PrimitiveValue : public Value {
 public:
  explicit PrimitiveValue(const lite::Primitive *prim) : primitive(prim) {}

  const lite::Primitive *GetPrimitive() const {
    return this->primitive;
  }
  MS_DECLARE_PARENT(PrimitiveValue, Value)
  bool operator==(const Value &rhs) const override {
    if (rhs.isa<PrimitiveValue>()) {
      auto other_prim = static_cast<const PrimitiveValue &>(rhs);
      return *this == other_prim;
    } else {
      return false;
    }
  }

 protected:
  const lite::Primitive *primitive = nullptr;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_ANF_IMPORTER_PRIMITIVE_H_

