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
#include "src/ops/dequant.h"
#include <vector>
#include <memory>

namespace mindspore {
namespace lite {
#ifdef PRIMITIVE_WRITEABLE
int Dequant::UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) {
  this->primitive_ = new (schema::PrimitiveT);
  auto attr = std::make_unique<schema::OnnxInt8DequantizeT>();
  this->primitive_->value.type = schema::PrimitiveType_OnnxInt8Dequantize;
  this->primitive_->value.value = attr.release();
  return RET_OK;
}
#endif
}  // namespace lite
}  // namespace mindspore
