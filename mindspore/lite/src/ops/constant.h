/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifdef PRIMITIVE_WRITEABLE
#ifndef LITE_MINDSPORE_LITE_C_OPS_CONSTANT_H_
#define LITE_MINDSPORE_LITE_C_OPS_CONSTANT_H_

#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class Constant : public PrimitiveC {
 public:
  Constant() = default;
  ~Constant() = default;
  MS_DECLARE_PARENT(Constant, PrimitiveC);
  explicit Constant(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_CONSTANT_H_
#endif
