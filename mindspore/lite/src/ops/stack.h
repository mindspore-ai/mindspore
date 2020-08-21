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

#ifndef LITE_MINDSPORE_LITE_C_OPS_STACK_H_
#define LITE_MINDSPORE_LITE_C_OPS_STACK_H_

#include <vector>
#include <set>
#include <cmath>
#include "ir/dtype/type_id.h"
#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class Stack : public PrimitiveC {
 public:
#ifdef PRIMITIVE_WRITEABLE
  explicit Stack(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
#endif
  explicit Stack(schema::Primitive *primitive) : PrimitiveC(primitive) {}

  int InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) override;
  int GetAxis() const;
  int GetN() const;
  std::vector<int> GetIsScale() const;
  void SetAxis(int axis);
  void SetN(int n);
  void SetIsScale(const std::vector<int> &is_scale);
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_STACK_H_
