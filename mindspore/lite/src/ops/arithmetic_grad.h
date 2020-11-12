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

#ifndef MINDSPORE_LITE_SRC_OPS_ARITHMETIC_GRAD_H_
#define MINDSPORE_LITE_SRC_OPS_ARITHMETIC_GRAD_H_

#include <vector>
#include <set>
#include <cmath>
#include "src/ops/primitive_c.h"
#include "nnacl/arithmetic_self_parameter.h"

namespace mindspore {
namespace lite {
class ArithmeticGrad : public PrimitiveC {
 public:
  ArithmeticGrad() = default;
  ~ArithmeticGrad() = default;
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(ArithmeticGrad, PrimitiveC);
  explicit ArithmeticGrad(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
#else
  // explicit ArithmeticGrad(const schema::Primitive &primitive) : PrimitiveC(primitive) {}
  int UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) override {
    return RET_ERROR;
  }
#endif
  int InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) override;
  bool Broadcasting() { return this->broadcasting_; }
  int NDims() { return this->ndim_; }
  std::vector<int> dyShape() { return this->dy_shape_; }
  std::vector<int> x1Shape() { return this->x1_shape_; }
  std::vector<int> x2Shape() { return this->x2_shape_; }

 protected:
  bool broadcasting_ = false;
  int ndim_;
  std::vector<int> dy_shape_;
  std::vector<int> x1_shape_;
  std::vector<int> x2_shape_;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_OPS_ARITHMETIC_GRAD_H_
