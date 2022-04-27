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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_OPS_ARITHMETIC_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_OPS_ARITHMETIC_H_

#include <vector>
#include <memory>
#include "src/expression/net.h"
#include "inner/model_generated.h"

namespace mindspore {
namespace lite {
class ArithmeticM : public Node {
 public:
  ArithmeticM() = default;
  explicit ArithmeticM(schema::PrimitiveType type);

 protected:
  std::vector<EXPR *> binop_grad_common(EXPR *x, EXPR *y, EXPR *dx, EXPR *dy);
};

class AddM : public ArithmeticM {
 public:
  AddM() = default;
  explicit AddM(int dummy);
  std::vector<EXPR *> Grad(EXPR *expr) override;
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;
};

class DivM : public ArithmeticM {
 public:
  DivM() = default;
  explicit DivM(int dummy);
  std::vector<EXPR *> Grad(EXPR *expr) override;
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;
};

class MulM : public ArithmeticM {
 public:
  MulM() = default;
  explicit MulM(int dummy);
  std::vector<EXPR *> Grad(EXPR *expr) override;
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;
};

class SubM : public ArithmeticM {
 public:
  SubM() = default;
  explicit SubM(int dummy);
  std::vector<EXPR *> Grad(EXPR *expr) override;
  int UnPopulate(const std::unique_ptr<schema::CNodeT> &cnode) override;
};
namespace NN {
Node *Add();
Node *Sub();
Node *Mul();
Node *Div();
}  // namespace NN
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXPRESSION_OPS_ARITHMETIC_H_
