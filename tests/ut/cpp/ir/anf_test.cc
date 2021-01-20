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
#include <iostream>
#include <memory>

#include "common/common_test.h"

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "frontend/operator/ops.h"
#include "base/core_ops.h"

namespace mindspore {

using Named = Named;

class TestAnf : public UT::Common {
 public:
  TestAnf() {}
};

TEST_F(TestAnf, test_ValueNode) {
  auto prim = std::make_shared<Primitive>(prim::kScalarAdd);
  ValueNodePtr c = NewValueNode(prim);
  ASSERT_EQ(c->isa<ValueNode>(), true);
  ASSERT_EQ(IsValueNode<Primitive>(c), true);
  ASSERT_EQ(IsValueNode<FuncGraph>(c), false);

  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  ValueNode c1(fg);
  ASSERT_EQ(c1.value()->isa<FuncGraph>(), true);
}

TEST_F(TestAnf, test_Parameter) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  Parameter a(fg);
  assert(a.isa<Parameter>());
}

TEST_F(TestAnf, test_CNode) {
  auto primitive = prim::kPrimScalarAdd;

  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  std::string s = fg->ToString();

  Parameter param(fg);
  std::vector<AnfNodePtr> params;
  CNode app_1(params, fg);
  params.push_back(NewValueNode(primitive));
  params.push_back(AnfNodePtr(new Parameter(param)));
  CNode app(params, fg);
  assert(app.isa<CNode>());
  assert(app.IsApply(primitive));
}

TEST_F(TestAnf, is_exception) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  Parameter a(fg);
  assert(!a.isa<CNode>());
  assert(!a.isa<ValueNode>());
}

}  // namespace mindspore
