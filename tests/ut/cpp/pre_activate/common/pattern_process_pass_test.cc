/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <sstream>
#include <vector>
#include <memory>
#include <algorithm>
#include "common/common_test.h"
#define private public
#define protected public
#include "backend/common/optimizer/optimizer.h"
#undef private
#undef protected

#include "mindspore/core/ops/core_ops.h"
#include "ir/anf.h"
#include "ir/value.h"
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
class TestPass : public PatternProcessPass {
 public:
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const { return nullptr; };
};

class TestPatternProcessPass : public UT::Common {
 public:
  TestPatternProcessPass() : TU() { fg = std::make_shared<FuncGraph>(); };

 public:
  TestPass TU;
  FuncGraphPtr fg;
};

/// Feature: Backend support dump flag
/// Description: Get orig nodes according to primitive_vars_ and equiv_
/// Expectation: Get correct orig nodes
TEST_F(TestPatternProcessPass, test_GetOrigNodes) {
  TU.primitive_vars_->clear();
  TU.equiv_->clear();
  VarPtr mul1 = std::make_shared<Var>(std::make_shared<Primitive>(kMulOpName));
  VarPtr v1 = std::make_shared<Var>();
  VarPtr v2 = std::make_shared<Var>();
  (*TU.primitive_vars_)[mul1->primitive()] = mul1;

  auto mul1_node = std::make_shared<CNode>(std::vector<AnfNodePtr>{NewValueNode(prim::kPrimMul)}, fg);
  auto anode1 = std::make_shared<AnfNode>(fg);
  auto anode2 = std::make_shared<AnfNode>(fg);
  (*TU.equiv_)[mul1] = mul1_node;
  (*TU.equiv_)[v1] = anode1;
  (*TU.equiv_)[v2] = anode2;

  auto orig_nodes = TU.GetOrigNodes();
  ASSERT_EQ(orig_nodes.size(), std::size_t(1));
  ASSERT_EQ(orig_nodes[0], mul1_node);

  VarPtr mul2 = std::make_shared<Var>(std::make_shared<Primitive>(kMulOpName));
  (*TU.primitive_vars_)[mul2->primitive()] = mul2;
  orig_nodes = TU.GetOrigNodes();
  ASSERT_EQ(orig_nodes.size(), std::size_t(1));

  auto mul2_node = std::make_shared<CNode>(std::vector<AnfNodePtr>{NewValueNode(prim::kPrimMul)}, fg);
  (*TU.equiv_)[mul2] = mul2_node;
  orig_nodes = TU.GetOrigNodes();
  ASSERT_EQ(orig_nodes.size(), std::size_t(2));
}
}  // namespace opt
}  // namespace mindspore
