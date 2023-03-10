/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "include/backend/optimizer/pattern_engine.h"
#include "include/backend/optimizer/visitor.h"
#include "include/backend/optimizer/helper.h"
#include "base/base_ref.h"
#include "mindspore/core/ops/core_ops.h"
#include "ir/anf.h"
#include "include/common/utils/utils.h"

namespace mindspore {
using PatternListType = std::initializer_list<BaseRef>;

std::shared_ptr<VectorRef> ExpandList(const std::vector<BaseRef> &list);

bool Equal(const BaseRef &a, const BaseRef &b) { return a == b; }

class TestMatchEngine : public UT::Common {
 public:
  TestMatchEngine() : TU(std::make_shared<Visitor>()) {
    equiv_null = std::make_shared<Equiv>();
    fg = std::make_shared<FuncGraph>();
  };

 public:
  PatternEngine TU;
  EquivPtr equiv_null;
  PrimitiveVarMap primitive_vars_null;
  FuncGraphPtr fg;
};

TEST_F(TestMatchEngine, Var) {
  VarPtr v1 = std::make_shared<Var>();
  VarPtr v2 = std::make_shared<Var>();
  VarPtr v3 = std::make_shared<Var>("name");
  ASSERT_TRUE(!(v1 == v2));
  ASSERT_TRUE(v1->matches(v2));
  ASSERT_TRUE(v1->matches(static_cast<int64_t>(1)));
  ASSERT_EQ(v3->ToString(), "Var(name)");
  ASSERT_NE(v1->tag(), v2->tag());
}

TEST_F(TestMatchEngine, CondVar) {
  auto Pos = static_cast<bool (*)(const BaseRef &)>([](const BaseRef &any) -> bool {
    float v = 0;
    if (utils::isa<float>(any)) {
      v = utils::cast<float>(any);
    } else if (utils::isa<int>(any)) {
      v = utils::cast<int>(any);
    } else if (utils::isa<int64_t>(any)) {
      v = utils::cast<int64_t>(any);
    } else {
      return false;
    }
    return v > 0;
  });

  VarPtr fv1 = std::make_shared<CondVar>(Pos);

  ASSERT_TRUE(fv1->matches(1.0f));
  ASSERT_FALSE(fv1->matches(0.0f));
}

TEST_F(TestMatchEngine, Seq) {
  auto seq = Seq({static_cast<int64_t>(1), static_cast<int64_t>(2), static_cast<int64_t>(3)});
  MS_LOG(INFO) << "seq:" << seq.ToString();
  ASSERT_EQ(seq.ToString(), "vector[Int64Imm value:1, Int64Imm value:2, Int64Imm value:3]");
}

TEST_F(TestMatchEngine, SeqVar) {
  VarPtr sv1 = std::make_shared<SeqVar>();
  auto seq1 = std::make_shared<Seq>(PatternListType({1, 2}));
  ASSERT_FALSE(sv1->matches(1));
  ASSERT_FALSE(sv1->matches(1.0f));

  ASSERT_TRUE(sv1->matches(seq1));

  std::cout << sv1->ToString() << std::endl;
}

TEST_F(TestMatchEngine, ExpandList) {
  auto v1 = VectorRef({1, 2, 3});
  auto v2 = VectorRef({1, PatternListType({2, 3, 4}), 5});
  auto p1 = ExpandList(v1.elements());
  auto p2 = ExpandList(v2.elements());
  ASSERT_EQ(*p1, VectorRef({1, 2, 3}));
  ASSERT_EQ(*p2, VectorRef({1, 2, 3, 4, 5}));
}

TEST_F(TestMatchEngine, MatchRaw_Var) {
  VarPtr v1 = std::make_shared<Var>();
  VarPtr v2 = std::make_shared<Var>();
  VarPtr v3 = std::make_shared<Var>();
  EquivPtr d;

  // common
  equiv_null->clear();
  d = TU.Match(v1, 1, primitive_vars_null, equiv_null);
  ASSERT_EQ((*d)[v1], 1);

  equiv_null->clear();
  (*equiv_null)[v1] = v2;
  d = TU.Match(v1, 1, primitive_vars_null, equiv_null);
  ASSERT_EQ(d->count(v2), std::size_t(1));
  ASSERT_EQ((*d)[v2], 1);

  equiv_null->clear();
  (*equiv_null)[v1] = v2;
  (*equiv_null)[v3] = 1;
  d = TU.Match(v1, 1, primitive_vars_null, equiv_null);
  ASSERT_EQ(d->count(v2), std::size_t(1));
  ASSERT_EQ((*d)[v2], 1);

  equiv_null->clear();
  d = TU.Match(VectorRef({v1}), VectorRef({1}), primitive_vars_null, equiv_null);
  ASSERT_EQ(d->size(), std::size_t(1));
  ASSERT_EQ(d->count(v1), std::size_t(1));
  ASSERT_EQ((*d)[v1], 1);

  equiv_null->clear();
  ASSERT_EQ(TU.Match(1, 2, primitive_vars_null, equiv_null), nullptr);
}

TEST_F(TestMatchEngine, MatchRaw_SVar) {
  VarPtr v1 = std::make_shared<Var>();
  VarPtr sv1 = std::make_shared<SeqVar>();
  VarPtr sv2 = std::make_shared<SeqVar>();
  EquivPtr d;

  equiv_null->clear();
  d = TU.Match(VectorRef({sv1}), VectorRef({1, 2}), primitive_vars_null, equiv_null);
  ASSERT_EQ(d->size(), std::size_t(1));
  ASSERT_EQ(d->count(sv1), std::size_t(1));
  ASSERT_EQ(utils::cast<Seq>((*d)[sv1]), Seq({1, 2}));

  equiv_null->clear();
  d = TU.Match(VectorRef({v1, sv1}), VectorRef({1, 2}), primitive_vars_null, equiv_null);
  ASSERT_EQ(d->size(), std::size_t(2));
  ASSERT_EQ(utils::cast<Seq>((*d)[sv1]), Seq({2}));

  equiv_null->clear();
  ASSERT_EQ(TU.Match(VectorRef({sv1, sv2}), VectorRef({1, 2}), primitive_vars_null, equiv_null), nullptr);

  equiv_null->clear();
  (*equiv_null)[sv1] = std::make_shared<Seq>(PatternListType{1, 2});
  d = TU.Match(VectorRef({v1, sv1}), VectorRef({1, 1, 2}), primitive_vars_null, equiv_null);
  ASSERT_EQ(d->size(), std::size_t(2));
  ASSERT_EQ((*d)[v1], 1);
}

TEST_F(TestMatchEngine, Match) {
  VarPtr v1 = std::make_shared<Var>();
  VarPtr v2 = std::make_shared<Var>();
  VarPtr v3 = std::make_shared<Var>();

  EquivPtr d;

  equiv_null->clear();
  d = TU.Match(VectorRef({v1, v1, v2}), VectorRef({1, 1, 2}), primitive_vars_null, equiv_null);
  ASSERT_EQ(d->size(), std::size_t(2));
  ASSERT_EQ((*d)[v1], 1);
  ASSERT_EQ((*d)[v2], 2);

  equiv_null->clear();
  d = TU.Match(static_cast<int>(1), static_cast<float>(1), primitive_vars_null, equiv_null);
  ASSERT_EQ(d, nullptr);
}

TEST_F(TestMatchEngine, Match_CondVar) {
  auto floats =
    static_cast<bool (*)(const BaseRef &)>([](const BaseRef &any) -> bool { return utils::isa<float>(any); });
  auto neg = static_cast<bool (*)(const BaseRef &)>([](const BaseRef &any) -> bool {
    float v = 0;
    if (utils::isa<float>(any)) {
      v = utils::cast<float>(any);
    } else if (utils::isa<int>(any)) {
      v = utils::cast<int>(any);
    } else {
      return false;
    }
    return v < 0;
  });

  VarPtr vf = std::make_shared<CondVar>(floats);
  VarPtr vn = std::make_shared<CondVar>(neg);
  EquivPtr d;

  equiv_null->clear();
  d = TU.Match(VectorRef({vf, vn}), VectorRef({static_cast<float>(1.0), -1}), primitive_vars_null, equiv_null);
  ASSERT_GE(d->size(), std::size_t(0));
  auto vfn = (*d)[vf];
  ASSERT_EQ((*d)[vf], static_cast<float>(1.0));
  ASSERT_EQ((*d)[vn], -1);

  equiv_null->clear();
  d = TU.Match(VectorRef({vf, vn}), VectorRef({1, static_cast<float>(-1.0)}), primitive_vars_null, equiv_null);
  ASSERT_EQ(d, nullptr);

  equiv_null->clear();
  d = TU.Match(VectorRef({vf, vn}), VectorRef({static_cast<float>(1.0), static_cast<int>(1)}), primitive_vars_null,
               equiv_null);
  ASSERT_EQ(d, nullptr);
}

/// Feature: Backend support dump flag
/// Description: PatternEngine match var with primitive
/// Expectation: Get correct Equiv map
TEST_F(TestMatchEngine, Match_PrimVar) {
  VarPtr mul1 = std::make_shared<Var>(std::make_shared<Primitive>(kMulOpName));
  VarPtr mul2 = std::make_shared<Var>(std::make_shared<Primitive>(kMulOpName));
  VarPtr v1 = std::make_shared<Var>();
  VarPtr sv2 = std::make_shared<SeqVar>();
  auto pattern_ref = VectorRef({mul1, v1, VectorRef({mul2, sv2})});
  PrimitiveVarMapPtr primitive_vars = std::make_shared<PrimitiveVarMap>();
  auto pattern_node = opt::SexpToNode(pattern_ref, fg, primitive_vars.get(), true);
  ASSERT_EQ(primitive_vars->size(), std::size_t(2));

  auto anode1 = std::make_shared<AnfNode>(fg);
  auto anode2 = std::make_shared<AnfNode>(fg);
  auto anode3 = std::make_shared<AnfNode>(fg);
  AnfNodePtr mul2_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode2, anode3}, fg);
  AnfNodePtr mul1_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode1, mul2_cnode}, fg);

  EquivPtr d;
  equiv_null->clear();
  d = TU.Match(pattern_node, mul1_cnode, *primitive_vars, equiv_null);
  ASSERT_EQ(d->size(), std::size_t(4));
  ASSERT_EQ((*d)[mul2], mul2_cnode);
  ASSERT_EQ((*d)[mul1], mul1_cnode);

  AnfNodePtr sub_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kSubOpName)), anode1, mul2_cnode}, fg);

  equiv_null->clear();
  d = TU.Match(pattern_node, sub_cnode, *primitive_vars, equiv_null);
  ASSERT_EQ(d, nullptr);
}

/// Feature: Backend support dump flag
/// Description: PatternEngine match primitive
/// Expectation: Get correct Equiv map
TEST_F(TestMatchEngine, Match_Prim) {
  VarPtr v1 = std::make_shared<Var>();
  VarPtr sv2 = std::make_shared<SeqVar>();
  auto pattern_ref = VectorRef({prim::kPrimMul, v1, VectorRef({prim::kPrimMul, sv2})});
  PrimitiveVarMapPtr primitive_vars = std::make_shared<PrimitiveVarMap>();
  auto pattern_node = opt::SexpToNode(pattern_ref, fg, primitive_vars.get(), true);
  ASSERT_EQ(primitive_vars->size(), std::size_t(2));

  auto anode1 = std::make_shared<AnfNode>(fg);
  auto anode2 = std::make_shared<AnfNode>(fg);
  auto anode3 = std::make_shared<AnfNode>(fg);
  AnfNodePtr mul2_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode2, anode3}, fg);
  AnfNodePtr mul1_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode1, mul2_cnode}, fg);

  EquivPtr d;
  equiv_null->clear();
  d = TU.Match(pattern_node, mul1_cnode, *primitive_vars, equiv_null);
  ASSERT_EQ(d->size(), std::size_t(4));
  for (auto &prim_var : *primitive_vars) {
    if (prim_var.first == prim::kPrimMul) {
      ASSERT_EQ((*d)[prim_var.second], mul1_cnode);
    } else {
      ASSERT_EQ((*d)[prim_var.second], mul2_cnode);
    }
  }

  AnfNodePtr sub_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kSubOpName)), anode1, mul2_cnode}, fg);

  equiv_null->clear();
  d = TU.Match(pattern_node, sub_cnode, *primitive_vars, equiv_null);
  ASSERT_EQ(d, nullptr);
}
}  // namespace mindspore
