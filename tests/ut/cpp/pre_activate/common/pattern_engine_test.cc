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
#include <sstream>
#include <memory>
#include <algorithm>

#include "common/common_test.h"
#include "pre_activate/common/pattern_engine.h"
#include "pre_activate/common/visit.h"
#include "utils/base_ref.h"
#include "ir/anf.h"

namespace mindspore {
using PatternListType = std::initializer_list<BaseRef>;

bool Equal(const BaseRef &a, const BaseRef &b) { return a == b; }

class TestMatchEngine : public UT::Common {
 public:
  TestMatchEngine()
      : TU(std::make_shared<DefaultVisitor>(), std::function<bool(const BaseRef &, const BaseRef &)>(Equal)) {
    equiv_null = std::make_shared<Equiv>();
  };

 public:
  PatternEngine TU;
  EquivPtr equiv_null;
};

TEST_F(TestMatchEngine, Var) {
  VarPtr v1 = std::make_shared<Var>();
  VarPtr v2 = std::make_shared<Var>();
  VarPtr v3 = std::make_shared<Var>("name");
  ASSERT_TRUE(!(v1 == v2));
  ASSERT_TRUE(v1->matches(v2));
  ASSERT_TRUE(v1->matches(1));
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
  auto seq = Seq({1, 2, 3});
  MS_LOG(INFO) << "seq:" << seq.ToString();
  ASSERT_EQ(seq.ToString(), "vector[Int32Imm value:1, Int32Imm value:2, Int32Imm value:3]");
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
  d = TU.Match(v1, 1, equiv_null);
  ASSERT_EQ((*d)[v1], 1);

  equiv_null->clear();
  (*equiv_null)[v1] = v2;
  d = TU.Match(v1, 1, equiv_null);
  ASSERT_EQ(d->count(v2), std::size_t(1));
  ASSERT_EQ((*d)[v2], 1);

  equiv_null->clear();
  (*equiv_null)[v1] = v2;
  (*equiv_null)[v3] = 1;
  d = TU.Match(v1, 1, equiv_null);
  ASSERT_EQ(d->count(v2), std::size_t(1));
  ASSERT_EQ((*d)[v2], 1);

  equiv_null->clear();
  d = TU.Match(VectorRef({v1}), VectorRef({1}), equiv_null);
  ASSERT_EQ(d->size(), std::size_t(1));
  ASSERT_EQ(d->count(v1), std::size_t(1));
  ASSERT_EQ((*d)[v1], 1);

  equiv_null->clear();
  ASSERT_EQ(TU.Match(1, 2, equiv_null), nullptr);
}

TEST_F(TestMatchEngine, MatchRaw_SVar) {
  VarPtr v1 = std::make_shared<Var>();
  VarPtr sv1 = std::make_shared<SeqVar>();
  VarPtr sv2 = std::make_shared<SeqVar>();
  EquivPtr d;

  equiv_null->clear();
  d = TU.Match(VectorRef({sv1}), VectorRef({1, 2}), equiv_null);
  ASSERT_EQ(d->size(), std::size_t(1));
  ASSERT_EQ(d->count(sv1), std::size_t(1));
  ASSERT_EQ(utils::cast<Seq>((*d)[sv1]), Seq({1, 2}));

  equiv_null->clear();
  d = TU.Match(VectorRef({v1, sv1}), VectorRef({1, 2}), equiv_null);
  ASSERT_EQ(d->size(), std::size_t(2));
  ASSERT_EQ(utils::cast<Seq>((*d)[sv1]), Seq({2}));

  equiv_null->clear();
  ASSERT_EQ(TU.Match(VectorRef({sv1, sv2}), VectorRef({1, 2}), equiv_null), nullptr);

  equiv_null->clear();
  (*equiv_null)[sv1] = std::make_shared<Seq>(PatternListType{1, 2});
  d = TU.Match(VectorRef({v1, sv1}), VectorRef({1, 1, 2}), equiv_null);
  ASSERT_EQ(d->size(), std::size_t(2));
  ASSERT_EQ((*d)[v1], 1);
}

TEST_F(TestMatchEngine, Match) {
  VarPtr v1 = std::make_shared<Var>();
  VarPtr v2 = std::make_shared<Var>();
  VarPtr v3 = std::make_shared<Var>();

  EquivPtr d;

  equiv_null->clear();
  d = TU.Match(VectorRef({v1, v1, v2}), VectorRef({1, 1, 2}), equiv_null);
  ASSERT_EQ(d->size(), std::size_t(2));
  ASSERT_EQ((*d)[v1], 1);
  ASSERT_EQ((*d)[v2], 2);

  equiv_null->clear();
  d = TU.Match(static_cast<int>(1), static_cast<float>(1), equiv_null);
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
  d = TU.Match(VectorRef({vf, vn}), VectorRef({static_cast<float>(1.0), -1}), equiv_null);
  ASSERT_GE(d->size(), std::size_t(0));
  auto vfn = (*d)[vf];
  ASSERT_EQ((*d)[vf], static_cast<float>(1.0));
  ASSERT_EQ((*d)[vn], -1);

  equiv_null->clear();
  d = TU.Match(VectorRef({vf, vn}), VectorRef({1, static_cast<float>(-1.0)}), equiv_null);
  ASSERT_EQ(d, nullptr);

  equiv_null->clear();
  d = TU.Match(VectorRef({vf, vn}), VectorRef({static_cast<float>(1.0), static_cast<int>(1)}), equiv_null);
  ASSERT_EQ(d, nullptr);
}

TEST_F(TestMatchEngine, Match_Reify) {
  VarPtr v1 = std::make_shared<Var>();
  VarPtr sv = std::make_shared<SeqVar>();

  BaseRef t;

  equiv_null->clear();
  (*equiv_null)[sv] = BaseRef(std::make_shared<Seq>(PatternListType{3, 4}));
  t = TU.Replace(VectorRef({1, 2, sv}), equiv_null);
  ASSERT_EQ(t, BaseRef(VectorRef({1, 2, 3, 4})));
}

}  // namespace mindspore
