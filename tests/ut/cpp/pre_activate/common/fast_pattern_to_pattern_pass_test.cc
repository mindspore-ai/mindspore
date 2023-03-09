/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "pattern_to_pattern_pass_utils.h"
#include "include/backend/optimizer/node_pass.h"

namespace mindspore {
namespace opt {
namespace {
const auto kZero = 0;
const auto kOne = 1;
const auto kTwo = 2;
const auto kThree = 3;

const auto kA = "a";
const auto kB = "b";
const auto kC = "c";
const auto kD = "d";
const auto kE = "e";
const auto kAAddB = "a_add_b";
const auto kCAddD = "c_add_d";
const auto kMul = "mul";
const auto kAdd = "add";

class TestFastMul0 : public PatternToPatternPass {
  // a*b + a*c -> a*(b+c)
 public:
  explicit TestFastMul0() : PatternToPatternPass("test_fast_mul0") {}
  ~TestFastMul0() override = default;

  void DefineSrcPattern(SrcPattern *src_pattern) override {
    (*src_pattern)
      .AddVar("a")
      .AddVar("b")
      .AddVar("c")
      .AddCNode("ab", {std::make_shared<Primitive>(kMulOpName), "a", "b"})
      .AddCNode("ac", {std::make_shared<Primitive>(kMulOpName), "a", "c"})
      .AddCNode("add", {std::make_shared<Primitive>(kAddOpName), "ab", "ac"});
  }
  void DefineDstPattern(DstPattern *dst_pattern) override {
    (*dst_pattern)
      .AddCNode("bc", {std::make_shared<Primitive>(kAddOpName), "b", "c"})
      .AddCNode("mul", {std::make_shared<Primitive>(kMulOpName), "a", "bc"});
  }
  bool CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const override { return true; }
};

class TestFastMul1 : public PatternToPatternPass {
  // a*b + c*d -> a*c
 public:
  explicit TestFastMul1() : PatternToPatternPass("test_fast_mul1") {}
  ~TestFastMul1() override = default;

  void DefineSrcPattern(SrcPattern *src_pattern) override {
    (*src_pattern)
      .AddVar("a")
      .AddVar("b")
      .AddVar("c")
      .AddVar("d")
      .AddCNode("ab", {std::make_shared<Primitive>(kMulOpName), "a", "b"})
      .AddCNode("cd", {std::make_shared<Primitive>(kMulOpName), "c", "d"})
      .AddCNode("add", {std::make_shared<Primitive>(kAddOpName), "ab", "cd"});
  }
  void DefineDstPattern(DstPattern *dst_pattern) override {
    (*dst_pattern).AddCNode("ad", {std::make_shared<Primitive>(kMulOpName), "a", "d"});
  }
  bool CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const override { return true; }
};

class TestFastMul2 : public PatternToPatternPass {
  // a*b -> b*a
 public:
  explicit TestFastMul2() : PatternToPatternPass("test_fast_mul2") {}
  ~TestFastMul2() override = default;

  void DefineSrcPattern(SrcPattern *src_pattern) override {
    (*src_pattern).AddSeqVar("Sv").AddCNode("ab", {std::make_shared<Primitive>(kMulOpName), "Sv"});
  }
  void DefineDstPattern(DstPattern *dst_pattern) override {
    auto ba = Unpacking("Sv");
    auto ab = Unpacking("Sv");
    ba[0] = ab[1];
    ba[1] = ab[0];
    (*dst_pattern).AddCNode("mul", {std::make_shared<Primitive>(kMulOpName), ba});
  }
  bool CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const override { return true; }
};
}  // namespace

class TestFastPatternToPatternPass : public UT::Common {
 public:
  TestFastPatternToPatternPass() : fg_(std::make_shared<FuncGraph>()){};

 public:
  FuncGraphPtr fg_;
};

/// Feature: Fast PatternToPattern Pass
/// Description: Fast PatternToPattern Pass rewrite graph
/// Expectation: Get correct Graph
TEST_F(TestFastPatternToPatternPass, Mul0) {
  // a*b + a*c -> a*(b+c)
  // init
  auto check = CheckPattern();
  auto pass = TestFastMul0();

  // build func graph
  auto a = std::make_shared<AnfNode>(fg_);
  auto b = std::make_shared<AnfNode>(fg_);
  auto c = std::make_shared<AnfNode>(fg_);
  AnfNodePtr ab =
    std::make_shared<CNode>(std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), a, b}, fg_);
  AnfNodePtr ac =
    std::make_shared<CNode>(std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), a, c}, fg_);
  AnfNodePtr add = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kAddOpName)), ab, ac}, fg_);

  fg_->set_output(add);
  auto manager = MakeManager({fg_});
  if (manager) {
    manager->AddFuncGraph(fg_);
    fg_->set_manager(manager);
  }
  auto func_graph_index = manager->func_graph_index(fg_);
  GenIndex(fg_, func_graph_index);

  ASSERT_TRUE(func_graph_index->node_degree_.at(add) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(ab) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(ac) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(c) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(b) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(a) == 2);

  ASSERT_TRUE(func_graph_index->name_to_cnode_.size() == 2);
  ASSERT_TRUE(func_graph_index->name_to_cnode_.find(kAddOpName) != func_graph_index->name_to_cnode_.end());
  ASSERT_TRUE(func_graph_index->name_to_cnode_.find(kMulOpName) != func_graph_index->name_to_cnode_.end());

  auto &add_set = func_graph_index->name_to_cnode_[kAddOpName];
  auto &mul_set = func_graph_index->name_to_cnode_[kMulOpName];

  ASSERT_TRUE(add_set.size() == 1);
  ASSERT_TRUE(mul_set.size() == 2);
  ASSERT_TRUE(add_set.find(add) != add_set.end());
  ASSERT_TRUE(mul_set.find(ab) != mul_set.end());
  ASSERT_TRUE(mul_set.find(ac) != mul_set.end());

  auto new_node = pass.Run(fg_, add);
  ASSERT_NE(new_node, nullptr);
  (void)manager->Replace(add, new_node);
  pass.AfterProcess(add, new_node, fg_, func_graph_index);

  ASSERT_TRUE(func_graph_index->node_degree_.at(add) == 0);
  ASSERT_TRUE(func_graph_index->node_degree_.at(ab) == 0);
  ASSERT_TRUE(func_graph_index->node_degree_.at(ac) == 0);
  ASSERT_TRUE(func_graph_index->node_degree_.at(c) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(b) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(a) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(pass.m_->Get("bc")) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(pass.m_->Get("mul")) == 1);

  ASSERT_TRUE(func_graph_index->name_to_cnode_.size() == 2);
  ASSERT_TRUE(func_graph_index->name_to_cnode_.find(kAddOpName) != func_graph_index->name_to_cnode_.end());
  ASSERT_TRUE(func_graph_index->name_to_cnode_.find(kMulOpName) != func_graph_index->name_to_cnode_.end());

  auto &add_set_2 = func_graph_index->name_to_cnode_[kAddOpName];
  auto &mul_set_2 = func_graph_index->name_to_cnode_[kMulOpName];

  ASSERT_TRUE(add_set_2.size() == 1);
  ASSERT_TRUE(mul_set_2.size() == 1);
  ASSERT_TRUE(add_set_2.find(pass.m_->Get("bc")) != add_set_2.end());
  ASSERT_TRUE(mul_set_2.find(pass.m_->Get("mul")) != mul_set_2.end());

  // build pattern
  check.src_pattern_.AddVar("a")
    .AddVar("b")
    .AddVar("c")
    .AddCNode("bc", {std::make_shared<Primitive>(kAddOpName), "b", "c"})
    .AddCNode("mul", {std::make_shared<Primitive>(kMulOpName), "a", "bc"});

  // pattern engine
  ASSERT_TRUE(check.build_pattern_map(new_node));

  // check
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("a"), a));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("b"), b));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("c"), c));
  ASSERT_EQ(check.m_->Get("bc")->cast<CNodePtr>()->inputs().size(), 3);
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("bc")->cast<CNodePtr>()->input(0),
                            NewValueNode(std::make_shared<Primitive>(kAddOpName))));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("bc")->cast<CNodePtr>()->input(1), b));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("bc")->cast<CNodePtr>()->input(2), c));
  ASSERT_EQ(check.m_->Get("mul")->cast<CNodePtr>()->inputs().size(), 3);
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("mul")->cast<CNodePtr>()->input(0),
                            NewValueNode(std::make_shared<Primitive>(kMulOpName))));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("mul")->cast<CNodePtr>()->input(1), a));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("mul")->cast<CNodePtr>()->input(2), check.m_->Get("bc")));
}

/// Feature: Fast PatternToPattern Pass
/// Description: Fast PatternToPattern Pass rewrite graph
/// Expectation: Get correct Graph
TEST_F(TestFastPatternToPatternPass, Mul0NotRoot) {
  // (a*b + a*c) + d -> a*(b+c) + d
  // init
  auto check = CheckPattern();
  auto pass = TestFastMul0();

  // build func graph
  auto a = std::make_shared<AnfNode>(fg_);
  auto b = std::make_shared<AnfNode>(fg_);
  auto c = std::make_shared<AnfNode>(fg_);
  auto d = std::make_shared<AnfNode>(fg_);
  AnfNodePtr ab =
    std::make_shared<CNode>(std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), a, b}, fg_);
  AnfNodePtr ac =
    std::make_shared<CNode>(std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), a, c}, fg_);
  AnfNodePtr add = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kAddOpName)), ab, ac}, fg_);
  AnfNodePtr add1 = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kAddOpName)), add, d}, fg_);

  fg_->set_output(add1);
  auto manager = MakeManager({fg_});
  if (manager) {
    manager->AddFuncGraph(fg_);
    fg_->set_manager(manager);
  }
  auto func_graph_index = manager->func_graph_index(fg_);
  GenIndex(fg_, func_graph_index);

  ASSERT_TRUE(func_graph_index->node_degree_.at(add1) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(add) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(ab) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(ac) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(d) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(c) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(b) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(a) == 2);

  ASSERT_TRUE(func_graph_index->name_to_cnode_.size() == 2);
  ASSERT_TRUE(func_graph_index->name_to_cnode_.find(kAddOpName) != func_graph_index->name_to_cnode_.end());
  ASSERT_TRUE(func_graph_index->name_to_cnode_.find(kMulOpName) != func_graph_index->name_to_cnode_.end());

  auto &add_set = func_graph_index->name_to_cnode_[kAddOpName];
  auto &mul_set = func_graph_index->name_to_cnode_[kMulOpName];

  ASSERT_TRUE(add_set.size() == 2);
  ASSERT_TRUE(mul_set.size() == 2);
  ASSERT_TRUE(add_set.find(add1) != add_set.end());
  ASSERT_TRUE(add_set.find(add) != add_set.end());
  ASSERT_TRUE(mul_set.find(ab) != mul_set.end());
  ASSERT_TRUE(mul_set.find(ac) != mul_set.end());

  auto new_node = pass.Run(fg_, add);
  ASSERT_NE(new_node, nullptr);
  (void)manager->Replace(add, new_node);
  pass.AfterProcess(add, new_node, fg_, func_graph_index);

  ASSERT_TRUE(func_graph_index->node_degree_.at(add) == 0);
  ASSERT_TRUE(func_graph_index->node_degree_.at(ab) == 0);
  ASSERT_TRUE(func_graph_index->node_degree_.at(ac) == 0);
  ASSERT_TRUE(func_graph_index->node_degree_.at(add1) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(d) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(c) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(b) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(a) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(pass.m_->Get("bc")) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(pass.m_->Get("mul")) == 1);

  ASSERT_TRUE(func_graph_index->name_to_cnode_.size() == 2);
  ASSERT_TRUE(func_graph_index->name_to_cnode_.find(kAddOpName) != func_graph_index->name_to_cnode_.end());
  ASSERT_TRUE(func_graph_index->name_to_cnode_.find(kMulOpName) != func_graph_index->name_to_cnode_.end());

  auto &add_set_2 = func_graph_index->name_to_cnode_[kAddOpName];
  auto &mul_set_2 = func_graph_index->name_to_cnode_[kMulOpName];

  ASSERT_TRUE(add_set_2.size() == 2);
  ASSERT_TRUE(mul_set_2.size() == 1);
  ASSERT_TRUE(add_set_2.find(add1) != add_set_2.end());
  ASSERT_TRUE(add_set_2.find(pass.m_->Get("bc")) != add_set_2.end());
  ASSERT_TRUE(mul_set_2.find(pass.m_->Get("mul")) != mul_set_2.end());

  // build pattern
  check.src_pattern_.AddVar("a")
    .AddVar("b")
    .AddVar("c")
    .AddVar("d")
    .AddCNode("bc", {std::make_shared<Primitive>(kAddOpName), "b", "c"})
    .AddCNode("mul", {std::make_shared<Primitive>(kMulOpName), "a", "bc"})
    .AddCNode("add1", {std::make_shared<Primitive>(kAddOpName), "mul", "d"});

  // pattern engine
  ASSERT_TRUE(check.build_pattern_map(add1));

  // check
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("a"), a));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("b"), b));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("c"), c));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("d"), d));

  ASSERT_EQ(check.m_->Get("bc")->cast<CNodePtr>()->inputs().size(), 3);
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("bc")->cast<CNodePtr>()->input(0),
                            NewValueNode(std::make_shared<Primitive>(kAddOpName))));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("bc")->cast<CNodePtr>()->input(1), b));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("bc")->cast<CNodePtr>()->input(2), c));

  ASSERT_EQ(check.m_->Get("mul")->cast<CNodePtr>()->inputs().size(), 3);
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("mul")->cast<CNodePtr>()->input(0),
                            NewValueNode(std::make_shared<Primitive>(kMulOpName))));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("mul")->cast<CNodePtr>()->input(1), a));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("mul")->cast<CNodePtr>()->input(2), check.m_->Get("bc")));

  ASSERT_EQ(check.m_->Get("add1")->cast<CNodePtr>()->inputs().size(), 3);
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("add1")->cast<CNodePtr>()->input(0),
                            NewValueNode(std::make_shared<Primitive>(kAddOpName))));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("add1")->cast<CNodePtr>()->input(1), check.m_->Get("mul")));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("add1")->cast<CNodePtr>()->input(2), d));
}

/// Feature: Fast PatternToPattern Pass
/// Description: Fast PatternToPattern Pass rewrite graph
/// Expectation: Get correct Graph
TEST_F(TestFastPatternToPatternPass, Mul1) {
  // (a * (b1 + d) + (c1 * c2) * d) + e -> (a + d) + e
  // init
  auto check = CheckPattern();
  auto pass = TestFastMul1();

  // build func graph
  auto a = std::make_shared<AnfNode>(fg_);
  auto b = std::make_shared<AnfNode>(fg_);
  auto c1 = std::make_shared<AnfNode>(fg_);
  auto c2 = std::make_shared<AnfNode>(fg_);
  auto d = std::make_shared<AnfNode>(fg_);
  auto e = std::make_shared<AnfNode>(fg_);

  AnfNodePtr b_add_d =
    std::make_shared<CNode>(std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kAddOpName)), b, d}, fg_);
  AnfNodePtr c1_mul_c2 = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), c1, c2}, fg_);
  AnfNodePtr a_mul = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), a, b_add_d}, fg_);
  AnfNodePtr d_mul = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), c1_mul_c2, d}, fg_);
  AnfNodePtr add = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kAddOpName)), a_mul, d_mul}, fg_);
  AnfNodePtr add1 = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kAddOpName)), add, e}, fg_);

  fg_->set_output(add1);
  auto manager = MakeManager({fg_});
  if (manager) {
    manager->AddFuncGraph(fg_);
    fg_->set_manager(manager);
  }
  auto func_graph_index = manager->func_graph_index(fg_);
  GenIndex(fg_, func_graph_index);

  ASSERT_TRUE(func_graph_index->node_degree_.at(b_add_d) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(c1_mul_c2) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(a_mul) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(d_mul) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(add) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(add1) == 1);

  ASSERT_TRUE(func_graph_index->node_degree_.at(a) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(b) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(c1) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(c2) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(d) == 2);
  ASSERT_TRUE(func_graph_index->node_degree_.at(e) == 1);

  ASSERT_TRUE(func_graph_index->name_to_cnode_.size() == 2);
  ASSERT_TRUE(func_graph_index->name_to_cnode_.find(kAddOpName) != func_graph_index->name_to_cnode_.end());
  ASSERT_TRUE(func_graph_index->name_to_cnode_.find(kMulOpName) != func_graph_index->name_to_cnode_.end());

  auto &add_set = func_graph_index->name_to_cnode_[kAddOpName];
  auto &mul_set = func_graph_index->name_to_cnode_[kMulOpName];

  ASSERT_TRUE(add_set.size() == 3);
  ASSERT_TRUE(mul_set.size() == 3);
  ASSERT_TRUE(add_set.find(add1) != add_set.end());
  ASSERT_TRUE(add_set.find(add) != add_set.end());
  ASSERT_TRUE(add_set.find(b_add_d) != add_set.end());
  ASSERT_TRUE(mul_set.find(a_mul) != mul_set.end());
  ASSERT_TRUE(mul_set.find(d_mul) != mul_set.end());
  ASSERT_TRUE(mul_set.find(c1_mul_c2) != mul_set.end());

  auto new_node = pass.Run(fg_, add);
  ASSERT_NE(new_node, nullptr);
  (void)manager->Replace(add, new_node);
  pass.AfterProcess(add, new_node, fg_, func_graph_index);

  ASSERT_TRUE(func_graph_index->node_degree_.at(b_add_d) == 0);
  ASSERT_TRUE(func_graph_index->node_degree_.at(c1_mul_c2) == 0);
  ASSERT_TRUE(func_graph_index->node_degree_.at(a_mul) == 0);
  ASSERT_TRUE(func_graph_index->node_degree_.at(d_mul) == 0);
  ASSERT_TRUE(func_graph_index->node_degree_.at(add) == 0);
  ASSERT_TRUE(func_graph_index->node_degree_.at(add1) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(pass.m_->Get("ad")) == 1);

  ASSERT_TRUE(func_graph_index->node_degree_.at(a) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(b) == 0);
  ASSERT_TRUE(func_graph_index->node_degree_.at(c1) == 0);
  ASSERT_TRUE(func_graph_index->node_degree_.at(c2) == 0);
  ASSERT_TRUE(func_graph_index->node_degree_.at(d) == 1);
  ASSERT_TRUE(func_graph_index->node_degree_.at(e) == 1);

  ASSERT_TRUE(func_graph_index->name_to_cnode_.size() == 2);
  ASSERT_TRUE(func_graph_index->name_to_cnode_.find(kAddOpName) != func_graph_index->name_to_cnode_.end());
  ASSERT_TRUE(func_graph_index->name_to_cnode_.find(kMulOpName) != func_graph_index->name_to_cnode_.end());

  auto &add_set_2 = func_graph_index->name_to_cnode_[kAddOpName];
  auto &mul_set_2 = func_graph_index->name_to_cnode_[kMulOpName];

  ASSERT_TRUE(add_set_2.size() == 1);
  ASSERT_TRUE(mul_set_2.size() == 1);
  ASSERT_TRUE(add_set_2.find(add1) != add_set_2.end());
  ASSERT_TRUE(mul_set_2.find(pass.m_->Get("ad")) != mul_set_2.end());

  // build pattern
  check.src_pattern_.AddVar("a")
    .AddVar("d")
    .AddVar("e")
    .AddCNode("ad", {std::make_shared<Primitive>(kMulOpName), "a", "d"})
    .AddCNode("add1", {std::make_shared<Primitive>(kAddOpName), "ad", "e"});

  // pattern engine
  ASSERT_TRUE(check.build_pattern_map(add1));

  // check
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("a"), a));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("d"), d));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("e"), e));

  ASSERT_EQ(check.m_->Get("ad")->cast<CNodePtr>()->inputs().size(), 3);
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("ad")->cast<CNodePtr>()->input(0),
                            NewValueNode(std::make_shared<Primitive>(kMulOpName))));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("ad")->cast<CNodePtr>()->input(1), a));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("ad")->cast<CNodePtr>()->input(2), d));

  ASSERT_EQ(check.m_->Get("add1")->cast<CNodePtr>()->inputs().size(), 3);
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("add1")->cast<CNodePtr>()->input(0),
                            NewValueNode(std::make_shared<Primitive>(kAddOpName))));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("add1")->cast<CNodePtr>()->input(1), check.m_->Get("ad")));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("add1")->cast<CNodePtr>()->input(2), e));
}

namespace {
void Check0(const FuncGraphIndexPtr &fg, const std::map<std::string, AnfNodePtr> &node_map) {
  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kAAddB)) == kOne);
  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kCAddD)) == kOne);
  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kMul)) == kOne);
  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kAdd)) == kOne);

  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kA)) == kOne);
  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kB)) == kOne);
  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kC)) == kOne);
  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kD)) == kOne);
  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kE)) == kOne);

  ASSERT_TRUE(fg->name_to_cnode_.size() == kTwo);
  ASSERT_TRUE(fg->name_to_cnode_.find(kAddOpName) != fg->name_to_cnode_.end());
  ASSERT_TRUE(fg->name_to_cnode_.find(kMulOpName) != fg->name_to_cnode_.end());

  auto &add_set = fg->name_to_cnode_[kAddOpName];
  auto &mul_set = fg->name_to_cnode_[kMulOpName];

  ASSERT_TRUE(add_set.size() == kThree);
  ASSERT_TRUE(mul_set.size() == kOne);
  ASSERT_TRUE(add_set.find(node_map.at(kAdd)) != add_set.end());
  ASSERT_TRUE(add_set.find(node_map.at(kAAddB)) != add_set.end());
  ASSERT_TRUE(add_set.find(node_map.at(kCAddD)) != add_set.end());
  ASSERT_TRUE(mul_set.find(node_map.at(kMul)) != mul_set.end());
}
void Check1(const TestFastMul2 &pass, const FuncGraphIndexPtr &fg, const std::map<std::string, AnfNodePtr> &node_map) {
  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kAAddB)) == kOne);
  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kCAddD)) == kOne);
  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kMul)) == kZero);
  ASSERT_TRUE(fg->node_degree_.at(pass.m_->Get(kMul)) == kOne);
  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kAdd)) == kOne);

  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kA)) == kOne);
  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kB)) == kOne);
  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kC)) == kOne);
  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kD)) == kOne);
  ASSERT_TRUE(fg->node_degree_.at(node_map.at(kE)) == kOne);

  ASSERT_TRUE(fg->name_to_cnode_.size() == kTwo);
  ASSERT_TRUE(fg->name_to_cnode_.find(kAddOpName) != fg->name_to_cnode_.end());
  ASSERT_TRUE(fg->name_to_cnode_.find(kMulOpName) != fg->name_to_cnode_.end());

  auto &add_set_2 = fg->name_to_cnode_[kAddOpName];
  auto &mul_set_2 = fg->name_to_cnode_[kMulOpName];

  ASSERT_TRUE(add_set_2.size() == kThree);
  ASSERT_TRUE(mul_set_2.size() == kOne);
  ASSERT_TRUE(add_set_2.find(node_map.at(kAAddB)) != add_set_2.end());
  ASSERT_TRUE(add_set_2.find(node_map.at(kCAddD)) != add_set_2.end());
  ASSERT_TRUE(mul_set_2.find(pass.m_->Get(kMul)) != mul_set_2.end());
}

void Check2(const CheckPattern &check, const std::map<std::string, AnfNodePtr> &node_map) {
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kA), node_map.at(kA)));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kB), node_map.at(kB)));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kC), node_map.at(kC)));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kD), node_map.at(kD)));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kE), node_map.at(kE)));

  ASSERT_EQ(check.m_->Get(kAAddB)->cast<CNodePtr>()->inputs().size(), kThree);
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kAAddB)->cast<CNodePtr>()->input(kZero),
                            NewValueNode(std::make_shared<Primitive>(kAddOpName))));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kAAddB)->cast<CNodePtr>()->input(kOne), node_map.at(kA)));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kAAddB)->cast<CNodePtr>()->input(kTwo), node_map.at(kB)));

  ASSERT_EQ(check.m_->Get(kCAddD)->cast<CNodePtr>()->inputs().size(), kThree);
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kCAddD)->cast<CNodePtr>()->input(kZero),
                            NewValueNode(std::make_shared<Primitive>(kAddOpName))));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kCAddD)->cast<CNodePtr>()->input(kOne), node_map.at(kC)));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kCAddD)->cast<CNodePtr>()->input(kTwo), node_map.at(kD)));

  ASSERT_EQ(check.m_->Get(kMul)->cast<CNodePtr>()->inputs().size(), kThree);
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kMul)->cast<CNodePtr>()->input(kZero),
                            NewValueNode(std::make_shared<Primitive>(kMulOpName))));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kMul)->cast<CNodePtr>()->input(kOne), node_map.at(kCAddD)));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kMul)->cast<CNodePtr>()->input(kTwo), node_map.at(kAAddB)));

  ASSERT_EQ(check.m_->Get(kAdd)->cast<CNodePtr>()->inputs().size(), kThree);
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kAdd)->cast<CNodePtr>()->input(kZero),
                            NewValueNode(std::make_shared<Primitive>(kAddOpName))));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kAdd)->cast<CNodePtr>()->input(kOne), check.m_->Get(kMul)));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get(kAdd)->cast<CNodePtr>()->input(kTwo), node_map.at(kE)));
}
}  // namespace

/// Feature: Fast PatternToPattern Pass
/// Description: Fast PatternToPattern Pass rewrite graph
/// Expectation: Get correct Graph
TEST_F(TestFastPatternToPatternPass, Mul2) {
  // ((a + b) * (c + d)) + e -> ((c + d) * (a + b)) + e
  // init
  auto check = CheckPattern();
  auto pass = TestFastMul2();

  // build func graph
  auto a = std::make_shared<AnfNode>(fg_);
  auto b = std::make_shared<AnfNode>(fg_);
  auto c = std::make_shared<AnfNode>(fg_);
  auto d = std::make_shared<AnfNode>(fg_);
  auto e = std::make_shared<AnfNode>(fg_);

  AnfNodePtr a_add_b =
    std::make_shared<CNode>(std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kAddOpName)), a, b}, fg_);
  AnfNodePtr c_add_d =
    std::make_shared<CNode>(std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kAddOpName)), c, d}, fg_);
  AnfNodePtr mul = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), a_add_b, c_add_d}, fg_);
  AnfNodePtr add = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kAddOpName)), mul, e}, fg_);

  std::map<std::string, AnfNodePtr> node_map;
  node_map.emplace("a", a);
  node_map.emplace("b", b);
  node_map.emplace("c", c);
  node_map.emplace("d", d);
  node_map.emplace("e", e);
  node_map.emplace("a_add_b", a_add_b);
  node_map.emplace("c_add_d", c_add_d);
  node_map.emplace("mul", mul);
  node_map.emplace("add", add);

  fg_->set_output(add);
  auto manager = MakeManager({fg_});
  if (manager) {
    manager->AddFuncGraph(fg_);
    fg_->set_manager(manager);
  }
  auto func_graph_index = manager->func_graph_index(fg_);
  GenIndex(fg_, func_graph_index);

  Check0(func_graph_index, node_map);
  auto new_node = pass.Run(fg_, mul);
  ASSERT_NE(new_node, nullptr);
  (void)manager->Replace(mul, new_node);
  pass.AfterProcess(mul, new_node, fg_, func_graph_index);
  Check1(pass, func_graph_index, node_map);

  // build pattern
  check.src_pattern_.AddVar("a")
    .AddVar("b")
    .AddVar("c")
    .AddVar("d")
    .AddVar("e")
    .AddCNode("a_add_b", {std::make_shared<Primitive>(kAddOpName), "a", "b"})
    .AddCNode("c_add_d", {std::make_shared<Primitive>(kAddOpName), "c", "d"})
    .AddCNode("mul", {std::make_shared<Primitive>(kMulOpName), "c_add_d", "a_add_b"})
    .AddCNode("add", {std::make_shared<Primitive>(kAddOpName), "mul", "e"});

  // pattern engine
  ASSERT_TRUE(check.build_pattern_map(add));
  Check2(check, node_map);
}
}  // namespace opt
}  // namespace mindspore
