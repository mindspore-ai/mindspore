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

#include "pattern_to_pattern_pass_utils.h"

namespace mindspore {
namespace opt {
namespace {
class TestMul0 : public PatternToPatternPass {
  // a*b + a*c -> a*(b+c)
 public:
  explicit TestMul0() : PatternToPatternPass("test_mul0") {}
  ~TestMul0() override = default;

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

class TestMul1 : public PatternToPatternPass {
  // a*b -> a*c
 public:
  explicit TestMul1() : PatternToPatternPass("test_mul1") {}
  ~TestMul1() override = default;

  void DefineSrcPattern(SrcPattern *src_pattern) override {
    (*src_pattern).AddSeqVar("Sv").AddCNode("ab", {std::make_shared<Primitive>(kMulOpName), "Sv"});
  }
  void DefineDstPattern(DstPattern *dst_pattern) override {
    auto ab = Unpacking("Sv");
    ab[1] = "c";
    (*dst_pattern)
      .AddValueNode("c", DefaultValueFunc(MakeValue(1)))
      .AddCNode("mul", {std::make_shared<Primitive>(kMulOpName), ab});
  }
  bool CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const override { return true; }
};

class TestMul2 : public PatternToPatternPass {
  // a*b -> b*a
 public:
  explicit TestMul2() : PatternToPatternPass("test_mul2") {}
  ~TestMul2() override = default;

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

class TestMul3 : public PatternToPatternPass {
  // a*b -> a+b
 public:
  explicit TestMul3() : PatternToPatternPass("test_mul3") {}
  ~TestMul3() override = default;

  void DefineSrcPattern(SrcPattern *src_pattern) override {
    (*src_pattern).AddSeqVar("Sv").AddCNode("ab", {std::make_shared<Primitive>(kMulOpName), "Sv"});
  }
  void DefineDstPattern(DstPattern *dst_pattern) override {
    (*dst_pattern).AddCNode("mul", {std::make_shared<Primitive>(kAddOpName), "Sv"});
  }
  bool CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const override { return true; }
};

class TestEmptySeq : public PatternToPatternPass {
 public:
  explicit TestEmptySeq() : PatternToPatternPass("test_empty_seq") {}
  ~TestEmptySeq() override = default;

  void DefineSrcPattern(SrcPattern *src_pattern) override {
    (*src_pattern).AddVar("V").AddSeqVar("Sv").AddCNode("s_a", {"V", "Sv"});
  }
  void DefineDstPattern(DstPattern *dst_pattern) override {
    (*dst_pattern).AddCNode("d_a", {"V", "Sv"}, InplaceCNodeFunc("s_a"));
  }
  bool CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const override { return true; }
};

AnfNodePtr BuildNull(const PatternMap &, const AnfNodePtr &) { return nullptr; }

class TestNull : public PatternToPatternPass {
 public:
  explicit TestNull() : PatternToPatternPass("test_null") {}
  ~TestNull() override = default;

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
      .AddCNode("bc", {std::make_shared<Primitive>(kAddOpName), "b", "c"}, BuildNull)
      .AddCNode("mul", {std::make_shared<Primitive>(kMulOpName), "a", "bc"});
  }
  bool CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const override { return true; }
};

class TestError0 : public PatternToPatternPass {
 public:
  explicit TestError0() : PatternToPatternPass("test_error0") {}
  ~TestError0() override = default;

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
      .AddCNode("add", {std::make_shared<Primitive>(kMulOpName), "a", "bc"});
  }
  bool CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const override { return true; }
};

class TestError1 : public PatternToPatternPass {
 public:
  explicit TestError1() : PatternToPatternPass("test_error1") {}
  ~TestError1() override = default;

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
      .AddCNode("mul", {std::make_shared<Primitive>(kMulOpName), "a", "bc"}, InplaceCNodeFunc("add"));
  }
  bool CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &) const override { return true; }
};
}  // namespace

class TestPatternToPatternPass : public UT::Common {
 public:
  TestPatternToPatternPass() : fg_(std::make_shared<FuncGraph>()){};

 public:
  FuncGraphPtr fg_;
};

/// Feature: PatternToPattern Pass
/// Description: PatternToPattern Pass rewrite graph
/// Expectation: Get correct Graph
TEST_F(TestPatternToPatternPass, Mul0) {
  // a*b + a*c -> a*(b+c)
  // init
  auto check = CheckPattern();
  auto pass = TestMul0();

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

  auto new_node = pass.Run(fg_, add);
  ASSERT_NE(new_node, nullptr);

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

/// Feature: PatternToPattern Pass
/// Description: PatternToPattern Pass rewrite graph
/// Expectation: Get correct Graph
TEST_F(TestPatternToPatternPass, Mul1) {
  // a*b -> a*1
  // init
  auto check = CheckPattern();
  auto pass = TestMul1();

  // build func graph
  auto a = std::make_shared<AnfNode>(fg_);
  auto b = std::make_shared<AnfNode>(fg_);
  AnfNodePtr ab =
    std::make_shared<CNode>(std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), a, b}, fg_);

  auto new_node = pass.Run(fg_, ab);
  ASSERT_NE(new_node, nullptr);

  // build pattern
  check.src_pattern_.AddVar("a").AddVar("b").AddCNode("mul", {std::make_shared<Primitive>(kMulOpName), "a", "b"});

  // pattern engine
  ASSERT_TRUE(check.build_pattern_map(new_node));

  // check
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("a"), a));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("b"), NewValueNode(1)));
  ASSERT_EQ(check.m_->Get("mul")->cast<CNodePtr>()->inputs().size(), 3);
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("mul")->cast<CNodePtr>()->input(0),
                            NewValueNode(std::make_shared<Primitive>(kMulOpName))));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("mul")->cast<CNodePtr>()->input(1), a));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("mul")->cast<CNodePtr>()->input(2), NewValueNode(1)));
}

/// Feature: PatternToPattern Pass
/// Description: PatternToPattern Pass rewrite graph
/// Expectation: Get correct Graph
TEST_F(TestPatternToPatternPass, Mul2) {
  // a*b -> b*a
  // init
  auto check = CheckPattern();
  auto pass = TestMul2();

  // build func graph
  auto a = std::make_shared<AnfNode>(fg_);
  auto b = std::make_shared<AnfNode>(fg_);
  AnfNodePtr ab =
    std::make_shared<CNode>(std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), a, b}, fg_);

  auto new_node = pass.Run(fg_, ab);
  ASSERT_NE(new_node, nullptr);

  // build pattern
  check.src_pattern_.AddVar("a").AddVar("b").AddCNode("mul", {std::make_shared<Primitive>(kMulOpName), "b", "a"});

  // pattern engine
  ASSERT_TRUE(check.build_pattern_map(new_node));

  // check
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("a"), a));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("b"), b));
  ASSERT_EQ(check.m_->Get("mul")->cast<CNodePtr>()->inputs().size(), 3);
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("mul")->cast<CNodePtr>()->input(0),
                            NewValueNode(std::make_shared<Primitive>(kMulOpName))));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("mul")->cast<CNodePtr>()->input(1), b));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("mul")->cast<CNodePtr>()->input(2), a));
}

/// Feature: PatternToPattern Pass
/// Description: PatternToPattern Pass rewrite graph
/// Expectation: Get correct Graph
TEST_F(TestPatternToPatternPass, Mul3) {
  // a*b -> a+b
  // init
  auto check = CheckPattern();
  auto pass = TestMul3();

  // build func graph
  auto a = std::make_shared<AnfNode>(fg_);
  auto b = std::make_shared<AnfNode>(fg_);
  AnfNodePtr ab =
    std::make_shared<CNode>(std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), a, b}, fg_);

  auto new_node = pass.Run(fg_, ab);
  ASSERT_NE(new_node, nullptr);

  // build pattern
  check.src_pattern_.AddVar("a").AddVar("b").AddCNode("add", {std::make_shared<Primitive>(kAddOpName), "a", "b"});

  // pattern engine
  ASSERT_TRUE(check.build_pattern_map(new_node));

  // check
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("a"), a));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("b"), b));
  ASSERT_EQ(check.m_->Get("add")->cast<CNodePtr>()->inputs().size(), 3);
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("add")->cast<CNodePtr>()->input(0),
                            NewValueNode(std::make_shared<Primitive>(kAddOpName))));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("add")->cast<CNodePtr>()->input(1), a));
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("add")->cast<CNodePtr>()->input(2), b));
}

/// Feature: PatternToPattern Pass
/// Description: PatternToPattern Pass rewrite graph
/// Expectation: Get correct Graph
TEST_F(TestPatternToPatternPass, EmptySeq) {
  // init
  auto check = CheckPattern();
  auto pass = TestEmptySeq();

  // build func graph
  AnfNodePtr a = std::make_shared<CNode>(std::vector<AnfNodePtr>{NewValueNode(prim::kPrimNPUAllocFloatStatus)}, fg_);

  auto new_node = pass.Run(fg_, a);
  ASSERT_NE(new_node, nullptr);

  // build pattern
  check.src_pattern_.AddVar("V").AddSeqVar("Sv").AddCNode("c_a", {"V", "Sv"});

  // pattern engine
  ASSERT_TRUE(check.build_pattern_map(new_node));

  // check
  ASSERT_TRUE(opt::AnfEqual(check.m_->Get("c_a"), a));
}

/// Feature: PatternToPattern Pass
/// Description: PatternToPattern Pass failed to rewrite graph
/// Expectation: Get nullptr
TEST_F(TestPatternToPatternPass, Null) {
  // init
  auto check = CheckPattern();
  auto pass = TestNull();

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

  EXPECT_THROW(pass.Run(fg_, add), std::runtime_error);
}

/// Feature: PatternToPattern Pass
/// Description: PatternToPattern Pass failed to rewrite graph
/// Expectation: Get runtime error
TEST_F(TestPatternToPatternPass, Error0) {
  // init
  auto check = CheckPattern();
  auto pass = TestError0();

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

  EXPECT_THROW(pass.Run(fg_, add), std::runtime_error);
}

/// Feature: PatternToPattern Pass
/// Description: PatternToPattern Pass failed to rewrite graph
/// Expectation: Get runtime error
TEST_F(TestPatternToPatternPass, Error1) {
  // init
  auto check = CheckPattern();
  auto pass = TestError1();

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

  EXPECT_THROW(pass.Run(fg_, add), std::runtime_error);
}
}  // namespace opt
}  // namespace mindspore
