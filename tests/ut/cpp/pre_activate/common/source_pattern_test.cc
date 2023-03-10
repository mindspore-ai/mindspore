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

#include <vector>
#include <memory>
#include "common/common_test.h"
#define private public
#define protected public
#include "include/backend/optimizer/pattern_to_pattern.h"
#undef private
#undef protected

#include "mindspore/core/ops/core_ops.h"
#include "ir/anf.h"
#include "ir/value.h"
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
class TestSrcPattern : public UT::Common {
 public:
  TestSrcPattern()
      : m_(std::make_shared<PatternMap>()),
        src_pattern_(SrcPattern(m_)),
        pattern_engine_(PatternEngine(std::make_shared<Visitor>())),
        primitive_vars_(std::make_shared<PrimitiveVarMap>()),
        equiv_(std::make_shared<Equiv>()),
        fg_(std::make_shared<FuncGraph>()){};
  bool build_pattern_map(const AnfNodePtr &node) {
    VarPtr root_g = std::make_shared<Var>("RootG");
    auto src_pattern_root = SexpToNode(src_pattern_.GetRoot(), root_g, primitive_vars_.get(), multigraph_);
    auto primitive = GetCNodePrimitive(src_pattern_root);
    if (IsPrimitiveCNode(node, primitive)) {
      MS_EXCEPTION_IF_NULL(primitive_vars_);
      MS_EXCEPTION_IF_NULL(equiv_);
      equiv_->clear();
      EquivPtr equiv = pattern_engine_.Match(src_pattern_root, node, *primitive_vars_, equiv_);
      if (equiv != nullptr && !equiv->empty()) {
        return src_pattern_.build_pattern_map(node, equiv);
      }
    }
    return false;
  }
  PatternMapPtr m_;
  SrcPattern src_pattern_;
  PatternEngine pattern_engine_;
  PrimitiveVarMapPtr primitive_vars_;
  EquivPtr equiv_;
  bool multigraph_ = true;
  FuncGraphPtr fg_;
};

/// Feature: PatternToPattern Pass
/// Description: SrcPattern match Var with primitive
/// Expectation: Get correct PatternMap
TEST_F(TestSrcPattern, Var) {
  // build func graph
  auto anode1 = std::make_shared<AnfNode>(fg_);
  auto anode2 = std::make_shared<AnfNode>(fg_);
  auto anode3 = std::make_shared<AnfNode>(fg_);
  AnfNodePtr mul2_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode2, anode3}, fg_);
  AnfNodePtr mul1_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode1, mul2_cnode}, fg_);

  // build pattern
  src_pattern_.AddVar("anode1")
    .AddVar("anode2")
    .AddVar("anode3")
    .AddCNode("mul2_cnode", {std::make_shared<Primitive>(kMulOpName), "anode2", "anode3"})
    .AddCNode("mul1_cnode", {std::make_shared<Primitive>(kMulOpName), "anode1", "mul2_cnode"});

  // pattern engine
  ASSERT_TRUE(build_pattern_map(mul1_cnode));

  // check
  ASSERT_TRUE(opt::AnfEqual(m_->Get("anode1"), anode1));
  ASSERT_TRUE(opt::AnfEqual(m_->Get("anode2"), anode2));
  ASSERT_TRUE(opt::AnfEqual(m_->Get("anode3"), anode3));
  ASSERT_TRUE(opt::AnfEqual(m_->Get("mul1_cnode"), mul1_cnode));
  ASSERT_TRUE(opt::AnfEqual(m_->Get("mul2_cnode"), mul2_cnode));
}

/// Feature: PatternToPattern Pass
/// Description: SrcPattern match SeqVar with primitive
/// Expectation: Get correct PatternMap
TEST_F(TestSrcPattern, SeqVar) {
  // build func graph
  auto anode1 = std::make_shared<AnfNode>(fg_);
  auto anode2 = std::make_shared<AnfNode>(fg_);
  auto anode3 = std::make_shared<AnfNode>(fg_);
  AnfNodePtr mul2_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode2, anode3}, fg_);
  AnfNodePtr mul1_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode1, mul2_cnode}, fg_);

  // build pattern
  src_pattern_.AddVar("anode1")
    .AddSeqVar("Sv")
    .AddCNode("mul2_cnode", {std::make_shared<Primitive>(kMulOpName), "Sv"})
    .AddCNode("mul1_cnode", {std::make_shared<Primitive>(kMulOpName), "anode1", "mul2_cnode"});

  // pattern engine
  ASSERT_TRUE(build_pattern_map(mul1_cnode));

  // check
  ASSERT_TRUE(opt::AnfEqual(m_->Get("anode1"), anode1));
  ASSERT_TRUE(opt::AnfEqual(m_->Get("mul1_cnode"), mul1_cnode));
  ASSERT_TRUE(opt::AnfEqual(m_->Get("mul2_cnode"), mul2_cnode));
  auto &v = m_->GetSeq("Sv");
  ASSERT_EQ(v.size(), std::size_t(2));
  ASSERT_TRUE(opt::AnfEqual(v[0], anode2));
  ASSERT_TRUE(opt::AnfEqual(v[1], anode3));
}

/// Feature: PatternToPattern Pass
/// Description: SrcPattern match Repeated Var with primitive
/// Expectation: Get correct PatternMap
TEST_F(TestSrcPattern, RepeatedVar) {
  // build func graph
  auto anode1 = std::make_shared<AnfNode>(fg_);
  auto anode2 = std::make_shared<AnfNode>(fg_);
  auto anode3 = std::make_shared<AnfNode>(fg_);
  AnfNodePtr mul2_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode2, anode3}, fg_);
  AnfNodePtr mul1_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode1, mul2_cnode}, fg_);
  AnfNodePtr mul3_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), mul1_cnode, mul2_cnode}, fg_);

  /*
   (Mul, anode2, anode3) -> mul2_cnode
   (Mul, anode1, mul2_cnode) -> mul1_cnode
   (Mul, mul1_cnode, mul2_cnode) -> mul3_cnode
   */
  // build pattern
  src_pattern_.AddVar("anode1")
    .AddVar("anode2")
    .AddVar("anode3")
    .AddCNode("mul2_cnode", {std::make_shared<Primitive>(kMulOpName), "anode2", "anode3"})
    .AddCNode("mul1_cnode", {std::make_shared<Primitive>(kMulOpName), "anode1", "mul2_cnode"})
    .AddCNode("mul3_cnode", {std::make_shared<Primitive>(kMulOpName), "mul1_cnode", "mul2_cnode"});

  // pattern engine
  ASSERT_TRUE(build_pattern_map(mul3_cnode));

  // check
  ASSERT_TRUE(opt::AnfEqual(m_->Get("anode1"), anode1));
  ASSERT_TRUE(opt::AnfEqual(m_->Get("anode2"), anode2));
  ASSERT_TRUE(opt::AnfEqual(m_->Get("anode3"), anode3));
  ASSERT_TRUE(opt::AnfEqual(m_->Get("mul1_cnode"), mul1_cnode));
  ASSERT_TRUE(opt::AnfEqual(m_->Get("mul2_cnode"), mul2_cnode));
  ASSERT_TRUE(opt::AnfEqual(m_->Get("mul3_cnode"), mul3_cnode));
}

/// Feature: PatternToPattern Pass
/// Description: SrcPattern match Repeated SeqVar with primitive
/// Expectation: Get correct PatternMap
TEST_F(TestSrcPattern, RepeatedSeqVar) {
  // build func graph
  auto anode1 = std::make_shared<AnfNode>(fg_);
  auto anode2 = std::make_shared<AnfNode>(fg_);
  auto anode3 = std::make_shared<AnfNode>(fg_);
  AnfNodePtr mul2_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode2, anode3}, fg_);
  AnfNodePtr mul3_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode2, anode3}, fg_);
  AnfNodePtr mul1_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode1, mul2_cnode}, fg_);
  AnfNodePtr mul4_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), mul3_cnode, mul1_cnode}, fg_);

  /*
   (Mul, *Sv) -> mul2_cnode
   (Mul, *Sv) -> mul3_cnode
   (Mul, anode1, mul2_cnode) -> mul1_cnode
   (Mul, mul3_cnode, mul1_cnode) -> mul4_cnode
   */
  // build pattern
  src_pattern_.AddVar("anode1")
    .AddSeqVar("Sv")
    .AddCNode("mul2_cnode", {std::make_shared<Primitive>(kMulOpName), "Sv"})
    .AddCNode("mul3_cnode", {std::make_shared<Primitive>(kMulOpName), "Sv"})
    .AddCNode("mul1_cnode", {std::make_shared<Primitive>(kMulOpName), "anode1", "mul2_cnode"})
    .AddCNode("mul4_cnode", {std::make_shared<Primitive>(kMulOpName), "mul3_cnode", "mul1_cnode"});

  // pattern engine
  ASSERT_TRUE(build_pattern_map(mul4_cnode));

  // check
  ASSERT_TRUE(opt::AnfEqual(m_->Get("anode1"), anode1));
  ASSERT_TRUE(opt::AnfEqual(m_->Get("mul1_cnode"), mul1_cnode));
  ASSERT_TRUE(opt::AnfEqual(m_->Get("mul2_cnode"), mul2_cnode));
  ASSERT_TRUE(opt::AnfEqual(m_->Get("mul3_cnode"), mul3_cnode));
  ASSERT_TRUE(opt::AnfEqual(m_->Get("mul4_cnode"), mul4_cnode));
  auto &v = m_->GetSeq("Sv");
  ASSERT_EQ(v.size(), std::size_t(2));
  ASSERT_TRUE(opt::AnfEqual(v[0], anode2));
  ASSERT_TRUE(opt::AnfEqual(v[1], anode3));
}

/// Feature: PatternToPattern Pass
/// Description: SrcPattern match Wrong Var with primitive
/// Expectation: Get False
TEST_F(TestSrcPattern, WrongVar) {
  // build func graph
  auto anode1 = std::make_shared<AnfNode>(fg_);
  auto anode2 = std::make_shared<AnfNode>(fg_);
  auto anode3 = std::make_shared<AnfNode>(fg_);
  AnfNodePtr mul2_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode2, anode3}, fg_);
  AnfNodePtr mul1_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode1, mul2_cnode}, fg_);
  AnfNodePtr mul3_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), mul1_cnode, mul2_cnode}, fg_);

  /*
   (Mul, anode2, anode3) -> mul2_cnode
   (Mul, anode1, mul2_cnode) -> mul1_cnode
   (Mul, mul1_cnode, mul2_cnode) -> mul3_cnode
   */
  // build pattern
  src_pattern_.AddVar("anode2")
    .AddVar("anode3")
    .AddCNode("mul2_cnode", {std::make_shared<Primitive>(kMulOpName), "anode2", "anode3"})
    .AddCNode("mul1_cnode", {std::make_shared<Primitive>(kMulOpName), "anode2", "mul2_cnode"})
    .AddCNode("mul3_cnode", {std::make_shared<Primitive>(kMulOpName), "mul1_cnode", "mul2_cnode"});

  // pattern engine
  ASSERT_FALSE(build_pattern_map(mul3_cnode));
}

/// Feature: PatternToPattern Pass
/// Description: SrcPattern match Wrong SeqVar with primitive
/// Expectation: Get False
TEST_F(TestSrcPattern, WrongSeqVar) {
  // build func graph
  auto anode1 = std::make_shared<AnfNode>(fg_);
  auto anode2 = std::make_shared<AnfNode>(fg_);
  auto anode3 = std::make_shared<AnfNode>(fg_);
  AnfNodePtr mul2_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode2, anode3}, fg_);
  AnfNodePtr mul3_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode1, anode2}, fg_);
  AnfNodePtr mul1_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode1, mul2_cnode}, fg_);
  AnfNodePtr mul4_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), mul3_cnode, mul1_cnode}, fg_);

  /*
   (Mul, *Sv) -> mul2_cnode
   (Mul, *Sv) -> mul3_cnode
   (Mul, anode1, mul2_cnode) -> mul1_cnode
   (Mul, mul3_cnode, mul1_cnode) -> mul4_cnode
   */
  // build pattern
  src_pattern_.AddVar("anode1")
    .AddSeqVar("Sv")
    .AddCNode("mul2_cnode", {std::make_shared<Primitive>(kMulOpName), "Sv"})
    .AddCNode("mul3_cnode", {std::make_shared<Primitive>(kMulOpName), "Sv"})
    .AddCNode("mul1_cnode", {std::make_shared<Primitive>(kMulOpName), "anode1", "mul2_cnode"})
    .AddCNode("mul4_cnode", {std::make_shared<Primitive>(kMulOpName), "mul3_cnode", "mul1_cnode"});

  // pattern engine
  ASSERT_FALSE(build_pattern_map(mul4_cnode));
}

/// Feature: PatternToPattern Pass
/// Description: SrcPattern match Wrong CNode with primitive
/// Expectation: Get False
TEST_F(TestSrcPattern, WrongCNode) {
  // build func graph
  auto anode1 = std::make_shared<AnfNode>(fg_);
  auto anode2 = std::make_shared<AnfNode>(fg_);
  AnfNodePtr mul1_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode1, anode2}, fg_);
  AnfNodePtr mul2_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), anode1, anode2}, fg_);
  AnfNodePtr mul3_cnode = std::make_shared<CNode>(
    std::vector<AnfNodePtr>{NewValueNode(std::make_shared<Primitive>(kMulOpName)), mul1_cnode, mul2_cnode}, fg_);

  /*
   (Mul, anode1, anode2) -> mul1_cnode
   (Mul, mul1_cnode, mul1_cnode) -> mul3_cnode
   */
  // build pattern
  src_pattern_.AddVar("anode1")
    .AddVar("anode2")
    .AddCNode("mul1_cnode", {std::make_shared<Primitive>(kMulOpName), "anode1", "anode2"})
    .AddCNode("mul3_cnode", {std::make_shared<Primitive>(kMulOpName), "mul1_cnode", "mul1_cnode"});

  // pattern engine
  ASSERT_FALSE(build_pattern_map(mul3_cnode));
}
}  // namespace opt
}  // namespace mindspore
