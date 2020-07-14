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
#include "common/py_func_graph_fetcher.h"

#include "ir/anf.h"
#include "ir/visitor.h"
#include "ir/func_graph_cloner.h"
#include "optimizer/opt.h"
#include "optimizer/irpass.h"
#include "optimizer/irpass/arithmetic_simplify.h"

#include "debug/draw.h"
#include "operator/ops.h"
#include "optimizer/cse.h"

namespace mindspore {
namespace opt {
class TestOptOpt : public UT::Common {
 public:
  TestOptOpt() : getPyFun("gtest_input.optimizer.opt_test", true) {}

  class IdempotentEliminater : public AnfVisitor {
   public:
    AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
      x_ = nullptr;
      AnfVisitor::Match(P, {irpass::IsCNode})(node);
      if (x_ == nullptr || node->func_graph() == nullptr) {
        return nullptr;
      }

      return node->func_graph()->NewCNode({NewValueNode(P), x_});
    };

    void Visit(const CNodePtr &cnode) override {
      if (IsPrimitiveCNode(cnode, P) && cnode->inputs().size() == 2) {
        x_ = cnode->input(1);
      }
    }

   private:
    AnfNodePtr x_{nullptr};
  };

  class QctToP : public AnfVisitor {
   public:
    AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
      v_ = nullptr;
      AnfVisitor::Match(Q, {irpass::IsVNode})(node);
      if (v_ == nullptr || node->func_graph() == nullptr) {
        return nullptr;
      }

      return node->func_graph()->NewCNode({NewValueNode(P), v_});
    };

    void Visit(const ValueNodePtr &vnode) override { v_ = vnode; }

   private:
    AnfNodePtr v_{nullptr};
  };

  void SetUp() {
    elim_Z = MakeSubstitution(std::make_shared<irpass::AddByZero>(), "elim_Z", prim::kPrimScalarAdd);
    elim_R = MakeSubstitution(std::make_shared<irpass::PrimEliminater>(R), "elim_R", R);
    idempotent_P = MakeSubstitution(std::make_shared<IdempotentEliminater>(), "idempotent_P", P);
    Qct_to_P = MakeSubstitution(std::make_shared<QctToP>(), "Qct_to_P", Q);
  }

  bool CheckTransform(FuncGraphPtr gbefore, FuncGraphPtr gafter, const SubstitutionList &transform) {
    equiv_node.clear();
    equiv_graph.clear();

    FuncGraphPtr gbefore_clone = BasicClone(gbefore);
    OptimizerPtr optimizer = std::make_shared<Optimizer>("ut_test", std::make_shared<pipeline::Resource>());
    transform(gbefore_clone, optimizer);

    return Isomorphic(gbefore_clone, gafter, &equiv_graph, &equiv_node);
  }

  bool CheckOpt(FuncGraphPtr before, FuncGraphPtr after, std::vector<SubstitutionPtr> opts = {}) {
    SubstitutionList eq(opts);
    return CheckTransform(before, after, eq);
  }

 public:
  UT::PyFuncGraphFetcher getPyFun;

  FuncGraphPairMapEquiv equiv_graph;
  NodeMapEquiv equiv_node;

  static const PrimitivePtr P;
  static const PrimitivePtr Q;
  static const PrimitivePtr R;

  SubstitutionPtr elim_Z;
  SubstitutionPtr elim_R;
  SubstitutionPtr idempotent_P;
  SubstitutionPtr Qct_to_P;
};

const PrimitivePtr TestOptOpt::P = std::make_shared<Primitive>("P");
const PrimitivePtr TestOptOpt::Q = std::make_shared<Primitive>("Q");
const PrimitivePtr TestOptOpt::R = std::make_shared<Primitive>("R");

TEST_F(TestOptOpt, TestCheckOptIsClone) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_add_zero", "before_1");

  ASSERT_TRUE(nullptr != before);
  ASSERT_TRUE(CheckOpt(before, before));
  ASSERT_FALSE(CheckOpt(before, before, std::vector<SubstitutionPtr>({elim_Z})));
}

TEST_F(TestOptOpt, Elim) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_add_zero", "before_1");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_add_zero", "after");

  ASSERT_TRUE(nullptr != before);
  ASSERT_TRUE(nullptr != after);
  ASSERT_TRUE(CheckOpt(before, after, std::vector<SubstitutionPtr>({elim_Z})));
}

TEST_F(TestOptOpt, ElimTwo) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_add_zero", "before_2");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_add_zero", "after");

  ASSERT_TRUE(nullptr != before);
  ASSERT_TRUE(nullptr != after);
  ASSERT_TRUE(CheckOpt(before, after, std::vector<SubstitutionPtr>({elim_Z})));
}

TEST_F(TestOptOpt, ElimR) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_elimR", "before_1");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_elimR", "after");

  ASSERT_TRUE(nullptr != before);
  ASSERT_TRUE(nullptr != after);
  ASSERT_TRUE(CheckOpt(before, after, std::vector<SubstitutionPtr>({elim_R})));
}

TEST_F(TestOptOpt, idempotent) {
  FuncGraphPtr before_2 = getPyFun.CallAndParseRet("test_idempotent", "before_2");
  FuncGraphPtr before_1 = getPyFun.CallAndParseRet("test_idempotent", "before_1");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_idempotent", "after");

  ASSERT_TRUE(nullptr != before_2);
  ASSERT_TRUE(nullptr != before_1);
  ASSERT_TRUE(nullptr != after);

  ASSERT_TRUE(CheckOpt(before_1, after, std::vector<SubstitutionPtr>({idempotent_P})));
  ASSERT_TRUE(CheckOpt(before_2, after, std::vector<SubstitutionPtr>({idempotent_P})));
}

TEST_F(TestOptOpt, ConstantVariable) {
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_constant_variable", "before_1");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_constant_variable", "after");

  ASSERT_TRUE(nullptr != before);
  ASSERT_TRUE(nullptr != after);
  ASSERT_TRUE(CheckOpt(before, after, std::vector<SubstitutionPtr>({Qct_to_P})));
}

TEST_F(TestOptOpt, CSE) {
  // test a simple cse testcase test_f1
  FuncGraphPtr test_graph1 = getPyFun.CallAndParseRet("test_cse", "test_f1");

  ASSERT_TRUE(nullptr != test_graph1);

  // add func_graph the GraphManager
  FuncGraphManagerPtr manager1 = Manage(test_graph1);
  draw::Draw("opt_cse_before_1.dot", test_graph1);

  ASSERT_EQ(manager1->all_nodes().size(), 10);

  auto cse = std::make_shared<CSE>();
  ASSERT_TRUE(cse != nullptr);
  bool is_changed = cse->Cse(test_graph1, manager1);

  ASSERT_TRUE(is_changed);
  ASSERT_EQ(manager1->all_nodes().size(), 8);

  draw::Draw("opt_cse_after_1.dot", test_graph1);

  // test a more complicated case test_f2
  FuncGraphPtr test_graph2 = getPyFun.CallAndParseRet("test_cse", "test_f2");

  ASSERT_TRUE(nullptr != test_graph2);

  FuncGraphManagerPtr manager2 = Manage(test_graph2);
  draw::Draw("opt_cse_before_2.dot", test_graph2);
  ASSERT_EQ(manager2->all_nodes().size(), 22);
  is_changed = cse->Cse(test_graph2, manager2);
  ASSERT_TRUE(is_changed);
  ASSERT_EQ(manager2->all_nodes().size(), 12);
  draw::Draw("opt_cse_after_2.dot", test_graph2);
}

}  // namespace opt
}  // namespace mindspore
