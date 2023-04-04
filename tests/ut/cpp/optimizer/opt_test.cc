/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/opt.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/arithmetic_simplify.h"
#include "frontend/optimizer/irpass/pynative_no_grad_eliminate.h"
#include "pipeline/jit/action.h"

#include "include/common/debug/draw.h"
#include "frontend/operator/ops.h"
#include "include/common/utils/cse.h"
#include "include/common/utils/convert_utils.h"

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
    elim_Z = MakeSubstitution(std::make_shared<irpass::ArithmeticSimplify>(), "elim_Z", prim::kPrimScalarAdd);
    elim_R = MakeSubstitution(std::make_shared<irpass::PrimEliminater>(R), "elim_R", R);
    idempotent_P = MakeSubstitution(std::make_shared<IdempotentEliminater>(), "idempotent_P", P);
    Qct_to_P = MakeSubstitution(std::make_shared<QctToP>(), "Qct_to_P", Q);
    pynative_no_grad_elim = MakeSubstitution(std::make_shared<irpass::PynativeNoGradEliminater>(),
                                             "pynative_no_grad_eliminate", prim::kPrimMakeTuple);
  }

  bool CheckTransform(FuncGraphPtr gbefore, FuncGraphPtr gafter, const SubstitutionList &transform) {
    FuncGraphPtr graph_after_trans = TransformGraph(gbefore, transform);

    return Isomorphic(graph_after_trans, gafter, &equiv_graph, &equiv_node);
  }

  bool CheckOpt(FuncGraphPtr before, FuncGraphPtr after, std::vector<SubstitutionPtr> opts = {}) {
    SubstitutionList eq(opts);
    return CheckTransform(before, after, eq);
  }

  FuncGraphPtr TransformGraph(FuncGraphPtr gbefore, const SubstitutionList &transform) {
    equiv_node.clear();
    equiv_graph.clear();

    FuncGraphPtr gbefore_clone = BasicClone(gbefore);
    pipeline::ResourcePtr resource = std::make_shared<pipeline::Resource>();
    MS_EXCEPTION_IF_NULL(resource);
    resource->set_func_graph(gbefore_clone);
    auto manager = resource->manager();
    MS_EXCEPTION_IF_NULL(manager);
    manager->AddFuncGraph(gbefore_clone, true);

    OptimizerPtr optimizer = std::make_shared<Optimizer>("ut_test", resource);
    transform(gbefore_clone, optimizer);

    return gbefore_clone;
  }

 public:
  UT::PyFuncGraphFetcher getPyFun;

  FuncGraphPairMapEquiv equiv_graph;
  NodeMapEquiv equiv_node;

  irpass::OptimizeIRPassLib irpass_lib;

  static const PrimitivePtr P;
  static const PrimitivePtr Q;
  static const PrimitivePtr R;

  SubstitutionPtr elim_Z;
  SubstitutionPtr elim_R;
  SubstitutionPtr idempotent_P;
  SubstitutionPtr Qct_to_P;
  SubstitutionPtr pynative_no_grad_elim;
  SubstitutionPtr tuple_flatten = irpass_lib.call_graph_tuple_transform_;
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
  FuncGraphPtr before = getPyFun.CallAndParseRet("test_elim_r", "before_1");
  FuncGraphPtr after = getPyFun.CallAndParseRet("test_elim_r", "after");

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

  ASSERT_EQ(manager1->all_nodes().size(), 9);

  auto cse = std::make_shared<CSE>();
  ASSERT_TRUE(cse != nullptr);
  bool is_changed = cse->Cse(test_graph1, manager1);

  ASSERT_TRUE(is_changed);
  ASSERT_EQ(manager1->all_nodes().size(), 8);

  // test a more complicated case test_f2
  FuncGraphPtr test_graph2 = getPyFun.CallAndParseRet("test_cse", "test_f2");

  ASSERT_TRUE(nullptr != test_graph2);

  FuncGraphManagerPtr manager2 = Manage(test_graph2);
  ASSERT_EQ(manager2->all_nodes().size(), 16);
  is_changed = cse->Cse(test_graph2, manager2);
  ASSERT_TRUE(is_changed);
  ASSERT_EQ(manager2->all_nodes().size(), 12);
}

/// Feature: test no grad input net.
/// Description: test no grad input net.
/// Expectation: No exception.
TEST_F(TestOptOpt, PynativeNoGradElim) {
  FuncGraphPtr test_graph1 = getPyFun.CallAndParseRet("test_no_grad", "test_f1");

  ASSERT_TRUE(nullptr != test_graph1);

  auto all_nodes1 = TopoSort(test_graph1->return_node(), SuccDeeperSimple, AlwaysInclude);
  auto mul_node_num1 = std::count_if(all_nodes1.begin(), all_nodes1.end(),
                                     [](AnfNodePtr node) { return IsPrimitiveCNode(node, prim::kPrimMul); });

  ASSERT_EQ(mul_node_num1, 2);

  FuncGraphPtr test_graph2 = getPyFun.CallAndParseRet("test_no_grad", "test_f1");

  ASSERT_TRUE(nullptr != test_graph2);

  std::vector<bool> need_grad_flags{true, false};
  test_graph2->set_attr(kAttrNeedGradFlagOfInputs, MakeValue(need_grad_flags));

  auto tmp_substitution = std::vector<SubstitutionPtr>({pynative_no_grad_elim});
  SubstitutionList substitution_list(tmp_substitution);

  std::vector<int64_t> shape_vec = {1};
  AbstractBasePtr abs = std::make_shared<abstract::AbstractTensor>(kTensorType, shape_vec);
  auto graph_params = test_graph2->parameters();
  for (auto graph_input : graph_params) {
    graph_input->set_abstract(abs);
  }

  auto test_graph2_after_optmiz = TransformGraph(test_graph2, substitution_list);
  ASSERT_TRUE(nullptr != test_graph2_after_optmiz);
  auto all_nodes2 = TopoSort(test_graph2_after_optmiz->return_node(), SuccDeeperSimple, AlwaysInclude);
  auto mul_node_num2 = std::count_if(all_nodes2.begin(), all_nodes2.end(),
                                     [](AnfNodePtr node) { return IsPrimitiveCNode(node, prim::kPrimMul); });

  ASSERT_EQ(mul_node_num2, 1);
}

size_t TupleArgAndParamSum(const FuncGraphPtr &func_graph) {
  // Check tuple params and tuple args.
  auto all_nodes = TopoSort(func_graph->return_node(), SuccDeeperSimple, AlwaysInclude);
  size_t tuple_arg_param_num = 0;
  auto tuple_accumulate_func = [](size_t prev_num, const AnfNodePtr &node) -> size_t {
    auto abs = node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    return abs->isa<abstract::AbstractTuple>() ? prev_num + 1 : prev_num;
  };
  for (const auto &node : all_nodes) {
    // Count func graph call tuple args.
    if (node->isa<CNode>() && !IsValueNode<Primitive>(node->cast<CNodePtr>()->input(0))) {
      auto call_node = node->cast<CNodePtr>();
      tuple_arg_param_num = std::accumulate(call_node->inputs().begin() + 1, call_node->inputs().end(),
                                            tuple_arg_param_num, tuple_accumulate_func);
    }
    // Count partial tuple args.
    if (IsPrimitiveCNode(node, prim::kPrimPartial)) {
      auto partial = node->cast<CNodePtr>();
      constexpr auto kPartialFirstArgIdx = 2;
      tuple_arg_param_num = std::accumulate(partial->inputs().begin() + kPartialFirstArgIdx, partial->inputs().end(),
                                            tuple_arg_param_num, tuple_accumulate_func);
    }

    // Count tuple params.
    if (IsValueNode<FuncGraph>(node)) {
      auto fg = GetValueNode<FuncGraphPtr>(node);
      tuple_arg_param_num =
        std::accumulate(fg->parameters().begin(), fg->parameters().end(), tuple_arg_param_num, tuple_accumulate_func);
    }
  }
  return tuple_arg_param_num;
}

// Feature: Switch call tuple arg transform.
// Description: Test switch call's tuple arg transform.This case include partial's tuple arg and the call's tuple arg in
// the same time.
// Expectation: All tuple args are correctly transformed to tensor args.
TEST_F(TestOptOpt, SwitchPartialTupleTrans) {
  FuncGraphPtr test_graph = getPyFun.CallAndParseRet("test_tuple_flatten", "test_flatten_switch_partial_arg");
  ASSERT_TRUE(nullptr != test_graph);

  FuncGraphManagerPtr manager1 = Manage(test_graph);
  pipeline::ResourcePtr res = std::make_shared<pipeline::Resource>();
  std::vector<AbstractBasePtr> args_spec;

  // Renormalize firstly.
  auto renormalized_fg = pipeline::Renormalize(res, test_graph, args_spec);
  ASSERT_TRUE(TupleArgAndParamSum(renormalized_fg) != 0);

  // Flatten tuple param and args.
  OptimizerPtr optimizer = std::make_shared<Optimizer>("ut_test", res);
  SubstitutionList transform(std::vector<SubstitutionPtr>({tuple_flatten}));
  transform(renormalized_fg, optimizer);

  // Renormalize again.
  auto transformed_fg = pipeline::Renormalize(res, renormalized_fg, args_spec);
  ASSERT_TRUE(TupleArgAndParamSum(transformed_fg) == 0);

  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  abstract::AnalysisContext::ClearContext();
}

// Feature: Switch layer call tuple arg transform.
// Description: Test switch layer call's tuple arg transform.This case include partial's tuple arg and the partial's
// tensor arg in the same time.
// Expectation: All tuple args are correctly transformed to tensor args.
TEST_F(TestOptOpt, SwitchLayerPartialTupleTrans) {
  FuncGraphPtr test_graph = getPyFun.CallAndParseRet("test_tuple_flatten", "test_flatten_switch_layer_partial_arg");
  ASSERT_TRUE(nullptr != test_graph);

  FuncGraphManagerPtr manager1 = Manage(test_graph);
  pipeline::ResourcePtr res = std::make_shared<pipeline::Resource>();
  std::vector<AbstractBasePtr> args_spec;

  // Renormalize firstly.
  auto renormalized_fg = pipeline::Renormalize(res, test_graph, args_spec);
  ASSERT_TRUE(TupleArgAndParamSum(renormalized_fg) != 0);

  // Flatten tuple param and args.
  OptimizerPtr optimizer = std::make_shared<Optimizer>("ut_test", res);
  SubstitutionList transform(std::vector<SubstitutionPtr>({tuple_flatten}));
  transform(renormalized_fg, optimizer);

  // Renormalize again.
  auto transformed_fg = pipeline::Renormalize(res, renormalized_fg, args_spec);
  ASSERT_TRUE(TupleArgAndParamSum(transformed_fg) == 0);

  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  abstract::AnalysisContext::ClearContext();
}

// Feature: Single graph call tuple arg transform.
// Description: Test single graph call's tuple arg transform.This case include tuple in tuple args.
// Expectation: All tuple args are correctly transformed to tensor args.
TEST_F(TestOptOpt, SimpleCallTupleTupleTrans) {
  FuncGraphPtr test_graph =
    getPyFun.CallAndParseRet("test_tuple_flatten", "test_flatten_simple_call_tuple_in_tuple_arg");
  ASSERT_TRUE(nullptr != test_graph);

  FuncGraphManagerPtr manager1 = Manage(test_graph);
  pipeline::ResourcePtr res = std::make_shared<pipeline::Resource>();
  std::vector<AbstractBasePtr> args_spec;

  // Renormalize firstly.
  auto renormalized_fg = pipeline::Renormalize(res, test_graph, args_spec);
  ASSERT_TRUE(TupleArgAndParamSum(renormalized_fg) != 0);

  // Flatten tuple param and args.
  OptimizerPtr optimizer = std::make_shared<Optimizer>("ut_test", res);
  SubstitutionList transform(std::vector<SubstitutionPtr>({tuple_flatten}));
  transform(renormalized_fg, optimizer);

  // Renormalize again.
  auto transformed_fg = pipeline::Renormalize(res, renormalized_fg, args_spec);
  ASSERT_TRUE(TupleArgAndParamSum(transformed_fg) == 0);

  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  abstract::AnalysisContext::ClearContext();
}
}  // namespace opt
}  // namespace mindspore
