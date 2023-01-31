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

#include "pipeline/jit/static_analysis/prim.h"
#include "pipeline/static_analysis/helper.h"
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "ir/manager.h"
#include "ir/tensor.h"
#include "frontend/operator/ops.h"
#include "pipeline/jit/parse/parse.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/resource.h"
#include "include/common/debug/draw.h"
#include "utils/log_adapter.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
namespace abstract {
namespace {

AbstractBasePtr InferImplScalarAddStub(const AnalysisEnginePtr &engine, const PrimitivePtr &,
                                       const AbstractBasePtrList &args_spec_list) {
  if (args_spec_list.size() != 2) {
    // assert;
    return nullptr;
  }
  AbstractBasePtr abs_base = args_spec_list[0];
  AbstractBasePtr abs_base_1 = args_spec_list[1];
  return abs_base;
}

EvaluatorPtr InitPrimitiveScalarAddEvaluatorStub() {
  EvaluatorPtr PrimitiveScalarAddEvaluator = std::make_shared<StandardPrimEvaluator>(
    prim::kPrimScalarAdd, StandardPrimitiveImplReg{InferImplScalarAddStub, nullptr, true});
  return PrimitiveScalarAddEvaluator;
}

AbstractBasePtr InferImplReturnStub(const AnalysisEnginePtr &engine, const PrimitivePtr &prim,
                                    const AbstractBasePtrList &args_spec_list) {
  if (args_spec_list.size() != 1) {
    // assert;
    return nullptr;
  }
  AbstractBasePtr abs_base = args_spec_list[0];
  return abs_base;
}

EvaluatorPtr InitPrimitiveReturnEvaluatorStub() {
  EvaluatorPtr PrimitiveReturnEvaluator = std::make_shared<StandardPrimEvaluator>(
    prim::kPrimReturn, StandardPrimitiveImplReg{InferImplReturnStub, nullptr, true});
  return PrimitiveReturnEvaluator;
}

PrimEvaluatorMap PrimEvaluatorConstructorsStub;

/* These stubs is a stub for ut test cases which don't rely on prim module */
void InitPrimEvaluatorConstructorsStub() {
  PrimEvaluatorConstructorsStub[prim::kPrimReturn] = InitPrimitiveReturnEvaluatorStub();
  PrimEvaluatorConstructorsStub[prim::kPrimScalarAdd] = InitPrimitiveScalarAddEvaluatorStub();
}

/* stub for test which don't rely on prim module */
AnalysisEnginePtr SetupAnalysisEngineStub() {
  // init resource
  InitPrimEvaluatorConstructorsStub();
  std::shared_ptr<FuncGraphManager> graph_manager = MakeManager();
  AnalysisEnginePtr engine = std::make_shared<AnalysisEngine>(PrimEvaluatorConstructorsStub, graph_manager);
  return engine;
}

class MetaScalarAdd : public MetaFuncGraph {
 public:
  explicit MetaScalarAdd(std::string name) : MetaFuncGraph(name) {}

  ~MetaScalarAdd() {}
  /*
   * Generate a Graph for the given abstract arguments.
   */
  FuncGraphPtr GenerateFromTypes(const TypePtrList &types) override {
    FuncGraphPtr fg = std::make_shared<FuncGraph>();
    ParameterPtr x = fg->add_parameter();
    ParameterPtr y = fg->add_parameter();
    auto prim_scalar_add = std::make_shared<Primitive>(prim::kScalarAdd);
    std::vector<AnfNodePtr> inputs;
    inputs.push_back(NewValueNode(prim_scalar_add));
    inputs.push_back(x);
    inputs.push_back(y);
    CNodePtr cnode_add = fg->NewCNode(inputs);
    auto prim_return = prim::kPrimReturn;
    inputs.clear();
    inputs.push_back(NewValueNode(prim_return));
    inputs.push_back(cnode_add);
    CNodePtr cnode_return = fg->NewCNode(inputs);
    fg->set_return(cnode_return);
    return fg;
  }
};

}  // namespace

class TestInfer : public UT::Common {
 public:
  void SetUp();
  void TearDown();
  AnalysisEnginePtr engine_;
};

void TestInfer::SetUp() { engine_ = SetupAnalysisEngineStub(); }

static FuncGraphPtr MakeFuncGraph(PrimitivePtr prim) {
  // build the func_graph manually.
  /* python source code:
   * @mindspore
   * def f(x, y):
   *     return x + y
   * print64_t(f(1,2))
   */
  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  ParameterPtr x = func_graph->add_parameter();
  ParameterPtr y = func_graph->add_parameter();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim));
  inputs.push_back(x);
  inputs.push_back(y);
  CNodePtr cnode_add = func_graph->NewCNode(inputs);
  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimReturn));
  inputs.push_back(cnode_add);
  CNodePtr cnode_return = func_graph->NewCNode(inputs);
  func_graph->set_return(cnode_return);
  return func_graph;
}

void TestInfer::TearDown() {
  // destroy resource
}

TEST_F(TestInfer, test_inferred_scalar_add) {
  AbstractBasePtrList args_spec_list;
  int64_t v1 = 1;
  int64_t v2 = 2;

  AbstractBasePtr abstract_v1 = FromValue(v1, false);
  AbstractBasePtr abstract_v2 = FromValue(v2, false);
  args_spec_list.push_back(abstract_v1);
  args_spec_list.push_back(abstract_v2);

  auto prim_scalar_add = std::make_shared<Primitive>(prim::kScalarAdd);
  FuncGraphPtr func_graph = MakeFuncGraph(prim_scalar_add);
  AbstractBasePtr abs_base_got = engine_->Run(func_graph, args_spec_list).eval_result->abstract();
  ASSERT_TRUE(*abs_base_got->BuildValue() == *MakeValue(static_cast<int64_t>(3)));
}

class TestInferGraph : public UT::Common {
 public:
  void SetUp();
  void TearDown();
  // f(x) call g(x)
  FuncGraphPtr graph_f_;
  FuncGraphPtr graph_g_;
  // alpha(x) return beta(x) closure;
  FuncGraphPtr graph_alpha_;
  FuncGraphPtr graph_beta_;
  AnalysisEnginePtr engine_;
};

void TestInferGraph::SetUp() {
  // init resource
  engine_ = SetupAnalysisEngineStub();

  /*
   * def g(y):
   *   return y;
   */
  graph_g_ = std::make_shared<FuncGraph>();
  ParameterPtr y = graph_g_->add_parameter();
  auto prim_return = prim::kPrimReturn;
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim_return));
  inputs.push_back(y);
  CNodePtr cnode_g_ret = graph_g_->NewCNode(inputs);
  graph_g_->set_return(cnode_g_ret);

  /*
   * def f(x):
   *   return g(x)
   */
  graph_f_ = std::make_shared<FuncGraph>();
  ParameterPtr x = graph_f_->add_parameter();
  inputs.clear();
  inputs.push_back(NewValueNode(graph_g_));
  inputs.push_back(x);
  CNodePtr cnode_f = graph_f_->NewCNode(inputs);
  inputs.clear();
  inputs.push_back(NewValueNode(prim_return));
  inputs.push_back(cnode_f);
  CNodePtr cnode_f_ret = graph_f_->NewCNode(inputs);
  graph_f_->set_return(cnode_f_ret);

  /* build a closure func_graph */
  /*
   *def alpha(x, y):
   *    def beta(x1):
   *         return x1 + y
   *    return beta(x)
   */
  graph_alpha_ = std::make_shared<FuncGraph>();
  graph_beta_ = std::make_shared<FuncGraph>();
  x = graph_alpha_->add_parameter();
  y = graph_alpha_->add_parameter();

  // build func_graph beta
  ParameterPtr x1 = graph_beta_->add_parameter();
  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimScalarAdd));
  inputs.push_back(x1);
  inputs.push_back(y);
  CNodePtr cnode_add = graph_beta_->NewCNode(inputs);
  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimReturn));
  inputs.push_back(cnode_add);
  CNodePtr cnode_return = graph_beta_->NewCNode(inputs);
  graph_beta_->set_return(cnode_return);

  // build func_graph alpha
  inputs.clear();
  inputs.push_back(NewValueNode(graph_beta_));
  inputs.push_back(x);
  CNodePtr cnode_graph_beta_ = graph_alpha_->NewCNode(inputs);

  inputs.clear();
  inputs.push_back(NewValueNode(prim_return));
  inputs.push_back(cnode_graph_beta_);
  cnode_return = graph_alpha_->NewCNode(inputs);
  graph_alpha_->set_return(cnode_return);
}

void TestInferGraph::TearDown() {
  // destroy resource
}

TEST_F(TestInferGraph, test_inferred) {
  AbstractBasePtrList args_spec_list;
  MS_LOG(INFO) << "Begin TestInferGraph call other graph.";
  MS_LOG(INFO) << "" << graph_f_->get_return()->ToString();
  AbstractBasePtr abstract_v1 = FromValue(static_cast<int64_t>(1), false);
  args_spec_list.push_back(abstract_v1);
  AbstractBasePtr abs_base_got = engine_->Run(graph_f_, args_spec_list).eval_result->abstract();
  ASSERT_TRUE(abs_base_got.get() == abstract_v1.get());

  // now this test case failed randomly, have to debug.
  MS_LOG(INFO) << "Begin TestInferGraph closure.";
  MS_LOG(INFO) << "" << graph_alpha_->get_return()->ToString();

  AbstractBasePtr abstract_v2 = FromValue(static_cast<int64_t>(2), false);
  args_spec_list.clear();
  args_spec_list.push_back(abstract_v1);
  args_spec_list.push_back(abstract_v2);
  abs_base_got = engine_->Run(graph_alpha_, args_spec_list).eval_result->abstract();
  ASSERT_TRUE(*abs_base_got->BuildValue() == *MakeValue(static_cast<int64_t>(3)));
}

TEST_F(TestInferGraph, test_context) {
  std::shared_ptr<FuncGraphManager> graph_manager_ = MakeManager();

  graph_manager_->AddFuncGraph(graph_f_);
  graph_manager_->AddFuncGraph(graph_g_);

  graph_manager_->AddFuncGraph(graph_alpha_);
  graph_manager_->AddFuncGraph(graph_beta_);

  ASSERT_TRUE(graph_f_->parent() == nullptr);
  ASSERT_TRUE(graph_g_->parent() == nullptr);

  ASSERT_TRUE(graph_beta_->parent() == graph_alpha_);
  ASSERT_TRUE(graph_alpha_->parent() == nullptr);

  AnalysisContextPtr dummy_context = AnalysisContext::DummyContext();

  AnalysisContextPtr f_context = dummy_context->NewContext(graph_f_, AbstractBasePtrList());
  ASSERT_TRUE(f_context->FindOwnOrParentContext(graph_f_.get()) = f_context);
  ASSERT_TRUE(f_context->FindOwnOrParentContext(nullptr) = dummy_context);

  AnalysisContextPtr g_context = f_context->NewContext(graph_g_, AbstractBasePtrList());
  ASSERT_TRUE(g_context->FindOwnOrParentContext(graph_g_.get()) = g_context);
  ASSERT_TRUE(g_context->FindOwnOrParentContext(graph_f_.get()) = dummy_context);
  ASSERT_TRUE(g_context->FindOwnOrParentContext(nullptr) = dummy_context);

  AnalysisContextPtr alpha_context = dummy_context->NewContext(graph_alpha_, AbstractBasePtrList());
  ASSERT_TRUE(alpha_context->FindOwnOrParentContext(graph_alpha_.get()) = alpha_context);
  ASSERT_TRUE(alpha_context->FindOwnOrParentContext(nullptr) = dummy_context);

  AnalysisContextPtr beta_context = alpha_context->NewContext(graph_beta_, AbstractBasePtrList());
  ASSERT_TRUE(beta_context->FindOwnOrParentContext(graph_beta_.get()) = beta_context);
  ASSERT_TRUE(beta_context->FindOwnOrParentContext(graph_alpha_.get()) = alpha_context);
  ASSERT_TRUE(beta_context->FindOwnOrParentContext(nullptr) = dummy_context);
}

class TestInferMetaGraph : public UT::Common {
 public:
  void SetUp();
  void TearDown();
  FuncGraphPtr func_graph_;
  AnalysisEnginePtr engine_;
};

void TestInferMetaGraph::SetUp() {
  // init resource
  engine_ = SetupAnalysisEngineStub();

  /*
   * def f(x, y):
   *   return mata_scalar_add(x, y)
   */
  func_graph_ = std::make_shared<FuncGraph>();
  ParameterPtr x = func_graph_->add_parameter();
  ParameterPtr y = func_graph_->add_parameter();
  std::shared_ptr<MetaFuncGraph> meta_scalar_add = std::make_shared<MetaScalarAdd>("meta_scalar_add");
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(meta_scalar_add));
  inputs.push_back(x);
  inputs.push_back(y);
  CNodePtr cnode_add = func_graph_->NewCNode(inputs);
  auto prim_return = prim::kPrimReturn;
  inputs.clear();
  inputs.push_back(NewValueNode(prim_return));
  inputs.push_back(cnode_add);
  CNodePtr cnode_return = func_graph_->NewCNode(inputs);
  func_graph_->set_return(cnode_return);
}

void TestInferMetaGraph::TearDown() {
  // destroy resource
}

TEST_F(TestInferMetaGraph, test_inferred) {
  AbstractBasePtrList args_spec_list;
  int64_t v1 = 1;
  int64_t res = 2;
  std::cout << "Begin TestInferGraph." << std::endl;
  std::cout << func_graph_->get_return()->ToString() << std::endl;
  AbstractBasePtr abstract_v1 = FromValue(v1, false);
  AbstractBasePtr abstract_v2 = FromValue(v1, false);
  args_spec_list.push_back(abstract_v1);
  args_spec_list.push_back(abstract_v2);
  AbstractBasePtr abs_base_got = engine_->Run(func_graph_, args_spec_list).eval_result->abstract();
  ASSERT_TRUE(*abs_base_got->BuildValue() == *MakeValue(res));
}

class TestInferUniform : public UT::Common {
 public:
  void SetUp();
  void TearDown();
  AnalysisEnginePtr engine_;
};

void TestInferUniform::SetUp() {
  // init resource
  engine_ = SetupAnalysisEngine();
}

void TestInferUniform::TearDown() {
  // destroy resource
}

TEST_F(TestInferUniform, test_inferred_scalar_add) {
  AbstractBasePtrList args_spec;
  int64_t v1 = 1;
  int64_t v2 = 2;

  AbstractBasePtr abstract_v1 = FromValue(v1, false);
  AbstractBasePtr abstract_v2 = FromValue(v2, false);
  args_spec.push_back(abstract_v1);
  args_spec.push_back(abstract_v2);

  auto prim_scalar_add = std::make_shared<Primitive>(prim::kScalarAdd);
  FuncGraphPtr func_graph = MakeFuncGraph(prim_scalar_add);
  AbstractBasePtr abs_base_got = engine_->Run(func_graph, args_spec).eval_result->abstract();
  ASSERT_TRUE(*(abs_base_got->GetTypeTrack()) == *(abstract_v1->GetTypeTrack()));
  ASSERT_TRUE(abs_base_got->GetTypeTrack()->type_id() == kNumberTypeInt64);
}


class TestGraphEval : public UT::Common {
 public:
  TestGraphEval() : getPyFun("gtest_input.pipeline.infer.infer_test", true){};
  void SetUp();
  void TearDown();
  AnalysisEnginePtr engine_;
  UT::PyFuncGraphFetcher getPyFun;
};

void TestGraphEval::SetUp() { engine_ = SetupAnalysisEngine(); }

void TestGraphEval::TearDown() {
  // destroy resource
  engine_->ClearEvaluatorCache();
  parse::data_converter::ClearObjectCache();
}

/* skip ut test cases temporarily
TEST_F(TestGraphInfer, test_graph_infer_defaults) {
  FuncGraphPtr graph = getPyFun.CallAndParseRet("test_graph_infer_defaults");
  AbstractBasePtrList args_spec_list = {};
  AbstractBasePtr res = engine_->Run(graph, args_spec_list).eval_result->abstract();
  AbstractBasePtr expect = FromValue(MakeValue(50), false);
  ASSERT_EQ(*res, *expect);
}

TEST_F(TestGraphInfer, test_graph_infer_vararg_0) {
  FuncGraphPtr graph = getPyFun.CallAndParseRet("test_graph_infer_vararg_0");
  AbstractBasePtrList args_spec_list = {};
  AbstractBasePtr res = engine_->Run(graph, args_spec_list).eval_result->abstract();
  AbstractBasePtr expect = FromValue(MakeValue(1), false);
  ASSERT_EQ(*res, *expect);
}

TEST_F(TestGraphInfer, test_graph_infer_vararg) {
  FuncGraphPtr graph = getPyFun.CallAndParseRet("test_graph_infer_vararg");
  AbstractBasePtrList args_spec_list = {};
  AbstractBasePtr res = engine_->Run(graph, args_spec_list).eval_result->abstract();
  AbstractBasePtr expect = FromValue(MakeValue(9), false);
  ASSERT_EQ(*res, *expect);
}

TEST_F(TestGraphInfer, test_graph_infer_vararg_kwonlyargs) {
  FuncGraphPtr graph = getPyFun.CallAndParseRet("test_graph_infer_vararg_kwonlyargs");
  AbstractBasePtrList args_spec_list = {};
  AbstractBasePtr res = engine_->Run(graph, args_spec_list).eval_result->abstract();
  AbstractBasePtr expect = FromValue(MakeValue(48), false);
  ASSERT_EQ(*res, *expect);
}

TEST_F(TestGraphInfer, test_graph_infer_kwarg) {
  FuncGraphPtr graph = getPyFun.CallAndParseRet("test_graph_infer_kwarg");
  AbstractBasePtrList args_spec_list = {};
  AbstractBasePtr res = engine_->Run(graph, args_spec_list).eval_result->abstract();
  AbstractBasePtr expect = FromValue(MakeValue(7), false);
  ASSERT_EQ(*res, *expect);
}

TEST_F(TestGraphInfer, test_graph_infer_vararg_kwonlyargs_kwarg) {
  FuncGraphPtr graph = getPyFun.CallAndParseRet("test_graph_infer_vararg_kwonlyargs_kwarg");
  AbstractBasePtrList args_spec_list = {};
  AbstractBasePtr res = engine_->Run(graph, args_spec_list).eval_result->abstract();
  AbstractBasePtr expect = FromValue(MakeValue(46), false);
  ASSERT_EQ(*res, *expect);
}

TEST_F(TestGraphInfer, test_graph_infer_vararg_kwonlyargs_kwarg_defaults) {
  FuncGraphPtr graph = getPyFun.CallAndParseRet("test_graph_infer_vararg_kwonlyargs_kwarg_defaults");
  AbstractBasePtrList args_spec_list = {};
  AbstractBasePtr res = engine_->Run(graph, args_spec_list).eval_result->abstract();
  AbstractBasePtr expect = FromValue(MakeValue(57), false);
  ASSERT_EQ(*res, *expect);
}
*/

}  // namespace abstract
}  // namespace mindspore
