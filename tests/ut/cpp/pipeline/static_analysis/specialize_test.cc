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

#include "ir/manager.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "pipeline/jit/static_analysis/program_specialize.h"
#include "pipeline/static_analysis/helper.h"
#include "utils/log_adapter.h"
#include "ir/graph_utils.h"
#include "utils/misc.h"
#include "debug/draw.h"
#include "base/core_ops.h"

namespace mindspore {
namespace abstract {
class TestSpecializeGraph : public UT::Common {
 public:
  void SetUp();
  void TearDown();
  // f(x) call g(x)
  FuncGraphPtr graph_f_;
  FuncGraphPtr graph_g_;
  // alpha(x) return beta(x) closure;
  FuncGraphPtr graph_alpha_;
  FuncGraphPtr graph_beta_;
  std::shared_ptr<AnalysisEngine> engine_;
  std::shared_ptr<ProgramSpecializer> special_;
};

void TestSpecializeGraph::SetUp() {
  UT::InitPythonPath();
  // init resource
  engine_ = SetupAnalysisEngine();

  special_ = std::make_shared<ProgramSpecializer>(engine_);

  /*
   * def g(y):
   *   return y;
   */
  graph_g_ = std::make_shared<FuncGraph>();
  ParameterPtr y = graph_g_->add_parameter();
  auto prim_return = std::make_shared<Primitive>("Return");
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
  inputs.push_back(NewValueNode(std::make_shared<Primitive>(prim::kScalarAdd)));
  inputs.push_back(x1);
  inputs.push_back(y);
  CNodePtr cnode_add = graph_beta_->NewCNode(inputs);
  inputs.clear();
  inputs.push_back(NewValueNode(std::make_shared<Primitive>("Return")));
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

void TestSpecializeGraph::TearDown() {}

TEST_F(TestSpecializeGraph, test_specialize) {
  AbstractBasePtrList args_spec_list;
  MS_LOG(INFO) << "Begin TestSpecializeGraph call other graph.";
  MS_LOG(INFO) << "" << graph_f_->get_return()->ToString();
  AbstractBasePtr abstract_v1 = FromValue(static_cast<int64_t>(1), false);
  args_spec_list.push_back(abstract_v1);

  AnalysisResult result = engine_->Run(graph_f_, args_spec_list);
  FuncGraphPtr new_graph = special_->Run(graph_f_, result.context);
}

TEST_F(TestSpecializeGraph, test_specialize1) {
  AbstractBasePtrList args_spec_list;
  AbstractBasePtr abstract_v1 = FromValue(static_cast<int64_t>(1), true);
  AbstractBasePtr abstract_v2 = FromValue(static_cast<int64_t>(2), true);
  args_spec_list.push_back(abstract_v1);
  args_spec_list.push_back(abstract_v2);
  AnalysisResult result = engine_->Run(graph_alpha_, args_spec_list);
  draw::Draw("befor_graph_alpha.dot", graph_alpha_);
  FuncGraphPtr new_graph = special_->Run(graph_alpha_, result.context);
  if (new_graph) {
    draw::Draw("after_graph_alpha.dot", new_graph);
  }
}

class TestSpecializeMetaFuncGraph : public UT::Common {
 public:
  void SetUp();
  void TearDown();
  FuncGraphPtr graph_;
  std::shared_ptr<AnalysisEngine> engine_;
  std::shared_ptr<ProgramSpecializer> special_;
};

class MetaScalarAdd : public MetaFuncGraph {
 public:
  explicit MetaScalarAdd(std::string name) : MetaFuncGraph(name) {}

  ~MetaScalarAdd() {}
  /*
   * Generate a Graph for the given abstract arguments.
   */
  FuncGraphPtr GenerateFromTypes(const TypePtrList& types) override {
    FuncGraphPtr graph_g = std::make_shared<FuncGraph>();
    ParameterPtr x = graph_g->add_parameter();
    ParameterPtr y = graph_g->add_parameter();
    auto prim_scalar_add = std::make_shared<Primitive>(prim::kScalarAdd);
    std::vector<AnfNodePtr> inputs;
    inputs.push_back(NewValueNode(prim_scalar_add));
    inputs.push_back(x);
    inputs.push_back(y);
    CNodePtr cnode_add = graph_g->NewCNode(inputs);
    auto prim_return = std::make_shared<Primitive>("Return");
    inputs.clear();
    inputs.push_back(NewValueNode(prim_return));
    inputs.push_back(cnode_add);
    CNodePtr cnode_return = graph_g->NewCNode(inputs);
    graph_g->set_return(cnode_return);
    return graph_g;
  }
};

void TestSpecializeMetaFuncGraph::SetUp() {
  UT::InitPythonPath();
  // init resource
  engine_ = SetupAnalysisEngine();
  special_ = std::make_shared<ProgramSpecializer>(engine_);

  /*
   * def f(x, y):
   *   return mata_scalar_add(x, y)
   */
  graph_ = std::make_shared<FuncGraph>();
  ParameterPtr x = graph_->add_parameter();
  ParameterPtr y = graph_->add_parameter();
  std::shared_ptr<MetaFuncGraph> meta_scalar_add = std::make_shared<MetaScalarAdd>("meta_scalar_add");
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(meta_scalar_add));
  inputs.push_back(x);
  inputs.push_back(y);
  CNodePtr cnode_add = graph_->NewCNode(inputs);
  auto prim_return = std::make_shared<Primitive>("Return");
  inputs.clear();
  inputs.push_back(NewValueNode(prim_return));
  inputs.push_back(cnode_add);
  CNodePtr cnode_return = graph_->NewCNode(inputs);
  graph_->set_return(cnode_return);
}

void TestSpecializeMetaFuncGraph::TearDown() {}

TEST_F(TestSpecializeMetaFuncGraph, test_specialize) {
  AbstractBasePtrList args_spec_list;
  std::cout << graph_->get_return()->ToString() << std::endl;
  AbstractBasePtr abstract_v1 = FromValue(static_cast<int64_t>(1), true);
  AbstractBasePtr abstract_v2 = FromValue(static_cast<int64_t>(2), true);
  args_spec_list.push_back(abstract_v1);
  args_spec_list.push_back(abstract_v2);
  AnalysisResult result = engine_->Run(graph_, args_spec_list);

  draw::Draw("befor_graph.dot", graph_);
  FuncGraphPtr new_graph = special_->Run(graph_, result.context);
  if (new_graph) {
    draw::Draw("after_graph.dot", new_graph);
  }
}

}  // namespace abstract
}  // namespace mindspore
