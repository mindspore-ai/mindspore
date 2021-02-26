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
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "ir/dtype.h"
#include "ir/manager.h"
#include "ir/func_graph_cloner.h"
#include "pipeline/jit/parse/parse.h"
#include "frontend/operator/ops.h"
#include "utils/log_adapter.h"
#include "debug/draw.h"
#include "utils/label.h"

namespace mindspore {

namespace {
std::vector<std::string> SplitString(std::string str, std::string pattern) {
  std::string::size_type pos;
  std::vector<std::string> result;
  str += pattern;
  std::string::size_type size = str.size();

  for (std::string::size_type i = 0; i < size; ++i) {
    pos = str.find(pattern, i);
    if (pos < size) {
      std::string s = str.substr(i, pos - i);
      result.push_back(s);
      i = pos + pattern.size() - 1;
    }
  }

  return result;
}
}  // namespace
using std::dynamic_pointer_cast;

using TodoList = std::vector<std::vector<std::pair<std::set<std::pair<AnfNodePtr, int>>, AnfNodePtr>>>;
using TodoListItem = std::vector<std::pair<std::set<std::pair<AnfNodePtr, int>>, AnfNodePtr>>;

class NestingSpecs;

class Stage {
 public:
  explicit Stage(std::vector<std::string> specs) {
    for (auto arg : specs) {
      auto spec = SplitString(arg, "=");
      if (spec.size() <= 1) {
        continue;
      }
      std::shared_ptr<NestingSpecs> nesting = std::make_shared<NestingSpecs>(this, spec[1]);
      specs_[ToFullString(spec[0])] = nesting;
    }
  }

  ~Stage() {}

  std::map<std::string, std::string>& subs() { return subs_; }

  void set_subs(const std::map<std::string, std::string>& subs) { subs_ = subs; }

 private:
  std::string ToFullString(std::string s) {
    if (s.find("fv") != std::string::npos) {
      s = s.replace(s.find("fv"), 2, "free_variable");
    }

    if (s.find("deps") != std::string::npos) {
      s = s.replace(s.find("deps"), 4, "dependencies");
    }

    return s;
  }

  std::map<std::string, std::shared_ptr<NestingSpecs>> specs_;
  std::map<std::string, std::string> subs_;
};

class NestingSpecs {
 public:
  NestingSpecs(Stage* stage, std::string specs) : stage_(stage) { ParseSpecs(specs); }

  ~NestingSpecs() {}

  std::string Name(Any node) {
    std::string name = label_manage::Label(node.cast<AnfNodePtr>()->debug_info());
    if (stage_->subs().find(name) != stage_->subs().end()) {
      return stage_->subs()[name];
    }

    return name;
  }

  void Check(std::shared_ptr<DepComputer> results) {
    if (expected_.empty() && expected_recursive_.empty()) {
      return;
    }

    auto parent = dynamic_pointer_cast<ParentComputer>(results);
    if (parent != nullptr) {
      CheckParent(parent);
      return;
    }

    auto recursive = dynamic_pointer_cast<RecursiveComputer>(results);
    if (recursive != nullptr) {
      CheckRecursive(recursive);
      return;
    }
  }

 private:
  void ParseSpecs(std::string specs) {
    if (specs.empty()) {
      return;
    }

    std::vector<std::string> str_list = SplitString(specs, ";");
    for (auto spec : str_list) {
      spec.erase(0, spec.find_first_not_of(" "));
      spec.erase(spec.find_last_not_of(" ") + 1);
      if (spec.empty()) {
        continue;
      }
      if (spec.find("->") != std::string::npos) {
        auto substr = SplitString(spec, "->");
        ASSERT_GT(substr.size(), 1);
        auto key = substr[0];
        auto value = substr[1];
        if (!value.empty()) {
          expected_[key] = {value};
        }
      } else if (spec.find(":") != std::string::npos) {
        auto substr = SplitString(spec, ":");
        ASSERT_GT(substr.size(), 1);
        auto key = substr[0];
        auto values = SplitString(substr[1], ",");
        std::set<std::string> values_set(values.begin(), values.end());
        if (!values_set.empty()) {
          expected_[key] = values_set;
        }
      } else {
        expected_recursive_[spec] = true;
      }
    }
  }

  void CheckParent(std::shared_ptr<ParentComputer> results) {
    std::map<std::string, std::set<std::string>> clean_results;
    for (auto& iter : results->parent_analysis()) {
      auto key = iter.first;
      auto value = iter.second;
      if (key == nullptr) {
        continue;
      }
      std::string k = Name(key);

      std::set<std::string> v;
      if (value != nullptr && !Name(value).empty()) {
        v.insert(Name(value));
      }

      if (!v.empty()) {
        clean_results[k] = v;
      }
    }

    ASSERT_EQ(clean_results, expected_);
  }

  void CheckRecursive(std::shared_ptr<RecursiveComputer> results) {
    std::map<std::string, bool> clean_results;
    for (auto iter = results->recursive_analysis().begin(); iter != results->recursive_analysis().end(); ++iter) {
      auto key = iter->first;
      auto value = iter->second;
      if (key == nullptr) {
        continue;
      }
      std::string k = Name(key);

      clean_results[k] = value;
    }

    ASSERT_EQ(clean_results, expected_recursive_);
  }

 private:
  Stage* stage_;
  std::map<std::string, std::set<std::string>> expected_;
  std::map<std::string, bool> expected_recursive_;
};

bool CheckUsers(std::shared_ptr<FuncGraphManager> manager) {
  for (auto node : manager->all_nodes()) {
    if (node->isa<CNode>()) {
      auto& inputs = node->cast<CNodePtr>()->inputs();
      for (size_t i = 0; i < inputs.size(); ++i) {
        auto inp = inputs[i];
        if (!manager->all_nodes().contains(inp)) {
          return false;
        }

        if (manager->node_users().find(inp) != manager->node_users().end()) {
          auto users = manager->node_users()[inp];
          if (!users.contains(make_pair(node, i))) {
            return false;
          }
        }
      }
    }

    if (manager->node_users().find(node) != manager->node_users().end()) {
      auto users = manager->node_users()[node];
      for (auto iter = users.begin(); iter != users.end(); ++iter) {
        auto node2 = iter->first;
        auto key = iter->second;
        if (!manager->all_nodes().contains(node2)) {
          return false;
        }
        if (node2->cast<CNodePtr>()->input(key) != node) {
          return false;
        }
      }
    }
  }

  return true;
}

class TestManager : public UT::Common {
 public:
  TestManager() : getPyFun("gtest_input.ir.manager_test") {}

  void CheckAnalysisSize(std::shared_ptr<FuncGraphManager> mng);

 public:
  std::vector<PrimitivePtr> swaps;
  UT::PyFuncGraphFetcher getPyFun;
};

FuncGraphPtr MakeFuncGraph(PrimitivePtr prim) {
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

std::vector<FuncGraphPtr> MakeNestedGraph() {
  /*
   *def f(x):
   *    def g():
   *         return x
   *    return g
   */
  FuncGraphPtr f = std::make_shared<FuncGraph>();
  FuncGraphPtr fg = std::make_shared<FuncGraph>();

  ParameterPtr x = f->add_parameter();

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(fg));
  inputs.push_back(NewValueNode(prim::kPrimReturn));

  CNodePtr cnode_f = f->NewCNode(inputs);
  f->set_return(cnode_f);

  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimReturn));
  inputs.push_back(x);
  CNodePtr cnode_g = fg->NewCNode(inputs);
  fg->set_return(cnode_g);

  std::vector<FuncGraphPtr> result = {f, fg};
  return result;
}

std::vector<FuncGraphPtr> MakeNestedGraph2() {
  /* build a closure func_graph */
  /*
   *def foo(x, y):
   *    def bar(x1):
   *         return x1 + y
   *    return bar(x)
   */
  FuncGraphPtr graph_foo = std::make_shared<FuncGraph>();
  ParameterPtr x = graph_foo->add_parameter();
  ParameterPtr y = graph_foo->add_parameter();

  std::vector<AnfNodePtr> inputs;

  // build func_graph bar
  FuncGraphPtr graph_bar = std::make_shared<FuncGraph>();
  ParameterPtr x1 = graph_bar->add_parameter();
  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimScalarAdd));
  inputs.push_back(x1);
  inputs.push_back(y);
  CNodePtr cnode_add = graph_bar->NewCNode(inputs);
  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimReturn));
  inputs.push_back(cnode_add);
  CNodePtr cnode_return = graph_bar->NewCNode(inputs);
  graph_bar->set_return(cnode_return);

  // build func_graph foo
  inputs.clear();
  inputs.push_back(NewValueNode(graph_bar));
  inputs.push_back(x);
  CNodePtr cnode_graph_bar = graph_foo->NewCNode(inputs);

  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimReturn));
  inputs.push_back(cnode_graph_bar);
  cnode_return = graph_foo->NewCNode(inputs);
  graph_foo->set_return(cnode_return);

  std::vector<FuncGraphPtr> result = {graph_foo, graph_bar};
  return result;
}

// Add TestManager::CheckManager function to checkout the result
void TestManager::CheckAnalysisSize(std::shared_ptr<FuncGraphManager> mng) {
  auto size = mng->func_graphs().size();

  ASSERT_EQ(size, mng->free_variables_total().size());
}

TEST_F(TestManager, test_scalar_add_manual) {
  auto prim_scalar_add = prim::kPrimScalarAdd;
  FuncGraphPtr func_graph = MakeFuncGraph(prim_scalar_add);
  auto mng = Manage(func_graph);
}

TEST_F(TestManager, test_scalar_replace) {
  auto prim_scalar_add = prim::kPrimScalarAdd;

  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  ParameterPtr x = func_graph->add_parameter();
  ParameterPtr y = func_graph->add_parameter();
  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(prim_scalar_add));
  inputs.push_back(x);
  inputs.push_back(y);
  CNodePtr cnode_add = func_graph->NewCNode(inputs);
  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimReturn));
  inputs.push_back(cnode_add);
  CNodePtr cnode_return = func_graph->NewCNode(inputs);
  func_graph->set_return(cnode_return);

  auto mng = Manage(func_graph);
  std::cout << "start " << x->ToString() << std::endl;
  mng->Replace(cnode_add, x);
}

TEST_F(TestManager, test_nested_manual) {
  auto graphs = MakeNestedGraph();
  auto f = graphs[0];
  auto g = graphs[1];

  auto mng = Manage(f);

  ASSERT_EQ(6, mng->all_nodes().size());
  ASSERT_EQ(2, mng->func_graphs().size());
  ASSERT_EQ(4, mng->node_users().size());
  ASSERT_EQ(1, mng->roots().size());
  CheckAnalysisSize(mng);

  ASSERT_EQ(2, f->nodes().size());
  ASSERT_EQ(1, g->nodes().size());

  auto &users = mng->node_users();
  for (auto& iter : users) {
    ASSERT_EQ(1, iter.second.size());
  }

  ASSERT_EQ(1, f->func_graphs_used().size());
  ASSERT_EQ(0, g->func_graphs_used().size());

  ASSERT_EQ(0, f->free_variables().size());
  ASSERT_EQ(1, g->free_variables().size());

  auto fv_total = mng->free_variables_total();
  ASSERT_EQ(0, fv_total[f].size());
  ASSERT_EQ(1, fv_total[g].size());

  ASSERT_EQ(0, f->func_graph_cnodes_index().size());
  ASSERT_EQ(1, g->func_graph_cnodes_index().size());
}

TEST_F(TestManager, test_deep_nested2_manual) {
  // create parser
  FuncGraphPtr func_graph = getPyFun("test_custom");
  return;

  // parse ast to func graph
  FuncGraphPtr gfn = BasicClone(func_graph);
  if (gfn == nullptr) {
    return;
  }

  auto mng = Manage(gfn);

  ASSERT_EQ(3, mng->func_graphs().size());
  ASSERT_EQ(1, mng->roots().size());
  ASSERT_EQ(4, gfn->nodes().size());
  ASSERT_EQ(20, mng->all_nodes().size());
  ASSERT_EQ(25, mng->node_users().size());
  CheckAnalysisSize(mng);
}

TEST_F(TestManager, test_deep_nested_manual) {
  FuncGraphPtr f = std::make_shared<FuncGraph>();
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  FuncGraphPtr h = std::make_shared<FuncGraph>();

  ParameterPtr x = f->add_parameter();
  ParameterPtr y = f->add_parameter();
  ParameterPtr z = f->add_parameter();

  std::vector<AnfNodePtr> inputs;
  inputs.push_back(NewValueNode(fg));
  inputs.push_back(x);
  inputs.push_back(y);
  CNodePtr cnode_1 = f->NewCNode(inputs);

  inputs.clear();
  inputs.push_back(cnode_1);
  inputs.push_back(NewValueNode(prim::kPrimReturn));
  CNodePtr cnode_0 = f->NewCNode(inputs);
  f->set_return(cnode_0);

  ParameterPtr x1 = fg->add_parameter();
  ParameterPtr y1 = fg->add_parameter();
  inputs.clear();
  inputs.push_back(NewValueNode(h));
  inputs.push_back(x1);
  CNodePtr cnode_3 = fg->NewCNode(inputs);

  inputs.clear();
  inputs.push_back(cnode_3);
  inputs.push_back(NewValueNode(prim::kPrimReturn));
  CNodePtr cnode_2 = fg->NewCNode(inputs);
  fg->set_return(cnode_2);

  ParameterPtr x2 = h->add_parameter();

  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimScalarAdd));
  inputs.push_back(x2);
  inputs.push_back(y1);
  CNodePtr cnode_6 = h->NewCNode(inputs);

  inputs.clear();
  inputs.push_back(NewValueNode(prim::kPrimScalarAdd));
  inputs.push_back(z);
  inputs.push_back(cnode_6);
  CNodePtr cnode_5 = h->NewCNode(inputs);

  inputs.clear();
  inputs.push_back(cnode_5);
  inputs.push_back(NewValueNode(prim::kPrimReturn));
  CNodePtr cnode_4 = h->NewCNode(inputs);
  h->set_return(cnode_4);

  auto mng = Manage(f);

  ASSERT_EQ(3, mng->func_graphs().size());
  ASSERT_EQ(1, mng->roots().size());
  ASSERT_EQ(20, mng->all_nodes().size());
  CheckAnalysisSize(mng);
}

TEST_F(TestManager, test_parent1_manual) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();

  Parameter param(fg);
  std::vector<AnfNodePtr> params;
  CNodePtr app = std::make_shared<CNode>(params, fg);
  fg->set_return(app);
  fg->set_parameters(params);

  std::shared_ptr<FuncGraphManager> manager = MakeManager();
  manager->AddFuncGraph(fg, true);
  FuncGraphPtr p = fg->parent();
  assert(p == nullptr);
}

TEST_F(TestManager, test_parent_manual) {
  auto prim_scalar_add = prim::kPrimScalarAdd;
  FuncGraphPtr fg = MakeFuncGraph(prim_scalar_add);

  std::shared_ptr<FuncGraphManager> manager = MakeManager();
  manager->AddFuncGraph(fg);
  FuncGraphPtr p = fg->parent();
  assert(p == nullptr);
}

TEST_F(TestManager, test_flat) {
  std::vector<std::shared_ptr<Stage>> stages;
  std::vector<std::string> specs = {"nodes=X:x", "parents=", "fvs_direct="};
  std::map<std::string, int> size_list;
  size_list["nodes"] = 2;
}

TEST_F(TestManager, test_nested) {
  std::vector<std::shared_ptr<Stage>> stages;
  std::vector<std::string> specs = {"nodes=X:x", "parent=g->X", "fvs_direct=g:x"};
  std::map<std::string, int> size_list;
  return;
}

TEST_F(TestManager, test_calls) {
  std::vector<std::shared_ptr<Stage>> stages;
  std::vector<std::string> specs = {"parents=g->X; h->X", "children=X:g,h", "scopes=X:X,g,h; g:g; h:h",
                                    "fvs_direct=h:a", "fvs_total=h:a; g:h"};
  std::map<std::string, int> size_list;
  return;
}

TEST_F(TestManager, test_unused_param) {
  std::vector<std::shared_ptr<Stage>> stages;
  std::vector<std::string> specs = {"nodes=X:x,y"};
  std::map<std::string, int> size_list;
}

TEST_F(TestManager, test_cannot_replace_return) {
  FuncGraphPtr fg = getPyFun("test_cannot_replace_return");
  ASSERT_NE(fg, nullptr);

  auto mng = Manage(fg);
  ASSERT_EQ(fg->manager(), mng);

  ASSERT_NE(mng, nullptr);
  ASSERT_GT(fg->parameters().size(), 0);
  ASSERT_FALSE(mng->Replace(fg->get_return(), fg->parameters()[0]));
}

TEST_F(TestManager, test_weak_manager) {
  FuncGraphPtr fg = getPyFun("ir_get_fn");

  auto mng1 = MakeManager({fg}, false);
  ASSERT_EQ(fg->manager(), nullptr);
  auto mng2 = MakeManager({fg}, true);
  ASSERT_EQ(fg->manager(), mng2);
  auto mng3 = MakeManager({fg}, false);
  ASSERT_EQ(fg->manager(), mng2);
}

TEST_F(TestManager, test_drop_root) {
  FuncGraphPtr fg = getPyFun("ir_get_fn");

  auto mng = Manage(fg);
  const auto &fgs = mng->func_graphs();
  ASSERT_TRUE(fgs.contains(fg));
  FuncGraphSet s;
  s.add(fg);
  mng->MaybeDropFuncGraphs(s);
  ASSERT_TRUE(fgs.contains(fg));
}

TEST_F(TestManager, test_keep_roots) {
  FuncGraphPtr fg1 = getPyFun("ir_get_fn");
  FuncGraphPtr fg2 = getPyFun("test_cannot_replace_return");

  auto mng = Manage(fg1);
  ASSERT_EQ(mng->func_graphs().size(), (size_t)1);
  ASSERT_TRUE(mng->func_graphs().contains(fg1));

  mng->AddFuncGraph(fg2);
  ASSERT_EQ(mng->func_graphs().size(), 2);
  ASSERT_TRUE(mng->func_graphs().contains(fg2));

  mng->KeepRoots();
  ASSERT_EQ(mng->func_graphs().size(), 1);
  ASSERT_TRUE(mng->func_graphs().contains(fg1));

  mng->KeepRoots({fg2});
  ASSERT_EQ(mng->func_graphs().size(), 1);
  ASSERT_TRUE(mng->func_graphs().contains(fg2));
}

TEST_F(TestManager, test_keep_roots_recursion) {
  return;

  FuncGraphPtr fg = getPyFun("test_keep_roots_recursion");
  ASSERT_NE(fg, nullptr);
  auto mng = Manage(fg);
  parse::ResolveAll(mng);

  ASSERT_NE(mng, nullptr);
  ASSERT_EQ(mng->func_graphs().size(), 4);

  ASSERT_GT(fg->parameters().size(), 0);
  mng->Replace(fg->output(), fg->parameters()[0]);
  ASSERT_EQ(mng->func_graphs().size(), 3);

  mng->KeepRoots();
  ASSERT_EQ(mng->func_graphs().size(), 1);
}

}  // namespace mindspore
