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
#include <algorithm>

#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "ops/arithmetic_ops.h"

#include "include/common/debug/draw.h"
#include "ir/func_graph_cloner.h"
#include "ir/graph_utils.h"
#include "ir/manager.h"
#include "pipeline/jit/ps/parse/parse.h"
#include "utils/log_adapter.h"

namespace mindspore {
class FuncGraphIndex {
 public:
  explicit FuncGraphIndex(const FuncGraphPtr &fg, const SearchFunc &search = DeepScopedGraphSearch,
                          const IncludeFunc &include = AlwaysInclude);
  FuncGraphIndex(const FuncGraphIndex &) = delete;
  FuncGraphIndex &operator=(const FuncGraphIndex &) = delete;

  virtual ~FuncGraphIndex() {}

  std::set<FuncGraphPtr> GetFuncGraphs(const std::string &key);
  std::set<AnfNodePtr> GetNodes(const std::string &key);
  FuncGraphPtr GetFirstFuncGraph(const std::string &key);
  AnfNodePtr GetFirstNode(const std::string &key);

 private:
  void Acquire(const FuncGraphPtr &key);
  void Acquire(const AnfNodePtr &key);

  std::map<std::string, std::set<FuncGraphPtr>> index_func_graph_;
  std::map<std::string, std::set<AnfNodePtr>> index_node_;
};

FuncGraphIndex::FuncGraphIndex(const FuncGraphPtr &fg, const SearchFunc &search, const IncludeFunc &include) {
  MS_EXCEPTION_IF_NULL(fg);
  Acquire(fg);
  auto vec = search(fg->get_return(), include);
  for (auto &node : vec) {
    MS_EXCEPTION_IF_NULL(node);
    Acquire(node);
    if (node->func_graph() != nullptr) {
      Acquire(node->func_graph());
    }
  }
}

std::set<FuncGraphPtr> FuncGraphIndex::GetFuncGraphs(const std::string &key) {
  std::set<FuncGraphPtr> func_graphs;
  if (index_func_graph_.find(key) != index_func_graph_.end()) {
    func_graphs = index_func_graph_[key];
  }
  return func_graphs;
}

std::set<AnfNodePtr> FuncGraphIndex::GetNodes(const std::string &key) {
  if (index_node_.find(key) != index_node_.end()) {
    return index_node_[key];
  }
  return std::set<AnfNodePtr>();
}

FuncGraphPtr FuncGraphIndex::GetFirstFuncGraph(const std::string &key) {
  if (GetFuncGraphs(key).empty()) {
    return nullptr;
  }
  auto fg = *GetFuncGraphs(key).begin();
  return fg;
}

AnfNodePtr FuncGraphIndex::GetFirstNode(const std::string &key) {
  if (GetNodes(key).empty()) {
    return nullptr;
  }
  auto node = *GetNodes(key).begin();
  return node;
}

void FuncGraphIndex::Acquire(const FuncGraphPtr &key) {
  std::string name = trace::Label(key->debug_info());
  if (!name.empty()) {
    (void)index_func_graph_[name].insert(key);
  }
}

void FuncGraphIndex::Acquire(const AnfNodePtr &key) {
  std::string name = trace::Label(key->debug_info());
  if (!name.empty()) {
    (void)index_node_[name].insert(key);
  }
}

class TestCloner : public UT::Common {
 public:
  TestCloner() : getPyFun("gtest_input.ir.clone_test", true) {
    one = NewValueNode(static_cast<int64_t>(1));
    two = NewValueNode(static_cast<int64_t>(2));
    three = NewValueNode(static_cast<int64_t>(3));
  }

  FuncGraphPtr GraphForInline() { return nullptr; }
  void SuccessfulInlining(const std::shared_ptr<Cloner> cl, FuncGraphPtr orig, const std::vector<AnfNodePtr> &params,
                          FuncGraphPtr target);

 public:
  UT::PyFuncGraphFetcher getPyFun;

  ValueNodePtr one;
  ValueNodePtr two;
  ValueNodePtr three;
};

void TestCloner::SuccessfulInlining(const std::shared_ptr<Cloner> cl, FuncGraphPtr orig,
                                    const std::vector<AnfNodePtr> &params, FuncGraphPtr target) {
  auto g = (*cl)[orig];
  ASSERT_TRUE(g != target);
  ASSERT_TRUE(g == orig);

  auto new_root = (*cl)[orig->output()];
  ASSERT_TRUE(new_root != orig->output());

  AnfNodeSet orig_nodes = AnfNodeSet(DeepLinkedGraphSearch(orig->output()));
  AnfNodeSet new_nodes = AnfNodeSet(DeepLinkedGraphSearch(new_root));

  for (auto &p : params) {
    ASSERT_TRUE(new_nodes.contains(p));
  }

  for (auto &node : orig_nodes) {
    if (node->func_graph() == orig) {
      ASSERT_TRUE((*cl)[node]);
    }
  }
  ASSERT_TRUE(target->output() == three);
}

TEST_F(TestCloner, test_clone_simple) {
  std::string py_code = "test_clone_simple";

  FuncGraphPtr g = getPyFun.CallAndParseRet(py_code);
  ASSERT_TRUE(g != nullptr);

  std::vector<FuncGraphPtr> gs = {g};
  Cloner cl(gs, true);
  auto g2 = cl[g];

  AnfNodeSet d1 = AnfNodeSet(DeepScopedGraphSearch(g->get_return()));
  AnfNodeSet d2 = AnfNodeSet(DeepScopedGraphSearch(g2->get_return()));

  auto common = d1 & d2;
  ASSERT_EQ((size_t)0, common.size());

  Cloner cl2(gs);
  auto g3 = cl2[g];

  std::vector<Primitive> results = {Primitive(kScalarAddOpName), Primitive(kScalarMulOpName), Primitive("Return")};
  AnfNodeSet d3 = AnfNodeSet(DeepScopedGraphSearch(g3->get_return()));
  common = d1 & d3;
  for (auto &x : common) {
    ASSERT_TRUE(x->isa<ValueNode>());
    ASSERT_TRUE(find(results.begin(), results.end(), *x->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>()) !=
                results.end());
  }
}

TEST_F(TestCloner, test_clone_closure) {
  std::string py_code = "test_clone_closure";

  // parse ast to graph
  FuncGraphPtr parsed_f = getPyFun(py_code);

  FuncGraphIndex idx(parsed_f);
  auto g = idx.GetFirstFuncGraph("j");

  std::vector<FuncGraphPtr> gs = {g};
  Cloner cl(gs, true);

  auto g_clone = cl[g];
  FuncGraphIndex idx2(g_clone, DeepLinkedGraphSearch);

  std::string name_list = "xy";
  for (auto name : name_list) {
    ASSERT_EQ(idx.GetFirstNode(std::string(1, name)), idx2.GetFirstNode(std::string(1, name)));
  }

  ASSERT_FALSE(idx.GetFirstNode("z") == idx2.GetFirstNode("z"));
  ASSERT_FALSE(idx.GetFirstFuncGraph("j") == idx2.GetFirstFuncGraph("j"));
}

TEST_F(TestCloner, test_clone_lifting) {
  std::string py_code = "test_clone_closure";

  // parse ast to graph
  FuncGraphPtr parsed_f = getPyFun(py_code);

  auto g_lifting = LiftingClone(parsed_f);

  FuncGraphIndex idx(g_lifting);
  auto g = idx.GetFirstFuncGraph("j");

  auto params = g_lifting->parameters();
  auto child_params = g->parameters();
  ASSERT_TRUE(params.size() + 1 == child_params.size());
}

TEST_F(TestCloner, test_clone_scoping) {
  std::string py_code = "test_clone_scoping";

  // parse ast to graph
  FuncGraphPtr g = getPyFun.CallAndParseRet(py_code);

  std::vector<FuncGraphPtr> gs = {g};
  Cloner cl(gs, true);

  auto g2 = cl[g];

  FuncGraphIndex idx1(g);
  FuncGraphIndex idx2(g2);

  std::string name_list = "fgi";
  for (auto name : name_list) {
    auto result1 = idx1.GetFirstFuncGraph(std::string(1, name));
    auto result2 = idx2.GetFirstFuncGraph(std::string(1, name));
    ASSERT_FALSE(result1 == result2);
  }

  name_list = "h";
  for (auto name : name_list) {
    ASSERT_TRUE(idx1.GetFirstFuncGraph(std::string(1, name)) == idx2.GetFirstFuncGraph(std::string(1, name)));
  }
}

TEST_F(TestCloner, test_clone_total) {
  std::string py_code = "test_clone_total";

  // parse ast to graph
  getPyFun.SetDoResolve();
  FuncGraphPtr g = getPyFun.CallAndParseRet(py_code);
  if (g == nullptr) {
    return;
  }

  FuncGraphIndex idx0(g);

  std::vector<FuncGraphPtr> gs = {g};
  Cloner cl1(gs, true, true, true);
  auto g2 = cl1[g];
  FuncGraphIndex idx1(g2);

  ASSERT_FALSE(idx0.GetFirstFuncGraph("clone_total_sub") == idx1.GetFirstFuncGraph("clone_total_sub"));
  ASSERT_FALSE(idx0.GetFirstFuncGraph("clone_total") == idx1.GetFirstFuncGraph("clone_total"));

  Cloner cl2(gs, true);
  FuncGraphIndex idx2(cl2[g]);

  ASSERT_FALSE(idx0.GetFirstFuncGraph("clone_total") == idx2.GetFirstFuncGraph("clone_total"));
  ASSERT_TRUE(idx0.GetFirstFuncGraph("clone_total_sub") == idx2.GetFirstFuncGraph("clone_total_sub"));
}

}  // namespace mindspore
