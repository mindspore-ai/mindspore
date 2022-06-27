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

#include "backend/common/pass/add_dropout_attrs.h"
#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
class TestHWAddDropoutAttrs : public BackendCommon {
 public:
  TestHWAddDropoutAttrs() : get_py_fun_("gtest_input.pre_activate.add_dropout_usage_attrs_test", true) {}
  ~TestHWAddDropoutAttrs() override = default;

  FuncGraphPtr RunPass(const FuncGraphPtr &input_graph);

  UT::PyFuncGraphFetcher get_py_fun_;
};

FuncGraphPtr TestHWAddDropoutAttrs::RunPass(const FuncGraphPtr &input_graph) {
  std::vector<int64_t> input_shape{2, 3, 4, 4};
  auto abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, input_shape);
  AbstractBasePtrList args_spec_list{abstract};
  auto kg = GetKernelGraph(input_graph, args_spec_list);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AddDropoutAttrs>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);
  return new_graph;
}

bool IsDropout(const AnfNodePtr &node) {
  if (node != nullptr && node->isa<CNode>() && common::AnfAlgo::GetCNodeName(node) == "Dropout") {
    return true;
  }
  return false;
}

/// Feature: Add dropout usages attr
/// Description: Only first output been used
/// Expectation: pass
TEST_F(TestHWAddDropoutAttrs, test_add_dropout_usage_only_first_output) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("add_dropout_usage_attrs_graph", "only_first_output");
  EXPECT_NE(g, nullptr);
  FuncGraphPtr g_after = RunPass(g);
  auto node_list = TopoSort(g_after->get_return());
  for (auto &node : node_list) {
    if (node == nullptr) {
      continue;
    }
    if (IsDropout(node)) {
      auto cnode = node->cast<CNodePtr>();
      EXPECT_TRUE(cnode != nullptr);
      EXPECT_TRUE(GetValue<bool>(cnode->GetAttr(kAttrOnlyUseFirstOutput)));
    }
  }
}

/// Feature: Add dropout usages attr
/// Description: Only second output been used
/// Expectation: pass
TEST_F(TestHWAddDropoutAttrs, test_add_dropout_usage_only_second_output) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("add_dropout_usage_attrs_graph", "only_second_output");
  EXPECT_NE(g, nullptr);
  FuncGraphPtr g_after = RunPass(g);
  auto node_list = TopoSort(g_after->get_return());
  for (auto &node : node_list) {
    if (node == nullptr) {
      continue;
    }
    if (IsDropout(node)) {
      auto cnode = node->cast<CNodePtr>();
      EXPECT_TRUE(cnode != nullptr);
      EXPECT_TRUE(GetValue<bool>(cnode->GetAttr(kAttrOnlyUseSecondOutput)));
    }
  }
}

/// Feature: Add dropout usages attr
/// Description: All output have been used
/// Expectation: pass
TEST_F(TestHWAddDropoutAttrs, test_add_dropout_usage_all_output) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("add_dropout_usage_attrs_graph", "all_output");
  EXPECT_NE(g, nullptr);
  FuncGraphPtr g_after = RunPass(g);
  auto node_list = TopoSort(g_after->get_return());
  for (auto &node : node_list) {
    if (node == nullptr) {
      continue;
    }
    if (IsDropout(node)) {
      auto cnode = node->cast<CNodePtr>();
      EXPECT_TRUE(cnode != nullptr);
      EXPECT_FALSE(cnode->HasAttr(kAttrOnlyUseFirstOutput));
      EXPECT_FALSE(cnode->HasAttr(kAttrOnlyUseSecondOutput));
    }
  }
}
}  // namespace opt
}  // namespace mindspore
