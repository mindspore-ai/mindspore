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
#include <iostream>
#include <memory>

#include "common/common_test.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "common/py_func_graph_fetcher.h"

#include "ir/anf.h"
#include "ir/visitor.h"
#include "ir/func_graph_cloner.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/opt.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/irpass/arithmetic_simplify.h"
#include "pipeline/jit/ps/action.h"

#include "include/common/debug/draw.h"
#include "frontend/operator/ops.h"
#include "include/common/utils/cse.h"
#include "include/common/utils/convert_utils.h"

namespace mindspore {
namespace opt {
class TestRenormalize : public UT::Common {
 public:
  TestRenormalize() : getPyFun("gtest_input.optimizer.renormalize_test", true) {}
  void SetUp() {}

 public:
  UT::PyFuncGraphFetcher getPyFun;
};

// Feature: Specialize.
// Description: If a poly node's parent are not specialized, poly node should be delay specialized.
// Expectation: graph can be executed and no exception raised.
TEST_F(TestRenormalize, TestPolyDelaySpecialize) {
  FuncGraphPtr test_graph = getPyFun.CallAndParseRet("test_renormalize", "test_poly_delay_specialize_ut");
  ASSERT_TRUE(nullptr != test_graph);
  pipeline::ResourcePtr res = std::make_shared<pipeline::Resource>();
  std::vector<AbstractBasePtr> args_spec;
  pipeline::Renormalize(res, test_graph, args_spec);
  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  abstract::AnalysisContext::ClearContext();
}

// Feature: Static analysis of control flow.
// Description: IgnoreValue flag should not be tagged when a function called twice if the function is header of 'if'.
// Expectation: No tuple-getitem exist in specialized graph.
TEST_F(TestRenormalize, TestIgnoreValueTag) {
  FuncGraphPtr test_graph = getPyFun.CallAndParseRet("test_renormalize", "test_ignore_flag_with_twice_call_if");
  ASSERT_TRUE(nullptr != test_graph);
  pipeline::ResourcePtr res = std::make_shared<pipeline::Resource>();
  std::vector<AbstractBasePtr> args_spec;
  auto specialized_fg = pipeline::Renormalize(res, test_graph, args_spec);
  const auto all_nodes = TopoSort(specialized_fg->get_return(), SuccDeeperSimple, AlwaysInclude);
  auto exist_tuple_getitem = std::any_of(all_nodes.cbegin(), all_nodes.cend(), [](const AnfNodePtr &node) {
    return IsPrimitiveCNode(node, prim::kPrimTupleGetItem);
  });
  if (exist_tuple_getitem) {
    DumpIR("test_ignore_flag_with_twice_call_if_error_graph.ir", specialized_fg);
    MS_LOG(ERROR) << "Specialize graph failed, please see the wrong graph in "
                     "'test_ignore_flag_with_twice_call_if_error_graph_0000.ir'";
  }
  ASSERT_EQ(exist_tuple_getitem, false);
  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  abstract::AnalysisContext::ClearContext();
}
}  // namespace opt
}  // namespace mindspore
