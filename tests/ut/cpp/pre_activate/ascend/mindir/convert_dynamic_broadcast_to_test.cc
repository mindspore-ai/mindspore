/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "common/backend_common_test.h"
#include "frontend/operator/ops.h"
#include "include/common/debug/anf_ir_dump.h"
#include "common/py_func_graph_fetcher.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pass_manager.h"
#include "utils/ms_utils.h"
#include "backend/common/pass/convert_dynamic_broadcast_to.h"

namespace mindspore {
namespace opt {
namespace {
class TestConvertDynmicBroadcastToPass : public BackendCommon {
 public:
  TestConvertDynmicBroadcastToPass() : getPyFun_("gtest_input.pre_activate.convert_dynamic_broadcast_to_test", true) {}
  ~TestConvertDynmicBroadcastToPass() override = default;

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};

/// Feature: ConvertDynmicBroadcastTo Pass
/// Description: ConvertDynmicBroadcastTo rewrite graph
/// Expectation: Get correct Graph
TEST_F(TestConvertDynmicBroadcastToPass, TestConvert) {
  // build func graph
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_dyn_broadcast", "before");
  std::vector<int64_t> shpx{3};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shpx);
  std::vector<int64_t> shpy{2, 3};
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kInt64, shpy);
  AbstractBasePtrList args_spec_list{x_abstract, y_abstract};
  auto fg = GetFuncGraph(g, args_spec_list);

  bool has_dyn = false;
  for (const auto &n : TopoSort(fg->get_return())) {
    if (IsPrimitiveCNode(n, prim::kPrimDynamicBroadcastTo)) {
      has_dyn = true;
    }
  }
  ASSERT_TRUE(has_dyn);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<ConvertDynamicBroadcastTo>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  for (const auto &n : TopoSort(fg->get_return())) {
    ASSERT_FALSE(IsPrimitiveCNode(n, prim::kPrimDynamicBroadcastTo));
  }
}
}  // namespace
}  // namespace opt
}  // namespace mindspore