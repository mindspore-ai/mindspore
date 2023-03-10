/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "common/py_func_graph_fetcher.h"
#include "plugin/device/ascend/hal/hardware/ascend_session.h"
#include "pipeline/jit/resource.h"
#include "frontend/operator/ops.h"
#include "ir/manager.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/optimizer.h"
#include "backend/common/pass/getitem_tuple.h"

namespace mindspore {
namespace opt {
class TestHWGetitemTuple : public BackendCommon {
 public:
  TestHWGetitemTuple() : get_py_fun_("gtest_input.pre_activate.getitem_tuple", true) {}
  ~TestHWGetitemTuple() override = default;

 public:
  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWGetitemTuple, test_getitem_tuple) {
  FuncGraphPtr g_before = get_py_fun_.CallAndParseRet("test_getitem_tuple", "before");

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::GetitemTuple>());
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(g_before);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_getitem_tuple", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore