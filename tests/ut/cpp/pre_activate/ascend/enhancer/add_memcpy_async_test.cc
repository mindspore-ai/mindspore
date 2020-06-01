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
#include "common/backend_common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "session/anf_runtime_algorithm.h"
#include "operator/ops.h"
#include "ir/tensor.h"
#include "debug/anf_ir_dump.h"
#include "utils/utils.h"
#include "kernel/kernel_build_info.h"
#include "pre_activate/common/optimizer.h"
#include "pre_activate/ascend/enhancer/add_memcpy_async.h"

namespace mindspore {
namespace opt {
class TestHWAddMemcpyAsync : public BackendCommon {
 public:
  TestHWAddMemcpyAsync() : get_py_fun_("gtest_input.pre_activate.add_memcpy_async", true) {}

 public:
  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWAddMemcpyAsync, test_add_memcpy_async) {
  get_py_fun_.SetDoResolve(true);
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_add_memcpy_async", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int> shp_x{1, 64, 112, 112};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto func_graph = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(func_graph, nullptr);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::AddMemcpyAsync>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  auto new_graph = optimizer->Optimize(func_graph);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_add_memcpy_async", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
