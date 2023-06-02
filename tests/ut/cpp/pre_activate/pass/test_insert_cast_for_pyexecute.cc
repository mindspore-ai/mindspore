/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "mindspore/core/ops/array_ops.h"
#include "frontend/operator/ops.h"
#include "ir/tensor.h"
#include "ir/manager.h"
#include "include/common/debug/anf_ir_dump.h"
#include "common/py_func_graph_fetcher.h"
#include "plugin/device/cpu/optimizer/insert_cast_to_pyexecute.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pass_manager.h"
#include "include/backend/kernel_info.h"
#include "plugin/device/ascend/optimizer/format_type/insert_cast.h"
#include "kernel/kernel_build_info.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
class TestInsertCastForPyExecute : public BackendCommon {
 public:
  TestInsertCastForPyExecute() : getPyFun_("gtest_input.pre_activate.insert_cast_for_py_execute", true) {}
  ~TestInsertCastForPyExecute() override = default;

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};

// Feature: Test InsertCastToPyExecute pass
// Description: test pass is correct
// Expectation: success
TEST_F(TestInsertCastForPyExecute, test_insert_cast_for_py_execute) {
  /**
   *  def before(x, y):
   *      x = py_execute_need_cast(x)
   *      y = py_execute_do_not_cast(y)
   *      return x, y
   *  TO:
   *  def after(x, y):
   *      x = py_execute_need_cast(x)
   *      y = py_execute_do_not_cast(y)
   *      x = cast(x)
   *      return x, y
   **/
  common::SetEnv("MS_DEV_FALLBACK_USE_SUPPOSED_DTYPE", "0");
  FuncGraphPtr g = getPyFun_.CallAndParseRet("insert_cast_for_py_execute", "before");

  EXPECT_NE(g, nullptr);

  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract, x_abstract};
  auto fg = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(fg, nullptr);

  // Do insert cast pass of hardware opt
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  InsertCastFunction func = [](const FuncGraphPtr &func_graph, const AnfNodePtr &input, const std::string &,
                               const TypeId &, const TypeId &, const abstract::BaseShapePtr &) -> AnfNodePtr {
    return func_graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name())), input});
  };
  pm->AddPass(std::make_shared<opt::InsertCastToPyExecute>(func));
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);
  EXPECT_NE(new_graph, nullptr);

  // check result
  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("insert_cast_for_py_execute", "after");
  DumpIR("after_py.ir", g_after);
  DumpIR("after_opt.it", new_graph);
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
