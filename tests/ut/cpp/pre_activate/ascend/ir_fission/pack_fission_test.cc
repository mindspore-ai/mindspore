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
#define private public
#define protected public
#include "plugin/device/ascend/optimizer/ir_fission/pack_fission.h"
#include "plugin/device/ascend/optimizer/ir_fission/ascend_convert_tuple_input_to_dynamic_input.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
class TestHWPackFission : public BackendCommon {
 public:
  TestHWPackFission() : get_py_fun_("gtest_input.pre_activate.stack_fission_test", true) {}
  ~TestHWPackFission() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWPackFission, test_stack_fission_divided_by_3) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_stack_fission", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 9; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AscendConvertTupleInputToDynamicInput>());
  auto pack_fission = std::make_shared<opt::PackFission>();
  pack_fission->inputs_divisor_ = 3;
  pm->AddPass(pack_fission);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_stack_fission", "after_divided_by_3");
  EXPECT_NE(g_after, nullptr);
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWPackFission, test_stack_fission_divided_by_4) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_stack_fission", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int64_t> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 9; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AscendConvertTupleInputToDynamicInput>());
  auto pack_fission = std::make_shared<opt::PackFission>();
  pack_fission->inputs_divisor_ = 4;
  pm->AddPass(pack_fission);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_stack_fission", "after_divided_by_4");
  EXPECT_NE(g_after, nullptr);
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}
}  // namespace opt
}  // namespace mindspore
