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
#include "debug/anf_ir_dump.h"

#define private public
#define protected public
#include "backend/optimizer/ascend/ir_fusion/add_input_to_output.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
class TestHWAddInputToOutput : public BackendCommon {
 public:
  TestHWAddInputToOutput() : getPyFun_("gtest_input.pre_activate.add_input_to_output_test", true) {}
  ~TestHWAddInputToOutput() override = default;

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};

class MockOpFinder : public OpFinder {
 public:
  MockOpFinder() = default;
  ~MockOpFinder() override = default;
  int GetOpRegisteredOutputNum(const std::string &op_name) override { return 2; }
};

TEST_F(TestHWAddInputToOutput, test_add_input_to_output) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_add_input_to_output", "before");
  EXPECT_NE(g, nullptr);
  std::vector<int> shp{2, 32, 224, 224};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list;
  for (size_t i = 0; i < 5; ++i) {
    args_spec_list.push_back(x_abstract);
  }
  auto kg = GetKernelGraph(g, args_spec_list);
  EXPECT_NE(kg, nullptr);
  auto ret = kg->get_return();
  EXPECT_NE(ret, nullptr);
  auto make_tuple = ret->input(1);
  EXPECT_NE(make_tuple, nullptr);
  auto momentum = make_tuple->cast<CNodePtr>()->input(1);
  EXPECT_NE(momentum, nullptr);
  EXPECT_NE(momentum->abstract(), nullptr);
  EXPECT_FALSE(momentum->abstract()->isa<abstract::AbstractTuple>());

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  auto pass = std::make_shared<opt::AddInputToOutput>();
  pass->op_finder_ = std::make_shared<MockOpFinder>();
  pm->AddPass(pass);
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kg);
  EXPECT_TRUE(momentum->abstract()->isa<abstract::AbstractTuple>());
}
}  // namespace opt
}  // namespace mindspore
