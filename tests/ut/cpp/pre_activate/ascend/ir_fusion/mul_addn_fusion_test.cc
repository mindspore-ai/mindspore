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
#include "plugin/device/ascend/optimizer/ir_fusion/mul_addn_fusion.h"
#include "plugin/device/ascend/optimizer/ir_fission/ascend_convert_tuple_input_to_dynamic_input.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
class TestHWMulAddNFusion : public BackendCommon {
 public:
  TestHWMulAddNFusion() : get_py_fun_("gtest_input.pre_activate.mul_addn_fusion_test", true) {}
  ~TestHWMulAddNFusion() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWMulAddNFusion, test_mul_addn_fusion) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_mul_addn_fusion", "before");
  std::vector<int64_t> shp{2, 2, 2, 2};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list({x_abstract, x_abstract});
  auto fg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AscendConvertTupleInputToDynamicInput>());
  pm->AddPass(std::make_shared<opt::MulAddNFusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_mul_addn_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWMulAddNFusion, test_unmatch) {
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_mul_addn_fusion", "unmatch");
  std::vector<int64_t> shp{2, 2, 2, 2};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list({x_abstract, x_abstract, x_abstract});
  auto fg = GetKernelGraph(g, args_spec_list);
  auto origin_fg = std::make_shared<session::KernelGraph>(*fg);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::AscendConvertTupleInputToDynamicInput>());
  pm->AddPass(std::make_shared<opt::MulAddNFusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(fg);

  EXPECT_TRUE(CheckEqualGraph(origin_fg, new_graph));
}
}  // namespace opt
}  // namespace mindspore
