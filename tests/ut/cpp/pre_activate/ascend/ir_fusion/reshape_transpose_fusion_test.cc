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
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/optimizer/ir_fusion/reshape_transpose_fusion.h"

namespace mindspore {
namespace opt {

class TestHWReshapeTransposeFusion : public BackendCommon {
 public:
  TestHWReshapeTransposeFusion() : get_py_fun_("gtest_input.pre_activate.reshape_transpose_fusion_test", true) {}
  ~TestHWReshapeTransposeFusion() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWReshapeTransposeFusion, test_reshape_transpose_fusion) {
  /*
   * x = MatMul(input0, input1)
   * reshape = Reshape(x, (2, 2, 16, 16))
   * transpose = Transpose(reshape, (1, 0, 2, 3))
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_reshape_transpose_fusion", "before");
  std::vector<int64_t> shpx{16, 64};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shpx);
  std::vector<int64_t> shpy{64, 64};
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shpy);
  AbstractBasePtrList args_spec_list{x_abstract, y_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  // Set Attr for transpose
  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto transpose = ret->input(1)->cast<CNodePtr>();
  common::AnfAlgo::SetNodeAttr(kAttrPerm, MakeValue("perm"), transpose);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ReshapeTransposeFusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_reshape_transpose_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWReshapeTransposeFusion, test_reshape_transpose_no_fusion) {
  /*
   * x = MatMul(input0, input1)
   * reshape = Reshape(x, (2, 2, 16, 16))
   * transpose = Transpose(reshape, (1, 0, 2, 3))
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_reshape_transpose_fusion", "before");
  std::vector<int64_t> shpx{4, 256};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shpx);
  std::vector<int64_t> shpy{256, 256};
  auto y_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shpy);
  AbstractBasePtrList args_spec_list{x_abstract, y_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ReshapeTransposeFusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);
  EXPECT_TRUE(CheckEqualGraph(kg, new_graph));
}
}  // namespace opt
}  // namespace mindspore
