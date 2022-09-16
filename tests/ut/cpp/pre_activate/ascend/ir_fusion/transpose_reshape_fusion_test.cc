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
#include "plugin/device/ascend/optimizer/ir_fusion/transpose_reshape_fusion.h"

namespace mindspore {
namespace opt {

class TestHWTransposeReshapeFusion : public BackendCommon {
 public:
  TestHWTransposeReshapeFusion() : get_py_fun_("gtest_input.pre_activate.transpose_reshape_fusion_test", true) {}
  ~TestHWTransposeReshapeFusion() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

TEST_F(TestHWTransposeReshapeFusion, test_transpose_reshape_fusion) {
  /*
   * transpose = Transpose(x, (1, 0, 2, 3))
   * reshape = Reshape(transpose, (2, 2, 16, 16))
   * res = BatchMatMul(reshape, reshape)
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_transpose_reshape_fusion", "before");
  std::vector<int64_t> shp{2, 2, 16, 16};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  // Set Attr for transpose
  auto ret = kg->get_return();
  EXPECT_NE(ret->input(1), nullptr);
  auto bmm = ret->input(1)->cast<CNodePtr>();
  EXPECT_NE(bmm, nullptr);
  auto reshape = bmm->input(1)->cast<CNodePtr>();
  EXPECT_NE(reshape->input(1), nullptr);
  auto transpose = reshape->input(1)->cast<CNodePtr>();
  common::AnfAlgo::SetNodeAttr(kAttrPerm, MakeValue("perm"), transpose);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::TransposeReshapeFusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);

  FuncGraphPtr g_after = get_py_fun_.CallAndParseRet("test_transpose_reshape_fusion", "after");
  EXPECT_TRUE(CheckEqualGraph(g_after, new_graph));
}

TEST_F(TestHWTransposeReshapeFusion, test_transpose_reshape_no_fusion) {
  /*
   * transpose = Transpose(x, (1, 0, 2, 3))
   * reshape = Reshape(transpose, (2, 2, 16, 16))
   * res = BatchMatMul(reshape, reshape)
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_transpose_reshape_fusion", "before");
  std::vector<int64_t> shp{2, 4, 8, 16};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::TransposeReshapeFusion>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kg);
  EXPECT_TRUE(CheckEqualGraph(kg, new_graph));
}
}  // namespace opt
}  // namespace mindspore
