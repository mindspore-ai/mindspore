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
#include "frontend/operator/ops.h"
#include "debug/anf_ir_dump.h"
#include "common/py_func_graph_fetcher.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/optimizer/common/optimizer.h"
#include "backend/optimizer/common/pass_manager.h"
#include "backend/optimizer/pass/const_to_attr_strided_slice_grad.h"
#include "utils/utils.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace opt {
class TestHWConstToAttrStridedSliceGrad : public BackendCommon {
 public:
  TestHWConstToAttrStridedSliceGrad() : getPyFun_("gtest_input.pre_activate.const_to_attr_strided_slice_grad", true) {}
  ~TestHWConstToAttrStridedSliceGrad() override = default;

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};

TEST_F(TestHWConstToAttrStridedSliceGrad, test_strided_slice_grad) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_const_to_attr_strided_slice_grad", "before");
  ASSERT_TRUE(g != nullptr);
  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_const_to_attr_strided_slice_grad", "after");
  ASSERT_TRUE(g_after != nullptr);

  auto ret = g->get_return();
  ASSERT_TRUE(ret != nullptr);
  EXPECT_NE(ret->input(1), nullptr);
  EXPECT_NE(ret->input(1)->cast<CNodePtr>(), nullptr);
  auto cnode = ret->input(1)->cast<CNodePtr>();
  EXPECT_FALSE(AnfAlgo::HasNodeAttr("shapex", cnode));
  EXPECT_FALSE(AnfAlgo::HasNodeAttr("begin", cnode));
  EXPECT_FALSE(AnfAlgo::HasNodeAttr("end", cnode));
  EXPECT_FALSE(AnfAlgo::HasNodeAttr("strides", cnode));
  EXPECT_FALSE(CheckEqualGraph(g, g_after));

  std::vector<int64_t> shp_x{16, 1, 1024};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kg = GetKernelGraph(g, args_spec_list);
  ASSERT_TRUE(kg != nullptr);

  ret = kg->get_return();
  ASSERT_TRUE(ret != nullptr);
  EXPECT_NE(ret->input(1), nullptr);
  EXPECT_NE(ret->input(1)->cast<CNodePtr>(), nullptr);
  auto make_tuple = ret->input(1)->cast<CNodePtr>();
  ASSERT_TRUE(make_tuple != nullptr);
  EXPECT_NE(make_tuple->input(1), nullptr);
  EXPECT_NE(make_tuple->input(1)->cast<CNodePtr>(), nullptr);
  cnode = make_tuple->input(1)->cast<CNodePtr>();
  EXPECT_TRUE(AnfAlgo::HasNodeAttr("shapex", cnode));
  EXPECT_TRUE(AnfAlgo::HasNodeAttr("begin", cnode));
  EXPECT_TRUE(AnfAlgo::HasNodeAttr("end", cnode));
  EXPECT_TRUE(AnfAlgo::HasNodeAttr("strides", cnode));
  EXPECT_TRUE(CheckEqualGraph(kg, g_after));
}
}  // namespace opt
}  // namespace mindspore
