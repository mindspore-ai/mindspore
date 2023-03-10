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
#include "include/common/debug/anf_ir_dump.h"
#include "common/py_func_graph_fetcher.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pass_manager.h"
#include "backend/common/pass/convert_const_input_to_attr.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace opt {
class TestHWConstInputToAttr : public BackendCommon {
 public:
  TestHWConstInputToAttr() : getPyFun_("gtest_input.pre_activate.convert_const_input_test", true) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kAscendDevice);
  }
  ~TestHWConstInputToAttr() override = default;

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};

TEST_F(TestHWConstInputToAttr, test_reshape) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_convert_reshape_input_to_attr", "before");
  ASSERT_TRUE(g != nullptr);
  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_convert_reshape_input_to_attr", "after");
  ASSERT_TRUE(g_after != nullptr);
  std::vector<int64_t> shp_x{2, 3};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto func_graph = GetKernelGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);

  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}

TEST_F(TestHWConstInputToAttr, test_cast) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_convert_cast_input_to_attr", "before");
  ASSERT_TRUE(g != nullptr);
  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_convert_cast_input_to_attr", "after");
  ASSERT_TRUE(g_after != nullptr);
  std::vector<int64_t> shp_x{2, 3};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto func_graph = GetKernelGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);

  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}

TEST_F(TestHWConstInputToAttr, test_transpose) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_convert_transpose_input_to_attr", "before");
  ASSERT_TRUE(g != nullptr);
  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_convert_transpose_input_to_attr", "after");
  ASSERT_TRUE(g_after != nullptr);
  std::vector<int64_t> shp_x{2, 2, 3};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto func_graph = GetKernelGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);

  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}

/// Feature: Const input to attr.
/// Description: Test if can change const input to attr successfully.
/// Expectation: Success.
TEST_F(TestHWConstInputToAttr, onehot_case) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("convert_onehot_input_to_attr", "before");
  ASSERT_TRUE(g != nullptr);
  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("convert_onehot_input_to_attr", "after");
  ASSERT_TRUE(g_after != nullptr);

  auto ret = g->get_return();
  ASSERT_TRUE(ret != nullptr);
  EXPECT_NE(ret->input(1), nullptr);
  EXPECT_NE(ret->input(1)->cast<CNodePtr>(), nullptr);
  auto cnode = ret->input(1)->cast<CNodePtr>();
  EXPECT_FALSE(common::AnfAlgo::HasNodeAttr("depth", cnode));
  EXPECT_FALSE(CheckEqualGraph(g, g_after));

  std::vector<int64_t> shp_x{16};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto func_graph = GetKernelGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);

  ret = func_graph->get_return();
  ASSERT_TRUE(ret != nullptr);
  EXPECT_NE(ret->input(1), nullptr);
  EXPECT_NE(ret->input(1)->cast<CNodePtr>(), nullptr);
  auto make_tuple = ret->input(1)->cast<CNodePtr>();
  ASSERT_TRUE(make_tuple != nullptr);
  EXPECT_NE(make_tuple->input(1), nullptr);
  EXPECT_NE(make_tuple->input(1)->cast<CNodePtr>(), nullptr);
  cnode = make_tuple->input(1)->cast<CNodePtr>();
  EXPECT_TRUE(common::AnfAlgo::HasNodeAttr("depth", cnode));
  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}
}  // namespace opt
}  // namespace mindspore
