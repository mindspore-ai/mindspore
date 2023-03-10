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
#include "ir/anf.h"
#include "ir/tensor.h"
#include "include/common/debug/anf_ir_dump.h"
#include "common/py_func_graph_fetcher.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/optimizer.h"
#include "include/backend/optimizer/pass_manager.h"
#include "backend/common/pass/convert_const_input_to_tensor_input.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
class TestHWConstInputToTensorInput : public BackendCommon {
 public:
  TestHWConstInputToTensorInput() : getPyFun_("gtest_input.pre_activate.convert_const_input_test", true) {}
  ~TestHWConstInputToTensorInput() override = default;

 public:
  UT::PyFuncGraphFetcher getPyFun_;
};

TEST_F(TestHWConstInputToTensorInput, test_onehot_fg) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_convert_onehot_input_to_tensor1", "before");
  ASSERT_TRUE(g != nullptr);
  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("test_convert_onehot_input_to_tensor1", "after_func_graph");
  ASSERT_TRUE(g_after != nullptr);
  std::vector<int64_t> shp_x{16};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto func_graph = GetFuncGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);
  EXPECT_FALSE(CheckEqualGraph(func_graph, g_after));

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  MS_EXCEPTION_IF_NULL(optimizer);
  auto pm = std::make_shared<opt::PassManager>();
  MS_EXCEPTION_IF_NULL(pm);
  pm->AddPass(std::make_shared<opt::ConvertConstInputToTensorInput>());
  optimizer->AddPassManager(pm);
  optimizer->Optimize(func_graph);
  auto ret = func_graph->get_return();
  ASSERT_TRUE(ret != nullptr);
  EXPECT_NE(ret->input(1), nullptr);
  EXPECT_NE(ret->input(1)->cast<CNodePtr>(), nullptr);
  auto cnode = ret->input(1)->cast<CNodePtr>();
  EXPECT_FALSE(common::AnfAlgo::HasNodeAttr("depth", cnode));
  EXPECT_TRUE(IsValueNode<tensor::Tensor>(cnode->input(2)));
}

/// Feature: Const input to tensor input.
/// Description: Test if can change const input to tensor input successfully.
/// Expectation: Success.
TEST_F(TestHWConstInputToTensorInput, onehot_kg_case) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("convert_onehot_input_to_tensor2", "before");
  ASSERT_TRUE(g != nullptr);
  FuncGraphPtr g_after = getPyFun_.CallAndParseRet("convert_onehot_input_to_tensor2", "after_kernel_graph");
  ASSERT_TRUE(g_after != nullptr);
  EXPECT_FALSE(CheckEqualGraph(g, g_after));
  std::vector<int64_t> shp_x{16};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kInt32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto func_graph = GetKernelGraph(g, args_spec_list);
  ASSERT_TRUE(func_graph != nullptr);

  auto ret = func_graph->get_return();
  ASSERT_TRUE(ret != nullptr);
  EXPECT_NE(ret->input(1), nullptr);
  EXPECT_NE(ret->input(1)->cast<CNodePtr>(), nullptr);
  auto cnode = ret->input(1)->cast<CNodePtr>()->input(1)->cast<CNodePtr>();
  EXPECT_TRUE(common::AnfAlgo::HasNodeAttr("depth", cnode));
  EXPECT_TRUE(CheckEqualGraph(func_graph, g_after));
}

TEST_F(TestHWConstInputToTensorInput, test_value_tuple_tensor_input) {
  FuncGraphPtr g = getPyFun_.CallAndParseRet("test_convert_dropout_gen_mask_tuple_input_to_tensor", "before");
  ASSERT_TRUE(g != nullptr);
  std::vector<int64_t> shp_x{1};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp_x);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kernel_graph = GetKernelGraph(g, args_spec_list);
  ASSERT_TRUE(kernel_graph != nullptr);

  auto ret = kernel_graph->get_return();
  ASSERT_TRUE(ret != nullptr);
  EXPECT_NE(ret->input(1), nullptr);
  EXPECT_NE(ret->input(1)->cast<CNodePtr>(), nullptr);
  auto cnode = ret->input(1)->cast<CNodePtr>()->input(1)->cast<CNodePtr>();
  EXPECT_EQ(common::AnfAlgo::GetCNodeName(cnode), prim::kPrimDropoutGenMask->name());
  auto input1 = cnode->input(1);
  ASSERT_TRUE(input1 != nullptr);
  EXPECT_TRUE(IsValueNode<tensor::Tensor>(input1));
  auto tensor = input1->cast<ValueNodePtr>()->value()->cast<tensor::TensorPtr>();
  ASSERT_TRUE(tensor != nullptr);
  auto data = tensor->data_c();
  EXPECT_EQ(std::vector<int64_t>((int64_t *)data, (int64_t *)data + 4), std::vector<int64_t>({2, 4, 2, 2}));
}
}  // namespace opt
}  // namespace mindspore
