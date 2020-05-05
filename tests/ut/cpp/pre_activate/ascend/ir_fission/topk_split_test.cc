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
#include "device/kernel_info.h"
#include "pre_activate/pass/convert_const_input_to_attr.h"
#include "debug/anf_ir_dump.h"
#define private public
#define protected public
#include "pre_activate/ascend/ir_fission/topk_split.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
class TestHWTopKSplit : public BackendCommon {
 public:
  TestHWTopKSplit() : get_py_fun_("gtest_input.pre_activate.topk_split_test", true) {}
  ~TestHWTopKSplit() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

class MockSupportedChecker : public SupportedChecker {
 public:
  MockSupportedChecker() = default;
  ~MockSupportedChecker() override = default;
  bool CheckSupported(const AnfNodePtr &anf_node, const kernel::KernelBuildInfoPtr &select_kernel_build_info) override {
    return true;
  }
};  // namespace opt

TEST_F(TestHWTopKSplit, test_topk_split) {
  /*
   * def before(input):
   *     topk = TopKSplit(input)
   *     output = tuple_getitem(topk, 0)
   *     return output
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_topk_split", "before");
  std::vector<int> shp{4, 4};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kernel_graph = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConvertConstInputToAttr>());
  auto topk_split = std::make_shared<opt::TopKSplit>();
  topk_split->supported_checker_ = std::make_shared<MockSupportedChecker>();
  pm->AddPass(topk_split);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kernel_graph);

  auto ret = new_graph->get_return();
  EXPECT_NE(ret, nullptr);
  auto make_tuple = ret->input(1);
  EXPECT_NE(make_tuple, nullptr);
  auto tuple_getitem = make_tuple->cast<CNodePtr>()->input(1);
  EXPECT_NE(tuple_getitem, nullptr);
  auto topk = tuple_getitem->cast<CNodePtr>()->input(1);
  auto topk_cnode = topk->cast<CNodePtr>();
  EXPECT_EQ(topk_cnode->inputs().size(), 3);
  EXPECT_TRUE(topk_cnode->input(2)->isa<ValueNode>());
  auto value_node = topk_cnode->input(2)->cast<ValueNodePtr>();
  EXPECT_TRUE(value_node->value()->isa<tensor::Tensor>());
  auto tensor = value_node->value()->cast<tensor::TensorPtr>();
  EXPECT_EQ(tensor->shape().size(), 1);
  EXPECT_EQ(tensor->shape()[0], 4);
}
}  // namespace opt
}  // namespace mindspore
