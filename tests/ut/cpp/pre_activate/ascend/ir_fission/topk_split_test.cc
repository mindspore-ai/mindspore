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
#include "pre_activate/ascend/ir_fission/topk_split.h"
#include "debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
class TestHWTopKSplit : public BackendCommon {
 public:
  TestHWTopKSplit() : get_py_fun_("gtest_input.pre_activate.topk_split_test", true) {}
  ~TestHWTopKSplit() override = default;

  UT::PyFuncGraphFetcher get_py_fun_;
};

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
  g->parameters()[0]->set_abstract(x_abstract);
  auto ret = g->get_return();
  EXPECT_NE(ret, nullptr);
  auto tuple_getitem = ret->input(1);
  EXPECT_NE(tuple_getitem, nullptr);
  auto topk = tuple_getitem->cast<CNodePtr>()->input(1);
  topk->set_abstract(x_abstract);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::TopKSplit>());
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(g);
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
