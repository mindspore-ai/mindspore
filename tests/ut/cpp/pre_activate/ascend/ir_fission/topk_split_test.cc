/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "include/backend/kernel_info.h"
#include "backend/common/pass/const_input_to_attr.h"
#include "backend/common/pass/convert_const_input_to_attr.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

#define private public
#define protected public
#include "plugin/device/ascend/optimizer/ir_fission/topk_split.h"
#undef private
#undef protected

namespace mindspore {
namespace opt {
class TestHWTopKSplit : public BackendCommon {
 public:
  TestHWTopKSplit() : get_py_fun_("gtest_input.pre_activate.topk_split_test", true) {}
  ~TestHWTopKSplit() override = default;

  CNodePtr GetTopkCNodeFromKernelGraph(const FuncGraphPtr &func_graph) {
    MS_EXCEPTION_IF_NULL(func_graph);
    auto ret = func_graph->get_return();
    MS_EXCEPTION_IF_NULL(ret);
    auto make_tuple = ret->input(1);
    MS_EXCEPTION_IF_NULL(make_tuple);
    auto tuple_getitem = make_tuple->cast<CNodePtr>()->input(1);
    MS_EXCEPTION_IF_NULL(tuple_getitem);
    auto topk = tuple_getitem->cast<CNodePtr>()->input(1);
    MS_EXCEPTION_IF_NULL(topk);
    auto topk_cnode = topk->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(topk_cnode);
    return topk_cnode;
  }

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
  std::vector<int64_t> shp{4, 4};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kernel_graph = GetKernelGraph(g, args_spec_list);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConvertConstInputToAttr>());
  auto topk_split = std::make_shared<opt::TopKSplit>();
  pm->AddPass(topk_split);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kernel_graph);

  auto topk_cnode = GetTopkCNodeFromKernelGraph(new_graph);
  EXPECT_EQ(topk_cnode->inputs().size(), 3);
  EXPECT_TRUE(topk_cnode->input(2)->isa<ValueNode>());
  auto value_node = topk_cnode->input(2)->cast<ValueNodePtr>();
  EXPECT_TRUE(value_node->value()->isa<tensor::Tensor>());
  auto tensor = value_node->value()->cast<tensor::TensorPtr>();
  EXPECT_EQ(tensor->shape().size(), 1);
  EXPECT_EQ(tensor->shape()[0], 4096*2);
}

TEST_F(TestHWTopKSplit, test_topk_no_split) {
  /*
   * def before(input):
   *     topk = TopKSplit(input)
   *     output = tuple_getitem(topk, 0)
   *     return output
   */
  FuncGraphPtr g = get_py_fun_.CallAndParseRet("test_topk_split", "before");
  std::vector<int64_t> shp{4, 4};
  auto x_abstract = std::make_shared<abstract::AbstractTensor>(kFloat32, shp);
  AbstractBasePtrList args_spec_list{x_abstract};
  auto kernel_graph = GetKernelGraph(g, args_spec_list);

  CNodePtr topk_cnode = GetTopkCNodeFromKernelGraph(kernel_graph);
  EXPECT_EQ(topk_cnode->inputs().size(), 3);
  auto input_names_vec = common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(topk_cnode, kAttrInputNames);
  EXPECT_EQ(input_names_vec.size(), 2);
  mindspore::HashSet<size_t> attr_index{1};
  topk_cnode = ConstInputToAttr(topk_cnode, attr_index);
  EXPECT_EQ(topk_cnode->inputs().size(), 2);
  input_names_vec = common::AnfAlgo::GetNodeAttr<std::vector<std::string>>(topk_cnode, kAttrInputNames);
  EXPECT_EQ(input_names_vec.size(), 2);

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>();
  pm->AddPass(std::make_shared<opt::ConvertConstInputToAttr>());
  auto topk_split = std::make_shared<opt::TopKSplit>();
  pm->AddPass(topk_split);
  optimizer->AddPassManager(pm);
  FuncGraphPtr new_graph = optimizer->Optimize(kernel_graph);
  EXPECT_NE(topk_cnode, GetTopkCNodeFromKernelGraph(new_graph));
}
}  // namespace opt
}  // namespace mindspore
