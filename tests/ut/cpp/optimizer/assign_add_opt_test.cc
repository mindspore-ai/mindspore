/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include <iostream>
#include <memory>

#include "common/common_test.h"

#include "mindspore/core/ops/sequence_ops.h"
#include "common/py_func_graph_fetcher.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "ir/manager.h"
#include "ir/value.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "pipeline/jit/ps/resource.h"
#include "include/common/debug/draw.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "include/common/utils/convert_utils.h"
#include "mindspore/ccsrc/pipeline/jit/ps/static_analysis/static_analysis.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "frontend/parallel/pass/assign_add_opt.h"
#include "mindspore/ccsrc/frontend/parallel/device_manager.h"

namespace mindspore {
namespace opt {

class TestAssignAddOpt : public UT::Common {
 public:
  TestAssignAddOpt() {}
  void SetUp() {}
  void TearDown() {}
};

FuncGraphPtr GenerateBackwardFuncGraph() {
  FuncGraphPtr bg = std::make_shared<FuncGraph>();
  bg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  bg->debug_info()->set_name("Backward");
  std::vector<int64_t> shape = {64, 64};
  std::shared_ptr<tensor::Tensor> mock_input_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt32, shape);
  AbstractBasePtr abstract = abstract::FromValue(mock_input_tensor, true);
  AnfNodePtr param0 = bg->add_parameter();
  param0->set_abstract(abstract);
  AnfNodePtr param1 = bg->add_parameter();
  param1->set_abstract(abstract);
  AnfNodePtr param2 = bg->add_parameter();
  param2->set_abstract(abstract);

  AnfNodePtr param3 = bg->add_parameter();
  param3->set_abstract(abstract);
  AnfNodePtr param4 = bg->add_parameter();
  param4->set_abstract(abstract);
  std::vector<AnfNodePtr> inputs;
  (void)inputs.push_back(NewValueNode(prim::kPrimSqrt));
  (void)inputs.push_back(param3);
  CNodePtr square0 = bg->NewCNode(inputs);
  square0->set_abstract(abstract);

  inputs.clear();
  (void)inputs.push_back(NewValueNode(prim::kPrimSqrt));
  (void)inputs.push_back(param4);
  CNodePtr square1 = bg->NewCNode(inputs);
  square1->set_abstract(abstract);

  inputs.clear();
  (void)inputs.push_back(NewValueNode(prim::kPrimMatMul));
  (void)inputs.push_back(square0);
  (void)inputs.push_back(param1);
  CNodePtr matmul0 = bg->NewCNode(inputs);
  PrimitivePtr prim = matmul0->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
  ValuePtr transpose_a = MakeValue(false);
  ValuePtr transpose_b = MakeValue(false);
  prim->set_attr("transpose_a", transpose_a);
  prim->set_attr("transpose_b", transpose_b);
  matmul0->set_abstract(abstract);

  inputs.clear();
  (void)inputs.push_back(NewValueNode(prim::kPrimAssignAdd));
  (void)inputs.push_back(param0);
  (void)inputs.push_back(matmul0);
  CNodePtr assign_add0 = bg->NewCNode(inputs);
  assign_add0->set_abstract(abstract);

  inputs.clear();
  (void)inputs.push_back(NewValueNode(prim::kPrimMatMul));
  (void)inputs.push_back(square1);
  (void)inputs.push_back(param2);
  CNodePtr matmul1 = bg->NewCNode(inputs);
  prim = matmul1->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
  transpose_a = MakeValue(false);
  transpose_b = MakeValue(false);
  prim->set_attr("transpose_a", transpose_a);
  prim->set_attr("transpose_b", transpose_b);
  matmul1->set_abstract(abstract);

  inputs.clear();
  (void)inputs.push_back(NewValueNode(prim::kPrimAssignAdd));
  (void)inputs.push_back(param0);
  (void)inputs.push_back(matmul1);
  CNodePtr assign_add1 = bg->NewCNode(inputs);
  assign_add1->set_abstract(abstract);

  inputs.clear();
  (void)inputs.push_back(NewValueNode(prim::kPrimMakeTuple));
  (void)inputs.push_back(assign_add0);
  (void)inputs.push_back(assign_add1);
  CNodePtr res = bg->NewCNode(inputs);

  abstract::AbstractBasePtrList abs_list({abstract, abstract});
  AbstractBasePtr abstract_tuple = std::make_shared<abstract::AbstractTuple>(abs_list);
  res->set_abstract(abstract_tuple);

  bg->set_output(res);
  return bg;
}

FuncGraphPtr GenerateForwardGraph(FuncGraphPtr bg) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  fg->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  fg->debug_info()->set_name("Forward");
  std::vector<int64_t> shape = {64, 64};
  std::shared_ptr<tensor::Tensor> mock_input_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt32, shape);
  AbstractBasePtr abstract = abstract::FromValue(mock_input_tensor, true);
  AnfNodePtr param0 = fg->add_parameter();
  param0->set_abstract(abstract);
  AnfNodePtr param1 = fg->add_parameter();
  param1->set_abstract(abstract);
  AnfNodePtr param2 = fg->add_parameter();
  param2->set_abstract(abstract);
  std::vector<AnfNodePtr> inputs;
  (void)inputs.push_back(NewValueNode(prim::kPrimPartial));
  (void)inputs.push_back(NewValueNode(bg));
  (void)inputs.push_back(param0);
  (void)inputs.push_back(param1);
  (void)inputs.push_back(param2);
  CNodePtr partial = fg->NewCNode(inputs);
  partial->set_abstract(bg->abstract());
  fg->set_output(partial);
  return fg;
}

// Feature: Assign add and concat eliminate opt.
// Description: Merge matmul and move concat to forward for ge no_task opt.
// Expectation: Each graph has one concat.
TEST_F(TestAssignAddOpt, test_assign_add_opt) {
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<bool>(MS_CTX_ENABLE_CONCAT_ELIMINATE_OPT, true);
  mindspore::parallel::g_device_manager = std::make_shared<mindspore::parallel::DeviceManager>();
  mindspore::parallel::g_device_manager->Init({0, 1}, 0, {1, 1}, "hccl");
  auto bg = GenerateBackwardFuncGraph();
  auto fg = GenerateForwardGraph(bg);
  std::vector<FuncGraphPtr> func_graphs{fg};
  FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(func_graphs, true);
  manager->Init();
  manager->AddFuncGraph(bg);
  parallel::AssignAddOpt(fg);
  std::list<CNodePtr> fg_graph_orders = fg->GetOrderedCnodes();
  std::list<CNodePtr> bg_graph_orders = bg->GetOrderedCnodes();
  size_t fg_concat_size = 0;
  size_t bg_concat_size = 0;
  size_t assign_add_size = 0;
  for (auto node : fg_graph_orders) {
    if (IsPrimitiveCNode(node, prim::kPrimConcat)) {
      bg_concat_size += 1;
    }
  }
  for (auto node : bg_graph_orders) {
    if (IsPrimitiveCNode(node, prim::kPrimConcat)) {
      fg_concat_size += 1;
    }
    if (IsPrimitiveCNode(node, prim::kPrimAssignAdd)) {
      assign_add_size += 1;
    }
  }
  ASSERT_EQ(bg_concat_size, 1);
  ASSERT_EQ(fg_concat_size, 1);
  ASSERT_EQ(assign_add_size, 1);
}

}  // namespace opt
}  // namespace mindspore
