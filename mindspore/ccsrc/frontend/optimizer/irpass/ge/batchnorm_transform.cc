/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/ge/batchnorm_transform.h"

#include <vector>
#include <memory>

#include "pybind_api/pybind_patch.h"
#include "pybind_api/ir/tensor_py.h"
#include "pipeline/pynative/base.h"
#include "pipeline/jit/static_analysis/prim.h"
#include "include/common/utils/python_adapter.h"
#include "include/common/utils/anfalgo.h"
#include "ir/func_graph.h"
#include "frontend/operator/ops.h"

using std::vector;

namespace mindspore {
namespace opt {
namespace irpass {
namespace {
// 1 primitive input, 5 data input, 1 monad input
constexpr int64_t kBNDefaultInputNum = 7;
constexpr int64_t kOutputIndexZero = 0;
constexpr int64_t kOutputIndexOne = 1;
// y, batch_mean, batch_variance
constexpr size_t kBNOutputNum = 3;
constexpr size_t kTupleSize = 2;
constexpr size_t kSubInputTensorNum = 2;
constexpr size_t kMulInputTensorNum = 2;
constexpr char kOpsFunctionModelName[] = "mindspore.ops.function.math_func";
constexpr char kMomentum[] = "momentum";
}  // namespace

void CreateMultiOutputOfAnfNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, size_t output_num,
                                std::vector<AnfNodePtr> *outputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(outputs);
  auto type_ptr = node->Type();
  auto shape_ptr = node->Shape();
  for (size_t i = 0; i < output_num; i++) {
    int64_t temp = SizeToLong(i);
    auto idx = NewValueNode(temp);
    MS_EXCEPTION_IF_NULL(idx);
    auto imm = std::make_shared<Int64Imm>(temp);
    auto abstract_scalar = std::make_shared<abstract::AbstractScalar>(imm);
    idx->set_abstract(abstract_scalar);
    auto tuple_getitem = func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), node, idx});
    tuple_getitem->set_abstract(idx->abstract());
    MS_EXCEPTION_IF_NULL(tuple_getitem);
    common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(type_ptr, i)},
                                                {common::AnfAlgo::GetOutputInferShape(node, shape_ptr, i)},
                                                tuple_getitem.get());
    (*outputs).push_back(tuple_getitem);
  }
}

AnfNodePtr CreateSubNode(const FuncGraphPtr &fg, const vector<AnfNodePtr> &inputs) {
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == kSubInputTensorNum, "Check Sub input size fail!");
  auto mean = inputs[0];
  auto tuple_getitems = inputs[1];
  static py::object sub_prim = python_adapter::GetPyFn(kOpsFunctionModelName, "tensor_sub");
  const auto &sub_adapter = py::cast<PrimitivePyAdapterPtr>(sub_prim);
  MS_EXCEPTION_IF_NULL(sub_adapter);
  auto prim_sub = sub_adapter->attached_primitive();
  if (prim_sub == nullptr) {
    prim_sub = std::make_shared<PrimitivePy>(sub_prim, sub_adapter);
    sub_adapter->set_attached_primitive(prim_sub);
  }
  auto sub_prim_node = NewValueNode(prim_sub);
  MS_EXCEPTION_IF_NULL(sub_prim_node);
  auto sub_node = fg->NewCNode({sub_prim_node, mean, tuple_getitems});
  MS_EXCEPTION_IF_NULL(sub_node);
  sub_node->set_abstract(mean->abstract());
  return sub_node;
}

AnfNodePtr CreateDataNode(const CNodePtr &node) {
  auto prim_bn = GetValueNode<PrimitivePtr>(node->input(0));
  auto momentum_val = prim_bn->GetAttr(kMomentum);
  MS_EXCEPTION_IF_NULL(momentum_val);
  auto tensor_ptr = ScalarToTensor(momentum_val->cast<ScalarPtr>());
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  auto data_node = NewValueNode(MakeValue(tensor_ptr));
  MS_EXCEPTION_IF_NULL(data_node);
  data_node->set_abstract(tensor_ptr->ToAbstract());
  return data_node;
}

AnfNodePtr CreateMulNode(const FuncGraphPtr &fg, const vector<AnfNodePtr> &inputs) {
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == kMulInputTensorNum, "Check Sub input size fail!");
  auto data_node = inputs[0];
  auto sub_node = inputs[1];
  static py::object mul_prim = python_adapter::GetPyFn(kOpsFunctionModelName, "tensor_mul");
  const auto &mul_adapter = py::cast<PrimitivePyAdapterPtr>(mul_prim);
  MS_EXCEPTION_IF_NULL(mul_adapter);
  auto prim_mul = mul_adapter->attached_primitive();
  if (prim_mul == nullptr) {
    prim_mul = std::make_shared<PrimitivePy>(mul_prim, mul_adapter);
    mul_adapter->set_attached_primitive(prim_mul);
  }
  auto mul_prim_node = NewValueNode(prim_mul);
  MS_EXCEPTION_IF_NULL(mul_prim_node);
  auto mul_node = fg->NewCNode({mul_prim_node, data_node, sub_node});
  MS_EXCEPTION_IF_NULL(mul_node);
  mul_node->set_abstract(sub_node->abstract());
  return mul_node;
}

AnfNodePtr CreateAssignSubNode(const FuncGraphPtr &fg, const vector<AnfNodePtr> &inputs) {
  auto ref_var = inputs[0];
  auto mul_node = inputs[1];
  auto update_state_node = inputs[2];
  auto assignsub_prim = NewValueNode(prim::kPrimAssignSub);
  MS_EXCEPTION_IF_NULL(assignsub_prim);
  auto assignsub_node = fg->NewCNode({assignsub_prim, ref_var, mul_node, update_state_node});
  MS_EXCEPTION_IF_NULL(assignsub_node);
  assignsub_node->set_abstract(mul_node->abstract());
  auto mean_prim = GetValueNode<PrimitivePtr>(assignsub_node->input(0));
  MS_EXCEPTION_IF_NULL(mean_prim);
  mean_prim->set_attr("side_effect_mem", MakeValue(true));
  return assignsub_node;
}

void SetScopeForNewNodes(const vector<AnfNodePtr> &nodes, const AnfNodePtr &bn_node) {
  auto bn_scope = bn_node->scope();
  for (auto node : nodes) {
    node->set_scope(bn_scope);
  }
}

AnfNodePtr TransformBatchNorm(const AnfNodePtr &anf_node) {
  auto node = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(node);
  auto fg = node->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto mgr = fg->manager();
  MS_EXCEPTION_IF_NULL(mgr);
  auto bn_inputs = node->inputs();
  MS_EXCEPTION_IF_CHECK_FAIL(bn_inputs.size() == kBNDefaultInputNum, "BatchNorm input size check fail!");
  // process bn
  auto moving_mean = bn_inputs[kIndex4];
  auto moving_var = bn_inputs[kIndex5];
  auto u_monad = bn_inputs[kIndex6];
  // process none
  auto none1 = NewValueNode(std::make_shared<None>());
  MS_EXCEPTION_IF_NULL(none1);
  auto none2 = NewValueNode(std::make_shared<None>());
  MS_EXCEPTION_IF_NULL(none2);
  none1->set_abstract(std::make_shared<abstract::AbstractNone>());
  none2->set_abstract(std::make_shared<abstract::AbstractNone>());
  mgr->SetEdge(node, kIndex4, none1);
  mgr->SetEdge(node, kIndex5, none2);
  // create tuple_get_item node
  vector<AnfNodePtr> tuple_getitems;
  (void)CreateMultiOutputOfAnfNode(fg, node, kBNOutputNum, &tuple_getitems);
  MS_EXCEPTION_IF_CHECK_FAIL(tuple_getitems.size() == kBNOutputNum, "BatchNorm output size check fail!");
  // process load for moving_mean
  auto mean_load_prim = NewValueNode(prim::kPrimLoad);
  auto mean_load_node = fg->NewCNode({mean_load_prim, moving_mean, u_monad});
  MS_EXCEPTION_IF_NULL(mean_load_node);
  auto mean_ref_abs = dyn_cast<abstract::AbstractRef>(moving_mean->abstract());
  mean_load_node->set_abstract(mean_ref_abs->CloneAsTensor());
  // process UpdateState1 for moving_mean
  auto mean_update_state_prim = NewValueNode(prim::kPrimUpdateState);
  auto mean_update_state_node = fg->NewCNode({mean_update_state_prim, u_monad, mean_load_node});
  MS_EXCEPTION_IF_NULL(mean_update_state_node);
  mean_update_state_node->set_abstract(u_monad->abstract());
  // process load for moving_var
  auto var_load_prim = NewValueNode(prim::kPrimLoad);
  auto var_load_node = fg->NewCNode({var_load_prim, moving_var, mean_update_state_node});
  MS_EXCEPTION_IF_NULL(var_load_node);
  auto var_ref_abs = dyn_cast<abstract::AbstractRef>(moving_var->abstract());
  var_load_node->set_abstract(var_ref_abs->CloneAsTensor());
  // process UpdateState2 for moving_var
  auto var_update_state_prim = NewValueNode(prim::kPrimUpdateState);
  auto var_update_state_node = fg->NewCNode({var_update_state_prim, mean_update_state_node, var_load_node});
  MS_EXCEPTION_IF_NULL(var_update_state_node);
  var_update_state_node->set_abstract(mean_update_state_node->abstract());
  // process sub node
  vector<AnfNodePtr> inputs{mean_load_node, tuple_getitems[1]};
  auto mean_sub_node = CreateSubNode(fg, inputs);
  inputs = {var_load_node, tuple_getitems[2]};
  auto var_sub_node = CreateSubNode(fg, inputs);
  // process data node
  auto data_node = CreateDataNode(node);
  // process mul node
  inputs = {data_node, mean_sub_node};
  auto mean_mul_node = CreateMulNode(fg, inputs);
  inputs = {data_node, var_sub_node};
  auto var_mul_node = CreateMulNode(fg, inputs);
  // process assignsub node for moving mean
  inputs = {moving_mean, mean_mul_node, mean_update_state_node};
  auto mean_assignsub_node = CreateAssignSubNode(fg, inputs);
  // process assignsub node for moving variance
  inputs = {moving_var, var_mul_node, var_update_state_node};
  auto var_assignsub_node = CreateAssignSubNode(fg, inputs);
  // process maketuple
  auto make_tuple_prim = NewValueNode(prim::kPrimMakeTuple);
  auto make_tuple_node = fg->NewCNode({make_tuple_prim, mean_assignsub_node, var_assignsub_node});
  MS_EXCEPTION_IF_NULL(make_tuple_node);
  AbstractBasePtrList element_abstracts;
  (void)element_abstracts.emplace_back(mean_assignsub_node->abstract());
  (void)element_abstracts.emplace_back(var_assignsub_node->abstract());
  make_tuple_node->set_abstract(std::make_shared<abstract::AbstractTuple>(element_abstracts));
  // process UpdateState
  MS_EXCEPTION_IF_CHECK_FAIL(mgr->node_users().find(anf_node) != mgr->node_users().end(),
                             "Can't find node in nodes_users.");
  const auto users = mgr->node_users()[anf_node];
  for (auto &iter : users) {
    auto user_node = iter.first;
    auto name = GetCNodeFuncName(user_node->cast<CNodePtr>());
    if (name == prim::kPrimUpdateState->name()) {
      mgr->SetEdge(user_node, kIndex2, make_tuple_node);
    }
  }
  std::vector<AnfNodePtr> new_nodes{none1,
                                    none2,
                                    tuple_getitems[0],
                                    tuple_getitems[1],
                                    tuple_getitems[2],
                                    mean_load_node,
                                    var_load_node,
                                    mean_update_state_node,
                                    var_update_state_node,
                                    mean_sub_node,
                                    var_sub_node,
                                    data_node,
                                    mean_mul_node,
                                    var_mul_node,
                                    mean_assignsub_node,
                                    var_assignsub_node,
                                    make_tuple_node};
  (void)SetScopeForNewNodes(new_nodes, anf_node);
  return anf_node;
}

bool NeedBNTransform(const AnfNodePtr &node) {
  // pass only work in training process and be executed once
  // if executed before the 4th and 5th input of BatchNorm is None
  if (IsPrimitiveCNode(node, prim::kPrimBatchNorm)) {
    auto c_node = node->cast<CNodePtr>();
    auto bn_inputs = c_node->inputs();
    auto input_4 = bn_inputs[kIndex4];
    auto input_5 = bn_inputs[kIndex5];
    if (!IsValueNode<None>(input_4) && !IsValueNode<None>(input_5)) {
      return true;
    }
  }
  return false;
}

void BatchNormTransform(const FuncGraphPtr &fg, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(fg);
  AnfNodeSet all_node = manager->all_nodes();
  for (auto &node : all_node) {
    MS_EXCEPTION_IF_NULL(node);
    AnfNodePtr new_node = nullptr;
    if (NeedBNTransform(node)) {
      MS_LOG(INFO) << "Start to transform BatchNorm";
      new_node = TransformBatchNorm(node);
      // This transformation has to be successful
      if (new_node == nullptr) {
        MS_LOG(EXCEPTION) << "BatchNorm transformation failed!";
      }
      MS_LOG(INFO) << "BatchNorm transform success.";
    }
  }
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
