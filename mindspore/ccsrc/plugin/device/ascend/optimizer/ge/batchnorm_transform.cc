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

#include "plugin/device/ascend/optimizer/ge/batchnorm_transform.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <map>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/convert_utils.h"
using std::vector;

namespace mindspore {
namespace opt {
namespace {
// 1 primitive input, 5 data input, 1 monad input
constexpr int64_t kBNDefaultInputNum = 7;
constexpr int64_t kOutputIndexZero = 0;
constexpr int64_t kOutputIndexOne = 1;
// y, batch_mean, batch_variance
constexpr size_t kBNOutputNum = 3;
constexpr size_t kTupleSize = 2;
constexpr size_t kSubInputNum = 2;
constexpr size_t kMulInputNum = 2;
constexpr char kMomentum[] = "momentum";

void CreateMultiOutputOfAnfNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, size_t output_num,
                                std::vector<AnfNodePtr> *outputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(outputs);
  auto type_ptr = node->Type();
  auto shape_ptr = node->Shape();
  std::map<int64_t, AnfNodePtr> out;
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &bn = node;
  auto iter = manager->node_users().find(bn);
  if (iter == manager->node_users().end()) {
    return;
  }

  for (const auto &node_index : iter->second) {
    const AnfNodePtr &output = node_index.first;
    MS_EXCEPTION_IF_NULL(output);
    if (!IsPrimitiveCNode(output, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto index = GetGetitemIndex(output);
    out[index] = output;
  }

  for (size_t i = 0; i < output_num; i++) {
    int64_t temp = SizeToLong(i);
    auto fb_iter = out.find(temp);
    if (fb_iter == out.end()) {
      auto tuple_getitem = CreatTupleGetItemNode(func_graph, node, i);
      (*outputs).push_back(tuple_getitem);
    } else {
      (*outputs).push_back(fb_iter->second);
    }
  }
}

AnfNodePtr CreateSubNode(const FuncGraphPtr &fg, const vector<AnfNodePtr> &inputs) {
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == kSubInputNum, "Check Sub input size fail!");
  const auto &mean = inputs[0];
  MS_EXCEPTION_IF_NULL(mean);
  const auto &tuple_getitems = inputs[1];
  auto sub_node = fg->NewCNode({NewValueNode(std::make_shared<Primitive>(kSubOpName)), mean, tuple_getitems});
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
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == kMulInputNum, "Check Sub input size fail!");
  const auto &data_node = inputs[0];
  const auto &sub_node = inputs[1];
  MS_EXCEPTION_IF_NULL(data_node);
  MS_EXCEPTION_IF_NULL(sub_node);
  auto mul_node = fg->NewCNode({NewValueNode(std::make_shared<Primitive>(kMulOpName)), data_node, sub_node});
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
  CreateMultiOutputOfAnfNode(fg, node, kBNOutputNum, &tuple_getitems);
  MS_EXCEPTION_IF_CHECK_FAIL(tuple_getitems.size() == kBNOutputNum, "BatchNorm output size check fail!");
  sort(tuple_getitems.begin(), tuple_getitems.end(), CompareTupleGetitem);
  // process load for moving_mean
  auto mean_load_prim = NewValueNode(prim::kPrimLoad);
  auto mean_load_node = fg->NewCNode({mean_load_prim, moving_mean, u_monad});
  MS_EXCEPTION_IF_NULL(mean_load_node);
  auto mean_ref_abs = dyn_cast<abstract::AbstractRefTensor>(moving_mean->abstract());
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
  auto var_ref_abs = dyn_cast<abstract::AbstractRefTensor>(moving_var->abstract());
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
  SetScopeForNewNodes(new_nodes, anf_node);
  return anf_node;
}
}  // namespace
bool BatchNormTransform::NeedBNTransform(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    AnfNodePtr node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    // pass only work in training process and be executed once
    // if executed before the 4th and 5th input of BatchNorm is None
    if (IsPrimitiveCNode(node, prim::kPrimBatchNorm)) {
      auto c_node = node->cast<CNodePtr>();
      auto bn_inputs = c_node->inputs();
      if (c_node->inputs().size() != kBNDefaultInputNum) {
        return false;
      }
      auto input_4 = bn_inputs[kIndex4];
      auto input_5 = bn_inputs[kIndex5];
      if (!IsValueNode<None>(input_4) && !IsValueNode<None>(input_5)) {
        return true;
      }
    }
  }
  return false;
}

const BaseRef BatchNormTransform::DefinePattern() const {
  MS_LOG(INFO) << "BatchNormTransform::DefinePattern";
  VarPtr x = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimBatchNorm, x});
}

const AnfNodePtr BatchNormTransform::Process(const FuncGraphPtr &fg, const AnfNodePtr &node, const EquivPtr &) const {
  MS_LOG(INFO) << "Start to Process BatchNorm";
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
    return new_node;
  }
  return node;
}
}  // namespace opt
}  // namespace mindspore
