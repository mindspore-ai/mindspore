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

#include <unordered_map>
#include <set>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <memory>
#include "frontend/parallel/pipeline_transformer/pipeline_transformer.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/group_manager.h"
#include "frontend/parallel/context.h"
#include "utils/comm_manager.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace parallel {
static std::unordered_map<AnfNodePtr, std::set<int>> parameter_color_map;
static int send_tag = 0;
static int recv_tag = 0;

void PipelineTransformer::Coloring() {
  auto need_coloring = true;
  while (need_coloring) {
    need_coloring = false;
    for (auto &fg : manager_->func_graphs()) {
      auto value_nodes = fg->value_nodes();
      for (auto &value_pair : value_nodes) {
        auto node = value_pair.first;
        if (!IsValueNode<FuncGraph>(node)) {
          continue;
        }
        auto graph = GetValueNode<FuncGraphPtr>(node);
        auto node_users = manager_->node_users()[node];
        for (auto &user_pair : node_users) {
          auto user_node = user_pair.first->cast<CNodePtr>();
          user_node->set_stage(graph->stage());
          auto user_node_graph = user_node->func_graph();
          if (graph->stage() == stage_ && user_node_graph->stage() == -1) {
            user_node_graph->set_stage(graph->stage());
            need_coloring = true;
          }
        }
      }
    }
  }
  return;
}

void PipelineTransformer::BroadCastColoring() {
  for (auto &fg : manager_->func_graphs()) {
    DoBroadCast(fg);
  }
}

void PipelineTransformer::DoBroadCast(const FuncGraphPtr &func) {
  auto need_coloring = true;
  while (need_coloring) {
    need_coloring = false;
    auto all_nodes = func->nodes();
    for (auto &node : all_nodes) {
      // only cnode can broadcast color.
      if (!node->isa<CNode>()) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      if (cnode->stage() == -1) {
        // broadcast from inputs to outputs
        for (auto &input : cnode->inputs()) {
          if (input->isa<CNode>() && input->stage() == stage_) {
            cnode->set_stage(input->stage());
            need_coloring = true;
          }
        }
      } else if (cnode->stage() == stage_) {
        // broadcast from outputs to inputs
        for (auto &input : cnode->inputs()) {
          if (input->stage() != -1 || !input->isa<CNode>()) {
            continue;
          }
          auto input_cnode = input->cast<CNodePtr>();
          auto prim = GetValueNode<PrimitivePtr>(input_cnode->input(0));
          if (prim != nullptr && prim->name() == VIRTUAL_DATA_SET) {
            continue;
          }
          input->set_stage(cnode->stage());
          need_coloring = true;
        }
      }
    }
  }
}

void PipelineTransformer::HandleSharedParameter() {
  auto parameters = root_->parameters();
  for (auto &parameter : parameters) {
    auto parameter_stage = parameter_color_map[parameter];
    if (parameter_stage.size() <= 1) {
      continue;
    }
    auto users = manager_->node_users()[parameter];
    for (auto &user : users) {
      auto node = user.first;
      auto graph = node->func_graph();
      if (graph != root_ && graph->stage() == -1) {
        MS_LOG(EXCEPTION) << "Don't support this situation.";
      }
      if (graph == root_ || graph->stage() != stage_) {
        continue;
      }
      if (stage_ == *parameter_stage.begin()) {
        std::vector<AnfNodePtr> make_tuple_input = {NewValueNode(prim::kPrimMakeTuple)};
        for (auto &stage : parameter_stage) {
          if (stage == stage_) {
            continue;
          } else {
            auto send_out = InsertSend(graph, parameter, stage, stage_);
            make_tuple_input.push_back(send_out.depend);
          }
        }
        auto make_tuple = graph->NewCNode(make_tuple_input);
        OperatorAttrs depend_attrs;
        auto depend_op = CreatOpInstance(depend_attrs, DEPEND, "");
        std::vector<AnfNodePtr> depend_input = {NewValueNode(depend_op), parameter, make_tuple};
        auto depend = graph->NewCNode(depend_input);
        manager_->SetEdge(node, user.second, depend);
        break;
      } else {
        InsertReceive(graph, parameter, node, user.second, stage_, *parameter_stage.begin());
        break;
      }
    }
  }
}

void PipelineTransformer::ParameterColoring() {
  auto parameters = root_->parameters();
  for (auto &parameter : parameters) {
    auto users = manager_->node_users()[parameter];
    std::set<int> parameter_stage;
    for (auto &user : users) {
      auto node = user.first;
      auto graph = node->func_graph();
      if (graph != root_ && graph->stage() != -1) {
        parameter_stage.insert(graph->stage());
        parameter->set_stage(graph->stage());
      }
    }
    if (*parameter_stage.begin() == stage_ && !virtual_param_) {
      virtual_param_ = parameter;
    }
    parameter_color_map[parameter] = parameter_stage;
  }
}

static std::pair<ValueListPtr, TypePtr> GetShapeType(const AnfNodePtr &node) {
  abstract::ShapePtr shape_ptr;
  TypePtr type;
  std::vector<int64_t> shape;
  auto cnode = node->cast<CNodePtr>();
  if (cnode != nullptr && IsValueNode<FuncGraph>(cnode->input(0))) {
    auto graph = GetValueNode<FuncGraphPtr>(cnode->input(0));
    auto graph_return = graph->get_return();
    shape_ptr = dyn_cast<abstract::Shape>(graph_return->Shape());
    type = graph_return->Type();
  } else {
    shape_ptr = dyn_cast<abstract::Shape>(node->Shape());
    type = node->Type();
  }
  MS_EXCEPTION_IF_NULL(shape_ptr);
  MS_EXCEPTION_IF_NULL(type);
  auto shape_int = shape_ptr->shape();
  std::vector<ValuePtr> element;
  std::transform(shape_int.begin(), shape_int.end(), std::back_inserter(element),
                 [](int elem) { return MakeValue(elem); });
  auto shape_list = std::make_shared<ValueList>(element);
  auto tensor_type = type->cast<mindspore::TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto dtype = tensor_type->element();
  MS_EXCEPTION_IF_NULL(dtype);
  return std::make_pair(shape_list, dtype);
}

SendAttr PipelineTransformer::InsertSend(const FuncGraphPtr &graph, const AnfNodePtr &parameter,
                                         const int &user_node_stage, const int &node_stage) {
  Attr attr_tag = std::make_pair("sr_tag", MakeValue(send_tag));
  send_tag += 1;
  auto dest_rank = global_rank_ + (user_node_stage - node_stage) * per_stage_rank_num_;
  Attr attr_rank = std::make_pair("dest_rank", MakeValue(dest_rank));
  OperatorAttrs attrs = {attr_tag, attr_rank};
  auto send_op = CreatOpInstance(attrs, "Send", "send");
  auto send_node = NewValueNode(send_op);
  auto prim = GetValueNode<PrimitivePtr>(send_node);
  auto shape_type_pair = GetShapeType(parameter);
  prim->set_attr("shape", shape_type_pair.first);
  prim->set_attr("dtype", shape_type_pair.second);
  std::vector<AnfNodePtr> send_input = {send_node, parameter};
  auto send = graph->NewCNode(send_input);
  OperatorAttrs depend_attrs;
  auto depend_op = CreatOpInstance(depend_attrs, "Depend", "depend");
  std::vector<AnfNodePtr> depend_input = {NewValueNode(depend_op), parameter, send};
  auto depend = graph->NewCNode(depend_input);
  SendAttr send_out = {shape_type_pair.first, shape_type_pair.second, depend};
  return send_out;
}

void PipelineTransformer::InsertReceive(const FuncGraphPtr &graph, const AnfNodePtr &node, const AnfNodePtr &use_node,
                                        const int &index, const int &user_node_stage, const int &node_stage) {
  Attr attr_tag = std::make_pair("sr_tag", MakeValue(recv_tag));
  recv_tag += 1;
  auto src_rank = global_rank_ + (user_node_stage - node_stage) * per_stage_rank_num_;
  Attr attr_rank = std::make_pair("src_rank", MakeValue(src_rank));
  auto shape_type_pair = GetShapeType(node);
  Attr attr_shape = std::make_pair("shape", shape_type_pair.first);
  Attr attr_dtype = std::make_pair("dtype", shape_type_pair.second);
  OperatorAttrs attrs = {attr_tag, attr_rank, attr_shape, attr_dtype};
  auto recv_op = CreatOpInstance(attrs, "Receive", "recv");
  std::vector<AnfNodePtr> recv_input = {NewValueNode(recv_op), virtual_param_};
  auto recv = graph->NewCNode(recv_input);
  manager_->SetEdge(use_node, index, recv);
}

std::pair<bool, int> PipelineTransformer::IsSharedNode(const AnfNodePtr &node, const AnfNodeIndexSet &node_users) {
  std::set<int> tag_set;
  auto node_stage = node->stage();
  int min_tag = node_stage;
  for (auto &user_pair : node_users) {
    auto user_node = user_pair.first;
    auto user_node_stage = user_node->stage();
    tag_set.insert(user_node_stage);
    if (user_node_stage == -1) {
      continue;
    }
    min_tag = min_tag > user_node_stage ? user_node_stage : min_tag;
  }
  bool is_shared = tag_set.size() > 1;
  return std::make_pair(is_shared, min_tag);
}

void PipelineTransformer::CutBorder(const FuncGraphPtr &graph) {
  OperatorAttrs depend_attrs;
  auto depend_op = CreatOpInstance(depend_attrs, "Depend", "");
  std::vector<AnfNodePtr> out_input = {NewValueNode(depend_op)};
  auto all_nodes = graph->nodes();
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>() || node->stage() == -1) {
      continue;
    }
    auto node_users = manager_->node_users()[node];
    auto shared_min_tag_pair = IsSharedNode(node, node_users);
    auto is_shared = shared_min_tag_pair.first;
    auto min_tag = shared_min_tag_pair.second;
    for (auto &user_pair : node_users) {
      auto user_node = user_pair.first;
      auto node_stage = node->stage();
      auto user_node_stage = user_node->stage();
      if (node_stage != stage_ && user_node_stage != stage_) {
        continue;
      }
      if (node_stage < user_node_stage) {
        if (is_shared && (min_tag != node_stage)) {
          continue;
        }
        if (node_stage == stage_) {
          auto send_out = InsertSend(graph, node, user_node_stage, node_stage);
          out_input.insert(out_input.begin() + 1, send_out.depend);
          type_ptr_ = send_out.type;
          shape_ = send_out.shape;
        } else {
          InsertReceive(graph, node, user_node, user_pair.second, user_node_stage, node_stage);
        }
        continue;
      }
      if (node_stage == user_node_stage) {
        if (is_shared && (min_tag != node_stage)) {
          InsertReceive(graph, node, user_node, user_pair.second, stage_, min_tag);
        }
        continue;
      }
      if (node_stage > user_node_stage) {
        auto cnode = node->cast<CNodePtr>();
        auto user_cnode = user_node->cast<CNodePtr>();
        if (IsValueNode<FuncGraph>(cnode->input(0)) && IsValueNode<FuncGraph>(user_cnode->input(0))) {
          MS_LOG(EXCEPTION) << "Don't support this situation";
        }
        continue;
      }
    }
  }
  if (out_input.size() == 2) {
    manager_->Replace(graph->output(), out_input[1]);
  }
  if (out_input.size() > 2) {
    std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
    make_tuple_inputs.insert(make_tuple_inputs.begin() + 1, out_input.begin() + 2, out_input.end());
    auto make_tuple = graph->NewCNode(make_tuple_inputs);
    std::vector<AnfNodePtr> out_depend_inputs = {out_input[0], out_input[1], make_tuple};
    auto out_node = graph->NewCNode(out_depend_inputs);
    manager_->Replace(graph->output(), out_node);
  }
}

void PipelineTransformer::CutGraph() {
  for (auto &fg : manager_->func_graphs()) {
    if (fg == root_) {
      ElimRootParameter();
      continue;
    }
    CutBorder(fg);
  }
}

void PipelineTransformer::ElimRootParameter() {
  auto output = root_->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(output);
  auto prim = GetValueNode<PrimitivePtr>(output->input(0));
  if (prim->name() == DEPEND) {
    auto opt_cnode = output->input(2)->cast<CNodePtr>();
    auto prim_make_tuple = GetValueNode<PrimitivePtr>(opt_cnode->input(0));
    if (prim_make_tuple->name() == MAKE_TUPLE) {
      std::vector<AnfNodePtr> new_node_input = {opt_cnode->input(0)};
      for (auto &input : opt_cnode->inputs()) {
        if (input->isa<CNode>()) {
          if (IsStageNode(input->cast<CNodePtr>())) {
            new_node_input.push_back(input);
          }
        }
      }
      auto new_node = root_->NewCNode(new_node_input);
      manager_->Replace(opt_cnode, new_node);
    }
  }
}

bool PipelineTransformer::IsStageNode(const CNodePtr &node) {
  for (auto &input : node->inputs()) {
    if (input->isa<Parameter>()) {
      return (*parameter_color_map[input].begin() == stage_ || input->stage() == -1);
    } else if (input->isa<CNode>()) {
      auto pre_node = input->cast<CNodePtr>();
      return IsStageNode(pre_node);
    } else {
      continue;
    }
  }
  return true;
}

void PipelineTransformer::ElimGraphStage() {
  for (auto &fg : manager_->func_graphs()) {
    fg->set_stage(-1);
  }
}

bool PipelineTransformer::IsSomePrimitive(const CNodePtr &cnode, const std::string &name) {
  ValueNodePtr anf_node = cnode->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(anf_node);
  PrimitivePtr prim = anf_node->value()->cast<PrimitivePtr>();
  return (prim->name() == name);
}

std::pair<CNodePtr, FuncGraphPtr> PipelineTransformer::FindSensNode() {
  std::pair<CNodePtr, FuncGraphPtr> sens_graph_pair;
  CNodePtr sens_cnode;
  FuncGraphPtr func_graph;
  for (auto &node : root_->nodes()) {
    if (!node->isa<CNode>()) {
      continue;
    }
    sens_cnode = node->cast<CNodePtr>();
    AnfNodePtr expect_tuple_getitem = sens_cnode->input(0);
    MS_EXCEPTION_IF_NULL(expect_tuple_getitem);
    if (!expect_tuple_getitem->isa<CNode>()) {
      continue;
    }

    auto expect_tuple_getitem_cnode = expect_tuple_getitem->cast<CNodePtr>();
    if (!IsSomePrimitive(expect_tuple_getitem_cnode, TUPLE_GETITEM)) {
      continue;
    }
    auto expect_anonymous = expect_tuple_getitem_cnode->input(1);
    if (!expect_anonymous->isa<CNode>()) {
      continue;
    }
    auto expect_anonymous_cnode = expect_anonymous->cast<CNodePtr>();
    AnfNodePtr expect_j = expect_anonymous_cnode->input(0);
    if (!expect_j->isa<CNode>()) {
      continue;
    }
    auto expect_j_cnode = expect_j->cast<CNodePtr>();
    if (!IsSomePrimitive(expect_j_cnode, J)) {
      continue;
    }
    func_graph = GetValueNode<FuncGraphPtr>(expect_j_cnode->input(1));
    break;
  }
  sens_graph_pair = std::make_pair(sens_cnode, func_graph);
  return sens_graph_pair;
}

void PipelineTransformer::CoverSensShape() {
  auto sens_graph_pair = FindSensNode();
  auto sens_cnode = sens_graph_pair.first;
  MS_EXCEPTION_IF_NULL(sens_cnode);
  OperatorAttrs attrs;
  auto fill_op = CreatOpInstance(attrs, "Fill", "");
  std::vector<AnfNodePtr> fill_input = {NewValueNode(fill_op), NewValueNode(type_ptr_),
                                        NewValueNode(MakeValue(shape_->value())), NewValueNode(0)};
  auto fill = root_->NewCNode(fill_input);
  std::vector<AnfNodePtr> new_sens_input = {sens_cnode->input(0), fill};
  auto new_sens_node = root_->NewCNode(new_sens_input);
  manager_->Replace(sens_cnode, new_sens_node);
}

void PipelineTransformer::ElimParameter() {
  auto parameters = root_->parameters();
  std::vector<AnfNodePtr> parameter_list;
  for (auto &parameter : parameters) {
    if (!manager_->node_users()[parameter].empty()) {
      parameter_list.push_back(parameter);
    }
  }
  auto del_num = parameters.size() - parameter_list.size();
  root_->set_hyper_param_count(root_->hyper_param_count() - del_num);
  manager_->SetParameters(root_, parameter_list);
}
}  // namespace parallel
}  // namespace mindspore
