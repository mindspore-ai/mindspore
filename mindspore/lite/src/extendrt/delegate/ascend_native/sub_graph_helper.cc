/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "extendrt/delegate/ascend_native/sub_graph_helper.h"
#include <fstream>
#include <string>
#include "extendrt/utils/func_graph_utils.h"
#include "ops/op_name.h"
#include "ops/tuple_get_item.h"
#include "ops/make_tuple.h"
#include "extendrt/delegate/ascend_native/ops/ascend_native_composite.h"

namespace mindspore {
AnSubGraph::AnSubGraph(int index) : index_{index} { func_graph_ = std::make_shared<FuncGraph>(); }
void AnSubGraph::Add(const CNodePtr &cnode) {
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (prim->name() == ops::kNameAscendNativeComposite) {
    int idx = static_cast<int>(GetValue<int64_t>(prim->GetAttr(ops::kGroup)));
    MS_LOG(ERROR) << "cannot add composite in graph #" << index_ << "from " << idx;
    return;
  }
  cnode->set_func_graph(func_graph_);
  func_graph_->AddNode(cnode);
}

int AnSubGraph::Size() const { return func_graph_->nodes().size(); }

void AnSubGraph::AddInput(const AnfNodePtr &node) {
  if (input_set_.find(node) == input_set_.end()) {
    input_set_.insert(node);
    inputs_.push_back(node);
  }
}

void AnSubGraph::AddOutput(const AnfNodePtr &node) {
  if (output_set_.find(node) == output_set_.end()) {
    output_set_.insert(node);
    outputs_.push_back(node);
  }
}

int AnSubGraph::GetOutputId(const CNodePtr &cnode) const {
  auto it = std::find(outputs_.begin(), outputs_.end(), cnode);
  if (it != outputs_.end()) {
    return it - outputs_.begin();
  }
  return -1;
}

CNodePtr AnSubGraph::CreateTuple() {
  auto tuple_prim = std::make_shared<ops::MakeTuple>();
  auto tuple_prim_c = tuple_prim->GetPrim();
  CNodePtr tuple_cnode = func_graph_->NewCNode(tuple_prim_c, {outputs_});
  tuple_cnode->set_fullname_with_scope("composite_" + std::to_string(index_) + "/make_tuple");
  return tuple_cnode;
}

void AnSubGraph::FixGroup(SubGraphHelperPtr helper) {
  // handle inputs
  std::vector<AnfNodePtr> inputs;
  for (size_t i = 1; i < cnode_->inputs().size(); i++) {
    auto input = cnode_->input(i);
    auto is_ginput = helper->IsGraphInput(input);
    int group = helper->FindSubGraph(input);
    if (group >= 0) {
      input = helper->GetCNode(group);
      cnode_->set_input(i, input);
    } else {
      CNodePtr cin;
      auto prim = GetCNodePrimitive(input);
      bool is_copy = (prim != nullptr) && (prim->name() == ops::kNameCopy);
      if ((input->isa<CNode>() && !is_copy) || is_ginput) {
        auto connect_node = helper->CreateGetItemAndCopyUnique(input, 0, cin, ops::Copy::CopyFormatType::HOST_DEVICE);
        cnode_->set_input(i, connect_node);
      }
    }
    if (input->isa<CNode>() || is_ginput) {
      auto para = std::make_shared<Parameter>(func_graph_);
      const std::string name = input->fullname_with_scope() + "/input_" + std::to_string(i - 1);
      para->set_name(name);
      para->debug_info()->set_name(name);
      para->set_abstract(cnode_->input(i)->abstract());
      inputs.push_back(para);
      // replace all function node input to user func_graph inputs
      for (const auto &node : func_graph_->nodes()) {
        if (node->isa<CNode>()) {
          const auto &cnode = node->cast<CNodePtr>();
          for (size_t j = 1; j < cnode->inputs().size(); j++) {
            const auto &node_in = cnode->input(j);
            if (node_in == input) {
              cnode->set_input(j, para);
            }
          }
        }
      }
    }
  }
  func_graph_->set_parameters(inputs);
  // handle outputs
  int outCount = GetOutputsCount();
  if (outCount < 1) {
    MS_LOG(ERROR) << "composite don't have outputs";
  }
  if (GetOutputsCount() > 1) {
    auto tuple = CreateTuple();
    func_graph_->set_output(tuple);
  } else {
    func_graph_->set_output(outputs_.at(0));
  }
}

void AnSubGraph::DumpNode(const AnfNodePtr &node) {
  std::cout << node->fullname_with_scope() << " ";
  if (node->isa<CNode>()) {
    const auto &cnode = node->cast<CNodePtr>();
    const auto &prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    std::cout << prim->name();
  }
  std::cout << std::endl;
}

void AnSubGraph::Dump() {
  int count = 0;
  std::cout << "graph have " << func_graph_->get_inputs().size() << " inputs" << std::endl;
  for (const auto &in : func_graph_->get_inputs()) {
    DumpNode(in);
  }
  std::cout << "graph have " << func_graph_->nodes().size() << " nodes" << std::endl;
  for (const auto &node : func_graph_->nodes()) {
    std::cout << "node #" << count << std::endl;
    DumpNode(node);
    if (node->isa<CNode>()) {
      const auto &cnode = node->cast<CNodePtr>();
      std::cout << "node " << count << " have " << cnode->inputs().size() - 1 << " inputs" << std::endl;
      for (size_t i = 1; i < cnode->inputs().size(); i++) {
        const auto &input = cnode->input(i);
        DumpNode(input);
      }
    }
    count++;
  }
}

void AnSubGraph::SetAbstract() {
  if (outputs_.size() == 1) {
    auto node = outputs_.at(0);
    if (node->isa<CNode>()) {
      const auto cnode = node->cast<CNodePtr>();
      cnode_->set_abstract(cnode->abstract());
    }
  } else {
    AbstractBasePtrList abstract_list;
    for (const auto &output : outputs_) {
      auto abstract = output->abstract();
      if (abstract == nullptr) {
        MS_LOG(ERROR) << "Create tensor abstract for " << output->fullname_with_scope() << " failed";
        return;
      }
      auto data_type = abstract->BuildType()->type_id();
      while (data_type == TypeId::kObjectTypeTuple) {
        MS_LOG(WARNING) << "got tuple as output of " << output->fullname_with_scope() << " in composite #" << index_;
        auto tuple_abs = abstract->cast<abstract::AbstractTuplePtr>();
        abstract = tuple_abs->elements().at(0);
        if (abstract == nullptr) {
          MS_LOG(ERROR) << "Create tensor abstract failed in loop for " << output->fullname_with_scope();
          return;
        }
        data_type = abstract->BuildType()->type_id();
      }
      abstract_list.emplace_back(abstract);
    }
    auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
    if (abstract_tuple == nullptr) {
      MS_LOG(ERROR) << "create abstract_tuple failed";
      return;
    }
    cnode_->set_abstract(abstract_tuple);
  }
}

void SubGraphHelper::FixGroups() {
  for (const auto &sbg : sg_v_) {
    sbg->FixGroup(shared_from_this());
  }
}

int SubGraphHelper::CheckAllInputInSameSg(const CNodePtr &cnode) {
  int prev_id = -1;
  int subg_id = sg_v_.size() - 1;
  for (size_t i = 1; i < cnode->inputs().size(); i++) {
    auto &input = cnode->input(i);
    if (!input->isa<CNode>()) {
      continue;
    }
    subg_id = FindSubGraph(input);
    if ((subg_id == -1) || ((prev_id != -1) && (subg_id != prev_id))) {
      break;
    }
    prev_id = subg_id;
  }
  if ((prev_id == -1) || (subg_id == prev_id)) return subg_id;
  return -1;
}

int SubGraphHelper::GetOutputsCount(int group) {
  auto &subg = sg_v_.at(group);
  return subg->GetOutputsCount();
}

void SubGraphHelper::AddToSubGraph(int index, const CNodePtr &node, bool update) {
  auto &subg = sg_v_.at(index);
  subg->Add(node);
  // add as inputs all non cnode and cnodes not in subgraph
  if (update) {
    for (size_t i = 1; i < node->inputs().size(); i++) {
      const auto &input = node->input(i);
      if (!input->isa<CNode>()) {
        subg->AddInput(input);
      } else {
        auto cnode = input->cast<CNodePtr>();
        int group = FindSubGraph(cnode);
        if (group < 0) {
          subg->AddInput(input);
        } else {
          // if input not in the same of input group, its an output of group
          if (group != index) {
            subg->AddInput(cnode);
          }
        }
      }
    }
    map_[node] = index;
  }
}

void SubGraphHelper::AddSubGraph(const CNodePtr &node) {
  auto index = sg_v_.size();
  auto subg = std::make_shared<AnSubGraph>(index);
  sg_v_.push_back(subg);
  AddToSubGraph(index, node);
}

int SubGraphHelper::FindSubGraph(const AnfNodePtr &node) const {
  auto const &cnode = node->cast<CNodePtr>();
  auto it = map_.find(cnode);
  if (it != map_.end()) {
    return it->second;
  }
  return -1;
}

void SubGraphHelper::SetCNode(int idx, const CNodePtr &cnode) {
  auto sbg = GetSbg(idx);
  sbg->set_cnode(cnode);
}

const CNodePtr &SubGraphHelper::GetCNode(int idx) const {
  auto sbg = GetSbg(idx);
  return sbg->cnode();
}

void SubGraphHelper::FixOutput() {
  const auto &output = func_graph_->output();
  if (output->isa<CNode>()) {
    int group = FindSubGraph(output);
    if (group >= 0) {
      func_graph_->set_output(GetCNode(group));
    }
  }
}

CNodePtr SubGraphHelper::CreateGetItem(const AnfNodePtr &node, int id, const CNodePtr &input) {
  auto tuple_get_item_prim = std::make_shared<ops::TupleGetItem>();
  auto get_item_value = NewValueNode(MakeValue<int>(id));
  if (tuple_get_item_prim == nullptr || get_item_value == nullptr) {
    MS_LOG(ERROR) << "NewValueNode is nullptr";
    return nullptr;
  }
  auto tuple_get_item_prim_c = tuple_get_item_prim->GetPrim();
  MS_ASSERT(tuple_get_item_prim_c != nullptr);
  CNodePtr get_item_cnode = func_graph_->NewCNode(tuple_get_item_prim_c, {node, get_item_value});
  if (get_item_cnode == nullptr) {
    MS_LOG(ERROR) << "cannot create a new node for value " << id;
    return nullptr;
  }
  get_item_cnode->set_abstract(input->abstract());
  get_item_cnode->set_fullname_with_scope(input->fullname_with_scope() + "/output_getitem_" + std::to_string(id));
  return get_item_cnode;
}

int SubGraphHelper::GetOutputId(int group, const CNodePtr &input) const {
  const auto subg = GetSbg(group);
  return subg->GetOutputId(input);
}

CNodePtr SubGraphHelper::CreateCopyNode(const AnfNodePtr &input, ops::Copy::CopyFormatType type) {
  auto copy_prim = std::make_shared<ops::Copy>();
  if (copy_prim == nullptr) {
    MS_LOG(ERROR) << "NewValueNode is nullptr";
    return nullptr;
  }
  copy_prim->set_copy_format(type);
  auto copy_prim_c = copy_prim->GetPrim();
  MS_ASSERT(copy_prim_c != nullptr);
  CNodePtr copy_cnode = func_graph_->NewCNode(copy_prim_c, {input});
  if (copy_cnode == nullptr) {
    MS_LOG(ERROR) << "cannot create copy node ";
    return nullptr;
  }
  copy_cnode->set_abstract(input->abstract());
  copy_cnode->set_fullname_with_scope(input->fullname_with_scope() + "/copy");
  return copy_cnode;
}

bool SubGraphHelper::IsGraphInput(const AnfNodePtr &node) const {
  const auto &inputs = func_graph_->get_inputs();
  auto it = std::find(inputs.begin(), inputs.end(), node);
  if (it != inputs.end()) {
    return true;
  }
  return false;
}

void SubGraphHelper::SetOutputsAndAbstract(const AnfNodePtrList &nodes) {
  for (const auto &node : nodes) {
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      int group = FindSubGraph(cnode);
      for (const auto &input : cnode->inputs()) {
        if (input->isa<CNode>()) {
          auto cinput = input->cast<CNodePtr>();
          int in_group = FindSubGraph(cinput);
          if ((in_group >= 0) && (in_group != group)) {
            AddSubGraphOutput(in_group, cinput);
          }
        }
      }
    }
  }
  for (const auto &sbg : sg_v_) {
    sbg->SetAbstract();
  }
}

AnfNodePtr SubGraphHelper::CreateGetItemAndCopyUnique(const AnfNodePtr &node, int id, const CNodePtr &cinput,
                                                      ops::Copy::CopyFormatType type) {
  auto pair = std::make_pair(id, node);
  auto connect_node = node;
  if (connection_map_.find(pair) != connection_map_.end()) {
    connect_node = connection_map_.at(pair);
  } else {
    if (cinput != nullptr) {
      auto get_item = CreateGetItem(connect_node, id, cinput);
      if (get_item == nullptr) {
        MS_LOG(ERROR) << "could not create get_item";
        return nullptr;
      }
      connect_node = get_item;
    }
    if (type != ops::Copy::CopyFormatType::NONE) {
      auto copy_node = CreateCopyNode(connect_node, type);
      if (copy_node == nullptr) {
        MS_LOG(ERROR) << "could not create copy_node";
        return nullptr;
      }
      connect_node = copy_node;
    }
    (connection_map_)[pair] = connect_node;
  }
  return connect_node;
}

void SubGraphHelper::UpdateInput(const CNodePtr &cnode, int index, const AnfNodePtr &input) const {
  int group = FindSubGraph(cnode);
  if (group >= 0) {
    auto cnode_group = GetCNode(group);
    auto prev_input = cnode->input(index);
    for (size_t i = 1; i < cnode_group->inputs().size(); i++) {
      if (cnode_group->input(i) == prev_input) {
        // update group input
        cnode_group->set_input(i, input);
        break;
      }
    }
  }
  cnode->set_input(index, input);
}

void SubGraphHelper::FixAllNodes(const AnfNodePtrList &nodes) {
  // set up outputs
  SetOutputsAndAbstract(nodes);
  for (const auto &node : nodes) {
    if (node->isa<CNode>()) {
      int cnode_group = FindSubGraph(node);
      auto cnode = node->cast<CNodePtr>();
      for (size_t i = 1; i < cnode->inputs().size(); i++) {
        auto const &input = cnode->input(i);
        if (input->isa<CNode>()) {
          auto cinput = input->cast<CNodePtr>();
          int in_group = FindSubGraph(input);
          ops::Copy::CopyFormatType oper = ops::Copy::CopyFormatType::NONE;
          if (cnode_group < 0) {
            oper = ops::Copy::CopyFormatType::DEVICE_HOST;
          }
          if ((in_group >= 0) && (in_group != cnode_group)) {
            auto in_cnode = GetCNode(in_group);
            if (GetOutputsCount(in_group) > 1) {
              int id = GetOutputId(in_group, cinput);
              if (id < 0) {
                MS_LOG(ERROR) << "cannot find input " << input->fullname_with_scope() << " in group " << in_group
                              << "output list";
                return;
              }
              auto connect_node = CreateGetItemAndCopyUnique(in_cnode, id, cinput, oper);
              if (connect_node == nullptr) {
                MS_LOG(ERROR) << "could not create nodes";
                return;
              }
              UpdateInput(cnode, i, connect_node);
            } else {
              CNodePtr in;
              auto connect_node = CreateGetItemAndCopyUnique(in_cnode, 0, in, oper);
              UpdateInput(cnode, i, connect_node);
            }
          }
        }
      }
    }
  }
  FixOutput();
  FixGroups();
}

void SubGraphHelper::DrawConnction(const AnfNodePtr &in_node, bool src_composite, int src_idx, const AnfNodePtr &node,
                                   bool dst_composite, int dst_idx, std::ostream &out) const {
  constexpr std::string_view quote{"\""};
  if (!src_composite && !dst_composite) {
    out << quote << in_node->fullname_with_scope() << quote << "->" << quote << node->fullname_with_scope() << quote
        << std::endl;
  } else if (src_composite && !dst_composite) {
    auto src_name = sg_v_[src_idx]->func_graph()->nodes().front()->fullname_with_scope();
    out << quote << src_name << quote << "->" << quote << node->fullname_with_scope() << quote << "[ltail=cluster_"
        << src_idx << "]" << std::endl;
  } else if (src_composite && dst_composite) {
    auto src_name = sg_v_[src_idx]->func_graph()->nodes().front()->fullname_with_scope();
    auto dst_name = sg_v_[dst_idx]->func_graph()->nodes().front()->fullname_with_scope();
    out << quote << src_name << quote << "->" << quote << dst_name << quote << "[ltail=cluster_" << src_idx
        << " lhead=cluster_" << dst_idx << "]" << std::endl;
  } else {
    auto dst_name = sg_v_[dst_idx]->func_graph()->nodes().front()->fullname_with_scope();
    out << quote << in_node->fullname_with_scope() << quote << "->" << quote << dst_name << quote << "[lhead=cluster_"
        << dst_idx << "]" << std::endl;
  }
}

void SubGraphHelper::DrawGraph(const FuncGraphPtr &graph, std::ostream &out, bool recursive) const {
  constexpr std::string_view quote{"\""};
  auto is_composite = [](const AnfNodePtr &node, int *idx) {
    if (node->isa<CNode>()) {
      auto prim = GetCNodePrimitive(node);
      if (prim->name() == ops::kNameAscendNativeComposite) {
        *idx = static_cast<int>(GetValue<int64_t>(prim->GetAttr(ops::kGroup)));
        return true;
      }
    }
    return false;
  };
  auto nodes = TopoSort(graph->get_return());
  for (const auto &node : nodes) {
    if (node->isa<CNode>()) {
      auto prim = GetCNodePrimitive(node);
      std::string node_name = prim->name();
      int idx;
      std::string color;
      if (node_name == ops::kNameCopy) {
        auto value = prim->GetAttr(ops::kCopyFormat);
        if (value == nullptr) {
          MS_LOG(ERROR) << "value returned null";
          return;
        }
        auto type = static_cast<ops::Copy::CopyFormatType>(GetValue<int64_t>(value));
        switch (type) {
          case ops::Copy::CopyFormatType::DEVICE_HOST:
            color = "red";
            break;
          case ops::Copy::CopyFormatType::HOST_DEVICE:
            color = "green";
            break;
          default:
            color = "";
            break;
        }
      }
      if (!is_composite(node, &idx)) {
        out << quote << node->fullname_with_scope() << quote << "[label=" << quote << node_name << quote;
        if (!color.empty()) {
          out << " color=" << color;
        }
        out << " ]" << std::endl;
      }
    }
  }
  for (const auto &node : nodes) {
    int dst_idx;
    if (node->isa<CNode>()) {
      bool dst_composite = false;
      if (is_composite(node, &dst_idx) && recursive) {
        out << "subgraph cluster_" << dst_idx << " {" << std::endl;
        out << "label=\"composite #" << dst_idx << "\"" << std::endl;
        out << "style=rounded" << std::endl;
        DrawGraph(sg_v_[dst_idx]->func_graph(), out, recursive);
        out << "}" << std::endl;
        dst_composite = true;
      }
      auto cnode = node->cast<CNodePtr>();
      for (size_t i = 1; i < cnode->inputs().size(); i++) {
        auto &in_node = cnode->input(i);
        if (in_node->isa<CNode>() || IsGraphInput(in_node)) {
          bool src_composite = false;
          int src_idx;
          if (is_composite(in_node, &src_idx)) {
            src_composite = true;
          }
          DrawConnction(in_node, src_composite, src_idx, node, dst_composite, dst_idx, out);
        }
      }
    }
  }
}

void SubGraphHelper::DrawGraph(const std::string &file_name, const FuncGraphPtr &graph, bool recursive) const {
  std::ofstream out(file_name);
  out << "digraph ascend {" << std::endl;
  out << "compound=true" << std::endl;
  for (const auto &in : func_graph_->get_inputs()) {
    out << "\"" << in->fullname_with_scope() << "\"[shape=box]" << std::endl;
  }
  DrawGraph(graph, out, recursive);
  out << "}\n";
  out.close();
}

void SubGraphHelper::DumpNode(std::ofstream &out, const AnfNodePtr &node) const {
  out << node->fullname_with_scope() << " typeid=" << node->tid() << " ";
  if (node->isa<CNode>()) {
    const auto &cnode = node->cast<CNodePtr>();
    const auto &prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    out << prim->name();
  }
  if (node->isa<ValueNode>()) {
    out << "node is valueNode";
  }

  out << std::endl;
}

void SubGraphHelper::Dump(std::string file_name) const {
  std::ofstream out(file_name);
  int count = 0;
  out << "graph have " << func_graph_->get_inputs().size() << " inputs" << std::endl;
  for (const auto &in : func_graph_->get_inputs()) {
    DumpNode(out, in);
  }
  auto nodes = TopoSort(func_graph_->get_return());
  out << "graph have " << nodes.size() << " nodes" << std::endl;
  for (const auto &node : nodes) {
    out << "node #" << count << std::endl;
    DumpNode(out, node);
    if (node->isa<CNode>()) {
      const auto &cnode = node->cast<CNodePtr>();
      out << "node " << count << " have " << cnode->inputs().size() - 1 << " inputs" << std::endl;
      for (size_t i = 1; i < cnode->inputs().size(); i++) {
        const auto &input = cnode->input(i);
        DumpNode(out, input);
      }
    }
    count++;
  }
  out.close();
}

}  // namespace mindspore
