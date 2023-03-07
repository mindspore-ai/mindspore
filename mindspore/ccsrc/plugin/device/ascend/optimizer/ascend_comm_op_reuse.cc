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
#include <cstdlib>
#include "backend/common/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/comm_manager.h"
#include "include/common/utils/parallel_context.h"
#include "runtime/graph_scheduler/graph_compiler.h"
#include "plugin/device/ascend/hal/device/ascend_stream_assign.h"
#include "plugin/device/ascend/optimizer/ascend_comm_op_reuse.h"

namespace mindspore {
namespace opt {
namespace {
template <class T>
std::string VecToString(const std::vector<T> &vec) {
  std::string res = "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    if (i != 0) {
      res += ",";
    }
    res += std::to_string(vec[i]);
  }
  res += "]";
  return res;
}

std::string GenCommOpKey(const CNodePtr &node, const KernelGraphPtr &root_graph) {
  std::string op_key;
  MS_EXCEPTION_IF_NULL(node);
  auto comm_prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(comm_prim);
  // op name
  op_key += comm_prim->name();
  auto comm_abstract = node->abstract();
  // shape and dtype
  if (comm_abstract != nullptr) {
    op_key += "_" + comm_abstract->BuildShape()->ToString() + "_" + comm_abstract->BuildType()->ToString();
  }
  // group
  if (comm_prim->HasAttr(kAttrGroup)) {
    op_key += "_" + GetValue<std::string>(comm_prim->GetAttr(kAttrGroup));
  }
  // allreduce op
  if (comm_prim->HasAttr(kAttrOp)) {
    op_key += "_" + GetValue<std::string>(comm_prim->GetAttr(kAttrOp));
  }
  // alltoall send rank
  if (comm_prim->HasAttr(kAttrSendRankIds)) {
    op_key += "_" + VecToString(GetValue<std::vector<int64_t>>(comm_prim->GetAttr(kAttrSendRankIds)));
  }
  // alltoall recv rank
  if (comm_prim->HasAttr(kAttrRecvRankIds)) {
    op_key += "_" + VecToString(GetValue<std::vector<int64_t>>(comm_prim->GetAttr(kAttrRecvRankIds)));
  }
  // model identifier, aka. root_graph_id
  op_key += "_" + std::to_string(root_graph->root_graph_id());
  MS_LOG(INFO) << node->DebugString() << " key " << op_key;
  return op_key;
}

bool IsReusable(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  if (!prim->HasAttr(parallel::COMM_REUSE) || !GetValue<bool>(prim->GetAttr(parallel::COMM_REUSE))) {
    return false;
  }

  return true;
}
}  // namespace

void AscendCommOpReuse::Run() {
  MS_LOG(INFO) << "Start process comm op reuse.";
  FindAllCommOp();
  AnalyseCommOpReuse();
  InsertMonadForReusedCommOp();
  ReplaceCommOpToCallNode();
  MS_LOG(INFO) << "Reuse comm op success.";
}

void AscendCommOpReuse::FindAllCommOp() {
  std::set<KernelGraphPtr> memo;
  all_comm_ops_ = FindCommOpRecur(root_graph_, &memo);
}

std::vector<std::pair<CNodePtr, KernelGraphPtr>> AscendCommOpReuse::FindCommOpRecur(const KernelGraphPtr &kg,
                                                                                    std::set<KernelGraphPtr> *memo) {
  MS_EXCEPTION_IF_NULL(kg);
  MS_EXCEPTION_IF_NULL(memo);
  if (memo->find(kg) != memo->end()) {
    return {};
  }
  memo->emplace(kg);

  std::vector<std::pair<CNodePtr, KernelGraphPtr>> ret;
  auto list = TopoSort(kg->get_return());
  for (const auto &node : list) {
    if (common::AnfAlgo::IsCommunicationOp(node)) {
      ret.emplace_back(node->cast<CNodePtr>(), kg);
    } else if (IsValueNode<FuncGraph>(node)) {
      auto sub_graph = GetValueNode<FuncGraphPtr>(node);
      MS_EXCEPTION_IF_NULL(sub_graph);
      auto sub_kg = sub_graph->cast<KernelGraphPtr>();
      MS_EXCEPTION_IF_NULL(sub_kg);
      auto sub_ret = FindCommOpRecur(sub_kg, memo);
      ret.insert(ret.end(), sub_ret.begin(), sub_ret.end());
    }
  }

  return ret;
}

void AscendCommOpReuse::InsertMonadForReusedCommOp() {
  std::set<KernelGraphPtr> memo;
  InsertMonadForReusedCommOpRecur(root_graph_, &memo);
}

void AscendCommOpReuse::InsertMonadForReusedCommOpRecur(const KernelGraphPtr &kg, std::set<KernelGraphPtr> *memo) {
  MS_EXCEPTION_IF_NULL(kg);
  MS_EXCEPTION_IF_NULL(memo);
  if (memo->find(kg) != memo->end()) {
    return;
  }
  memo->emplace(kg);

  AnfNodePtr last_monad = nullptr;
  bool need_attach = false;
  auto nodes = TopoSort(kg->output());
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (HasAbstractUMonad(node)) {
      last_monad = node;
    } else if (IsValueNode<FuncGraph>(node)) {
      auto sub_graph = GetValueNode<FuncGraphPtr>(node);
      MS_EXCEPTION_IF_NULL(sub_graph);
      auto sub_kg = sub_graph->cast<KernelGraphPtr>();
      MS_EXCEPTION_IF_NULL(sub_kg);
      InsertMonadForReusedCommOpRecur(sub_kg, memo);
    } else if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      auto iter = reused_comm_sub_graphs_.find(cnode);
      if (iter == reused_comm_sub_graphs_.end()) {
        continue;
      }
      // if reused comm op
      if (last_monad == nullptr) {
        last_monad = kg->NewValueNode(kUMonad->ToAbstract(), kUMonad);
      }
      MS_EXCEPTION_IF_NULL(last_monad);
      cnode->add_input(last_monad);
      auto abstract = last_monad->abstract();
      last_monad = kg->NewCNode({NewValueNode(prim::kPrimUpdateState), last_monad, cnode});
      last_monad->set_abstract(abstract);
      need_attach = true;
    }
  }
  if (need_attach) {
    auto output = kg->output();
    auto depend = NewValueNode(prim::kPrimDepend);
    // If isolated nodes dependencies exist.
    if (IsPrimitiveCNode(output, prim::kPrimDepend) &&
        IsPrimitiveCNode(output->cast<CNodePtr>()->input(kDependAttachNodeIndex), prim::kPrimStopGradient)) {
      // Insert new Depend node before isolated Depend node.
      auto manager = kg->manager();
      auto isolated_depend = output->cast<CNodePtr>();
      auto &orig_output = isolated_depend->input(1);
      auto state_depend = kg->NewCNode({depend, orig_output, last_monad});
      state_depend->set_abstract(orig_output->abstract());
      manager->SetEdge(isolated_depend, 1, state_depend);
      return;
    }
    // Insert Depend node and set it as output, if no isolated nodes.
    auto depend_cnode = kg->NewCNode({depend, output, last_monad});
    depend_cnode->set_abstract(output->abstract());
    kg->set_output(depend_cnode);
  }
}

void AscendCommOpReuse::AnalyseCommOpReuse() {
  std::map<std::string, std::vector<CNodePtr>> reuse_map;
  for (const auto &iter : all_comm_ops_) {
    const auto &comm_op = iter.first;
    if (!IsReusable(comm_op)) {
      continue;
    }
    reuse_map[GenCommOpKey(comm_op, root_graph_)].push_back(comm_op);
  }

  for (const auto &[key, comm_op_set] : reuse_map) {
    if (comm_op_set.empty()) {
      continue;
    }
    if (total_comm_op_reuse_num_ + comm_op_set.size() > max_comm_op_reuse_num_) {
      continue;
    }
    CNodePtr one_of_node = *(comm_op_set.begin());
    MS_EXCEPTION_IF_NULL(one_of_node);
    if (IsPrimitiveCNode(one_of_node, prim::kPrimAllToAllv)) {
      // reuse alltoall op when more than 1
      if (comm_op_set.size() <= 1) {
        continue;
      }
    } else {
      // reuse other comm op when stream costs reduced
      if (comm_op_set.size() <= device::ascend::AscendStreamAssign::GetInstance().max_task_count() /
                                  device::ascend::AscendStreamAssign::GetHcomTaskNum(one_of_node)) {
        continue;
      }
    }
    MS_LOG(INFO) << "Start reuse " << comm_op_set.size() << " comm ops for key " << key;
    auto sub_comm_graph = CreateCommSubGraph(one_of_node);
    MS_EXCEPTION_IF_NULL(sub_comm_graph);
    for (const auto &comm_op : comm_op_set) {
      reused_comm_sub_graphs_.emplace(comm_op, sub_comm_graph);
      MS_LOG(INFO) << "Reuse comm op " << comm_op->DebugString() << " to sub comm graph " << sub_comm_graph->ToString();
    }
    total_comm_op_reuse_num_ += comm_op_set.size();
  }
}

KernelGraphPtr AscendCommOpReuse::CreateCommSubGraph(const CNodePtr &comm_op) {
  MS_EXCEPTION_IF_NULL(comm_op);
  MS_EXCEPTION_IF_ZERO("input size of comm_op " + comm_op->DebugString(), comm_op->size());
  // create sub graph
  auto graph = std::make_shared<session::KernelGraph>();
  graph->set_graph_id(comm_subgraph_sum_++);
  MS_EXCEPTION_IF_NULL(graph);
  auto sub_graph_inputs = graph->MutableInputs();
  MS_EXCEPTION_IF_NULL(sub_graph_inputs);
  // create inputs param for sub graph
  for (size_t i = 1; i < comm_op->size(); ++i) {
    auto input = comm_op->input(i);
    MS_EXCEPTION_IF_NULL(input);
    sub_graph_inputs->emplace_back(graph->NewParameter(input->abstract()));
  }
  // create comm op for sub graph
  std::vector<AnfNodePtr> new_comm_op_args = {comm_op->input(0)};
  new_comm_op_args.insert(new_comm_op_args.end(), sub_graph_inputs->begin(), sub_graph_inputs->end());
  auto new_comm_op = graph->NewCNode(new_comm_op_args);
  MS_EXCEPTION_IF_NULL(new_comm_op);
  new_comm_op->set_abstract(comm_op->abstract());

  std::string group_name = GenCommOpKey(comm_op, root_graph_);
  auto rank_list = common::AnfAlgo::GetNodeAttr<std::vector<unsigned int>>(comm_op, kAttrRankList);
  if (!CommManager::GetInstance().CreateGroupSync(group_name, rank_list)) {
    MS_LOG(EXCEPTION) << "Create new group " << group_name << " failed, rank list = " << VecToString(rank_list);
  }
  common::AnfAlgo::SetNodeAttr(kAttrGroup, MakeValue<std::string>(group_name), new_comm_op);

  // insert tuple_getitem if output is tuple
  if (auto abstract = comm_op->abstract(); abstract != nullptr && abstract->isa<abstract::AbstractTuple>()) {
    size_t out_size = abstract->cast<abstract::AbstractTuplePtr>()->size();
    std::vector<AnfNodePtr> comm_op_outputs;
    opt::CreateMultipleOutputsOfAnfNode(graph, new_comm_op, out_size, &comm_op_outputs);
    std::vector<AnfNodePtr> make_tuple_args = {NewValueNode(std::make_shared<Primitive>(prim::kPrimMakeTuple->name()))};
    make_tuple_args.insert(make_tuple_args.end(), comm_op_outputs.begin(), comm_op_outputs.end());
    auto make_tuple = graph->NewCNode(make_tuple_args);
    MS_EXCEPTION_IF_NULL(make_tuple);
    make_tuple->set_abstract(abstract);
    graph->set_output(make_tuple);
  } else {
    graph->set_output(new_comm_op);
  }

  sub_graph_inputs->emplace_back(graph->NewParameter(kUMonad->ToAbstract()));
  root_graph_->RecordNewCommSubGraphId(graph->graph_id());
  return graph;
}

void AscendCommOpReuse::ReplaceCommOpToCallNode() {
  MS_LOG(INFO) << "Reuse comm op reused_comm_sub_graphs_ size: " << reused_comm_sub_graphs_.size();
  for (const auto &[origin_comm_op, comm_sub_graph] : reused_comm_sub_graphs_) {
    MS_EXCEPTION_IF_NULL(origin_comm_op);
    MS_EXCEPTION_IF_NULL(comm_sub_graph);
    KernelGraphPtr origin_graph = nullptr;
    for (const auto &[n, g] : all_comm_ops_) {
      if (n == origin_comm_op) {
        origin_graph = g;
        break;
      }
    }
    MS_EXCEPTION_IF_NULL(origin_graph);
    auto origin_inputs = origin_comm_op->inputs();
    std::vector<AnfNodePtr> new_call_op_args = {NewValueNode(std::make_shared<Primitive>(prim::kPrimCall->name())),
                                                NewValueNode(comm_sub_graph)};
    new_call_op_args.insert(new_call_op_args.end(), origin_inputs.begin() + 1, origin_inputs.end());
    origin_comm_op->set_inputs(new_call_op_args);
  }
}
}  // namespace opt
}  // namespace mindspore
