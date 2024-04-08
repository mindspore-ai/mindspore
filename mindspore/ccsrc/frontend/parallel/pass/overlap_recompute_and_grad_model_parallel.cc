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

#include "frontend/parallel/pass/overlap_recompute_and_grad_model_parallel.h"
#include <memory>
#include <list>
#include <vector>
#include <string>
#include <queue>
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/utils/convert_utils_base.h"
#include "frontend/parallel/step_parallel.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace parallel {
constexpr size_t kSize2 = 2;
using BlockBorder = std::pair<CNodePtr, CNodePtr>;
using AnfNodeIndex = std::pair<AnfNodePtr, size_t>;
namespace {
constexpr auto kOverlapRecomputeAndGradNodesInsert = "overlap_recompute_and_grad_nodes_insert";
void ExtractRecomputeSubGraph(const std::vector<CNodePtr> &origin_nodes_topological,
                              mindspore::HashMap<int32_t, std::vector<CNodePtr>> *recomputed_block_node_in_orders,
                              mindspore::HashMap<int32_t, std::vector<CNodePtr>> *recompute_block_node_in_orders,
                              mindspore::HashMap<int32_t, std::vector<CNodePtr>> *recomputed_grad_node) {
  for (const auto &cnode : origin_nodes_topological) {
    if (!cnode->HasAttr(kAttrRecomputeSubGraph)) {
      continue;
    }
    auto recompute_block_id = GetValue<size_t>(cnode->GetAttr(kAttrRecomputeSubGraph));
    if (cnode->HasAttr(kAttrDuplicated)) {
      if ((*recompute_block_node_in_orders).find(recompute_block_id) == (*recompute_block_node_in_orders).end()) {
        (*recompute_block_node_in_orders)[recompute_block_id] = {cnode};
      } else {
        (*recompute_block_node_in_orders)[recompute_block_id].push_back(cnode);
      }
    } else if (!cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
      if ((*recomputed_block_node_in_orders).find(recompute_block_id) == (*recomputed_block_node_in_orders).end()) {
        (*recomputed_block_node_in_orders)[recompute_block_id] = {cnode};
      } else {
        (*recomputed_block_node_in_orders)[recompute_block_id].push_back(cnode);
      }
    } else {
      if ((*recomputed_grad_node).find(recompute_block_id) == (*recomputed_grad_node).end()) {
        (*recomputed_grad_node)[recompute_block_id] = {cnode};
      } else {
        (*recomputed_grad_node)[recompute_block_id].push_back(cnode);
      }
    }
  }
}

std::vector<CNodePtr> NodeUsersInRecomputeSubGraph(const CNodePtr &cnode, std::function<bool(const CNodePtr &)> match) {
  auto func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<CNodePtr> res;
  std::queue<CNodePtr> cnode_queue;
  cnode_queue.push(cnode);
  while (!cnode_queue.empty()) {
    auto queue_end = cnode_queue.front();
    cnode_queue.pop();
    auto user_set = manager->node_users()[queue_end];
    for (auto &pair : user_set) {
      if (!pair.first->isa<CNode>()) {
        continue;
      }
      auto user_cnode = pair.first->cast<CNodePtr>();
      if (std::find(res.begin(), res.end(), user_cnode) != res.end()) {
        continue;
      }
      if (match(user_cnode)) {
        cnode_queue.push(user_cnode);
        res.push_back(user_cnode);
        continue;
      }
    }
  }
  return res;
}

void ExtractCommNodes(const mindspore::HashMap<int32_t, std::vector<CNodePtr>> &origin_node_map,
                      mindspore::HashMap<int32_t, std::vector<CNodePtr>> *dst_node_map) {
  for (const auto &sub_graph : origin_node_map) {
    auto sub_graph_id = sub_graph.first;
    (*dst_node_map)[sub_graph_id] = {};
    for (const auto &sub_cnode : sub_graph.second) {
      if (!sub_cnode->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
        continue;
      }
      (*dst_node_map)[sub_graph_id].push_back(sub_cnode);
    }
  }
}

std::vector<CNodePtr> SrcNodeNoRelyInputs(const CNodePtr &src_node, const std::vector<CNodePtr> &dst_node_users) {
  std::vector<CNodePtr> res;
  std::queue<CNodePtr> cnode_queue;
  cnode_queue.push(src_node);
  while (!cnode_queue.empty()) {
    auto queue_end = cnode_queue.front();
    cnode_queue.pop();
    if (std::find(dst_node_users.begin(), dst_node_users.end(), queue_end) == dst_node_users.end()) {
      res.push_back(queue_end);
      continue;
    }
    for (size_t i = 1; i < queue_end->size(); ++i) {
      if (!queue_end->input(i)->isa<CNode>()) {
        continue;
      }
      auto input_cnode = queue_end->input(i)->cast<CNodePtr>();
      cnode_queue.push(input_cnode);
    }
  }
  return res;
}

bool IsNotCareCNode(const CNodePtr &cnode) {
  return IsPrimitiveCNode(cnode, prim::kPrimDepend) || IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) ||
         IsPrimitiveCNode(cnode, prim::kPrimLoad) || IsPrimitiveCNode(cnode, prim::kPrimUpdateState) ||
         IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem);
}

CNodePtr GetSrcNode(const CNodePtr &src_node_output) {
  CNodePtr src_node = nullptr;
  for (size_t i = 1; i < src_node_output->size(); ++i) {
    if (src_node_output->input(i)->isa<CNode>()) {
      src_node = src_node_output->input(i)->cast<CNodePtr>();
    }
  }
  return src_node;
}

void InsertDepend(const AnfNodePtr &prior_node, const AnfNodeIndex &post_node_index, const FuncGraphPtr &graph,
                  const std::string &attr_tag, size_t block_id) {
  MS_EXCEPTION_IF_NULL(prior_node);
  MS_EXCEPTION_IF_NULL(post_node_index.first);
  if (!post_node_index.first->isa<CNode>()) {
    return;
  }
  MS_LOG(INFO) << "Insert depend between " << prior_node->DebugString() << " and "
               << post_node_index.first->DebugString();
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto post_cnode = post_node_index.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(post_cnode);
  auto index = post_node_index.second;
  std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), post_cnode->input(index), prior_node};
  auto depend_node = graph->NewCNode(depend_input);
  const auto abstract = post_cnode->input(index)->abstract();

  depend_node->set_abstract(abstract);
  if (!attr_tag.empty()) {
    depend_node->AddAttr(attr_tag, MakeValue<size_t>(block_id));
  }
  manager->SetEdge(post_cnode, index, depend_node);
}

// find the node's last input node with index
AnfNodeIndex FindNodeLastInput(const CNodePtr &node, const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &topos,
                               const std::string &tag) {
  auto check_func = [tag](const AnfNodePtr &node) {
    if (!node->isa<CNode>()) {
      return false;
    }
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    return cnode->HasAttr(tag) || cnode->HasPrimalAttr(tag);
  };
  // the pair first is the node's last input
  AnfNodeIndex ret;
  std::vector<AnfNodePtr> node_inputs;
  std::copy_if(node->inputs().begin() + 1, node->inputs().end(), std::back_inserter(node_inputs), check_func);
  auto comp_func = [&topos](const auto &node1, const auto &node2) {
    auto index1 = std::distance(topos.begin(), std::find(topos.begin(), topos.end(), node1));
    auto index2 = std::distance(topos.begin(), std::find(topos.begin(), topos.end(), node2));
    return index1 < index2;
  };
  auto last_input = std::max_element(node_inputs.begin(), node_inputs.end(), comp_func);
  if (last_input != node_inputs.end()) {
    const auto &inputs = node->inputs();
    size_t index = IntToSize(std::distance(inputs.begin(), std::find(inputs.begin(), inputs.end(), *last_input)));
    ret = std::make_pair(*last_input, index);
  }
  return ret;
}

std::vector<BlockBorder> DivideIntoCommAndCalBlock(const std::vector<CNodePtr> &block) {
  // the divided comm and call block like this.
  // {cal_blk1_first, cal_blk1_last}, {comm_node1, comm_node1}, {cal_blk2_first, cal_blk2_last}
  std::vector<BlockBorder> borders;
  for (const auto &node : block) {
    if (node->HasAttr(kRecomputeInsert)) {
      continue;
    }
    if (IsNotCareCNode(node)) {
      continue;
    }
    if (node->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
      borders.push_back({node, node});
      continue;
    }
    if (borders.empty()) {
      borders.push_back({node, node});
      continue;
    }
    const auto &last_border = *(borders.rbegin());
    if (!last_border.first->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
      borders.rbegin()->second = node;
      continue;
    }
    borders.push_back({node, node});
  }
  return borders;
}

void DeleteRecomputeInsertedControlEdge(const std::vector<CNodePtr> &depend_block, const FuncGraphPtr &graph) {
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (const auto &node : depend_block) {
    auto node_users = manager->node_users();
    auto iter = node_users.find(node);
    if (iter == node_users.end()) {
      continue;
    }
    const auto &users = iter->second;
    for (const auto &[user, index] : users) {
      MS_LOG(INFO) << "Delete recompute inserted control edge: " << node->DebugString()
                   << ", user: " << user->DebugString();
      manager->SetEdge(user, index, node->input(kIndexOne));
    }
  }
}

bool ReorderCandidateOverlapBlock(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &topos,
                                  const std::vector<CNodePtr> &grad_block, const std::vector<CNodePtr> &recompute_block,
                                  const std::vector<CNodePtr> &depend_block, size_t block_id) {
  bool changed = false;
  if (grad_block.empty()) {
    return changed;
  }
  auto grad_has_comm = std::any_of(grad_block.begin(), grad_block.end(), [](const auto &node) {
    return node->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId);
  });
  auto recompute_has_cal = std::any_of(recompute_block.begin(), recompute_block.end(), [](const auto &node) {
    return !node->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId);
  });
  if (grad_has_comm && recompute_has_cal) {
    DeleteRecomputeInsertedControlEdge(depend_block, graph);
  } else {
    return changed;
  }
  // divide block into small block(cal block, comm clock)
  const auto &recompute_small_block = DivideIntoCommAndCalBlock(recompute_block);

  size_t recompute_small_block_size = recompute_small_block.size();
  size_t index = 0;
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (const auto &grad_comm : grad_block) {
    if (!grad_comm->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
      continue;
    }
    const auto &pioneer = FindNodeLastInput(grad_comm, graph, topos, kPrimalAttrForwardUniqueId);
    if (!pioneer.first) {
      continue;
    }
    if (index >= recompute_small_block_size) {
      break;
    }
    while (recompute_small_block.at(index).first->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
      ++index;
      if (index >= recompute_small_block_size) {
        return changed;
      }
    }
    auto recompute_cal_block = recompute_small_block.at(index);
    // the cal_block_first is the previous comm's first output if it exist
    AnfNodeIndex post = std::make_pair(recompute_cal_block.first, kIndexOne);
    AnfNodeIndexSet post_set;
    post_set.insert(post);
    if (index >= kIndexOne) {
      const auto &previous_comm = recompute_small_block.at(index - kIndexOne).second;
      post_set = manager->node_users()[previous_comm];
    }
    for (const auto &user : post_set) {
      // prior: grad_comm_last_input, post: cal_block_begins, post_input: cal_block_begins_input
      InsertDepend(pioneer.first, user, graph, kOverlapRecomputeAndGradNodesInsert, block_id);
    }

    // the cal_block_last is the next comm's last input if it exist
    AnfNodePtr prior = recompute_cal_block.second;
    if (index + kIndexOne < recompute_small_block_size) {
      const auto next_comm = recompute_small_block.at(index + kIndexOne).first;
      const auto &next_comm_last_input = FindNodeLastInput(next_comm, graph, topos, kAttrDuplicated);
      if (!next_comm_last_input.first) {
        prior = next_comm_last_input.first;
      }
    }
    const auto &grad_comm_users = manager->node_users()[grad_comm];
    for (const auto &user : grad_comm_users) {
      // prior: cal_last, post: grad_comm_output, post_input: grad_comm
      InsertDepend(prior, user, graph, kOverlapRecomputeAndGradNodesInsert, block_id);
    }
    changed = true;
    ++index;
  }
  return changed;
}

void ClusterNodes(const std::vector<AnfNodePtr> &nodes, const std::vector<AnfNodePtr> &all_depends,
                  mindspore::HashMap<size_t, std::vector<CNodePtr>> *recompute_block,
                  std::map<size_t, std::vector<CNodePtr>> *grad_block,
                  mindspore::HashMap<size_t, std::vector<CNodePtr>> *depend_block) {
  MS_EXCEPTION_IF_NULL(recompute_block);
  MS_EXCEPTION_IF_NULL(grad_block);
  MS_EXCEPTION_IF_NULL(depend_block);
  // clustering depends on the last input of its last input
  mindspore::HashMap<AnfNodePtr, std::vector<CNodePtr>> depend_cluster;
  for (const auto &depend : all_depends) {
    auto cdepend = depend->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cdepend);
    if (!cdepend->inputs().back()->isa<CNode>()) {
      continue;
    }
    auto tuple = cdepend->inputs().back()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple);
    if (!IsPrimitiveCNode(tuple, prim::kPrimMakeTuple)) {
      MS_LOG(EXCEPTION) << "The recompute inserted depend: " << depend->DebugString()
                        << "'s last input is not make_tuple.";
    }
    depend_cluster[tuple->inputs().back()].push_back(cdepend);
  }

  size_t id = 0;
  mindspore::HashSet<AnfNodePtr> visit;
  for (const auto &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    if (visit.find(node) != visit.end()) {
      continue;
    }
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsPrimitiveCNode(cnode, prim::kPrimDepend) && cnode->HasAttr(kRecomputeInsert)) {
      id++;
      auto tuple = cnode->inputs().back()->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(tuple);
      const auto &depend_vec = depend_cluster.at(tuple->inputs().back());
      std::for_each(depend_vec.begin(), depend_vec.end(), [&depend_block, id, &visit](const auto &depend) {
        (*depend_block)[id].push_back(depend);
        visit.insert(depend);
        // The attr is just for debug
        depend->AddAttr(kCondidateOverlapBlockId, MakeValue<size_t>(id));
      });
      continue;
    } else if (cnode->HasAttr(kAttrDuplicated)) {
      (*recompute_block)[id].push_back(cnode);
    } else if (cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
      (*grad_block)[id].push_back(cnode);
    }
    // The attr is just for debug
    cnode->AddAttr(kCondidateOverlapBlockId, MakeValue<size_t>(id));
    visit.insert(cnode);
  }
}

void ReorderRecomputeAndGradNodes(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Overlap recompute and grad comm nodes, the graph is " << graph->ToString();
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto node_users_map = manager->node_users();
  const auto &nodes = TopoSort(graph->get_return(), SuccDeeperSimple, AlwaysInclude);

  std::vector<AnfNodePtr> all_depends;
  std::copy_if(nodes.begin(), nodes.end(), std::back_inserter(all_depends), [](const auto &node) {
    if (!node->template isa<CNode>()) {
      return false;
    }
    auto cnode = node->template cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsPrimitiveCNode(cnode, prim::kPrimDepend) && cnode->HasAttr(kRecomputeInsert)) {
      return true;
    }
    return false;
  });
  if (all_depends.empty()) {
    return;
  }
  mindspore::HashMap<size_t, std::vector<CNodePtr>> recompute_block;
  std::map<size_t, std::vector<CNodePtr>> grad_block;
  mindspore::HashMap<size_t, std::vector<CNodePtr>> depend_block;
  ClusterNodes(nodes, all_depends, &recompute_block, &grad_block, &depend_block);

  bool has_overlap = false;
  for (const auto &block : grad_block) {
    auto block_id = block.first;
    auto condidate_recompute_block_id = block_id + 1;
    MS_LOG(INFO) << "Reorder recompute and grad nodes block_id: " << block_id
                 << ", condidate_recompute_block_id: " << condidate_recompute_block_id;
    if (recompute_block.find(condidate_recompute_block_id) == recompute_block.end()) {
      continue;
    }
    // divide grad/recompute block into small block and reorder them
    auto reorder =
      ReorderCandidateOverlapBlock(graph, nodes, block.second, recompute_block[condidate_recompute_block_id],
                                   depend_block[condidate_recompute_block_id], condidate_recompute_block_id);
    has_overlap |= reorder;
  }
  if (has_overlap) {
    MS_LOG(INFO) << "Overlap recompute and grad comm nodes success.";
  }
}

void HandleCellReuseScene(const FuncGraphPtr &graph) {
  MS_LOG(INFO) << "Overlap recompute and grad comm nodes in cell reuse scene.";
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool enable_save_graphs = context->CanDump(kIntroductory);
  if (enable_save_graphs) {
    DumpIR("before_overlap_recompute_and_grad_comm_nodes.ir", graph);
  }
#endif
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &graphs = manager->func_graphs();
  // find the grad func graph corresponding to reused cell
  // step1: find the reused cell
  // step2: find the grad func
  auto iter =
    std::find_if(graphs.begin(), graphs.end(), [](const auto &g) { return g->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE); });
  if (iter == graphs.end()) {
    MS_LOG(WARNING) << "The reused cell is not found.";
    return;
  }
  const auto &reused_graph = *iter;
  const auto &output = reused_graph->output();
  // the last input reused graph's output is grad reused graph
  if (!output->isa<CNode>()) {
    MS_LOG(WARNING) << "The reused cell's output is not a cnode.";
    return;
  }
  const auto &coutput = output->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(coutput);
  if (!IsPrimitiveCNode(coutput, prim::kPrimMakeTuple)) {
    return;
  }

  MS_LOG(INFO) << "The reused cell is " << reused_graph->ToString();
  auto bprop_func_node = *(coutput->inputs().rbegin());
  if (!bprop_func_node->isa<ValueNode>() && !IsPrimitiveCNode(bprop_func_node, prim::kPrimPartial)) {
    MS_LOG(WARNING) << "The reused cell's grad func is not found.";
    return;
  }
  FuncGraphPtr bprop_func = nullptr;
  if (IsPrimitiveCNode(bprop_func_node, prim::kPrimPartial)) {
    const auto &cpartial = bprop_func_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cpartial);
    bprop_func = GetValueNode<FuncGraphPtr>(cpartial->input(kIndexOne));
  } else {
    bprop_func = GetValueNode<FuncGraphPtr>(bprop_func_node);
  }
  if (!bprop_func) {
    MS_LOG(WARNING) << "The reused cell's grad func is null.";
    return;
  }
  ReorderRecomputeAndGradNodes(bprop_func);
#ifdef ENABLE_DUMP_IR
  if (enable_save_graphs) {
    DumpIR("after_overlap_recompute_and_grad_comm_nodes.ir", graph);
  }
#endif
}
}  // namespace

void OverlapRecomputeAndGradModelParallel(const FuncGraphPtr &graph) {
  if ((parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
       parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel)) {
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_enable = ms_context->get_param<bool>(MS_CTX_RECOMPUTE_COMM_OVERLAP);
  if (!is_enable) {
    return;
  }
  if (ms_context->CellReuseLevel() != CellReuseLevel::kNoCellReuse) {
    HandleCellReuseScene(graph);
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::list<CNodePtr> orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(orders.cbegin(), orders.cend());
  mindspore::HashMap<int32_t, std::vector<CNodePtr>> recomputed_block_node_in_orders;
  mindspore::HashMap<int32_t, std::vector<CNodePtr>> recompute_block_node_in_orders;
  mindspore::HashMap<int32_t, std::vector<CNodePtr>> recomputed_grad_node;
  mindspore::HashMap<int32_t, std::vector<CNodePtr>> recomputed_grad_comm_node_in_orders;
  mindspore::HashMap<int32_t, std::vector<CNodePtr>> recompute_block_comm_node_in_orders;
  ExtractRecomputeSubGraph(origin_nodes_topological, &recomputed_block_node_in_orders, &recompute_block_node_in_orders,
                           &recomputed_grad_node);
  ExtractCommNodes(recomputed_grad_node, &recomputed_grad_comm_node_in_orders);
  std::for_each(recomputed_grad_comm_node_in_orders.begin(), recomputed_grad_comm_node_in_orders.end(),
                [](auto &vector_pair) { std::reverse(vector_pair.second.begin(), vector_pair.second.end()); });
  ExtractCommNodes(recompute_block_node_in_orders, &recompute_block_comm_node_in_orders);
  // In recompute.cc, the grad_block input has a depend to the recompute block already.
  for (const auto &recompute_grad_sub_graph : recomputed_grad_comm_node_in_orders) {
    auto sub_graph_id = recompute_grad_sub_graph.first;
    auto recomputed_grad_comm_nodes = recomputed_grad_comm_node_in_orders[sub_graph_id];
    auto recompute_block_comm_nodes = recompute_block_comm_node_in_orders[sub_graph_id];
    size_t overlap_size = 2 * recompute_block_comm_nodes.size() + 1;
    size_t max_iter_num = recompute_block_comm_nodes.size();
    while (overlap_size > 0) {
      auto recompute_begin_index = max_iter_num - overlap_size / 2;
      auto grad_begin_index = max_iter_num - (overlap_size - 1) / 2;
      if (grad_begin_index >= recomputed_grad_comm_nodes.size()) {
        break;
      }
      CNodePtr src_node_output;
      CNodePtr dst_node_input;
      if (overlap_size % kSize2 == 1) {
        src_node_output = (recompute_begin_index == max_iter_num) ? recompute_block_node_in_orders[sub_graph_id].back()
                                                                  : recompute_block_comm_nodes[recompute_begin_index];
        dst_node_input = recomputed_grad_comm_nodes[grad_begin_index];
      } else {
        dst_node_input = recompute_block_comm_nodes[recompute_begin_index];
        src_node_output = recomputed_grad_comm_nodes[grad_begin_index];
      }
      CNodePtr src_node = GetSrcNode(src_node_output);
      if (src_node == nullptr) {
        continue;
      }
      auto dst_nodes = manager->node_users()[dst_node_input];
      for (const auto &dst_node_pair : dst_nodes) {
        if (!dst_node_pair.first->isa<CNode>()) {
          continue;
        }
        auto dst_node = dst_node_pair.first->cast<CNodePtr>();
        MS_LOG(INFO) << "The dst node is:" << dst_node->DebugString() << ", " << dst_node->fullname_with_scope();
        // Check whether src_node is the user of dst_node, if it is, adjust the src node toward its input.
        auto dst_node_users = NodeUsersInRecomputeSubGraph(dst_node, [&](const CNodePtr &cnode) {
          return std::find(recompute_block_node_in_orders[sub_graph_id].begin(),
                           recompute_block_node_in_orders[sub_graph_id].end(),
                           cnode) != recompute_block_node_in_orders[sub_graph_id].end() ||
                 std::find(recomputed_grad_node[sub_graph_id].begin(), recomputed_grad_node[sub_graph_id].end(),
                           cnode) != recomputed_grad_node[sub_graph_id].end() ||
                 IsNotCareCNode(cnode) || IsPrimitiveCNode(cnode, prim::kPrimAllGather) ||
                 IsPrimitiveCNode(cnode, prim::kPrimAddN);
        });
        dst_node_users.push_back(dst_node);
        // Insert depend src_node->depend->dst_node.
        auto src_node_no_rely_inputs = SrcNodeNoRelyInputs(src_node, dst_node_users);
        // Find the last input ordered by executed order.
        auto new_src_node = *std::max_element(
          src_node_no_rely_inputs.begin(), src_node_no_rely_inputs.end(),
          [&](const CNodePtr &cnode1, const CNodePtr &cnode2) {
            size_t cnode_iter1 =
              (size_t)(std::find(origin_nodes_topological.begin(), origin_nodes_topological.end(), cnode1) -
                       origin_nodes_topological.begin());
            size_t cnode_iter2 =
              (size_t)(std::find(origin_nodes_topological.begin(), origin_nodes_topological.end(), cnode2) -
                       origin_nodes_topological.begin());
            cnode_iter1 = IsNotCareCNode(cnode1) ? 0 : cnode_iter1;
            cnode_iter2 = IsNotCareCNode(cnode2) ? 0 : cnode_iter2;
            return cnode_iter1 < cnode_iter2;
          });
        MS_LOG(INFO) << "The origin_src_node is " << src_node->DebugString()
                     << "new_src_node is: " << new_src_node->DebugString();
        // Insert depend src_node->depend->dst_node.
        std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), dst_node_input, new_src_node};
        auto depend_node = graph->NewCNode(depend_input);
        depend_node->AddAttr("recompute_grad_depend", MakeValue<bool>(true));
        depend_node->set_abstract(dst_node_input->abstract()->Clone());
        manager->SetEdge(dst_node, dst_node_pair.second, depend_node);
      }
      overlap_size--;
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
