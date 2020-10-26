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

#include "backend/optimizer/gpu/cudnn_inplace_fusion.h"

#include <memory>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <utility>
#include <string>

#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "utils/contract.h"
#include "backend/optimizer/common/helper.h"
#include "runtime/device/gpu/kernel_info_setter.h"

namespace mindspore {
namespace opt {
namespace {
struct AnfNodeIndex {
  AnfNodeIndex() : node(nullptr), index(0) {}
  AnfNodeIndex(const AnfNodePtr &n, const int i) : node(n), index(i) {}
  AnfNodePtr node;
  uint32_t index;
};

// opname, output idx
std::map<string, uint32_t> kInplaceOpNames = {{kConv2DBackpropInputOpName, 0},
                                              {kFusedBatchNormGradExWithAddAndActivation, 3}};

std::set<string> kSkipOpNames = {
  kTensorAddOpName,
};

// opname, input idx
std::map<string, uint32_t> kAggregatesOpNames = {
  {kConv2DBackpropInputOpName, 0}, {kmaxPoolGradOpName, 2}, {kFusedBatchNormGradExWithAddAndActivation, 0}};

template <typename T>
void SetPrimAttr(AnfNodePtr inplace_node, const string &key, const T &value) {
  auto primitive = AnfAlgo::GetCNodePrimitive(inplace_node);
  MS_EXCEPTION_IF_NULL(primitive);
  primitive->AddAttr(key, MakeValue(value));
}

void SetNodeAttr(AnfNodeIndex aggregate_node, AnfNodePtr skip_node, std::vector<AnfNodeIndex> *inplace_node) {
  SetPrimAttr(aggregate_node.node, "aggregate", true);
  SetPrimAttr(aggregate_node.node, "aggregate_input_index", aggregate_node.index);
  SetPrimAttr(skip_node, "skip", true);

  static uint32_t group = 0;
  for (size_t i = 0; i < inplace_node->size(); i++) {
    auto algo = (i == 0) ? "cover" : "accumulation";
    SetPrimAttr((*inplace_node)[i].node, "inplace_algo", algo);
    SetPrimAttr((*inplace_node)[i].node, "inplace_group", group);
    SetPrimAttr((*inplace_node)[i].node, "inplace_output_index", (*inplace_node)[i].index);
  }
  group++;
}

void InsertControlDependToGraph(const FuncGraphPtr &graph, const std::vector<AnfNodeIndex> &inplace_nodes,
                                const AnfNodePtr aggregate_node) {
  std::vector<AnfNodePtr> inputs1 = {NewValueNode(std::make_shared<Primitive>(prim::kPrimControlDepend->name())),
                                     inplace_nodes[0].node, inplace_nodes[1].node};
  auto control_depend_node = graph->NewCNode(inputs1);

  std::vector<AnfNodePtr> inputs2 = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                     aggregate_node, control_depend_node};
  auto depend_node = graph->NewCNode(inputs2);

  auto users = GetRealNodeUsedList(graph, aggregate_node);
  if (users->size() == 0) {
    MS_LOG(EXCEPTION) << "No users found: " << aggregate_node->DebugString();
  }
  auto mount_node = users->at(0).first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(mount_node);
  mount_node->set_input(kFirstDataInputIndex, depend_node);
}

bool PatternMatch(const FuncGraphPtr &graph, const AnfNodePtr &node, AnfNodeIndex *aggregate, AnfNodePtr *skip_node,
                  std::vector<AnfNodeIndex> *inplace) {
  MS_EXCEPTION_IF_NULL(skip_node);
  MS_EXCEPTION_IF_NULL(aggregate);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto aggregate_iter = kAggregatesOpNames.find(AnfAlgo::GetCNodeName(node));
  if (aggregate_iter == kAggregatesOpNames.end()) {
    return false;
  }
  aggregate->node = node;
  aggregate->index = aggregate_iter->second;

  *skip_node = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), aggregate_iter->second);
  if (*skip_node == nullptr || !(*skip_node)->isa<CNode>() ||
      kSkipOpNames.count(AnfAlgo::GetCNodeName(*skip_node)) == 0 ||
      GetRealNodeUsedList(graph, *skip_node)->size() >= 2) {
    return false;
  }

  auto cnode = (*skip_node)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(cnode); i++) {
    auto inplace_node = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(*skip_node), i);
    if (!inplace_node->isa<CNode>()) {
      return false;
    }
    // Check Inplace nodes have no user except TensorAdd nodes
    if (GetRealNodeUsedList(graph, inplace_node)->size() >= 2) {
      return false;
    }

    // skip TupleGetItem node
    if (AnfAlgo::GetCNodeName(inplace_node) == prim::kPrimTupleGetItem->name()) {
      inplace_node = AnfAlgo::GetInputNode(utils::cast<CNodePtr>(inplace_node), 0);
    }

    auto inplace_iter = kInplaceOpNames.find(AnfAlgo::GetCNodeName(inplace_node));
    if (inplace_iter == kInplaceOpNames.end()) {
      return false;
    }

    inplace->push_back(AnfNodeIndex(inplace_node, inplace_iter->second));
  }

  return true;
}

std::map<AnfNodePtr, int> TopoIndex(const std::vector<AnfNodePtr> &node_list) {
  std::map<AnfNodePtr, int> topo_index;
  for (size_t i = 0; i < node_list.size(); i++) {
    topo_index.insert(make_pair(node_list[i], i));
  }
  return topo_index;
}
}  // namespace

bool CudnnInplaceAggregate::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  auto topo_index = TopoIndex(node_list);

  for (auto node : node_list) {
    AnfNodeIndex aggregate_node;
    AnfNodePtr skip_node;
    std::vector<AnfNodeIndex> inplace_node;
    // 1. Pattern Match.
    if (!PatternMatch(graph, node, &aggregate_node, &skip_node, &inplace_node)) {
      continue;
    }

    // 2. Keep the original topological order in case the dependence between inplace nodes
    std::sort(inplace_node.begin(), inplace_node.end(), [&topo_index](const AnfNodeIndex &n1, const AnfNodeIndex &n2) {
      auto iter1 = topo_index.find(n1.node);
      auto iter2 = topo_index.find(n2.node);
      if (iter1 == topo_index.end() || iter2 == topo_index.end()) {
        MS_LOG(EXCEPTION) << ": Node not existed in topo order. node1: " << n1.node->DebugString()
                          << ", node2: " << n2.node->DebugString();
      }

      if (iter1->second < iter2->second) {
        return true;
      }
      return false;
    });
    MS_LOG(INFO) << "[inplace optimizer] aggregate node: " << aggregate_node.index << ", "
                 << aggregate_node.node->DebugString() << "; skip node: " << skip_node->DebugString() << std::endl
                 << "; inplace node 0: " << inplace_node[0].index << ", " << inplace_node[0].node->DebugString()
                 << std::endl
                 << "; inplace node 1: " << inplace_node[1].index << ", " << inplace_node[1].node->DebugString()
                 << std::endl;
    // 2. Set Node attr
    SetNodeAttr(aggregate_node, skip_node, &inplace_node);
    // 3. Set dependence for inplace nodes
    InsertControlDependToGraph(graph, inplace_node, aggregate_node.node);
  }

  return true;
}
}  // namespace opt
}  // namespace mindspore
