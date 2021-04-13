/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
std::map<string, uint32_t> kInplaceOpNames = {{kConv2DBackpropInputOpName, 0}, {kBatchNormGradWithAddAndActivation, 3}};

std::set<string> kSkipOpNames = {
  kTensorAddOpName,
};

// opname, input idx
std::map<string, uint32_t> kAggregatesOpNames = {
  {kConv2DBackpropInputOpName, 0}, {kmaxPoolGradOpName, 2}, {kBatchNormGradWithAddAndActivation, 0}};

constexpr size_t inplace_node_size = 2;

template <typename T>
void SetPrimAttr(AnfNodePtr inplace_node, const string &key, const T &value) {
  auto primitive = AnfAlgo::GetCNodePrimitive(inplace_node);
  MS_EXCEPTION_IF_NULL(primitive);
  primitive->AddAttr(key, MakeValue(value));
}

std::pair<size_t, bool> GetCoverIndex(const std::vector<AnfNodeIndex> &inplace_node) {
  if (inplace_node.size() != inplace_node_size) {
    return {0, false};
  }
  auto first_node = inplace_node[0].node;
  auto second_node = inplace_node[1].node;
  if (AnfAlgo::GetCNodeName(first_node) != kConv2DBackpropInputOpName ||
      AnfAlgo::GetCNodeName(second_node) != kConv2DBackpropInputOpName) {
    return {0, false};
  }

  auto first_node_prim = AnfAlgo::GetCNodePrimitive(first_node);
  auto first_node_channel = first_node_prim.get()->GetAttr("out_channel");
  MS_EXCEPTION_IF_NULL(first_node_channel);
  size_t first_channel = first_node_channel->cast<Int64ImmPtr>()->value();
  auto second_node_prim = AnfAlgo::GetCNodePrimitive(second_node);
  auto second_node_channel = second_node_prim.get()->GetAttr("out_channel");
  MS_EXCEPTION_IF_NULL(second_node_channel);
  size_t second_channel = second_node_channel->cast<Int64ImmPtr>()->value();
  size_t cover_index = (first_channel >= second_channel) ? 0 : 1;
  return {cover_index, true};
}

void CopyKernelInfo(AnfNodePtr src, AnfNodePtr dst) {
  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(src);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, dst.get());
  size_t output_num = AnfAlgo::GetOutputTensorNum(src);
  std::vector<TypeId> types;
  std::vector<std::vector<size_t>> shapes;
  for (size_t i = 0; i < output_num; i++) {
    types.emplace_back(AnfAlgo::GetOutputInferDataType(src, i));
    shapes.emplace_back(AnfAlgo::GetOutputInferShape(src, i));
  }
  AnfAlgo::SetOutputInferTypeAndShape(types, shapes, dst.get());
}

void CheckInplaceNodeInputs(std::vector<AnfNodeIndex> *inplace_node, const FuncGraphPtr &graph) {
  if (inplace_node->size() == inplace_node_size) {
    auto first_cnode = (*inplace_node)[0].node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(first_cnode);
    auto first_node_input = first_cnode->input(1);
    auto second_cnode = (*inplace_node)[1].node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(second_cnode);
    auto second_node_input = second_cnode->input(1);
    // if two inplace nodes have same input, will be have loop after insert depend
    // so copy a new input for one of inplace node
    if (first_node_input == second_node_input) {
      auto cnode = first_node_input->cast<CNodePtr>();
      auto new_input = graph->NewCNode(cnode->inputs());
      new_input->set_abstract(first_node_input->abstract());
      CopyKernelInfo(first_node_input, new_input);
      auto new_inplace_node = graph->NewCNode({first_cnode->input(0), new_input, first_cnode->input(2)});
      new_inplace_node->set_abstract(first_cnode->abstract());
      CopyKernelInfo(first_cnode, new_inplace_node);
      auto manager = graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      manager->Replace(first_cnode, new_inplace_node);
      (*inplace_node)[0].node = new_inplace_node;
    }
  }
}

void SetNodeAttr(AnfNodeIndex aggregate_node, AnfNodePtr skip_node, std::vector<AnfNodeIndex> *inplace_node,
                 const FuncGraphPtr &graph) {
  SetPrimAttr(aggregate_node.node, "aggregate", true);
  SetPrimAttr(aggregate_node.node, "aggregate_input_index", aggregate_node.index);
  SetPrimAttr(skip_node, "skip", true);

  static uint32_t group = 0;
  auto [cover_index, order_required] = GetCoverIndex(*inplace_node);
  if (order_required) {
    CheckInplaceNodeInputs(inplace_node, graph);
  }
  for (size_t i = 0; i < inplace_node->size(); i++) {
    auto algo = (i == cover_index) ? "cover" : "accumulation";
    auto node = (*inplace_node)[i].node;
    SetPrimAttr(node, "inplace_algo", algo);
    SetPrimAttr(node, "inplace_group", group);
    SetPrimAttr(node, "inplace_output_index", (*inplace_node)[i].index);
    // for Conv2DBackpropInputOp, need insert depend node to keep order, set the larger channel to cover
    if (order_required && i != cover_index) {
      auto acc_node = node;
      auto cover_node = (*inplace_node)[cover_index].node;
      auto acc_node_input = acc_node->cast<CNodePtr>()->input(1);
      std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                        acc_node_input, cover_node};
      auto depend_node = graph->NewCNode(inputs);
      depend_node->set_abstract(acc_node_input->abstract());
      auto manager = graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      manager->Replace(acc_node_input, depend_node);
    }
  }
  group++;
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
  size_t input_num = AnfAlgo::GetInputTensorNum(cnode);
  for (size_t i = 0; i < input_num; i++) {
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
    SetNodeAttr(aggregate_node, skip_node, &inplace_node, graph);
  }

  return true;
}
}  // namespace opt
}  // namespace mindspore
