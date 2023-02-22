/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/optimizer/cudnn_inplace_fusion.h"

#include <memory>
#include <vector>
#include <set>
#include <map>
#include <queue>
#include <algorithm>
#include <utility>
#include <string>

#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/contract.h"
#include "backend/common/optimizer/helper.h"
#include "plugin/device/gpu/hal/device/kernel_info_setter.h"

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
  {kConv2DBackpropInputOpName, 0}, {kMaxPoolGradOpName, 2}, {kBatchNormGradWithAddAndActivation, 0}};

constexpr size_t inplace_node_size = 2;

template <typename T>
void SetPrimAttr(AnfNodePtr inplace_node, const string &key, const T &value) {
  auto primitive = common::AnfAlgo::GetCNodePrimitive(inplace_node);
  MS_EXCEPTION_IF_NULL(primitive);
  primitive->AddAttr(key, MakeValue(value));
}

// Check whether exist a route from src node to dst node.
bool ExistRoute(const CNodePtr &src, const CNodePtr &dst) {
  MS_EXCEPTION_IF_NULL(src);
  MS_EXCEPTION_IF_NULL(dst);

  if (src == dst) {
    return true;
  }

  auto seen = NewSeenGeneration();
  std::queue<CNodePtr> to_do;
  to_do.push(dst);
  while (!to_do.empty()) {
    const auto &current_node = to_do.front();
    size_t input_num = common::AnfAlgo::GetInputTensorNum(current_node);
    for (size_t input_index = 0; input_index < input_num; ++input_index) {
      const AnfNodePtr &input_node = common::AnfAlgo::GetInputNode(current_node, input_index);
      MS_EXCEPTION_IF_NULL(input_node);
      const auto &cnode = input_node->cast<CNodePtr>();
      if (cnode == nullptr) {
        continue;
      }
      if (cnode->seen_ == seen) {
        continue;
      }
      // Exist a route from src node to dst.
      if (cnode == src) {
        return true;
      }
      to_do.push(cnode);
      cnode->seen_ = seen;
    }
    to_do.pop();
  }
  return false;
}

// Check whether exist a route from accumulate node to cover node.
bool ExistDependencyFromAcc2Cover(const std::vector<AnfNodeIndex> &inplace_node, size_t cover_index) {
  if (inplace_node.size() != inplace_node_size) {
    return false;
  }

  size_t acc_index = cover_index == 1 ? 0 : 1;
  MS_EXCEPTION_IF_CHECK_FAIL((inplace_node.size() > cover_index), "The index is out of range.");
  MS_EXCEPTION_IF_NULL(inplace_node[cover_index].node);
  const CNodePtr &cover_node = inplace_node[cover_index].node->cast<CNodePtr>();
  const CNodePtr &acc_node = inplace_node[acc_index].node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cover_node);
  MS_EXCEPTION_IF_NULL(acc_node);
  return ExistRoute(acc_node, cover_node);
}

std::pair<size_t, bool> GetCoverIndex(const std::vector<AnfNodeIndex> &inplace_node) {
  if (inplace_node.size() != inplace_node_size) {
    return {0, false};
  }
  MS_EXCEPTION_IF_CHECK_FAIL((inplace_node.size() > 1), "The index is out of range.");
  auto first_node = inplace_node[0].node;
  auto second_node = inplace_node[1].node;
  if (common::AnfAlgo::GetCNodeName(first_node) != kConv2DBackpropInputOpName ||
      common::AnfAlgo::GetCNodeName(second_node) != kConv2DBackpropInputOpName) {
    return {0, false};
  }

  auto first_node_prim = common::AnfAlgo::GetCNodePrimitive(first_node);
  MS_EXCEPTION_IF_NULL(first_node_prim);
  auto first_node_channel = first_node_prim.get()->GetAttr("out_channel");
  MS_EXCEPTION_IF_NULL(first_node_channel);
  auto first_imm_ptr = first_node_channel->cast<Int64ImmPtr>();
  MS_EXCEPTION_IF_NULL(first_imm_ptr);
  size_t first_channel = first_imm_ptr->value();
  auto second_node_prim = common::AnfAlgo::GetCNodePrimitive(second_node);
  MS_EXCEPTION_IF_NULL(second_node_prim);
  auto second_node_channel = second_node_prim.get()->GetAttr("out_channel");
  MS_EXCEPTION_IF_NULL(second_node_channel);
  auto second_imm_ptr = second_node_channel->cast<Int64ImmPtr>();
  MS_EXCEPTION_IF_NULL(second_imm_ptr);
  size_t second_channel = second_imm_ptr->value();
  size_t cover_index = (first_channel >= second_channel) ? 0 : 1;
  bool ret = ExistDependencyFromAcc2Cover(inplace_node, cover_index);
  if (ret) {
    return {0, false};
  }
  return {cover_index, true};
}

void CopyKernelInfo(AnfNodePtr src, AnfNodePtr dst) {
  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(src);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, dst.get());
  size_t output_num = AnfAlgo::GetOutputTensorNum(src);
  std::vector<TypeId> types;
  std::vector<BaseShapePtr> shapes;
  for (size_t i = 0; i < output_num; i++) {
    types.emplace_back(common::AnfAlgo::GetOutputInferDataType(src, i));
    shapes.emplace_back(AnfAlgo::GetOutputDetailShape(src, i));
  }
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, dst.get());
}

void CheckInplaceNodeInputs(std::vector<AnfNodeIndex> *inplace_node, size_t cover_index, const FuncGraphPtr &graph) {
  // If two inplace nodes have same input, will be have loop after insert depend:
  //            A                              A     Cover <----+
  //          /    \                            \    /          |
  //         B      \            -->            Depend -------> B
  //        /        \                            |
  //      Cover      Acc                         Acc
  // so copy a new input for one of inplace node like this
  //        A         A'                          A           A'
  //        |         |                           |           |
  //        B         |          -->              B        Depend <-+
  //        |         |                           |           |     |
  //      Cover      Acc                          |          Acc    |
  //                                            Cover---------------+
  MS_EXCEPTION_IF_NULL(inplace_node);
  MS_EXCEPTION_IF_NULL(graph);
  size_t acc_index = cover_index == 1 ? 0 : 1;
  MS_EXCEPTION_IF_NULL(inplace_node->at(cover_index).node);
  const CNodePtr &cover_node = inplace_node->at(cover_index).node->cast<CNodePtr>();
  const CNodePtr &acc_node = inplace_node->at(acc_index).node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cover_node);
  MS_EXCEPTION_IF_NULL(acc_node);
  const auto &acc_input = acc_node->input(1)->cast<CNodePtr>();
  if (acc_input == nullptr) {
    return;
  }
  bool ret = ExistRoute(acc_input, cover_node);
  if (ret) {
    auto new_input = graph->NewCNode(acc_input->inputs());
    MS_EXCEPTION_IF_NULL(new_input);
    new_input->set_abstract(acc_input->abstract());
    CopyKernelInfo(acc_input, new_input);
    std::vector<AnfNodePtr> new_inplace_input = acc_node->inputs();
    new_inplace_input[1] = new_input;
    auto new_inplace_node = graph->NewCNode(new_inplace_input);
    MS_EXCEPTION_IF_NULL(new_inplace_node);
    new_inplace_node->set_abstract(acc_node->abstract());
    CopyKernelInfo(acc_node, new_inplace_node);
    auto manager = graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    manager->Replace(acc_node, new_inplace_node);
    (*inplace_node)[acc_index].node = new_inplace_node;
  }
}

void SetNodeAttr(AnfNodeIndex aggregate_node, AnfNodePtr skip_node, std::vector<AnfNodeIndex> *inplace_node,
                 const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(skip_node);
  MS_EXCEPTION_IF_NULL(inplace_node);
  MS_EXCEPTION_IF_NULL(graph);

  SetPrimAttr(aggregate_node.node, "aggregate", true);
  SetPrimAttr(aggregate_node.node, "aggregate_input_index", aggregate_node.index);
  SetPrimAttr(skip_node, "skip", true);

  static uint32_t group = 0;
  auto [cover_index, order_required] = GetCoverIndex(*inplace_node);
  CheckInplaceNodeInputs(inplace_node, cover_index, graph);

  for (size_t i = 0; i < inplace_node->size(); i++) {
    auto algo = (i == cover_index) ? "cover" : "accumulation";
    auto node = (*inplace_node)[i].node;
    MS_EXCEPTION_IF_NULL(node);
    SetPrimAttr(node, "inplace_algo", algo);
    SetPrimAttr(node, "inplace_group", group);
    SetPrimAttr(node, "inplace_output_index", (*inplace_node)[i].index);
    // for Conv2DBackpropInputOp, need insert depend node to keep order, set the larger channel to cover
    if (order_required && i != cover_index) {
      auto acc_node = node;
      auto cover_node = (*inplace_node)[cover_index].node;
      auto cnode = acc_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto acc_node_input = cnode->input(1);
      std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                        acc_node_input, cover_node};
      auto depend_node = graph->NewCNode(inputs);
      MS_EXCEPTION_IF_NULL(depend_node);
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
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(inplace);
  MS_EXCEPTION_IF_NULL(skip_node);
  MS_EXCEPTION_IF_NULL(aggregate);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto aggregate_iter = kAggregatesOpNames.find(common::AnfAlgo::GetCNodeName(node));
  if (aggregate_iter == kAggregatesOpNames.end()) {
    return false;
  }
  aggregate->node = node;
  aggregate->index = aggregate_iter->second;

  *skip_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), aggregate_iter->second);
  if (*skip_node == nullptr || !(*skip_node)->isa<CNode>() ||
      kSkipOpNames.count(common::AnfAlgo::GetCNodeName(*skip_node)) == 0 ||
      GetRealNodeUsedList(graph, *skip_node)->size() >= 2) {
    return false;
  }

  auto cnode = (*skip_node)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t i = 0; i < input_num; i++) {
    auto inplace_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(*skip_node), i);
    if (!inplace_node->isa<CNode>()) {
      return false;
    }
    // Check Inplace nodes have no user except TensorAdd nodes
    if (GetRealNodeUsedList(graph, inplace_node)->size() >= 2) {
      return false;
    }

    // skip TupleGetItem node
    if (common::AnfAlgo::GetCNodeName(inplace_node) == prim::kPrimTupleGetItem->name()) {
      inplace_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(inplace_node), 0);
    }

    auto inplace_iter = kInplaceOpNames.find(common::AnfAlgo::GetCNodeName(inplace_node));
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
    AnfNodePtr skip_node = nullptr;
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

    MS_EXCEPTION_IF_NULL(aggregate_node.node);
    MS_EXCEPTION_IF_NULL(skip_node);
    MS_EXCEPTION_IF_CHECK_FAIL((inplace_node.size() > 1), "The index is out of range.");
    MS_EXCEPTION_IF_NULL(inplace_node[0].node);
    MS_EXCEPTION_IF_NULL(inplace_node[1].node);
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
