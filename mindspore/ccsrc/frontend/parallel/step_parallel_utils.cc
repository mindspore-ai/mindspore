/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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
#include "frontend/parallel/step_parallel_utils.h"

#include <algorithm>
#include <cinttypes>

#include <map>
#include <memory>
#include <queue>
#include <set>
#include <unordered_map>
#include <string>
#include <utility>

#include "abstract/dshape.h"
#include "base/base.h"
#include "frontend/operator/ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/graph_util/parallel_tensordump.h"
#include "frontend/parallel/node_check.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/dynamic_shape/dynamic_shape.h"
#include "include/utils/comm_manager.h"
#include "include/utils/parallel_context.h"
#include "include/utils/anfalgo.h"
#include "ir/param_info.h"
#include "ir/tensor.h"
#include "ir/graph_utils.h"
#include "primitive/array_ops.h"
#include "primitive/framework_ops.h"
#include "primitive/structure_ops.h"
#include "primitive/nn_ops.h"
#include "primitive/other_ops.h"
#include "primitive/sequence_ops.h"
#include "primitive/arithmetic_ops.h"
#include "frontend/parallel/parallel_node_check.h"
#include "utils/hash_map.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "utils/trace_base.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"
#include "primitive/auto_generate/gen_ops_primitive_v.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
namespace parallel {
using mindspore::tensor::Tensor;
size_t TOTAL_OPS = 0;
static const std::set<std::string> INVALID_LOSS_OPS = {GET_NEXT, VIRTUALLOSS, LOAD, UPDATESTATE};
// g_RefMap, for CNode B input i is a RefKey[Parameter C],
// it will be one item in map with key: C, and value: (B, i)
std::map<AnfNodePtr, std::pair<AnfNodePtr, int64_t>> g_RefMap;

bool IsDynamicShapeInput(const CNodePtr &node, const AnfNodePtr &input) {
  if (IsSomePrimitiveList(node, CANDIDATE_DYNAMIC_VALUE_OPS) &&
      (IsPrimitiveCNode(input, prim::kPrimMakeTuple) || IsPrimitiveCNode(input, prim::kPrimShape))) {
    return true;
  }
  if (IsPrimitiveCNode(node, prim::kPrimCast) && IsPrimitiveCNode(input, prim::kPrimTupleGetItem)) {
    BaseShapePtr base_shape_ptr = node->Shape();
    if (base_shape_ptr == nullptr) {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "IsDynamicShapeInput: " << node->ToString()
                                        << " shape_ptr is nullptr, full name is " << node->fullname_with_scope();
    }
    auto shape_ptr = dyn_cast<abstract::Shape>(base_shape_ptr);
    MS_EXCEPTION_IF_NULL(shape_ptr);
    if (shape_ptr->shape().empty()) {
      return true;
    }
  }
  return false;
}

bool IsSomePrimitive(const CNodePtr &cnode, const std::string &name) {
  if (!cnode) {
    return false;
  }
  ValueNodePtr anf_node = cnode->input(0)->cast<ValueNodePtr>();
  if (!anf_node) {
    return false;
  }
  PrimitivePtr prim = anf_node->value()->cast<PrimitivePtr>();
  if (!prim) {
    return false;
  }
  return (prim->name() == name);
}

bool IsSomePrimitiveList(const CNodePtr &cnode, const std::set<string> &check_list) {
  if (!cnode) {
    return false;
  }
  ValueNodePtr anf_node = cnode->input(0)->cast<ValueNodePtr>();
  if (!anf_node) {
    return false;
  }
  PrimitivePtr prim = anf_node->value()->cast<PrimitivePtr>();
  if (!prim) {
    return false;
  }
  return std::any_of(check_list.begin(), check_list.end(), [prim](const string &in) { return prim->name() == in; });
}

bool IsIgnoreSplitTensor(const CNodePtr &node, int64_t index) {
  if (IsSomePrimitiveList(node, SPLIT_TENSOR_ONLY_FOR_FIRST_INPUT_OPS) && index > 0) {
    return true;
  }
  return false;
}

std::string GetPrimName(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto prim = GetCNodePrimitive(node);
  if (!prim) {
    return node->DebugString();
  }
  return prim->name();
}

bool IsTraining(const FuncGraphManagerPtr &manager) {
  for (auto &fg : manager->func_graphs()) {
    if (fg->has_flag(kTraining)) {
      return true;
    }
  }
  return false;
}

bool HasBackward(const FuncGraphPtr &root) {
  auto nodes = root->nodes();
  for (auto &node : nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimJ)) {
      return true;
    }
  }
  return false;
}

std::vector<TensorInfo> GetInputsTensorInfo(const std::pair<AnfNodePtr, int64_t> &param_info) {
  auto user_cnode = param_info.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(user_cnode);
  auto user_input_index = param_info.second;
  OperatorInfoPtr op_info = user_cnode->user_data<OperatorInfo>();
  MS_EXCEPTION_IF_NULL(op_info);

  std::vector<TensorInfo> tensor_info_vector;
  if (IsPrimitiveCNode(user_cnode, prim::kPrimSend)) {
    auto param_index = IntToSize(GetValue<int>(user_cnode->GetPrimalAttr(PARAM_INDEX)));
    auto tensor_info = op_info->inputs_tensor_info()[param_index];
    tensor_info_vector.push_back(tensor_info);
  } else {
    if (op_info->inputs_tensor_info_new().empty()) {
      size_t input_tensor_info_size = op_info->inputs_tensor_info().size();
      if (SizeToLong(input_tensor_info_size) <= user_input_index - 1) {
        MS_LOG_WITH_NODE(EXCEPTION, user_cnode)
          << op_info->name() << ": the size of inputs tensor info is " << input_tensor_info_size
          << ", but the index is " << (user_input_index - 1);
      }
      auto tensor_info = op_info->inputs_tensor_info()[LongToSize(user_input_index - 1)];
      tensor_info_vector.push_back(tensor_info);
    } else {
      size_t input_tensor_info_size = op_info->inputs_tensor_info_new().size();
      if (SizeToLong(input_tensor_info_size) <= user_input_index - 1) {
        MS_LOG_WITH_NODE(EXCEPTION, user_cnode)
          << op_info->name() << ": the size of inputs tensor info is " << input_tensor_info_size
          << ", but the index is " << (user_input_index - 1);
      }
      tensor_info_vector = op_info->inputs_tensor_info_new()[LongToSize(user_input_index - 1)]->GetAllElements();
    }
  }
  return tensor_info_vector;
}

static bool IsRealKernelNode(const AnfNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimDepend) || IsPrimitiveCNode(node, prim::kPrimLoad) ||
      IsPrimitiveCNode(node, prim::kPrimCast) || IsPrimitiveCNode(node, prim::kPrimVirtualDiv) ||
      IsPrimitiveCNode(node, prim::kPrimReceive) || IsPrimitiveCNode(node, prim::kPrimMicroStepAllGather) ||
      IsPrimitiveCNode(node, prim::kPrimSend) || IsPrimitiveCNode(node, prim::kPrimInsertGradientOf) ||
      IsPrimitiveCNode(node, prim::kPrimDumpGradient)) {
    return false;
  }
  return true;
}

std::pair<AnfNodePtr, int64_t> GetRealKernelNode(const AnfNodePtr &node, int64_t get_item_index, CNodePtr *call_node,
                                                 bool ignore_get_item) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsRealKernelNode(node)) {
    const auto &&cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    size_t travel_index = IsPrimitiveCNode(node, prim::kPrimDumpGradient) ? kDumpGradientSkipIndex : kIndex1;
    return GetRealKernelNode(cnode->input(travel_index), get_item_index, call_node, ignore_get_item);
  }
  if ((IsPrimitiveCNode(node, prim::kPrimTupleGetItem) || IsPrimitiveCNode(node, prim::kPrimInsertGradientOf)) &&
      ignore_get_item) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto cur_get_item_index = LongToInt(GetTupleGetItemIndex(cnode));
    auto tuple_getitem_input = cnode->input(1);
    return GetRealKernelNode(tuple_getitem_input, cur_get_item_index, call_node, ignore_get_item);
  }
  if (get_item_index != -1 &&
      (IsPrimitiveCNode(node, prim::kPrimMakeTuple) || IsPrimitiveCNode(node, prim::kPrimInsertGradientOf))) {
    auto make_tuple_cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple_cnode);
    auto make_tuple_input = make_tuple_cnode->input(LongToSize(get_item_index + 1));
    return GetRealKernelNode(make_tuple_input, -1, call_node, ignore_get_item);
  }
  if (IsControlFlowNode(node)) {
    const auto &&cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    MS_EXCEPTION_IF_NULL(cnode->input(0));
    auto switch_cnode = cnode->input(0)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(switch_cnode);
    auto fg = GetValueNode<FuncGraphPtr>(switch_cnode->input(3));
    MS_EXCEPTION_IF_NULL(fg);
    return GetRealKernelNode(fg->output(), get_item_index, call_node, ignore_get_item);
  }
  if (node->isa<CNode>() && IsValueNode<FuncGraph>(node->cast<CNodePtr>()->input(0))) {
    if (call_node != nullptr && *call_node == nullptr) {
      *call_node = node->cast<CNodePtr>();
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto graph = GetValueNode<FuncGraphPtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(graph);
    auto output = GetRealKernelNode(graph->output(), get_item_index, call_node, ignore_get_item).first;
    MS_EXCEPTION_IF_NULL(output);
    if (output->isa<Parameter>()) {
      auto param_graph = output->func_graph();
      auto parameter_list = param_graph->parameters();
      auto fg_used_map = param_graph->func_graph_cnodes_index();
      for (auto &cur_fg_use : fg_used_map) {
        if (cur_fg_use.first->second != 0) {
          continue;
        }
        auto cur_fg = cur_fg_use.first->first->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cur_fg);
        auto iter = std::find(parameter_list.begin(), parameter_list.end(), output);
        auto pos = std::distance(parameter_list.begin(), iter);
        auto argument = cur_fg->input(pos + 1);
        return GetRealKernelNode(argument, get_item_index, call_node, ignore_get_item);
      }
      return std::make_pair(output, get_item_index);
    }
    return std::make_pair(output, get_item_index);
  }
  return std::make_pair(node, get_item_index);
}

static bool IsWhileGraph(const FuncGraphPtr &cur_fg, const FuncGraphPtr &fg) {
  auto cur_fg_map = cur_fg->func_graph_cnodes_index();
  for (auto &cur_fg_use : cur_fg_map) {
    auto temp_node = cur_fg_use.first->first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(temp_node);
    if (temp_node->func_graph() == fg) {
      return true;
    }
  }
  return false;
}

static RankList GetDevListByTensorMapValue(const DeviceMatrix dev_matrix, const int64_t &tensor_map_value,
                                           const size_t &dev_matrix_size) {
  RankList rank_list;
  if (tensor_map_value >= SizeToLong(dev_matrix_size) || tensor_map_value < MAP_NONE) {
    MS_LOG(ERROR) << "The size of dev_matrix is " << dev_matrix_size << ", but the tensor map value is "
                  << tensor_map_value;
    return rank_list;
  }

  if (tensor_map_value == MAP_NONE) {
    rank_list.push_back(g_device_manager->global_rank());
    return rank_list;
  }

  uint64_t dim = dev_matrix_size - LongToSize(tensor_map_value) - 1;
  if (dev_matrix.GetDevicesAlongDim(dim, &rank_list) != SUCCESS) {
    MS_LOG(ERROR) << "Get devices along dim failed";
  }

  return rank_list;
}

static bool IsSameTensorLayout(const TensorLayout &a, const TensorLayout &b) {
  if (!a.IsSameTensorShape(b)) {
    return false;
  }
  if (a.IsSameDeviceArrangement(b) && a.IsSameTensorMap(b)) {
    return true;
  }

  Shape a_tensor_map = a.tensor_map().array();
  Shape b_tensor_map = b.tensor_map().array();
  if (a_tensor_map.size() != b_tensor_map.size()) {
    return false;
  }

  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix a_dev_matrix(rank, g_device_manager->GetDeviceListInThisStage(), a.device_arrangement().array());
  DeviceMatrix b_dev_matrix(rank, g_device_manager->GetDeviceListInThisStage(), b.device_arrangement().array());
  size_t a_dev_mat_size = a.device_arrangement().array().size();
  size_t b_dev_mat_size = b.device_arrangement().array().size();

  for (size_t i = 0; i < a_tensor_map.size(); ++i) {
    if (a_tensor_map[i] == MAP_NONE && b_tensor_map[i] == MAP_NONE) {
      continue;
    }

    RankList a_dev_list_by_dim = GetDevListByTensorMapValue(a_dev_matrix, a_tensor_map[i], a_dev_mat_size);
    RankList b_dev_list_by_dim = GetDevListByTensorMapValue(b_dev_matrix, b_tensor_map[i], b_dev_mat_size);
    if (a_dev_list_by_dim.empty() || b_dev_list_by_dim.empty()) {
      MS_LOG(EXCEPTION) << "Can not get device list by tensor map value, these layouts are " << a.ToString()
                        << std::endl
                        << " and " << b.ToString();
    }

    if (a_dev_list_by_dim != b_dev_list_by_dim) {
      return false;
    }
  }

  return true;
}

bool IsSameTensorInfo(const std::vector<TensorInfo> &a, const std::vector<TensorInfo> &b) {
  if (a.size() != b.size()) {
    MS_LOG(ERROR) << "The size of tensor info is different, they are " << a.size() << " and " << b.size()
                  << ", which means that one input is tuple, but another input is not";
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (!IsSameTensorLayout(a[i].tensor_layout(), b[i].tensor_layout())) {
      MS_LOG(ERROR) << "The " << i << "th tensor layout is different, they are " << a[i].tensor_layout().ToString()
                    << " and " << b[i].tensor_layout().ToString();
      return false;
    }
  }
  return true;
}

std::pair<AnfNodePtr, int> CheckMakeTupleSplit(const AnfNodePtr &node, const FuncGraphManagerPtr &manager) {
  auto node_users = manager->node_users()[node];
  if (node_users.size() == 1) {
    return std::make_pair(node_users.front().first, node_users.front().second);
  }

  bool is_first_tensor_info = true;
  std::vector<TensorInfo> first_tensor_info;
  AnfNodePtr first_node;
  int corresponding_input_pos;
  for (auto &node_user : node_users) {
    auto user_node = node_user.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_node);
    if (!user_node->has_user_data<OperatorInfo>()) {
      continue;
    }
    auto tensor_info = GetInputsTensorInfo(node_user);
    if (is_first_tensor_info) {
      is_first_tensor_info = false;
      first_tensor_info = tensor_info;
      first_node = node_user.first;
      corresponding_input_pos = node_user.second;
      continue;
    }
    if (IsSameTensorInfo(first_tensor_info, tensor_info)) {
      continue;
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "The node: " << node->DebugString()
                                        << " has multiple users, but the TensorInfo are different";
    }
  }
  return std::make_pair(first_node, corresponding_input_pos);
}

bool IsParallelCareNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  // Not skip Send Receive in pp interleave
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto is_pp_interleave = parallel_context->pipeline_interleave();
  if (is_pp_interleave && (IsPrimitiveCNode(cnode, prim::kPrimSend) || IsPrimitiveCNode(cnode, prim::kPrimReceive))) {
    return false;
  }
  ValueNodePtr prim_node = cnode->input(0)->cast<ValueNodePtr>();
  if (prim_node == nullptr) {
    return false;
  }
  PrimitivePtr prim = prim_node->value()->cast<PrimitivePtr>();
  if (prim == nullptr) {
    return false;
  }
  if (!IsParallelConsiderCNode(cnode)) {
    MS_LOG(DEBUG) << "Parallel don't care node: " << prim->name();
    return false;
  }
  // get_next is not in the forward graph, we need mark the get_next as the forward node
  if (prim->name() == GET_NEXT || prim->name() == VIRTUAL_OUTPUT) {
    return true;
  }
  if ((prim->name() == CAST) && !cnode->has_user_data<OperatorInfo>()) {
    return false;
  }

  return cnode->in_forward_flag();
}

bool IsEmbedShardNode(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtr ret = func_graph->get_return();
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  return std::any_of(all_nodes.begin(), all_nodes.end(), [&func_graph](const AnfNodePtr &node) {
    return IsPrimitiveCNode(node, prim::kPrimShard) && (node->func_graph() == func_graph);
  });
}

Shapes GetValueListShape(const AnfNodePtr &node, bool is_new_shape_base = false) {
  Shapes shapes;
  std::vector<ValuePtr> inputs_seq;
  if (IsValueNode<ValueList>(node)) {
    inputs_seq = node->cast<ValueNodePtr>()->value()->cast<ValueListPtr>()->value();
  } else if (IsValueNode<ValueTuple>(node)) {
    inputs_seq = node->cast<ValueNodePtr>()->value()->cast<ValueTuplePtr>()->value();
  } else {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "node is either ValueList or ValueTuple";
  }
  for (auto &ele : inputs_seq) {
    // Scalar Tuple like (1, 1)
    if (ele->isa<Scalar>() && is_new_shape_base) {
      shapes.push_back({});
      continue;
    }
    auto tensor = ele->cast<tensor::TensorPtr>();
    if (tensor == nullptr) {
      MS_LOG(WARNING) << "The value node is not a tensor";
      break;
    }
    auto one_shape = tensor->shape();
    shapes.push_back(one_shape);
  }
  return shapes;
}

bool IsControlFlowNode(const AnfNodePtr &node) {
  // Only switch or FuncCall nodes are control flow nodes
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // func node
  if (cnode->input(0)->isa<CNode>() && IsPrimitiveCNode(cnode->input(0), prim::kPrimSwitch)) {
    return true;
  }
  return false;
}

int64_t GetTupleGetItemIndex(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (!cnode->input(TUPLE_GETITEM_INDEX_POS)->isa<ValueNode>()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "The index of tuple getitem is not a value node";
  }

  ValuePtr tuple_index_value = GetValueNode(cnode->input(TUPLE_GETITEM_INDEX_POS));
  MS_EXCEPTION_IF_NULL(tuple_index_value);
  if (!tuple_index_value->isa<Int64Imm>()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "The index of tuple getitem is not int64";
  }
  return tuple_index_value->cast<Int64ImmPtr>()->value();
}

static bool IsNoNeedRedistribution(const CNodePtr &use_cnode, int use_index) {
  return (IsPrimitiveCNode(use_cnode, prim::kPrimDepend) && use_index != 1) || use_cnode->input(0)->isa<CNode>() ||
         IsOneOfPrimitiveCNode(use_cnode, {prim::kPrimUpdateState, prim::kPrimSwitch, prim::kPrimShape,
                                           prim::kPrimTensorShape, prim::kPrimDType});
}

std::vector<std::pair<AnfNodePtr, int>> FuncGraphNodeUsers(const std::pair<AnfNodePtr, int> &node_pair) {
  std::vector<std::pair<AnfNodePtr, int>> func_users_vector;
  if (!node_pair.first->isa<CNode>()) {
    return func_users_vector;
  }
  auto use_cnode = node_pair.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(use_cnode);
  if (IsValueNode<FuncGraph>(use_cnode->input(0))) {
    auto fg = GetValueNode<FuncGraphPtr>(use_cnode->input(0));
    auto fg_parameters = fg->parameters();
    auto param = fg_parameters[IntToSize(node_pair.second - 1)];
    auto manager = fg->manager();
    auto param_node_users = manager->node_users()[param];
    for (const auto &node_user : param_node_users) {
      auto cnode = node_user.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (IsValueNode<FuncGraph>(cnode->input(0))) {
        auto sub_graph_users = FuncGraphNodeUsers(node_user);
        (void)std::copy(sub_graph_users.begin(), sub_graph_users.end(), std::back_inserter(func_users_vector));
      } else {
        func_users_vector.emplace_back(node_user);
      }
    }
  }
  return func_users_vector;
}

std::vector<int> RemovePlaceholderIdx(const std::vector<int> &get_item_index) {
  if (get_item_index.size() == 1) {
    return get_item_index;
  }
  std::vector<int> new_get_item_index;
  std::copy_if(get_item_index.begin(), get_item_index.end(), std::back_inserter(new_get_item_index),
               [](int idx) { return idx != -1; });
  if (new_get_item_index.size() < 1) {
    return {-1};
  }
  return new_get_item_index;
}

void RedistributionNextNodeInMakeTuple(
  const CNodePtr &use_cnode, const std::pair<std::shared_ptr<AnfNode>, size_t> &node_pair,
  const std::vector<int> &get_item_index, int64_t *make_tuple_index,
  std::vector<std::pair<std::pair<AnfNodePtr, PosPair>, std::vector<int>>> *next_nodes) {
  auto modified_get_item_idx = RemovePlaceholderIdx(get_item_index);
  const std::string kIncreFlashAttentionName = "IncreFlashAttention";
  if (*make_tuple_index != -1) {
    int node_pos = node_pair.second;
    if (common::AnfAlgo::GetCNodeName(use_cnode) != kIncreFlashAttentionName) {
      node_pos = IsSupportNewShapeBaseNode(use_cnode) ? node_pair.second : 1;
    }
    MS_LOG(DEBUG) << "use_cnode name: " << common::AnfAlgo::GetCNodeName(use_cnode) << " ,node_pos: " << node_pos;
    auto real_node = GetRealKernelNode(use_cnode->input(node_pos), -1, nullptr);
    if (IsPrimitiveCNode(real_node.first, prim::kPrimMakeTuple) &&
        std::find_if(next_nodes->begin(), next_nodes->end(), [&](const auto n_pair) {
          auto next_node_input_pos = n_pair.first.second.first;
          return n_pair.first.first == real_node.first && next_node_input_pos == (*make_tuple_index) + 1 &&
                 n_pair.second == modified_get_item_idx;
        }) == next_nodes->end()) {
      next_nodes->push_back(std::make_pair(std::make_pair(real_node.first, std::make_pair((*make_tuple_index) + 1, 0)),
                                           modified_get_item_idx));
      *make_tuple_index = -1;
      return;
    }
  }
  if (std::find_if(next_nodes->begin(), next_nodes->end(), [&](const auto n_pair) {
        size_t next_node_input_pos = n_pair.first.second.first;
        return n_pair.first.first == node_pair.first && next_node_input_pos == node_pair.second &&
               n_pair.second == modified_get_item_idx;
      }) == next_nodes->end()) {
    const auto &pos_pair = std::make_pair(node_pair.second, 0);
    const auto &next_node = std::make_pair(node_pair.first, pos_pair);
    next_nodes->push_back(std::make_pair(next_node, modified_get_item_idx));
  }
}

std::vector<std::pair<AnfNodePtr, int>> NextNodeUsers(const AnfNodePtr &node) {
  auto node_set = GetOutputNodesWithFilter(node, [](auto anode) { return false; });
  if (parallel::ParallelContext::GetInstance()->fine_grained_micro_interleaved_size() > 0) {
    node_set = GetOutputNodesWithFilter(node, [&](auto anode) {
      return IsPrimitiveCNode(anode, prim::kPrimLoad) || IsPrimitiveCNode(anode, prim::kPrimCast);
    });
    (void)std::partition(node_set.begin(), node_set.end(), [](const std::pair<std::shared_ptr<AnfNode>, int> &pair) {
      if (!pair.first->has_user_data<OperatorInfo>() || IsPrimitiveCNode(pair.first, prim::kPrimReshape)) {
        return false;
      }
      auto tensor_info = GetInputsTensorInfo(pair);
      return !tensor_info[0].tensor_layout().IsInterleavedParallel();
    });
  }
  return node_set;
}

void FindFuncGraphNextNode(const CNodePtr &use_cnode, const std::pair<AnfNodePtr, int> &node_pair,
                           const FuncGraphManagerPtr &manager, const NodeUsersMap &node_users_map,
                           const std::vector<int> &get_item_index, int64_t make_tuple_index,
                           std::vector<std::pair<std::pair<AnfNodePtr, PosPair>, std::vector<int>>> *next_nodes) {
  auto cur_fg = use_cnode->func_graph();
  auto fg = GetValueNode<FuncGraphPtr>(use_cnode->input(0));
  MS_EXCEPTION_IF_NULL(fg);
  if (IsWhileGraph(cur_fg, fg)) {
    return;
  }
  auto fg_parameters = fg->parameters();
  auto param = fg_parameters[IntToSize(node_pair.second - 1)];
  MS_EXCEPTION_IF_NULL(param);
  size_t index = make_tuple_index == -1 ? 0 : static_cast<size_t>(make_tuple_index);
  bool has_operator_info = false;
  if (param->has_user_data(INDEX_OPERATOR_INFO)) {
    const auto &index_operator_map = param->user_data<IndexOperatorMap>(INDEX_OPERATOR_INFO);
    MS_EXCEPTION_IF_NULL(index_operator_map);
    if (index_operator_map->find(index) != index_operator_map->end()) {
      has_operator_info = true;
    }
  }
  if (has_operator_info && std::find_if(next_nodes->begin(), next_nodes->end(), [&](const auto n_pair) {
                             int next_node_input_pos = n_pair.first.second.first;
                             return n_pair.first.first == node_pair.first && next_node_input_pos == node_pair.second &&
                                    n_pair.second == RemovePlaceholderIdx(get_item_index);
                           }) == next_nodes->end()) {
    const auto &pos_pair = std::make_pair(node_pair.second, index);
    const auto &next_node = std::make_pair(node_pair.first, pos_pair);
    next_nodes->push_back(std::make_pair(next_node, RemovePlaceholderIdx(get_item_index)));
    return;
  }
  RedistributionNextNode(param, manager, node_users_map, get_item_index, make_tuple_index, next_nodes);
  for (const auto &next_node : *next_nodes) {
    auto param_out_index = std::make_shared<ParamOutIndex>(param, index);
    next_node.first.first->set_user_data<ParamOutIndex>(PARAM_OUT_INDEX, param_out_index);
  }
}

void RedistributionNextNode(const AnfNodePtr &node, const FuncGraphManagerPtr &manager,
                            const NodeUsersMap &node_users_map, const std::vector<int> &get_item_index,
                            int64_t make_tuple_index,
                            std::vector<std::pair<std::pair<AnfNodePtr, PosPair>, std::vector<int>>> *next_nodes) {
  MS_EXCEPTION_IF_NULL(node);
  if (node_users_map.count(node) == 0) {
    return;
  }
  auto node_set = node_users_map.at(node);
  for (auto &node_pair : node_set) {
    auto use_cnode = node_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(use_cnode);
    if (IsValueNode<FuncGraph>(use_cnode->input(0))) {
      FindFuncGraphNextNode(use_cnode, node_pair, manager, node_users_map, get_item_index, make_tuple_index,
                            next_nodes);
      continue;
    }
    if (IsPrimitiveCNode(use_cnode, prim::kPrimMakeTuple)) {
      make_tuple_index = node_pair.second - 1;
      RedistributionNextNode(use_cnode, manager, node_users_map, get_item_index, make_tuple_index, next_nodes);
      continue;
    }
    if (IsPrimitiveCNode(use_cnode, prim::kPrimTupleGetItem) || IsPrimitiveCNode(use_cnode, prim::kPrimListGetItem)) {
      auto temp = LongToInt(GetTupleGetItemIndex(use_cnode));
      if (temp != make_tuple_index && make_tuple_index != -1) {
        continue;
      }
      temp = make_tuple_index != -1 ? -1 : temp;
      std::vector<int> new_get_item_index;
      std::copy(get_item_index.begin(), get_item_index.end(), std::back_inserter(new_get_item_index));
      new_get_item_index.push_back(temp);
      RedistributionNextNode(use_cnode, manager, node_users_map, new_get_item_index, -1, next_nodes);
      continue;
    }
    if (IsPrimitiveCNode(use_cnode, prim::kPrimReturn)) {
      auto fg = use_cnode->func_graph();
      auto fg_map = fg->func_graph_cnodes_index();
      for (auto &fg_use : fg_map) {
        auto fg_node = fg_use.first->first->cast<CNodePtr>();
        constexpr int SWITCH_LAST_INPUT_INDEX = 3;
        if (IsWhileGraph(fg, fg) && fg_use.first->second != SWITCH_LAST_INPUT_INDEX) {
          continue;
        }
        RedistributionNextNode(fg_node, manager, node_users_map, get_item_index, make_tuple_index, next_nodes);
      }
    }
    // depend, auto monad and control flow op don't need to jump over
    if (IsNoNeedRedistribution(use_cnode, node_pair.second)) {
      continue;
    }
    if (IsParallelCareNode(use_cnode) && use_cnode->has_user_data<OperatorInfo>()) {
      RedistributionNextNodeInMakeTuple(use_cnode, node_pair, get_item_index, &make_tuple_index, next_nodes);
      continue;
    }
    // search recursively
    RedistributionNextNode(use_cnode, manager, node_users_map, get_item_index, make_tuple_index, next_nodes);
  }
}

void RedistributionPreNode(const CNodePtr &cnode, const FuncGraphManagerPtr &manager,
                           std::vector<AnfNodePtr> *pre_nodes) {
  if (IsValueNode<FuncGraph>(cnode->input(0))) {
    return;
  }
  if (IsControlFlowNode(cnode)) {
    auto switch_cnode = cnode->input(0)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(switch_cnode);
    // extract true branch, false branch is usually also a control flow graph
    auto fg = GetValueNode<FuncGraphPtr>(switch_cnode->input(2));
    MS_EXCEPTION_IF_NULL(fg);
    MS_EXCEPTION_IF_NULL(fg->output());
    auto fg_out = fg->output()->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(fg_out);
    // control flow node, need enter graph to find redistribution pre node.
    RedistributionPreNode(fg_out, manager, pre_nodes);
  }
  if (IsPrimitiveCNode(cnode, prim::kPrimDepend) || IsPrimitiveCNode(cnode, prim::kPrimLoad) ||
      IsPrimitiveCNode(cnode, prim::kPrimCast) || IsPrimitiveCNode(cnode, prim::kPrimAllReduce)) {
    auto cnode_input = cnode->input(1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode_input);
    RedistributionPreNode(cnode_input, manager, pre_nodes);
  }
  if (IsParallelCareNode(cnode) && cnode->has_user_data<OperatorInfo>()) {
    pre_nodes->push_back(cnode);
  }
}

Shapes GetNodeShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  Shapes shapes;
  if (IsValueNode<ValueList>(node) || IsValueNode<ValueTuple>(node)) {
    return GetValueListShape(node);
  }
  BaseShapePtr base_shape_ptr = node->Shape();
  if (base_shape_ptr == nullptr && node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    MS_EXCEPTION_IF_CHECK_FAIL(value_node->value() != nullptr, "ValueNode has no value.");
    auto abstract = value_node->value()->ToAbstract();
    MS_EXCEPTION_IF_CHECK_FAIL(abstract != nullptr, "ValueNode has no Abstract.");
    node->set_abstract(abstract);
    base_shape_ptr = node->Shape();
  }
  if (node->isa<CNode>() && !IsControlFlowNode(node)) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->input(0)->isa<CNode>()) {
      if (cnode->size() < 2) {
        MS_LOG_WITH_NODE(EXCEPTION, node) << "GetNodeShape: " << node->ToString() << " size is smaller than 2";
      }
      base_shape_ptr = cnode->input(1)->Shape();
    }
  }
  // If node is Depend, only first input should be used.
  if (node->isa<CNode>() && IsPrimitiveCNode(node->cast<CNodePtr>(), prim::kPrimDepend)) {
    auto depend_cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_cnode);
    MS_EXCEPTION_IF_NULL(depend_cnode->input(1));
    return GetNodeShape(depend_cnode->input(1));
  }
  if (base_shape_ptr == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "GetNodeShape: " << node->ToString() << " shape_ptr is nullptr, full name is "
                                      << node->fullname_with_scope();
  }
  MS_LOG(DEBUG) << "For node " << node->DebugString() << ", its base_shape_ptr is " << base_shape_ptr->ToString();
  auto tuple_shape_ptr = dyn_cast<abstract::SequenceShape>(base_shape_ptr);
  if (tuple_shape_ptr != nullptr) {
    if (tuple_shape_ptr->size() == 0) {
      shapes.push_back(Shape{0});
      return shapes;
    }
    auto tuple_shape = tuple_shape_ptr->shape();
    if (tuple_shape[0]->isa<abstract::NoShape>()) {
      shapes.push_back(Shape{SizeToLong(tuple_shape_ptr->size())});
      return shapes;
    }
    for (auto &shape : tuple_shape) {
      auto each_shape = dyn_cast<abstract::Shape>(shape);
      MS_EXCEPTION_IF_NULL(each_shape);
      shapes.push_back(each_shape->shape());
    }
  } else if (base_shape_ptr->isa<abstract::DynamicSequenceShape>()) {
    shapes.push_back(Shape{-1});
  } else if (base_shape_ptr->isa<abstract::Shape>()) {
    auto shape_ptr = dyn_cast<abstract::Shape>(base_shape_ptr);
    MS_EXCEPTION_IF_NULL(shape_ptr);
    shapes.push_back(shape_ptr->shape());
  } else if (base_shape_ptr->isa<abstract::NoShape>()) {
    shapes.push_back(Shape{});
  } else {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "GetNodeShape: " << node->ToString()
                                      << " should be Tuple/List/Tensor/Scalar, but got " << base_shape_ptr->ToString()
                                      << "full name is " << node->fullname_with_scope();
  }
  return shapes;
}

ShapeBasePtr ExtractNewShapeFromShape(const abstract::BaseShapePtr &shape) {
  ShapeBasePtr out_shape;
  if (dyn_cast<abstract::Shape>(shape) != nullptr) {
    auto casted_shape = dyn_cast<abstract::Shape>(shape);
    std::vector<int64_t> shape_value = casted_shape->shape();
    out_shape = std::make_shared<ShapeValue>(shape_value);
  } else if (dyn_cast<abstract::SequenceShape>(shape) != nullptr) {
    std::vector<ShapeBasePtr> tuple_shape;
    auto sequence_shape = dyn_cast<abstract::SequenceShape>(shape);
    std::transform(sequence_shape->shape().begin(), sequence_shape->shape().end(), std::back_inserter(tuple_shape),
                   ExtractNewShapeFromShape);
    out_shape = std::make_shared<ShapeList>(tuple_shape);
  } else {
    MS_LOG(EXCEPTION) << "each shape in tuple shape is not shape or sequenceshape";
  }
  return out_shape;
}

void UseAbstractForValueNode(const AnfNodePtr &node, BaseShapePtr *base_shape_ptr) {
  auto value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_CHECK_FAIL(value_node->value() != nullptr, "ValueNode has no value.");
  auto abstract = value_node->value()->ToAbstract();
  MS_EXCEPTION_IF_CHECK_FAIL(abstract != nullptr, "ValueNode has no Abstract.");
  node->set_abstract(abstract);
  *base_shape_ptr = node->Shape();
}

bool GenerateShapeForTupleShapeNode(const std::shared_ptr<abstract::SequenceShape> &tuple_shape_ptr,
                                    std::vector<ShapeBasePtr> *shapes) {
  if (tuple_shape_ptr->size() == 0) {
    std::vector<int64_t> shape_value = {0};
    shapes->emplace_back(std::make_shared<ShapeValue>(shape_value));
    return true;
  }
  auto tuple_shape = tuple_shape_ptr->shape();
  if (tuple_shape[0]->isa<abstract::NoShape>()) {
    std::vector<int64_t> shape_value = {SizeToLong(tuple_shape_ptr->size())};
    shapes->emplace_back(std::make_shared<ShapeValue>(shape_value));
    return true;
  }
  for (auto &shape : tuple_shape) {
    auto each_shape = ExtractNewShapeFromShape(shape);
    shapes->emplace_back(each_shape);
  }
  return false;
}

NewShapes GetNodeNewShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  NewShapes shapes;
  if (IsValueNode<ValueList>(node) || IsValueNode<ValueTuple>(node)) {
    NewShapes out_new_shapes;
    auto shapes_list = GetValueListShape(node, true);
    std::transform(shapes_list.begin(), shapes_list.end(), std::back_inserter(out_new_shapes),
                   [](const auto &shape) { return std::make_shared<ShapeValue>(shape); });
    return out_new_shapes;
  }
  BaseShapePtr base_shape_ptr = node->Shape();
  if (base_shape_ptr == nullptr && node->isa<ValueNode>()) {
    UseAbstractForValueNode(node, &base_shape_ptr);
  }
  if (node->isa<CNode>() && !IsControlFlowNode(node)) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->input(0)->isa<CNode>()) {
      if (cnode->size() < kSizeTwo) {
        MS_LOG_WITH_NODE(EXCEPTION, node) << "GetNodeShape: " << node->ToString() << " size is smaller than 2";
      }
      base_shape_ptr = cnode->input(1)->Shape();
    }
  }
  // If node is Depend, only first input should be used.
  if (node->isa<CNode>() && IsPrimitiveCNode(node->cast<CNodePtr>(), prim::kPrimDepend)) {
    auto depend_cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_cnode);
    MS_EXCEPTION_IF_NULL(depend_cnode->input(1));
    return GetNodeNewShape(depend_cnode->input(1));
  }
  if (base_shape_ptr == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "GetNodeShape: " << node->ToString() << " shape_ptr is nullptr, full name is "
                                      << node->fullname_with_scope();
  }
  auto tuple_shape_ptr = dyn_cast<abstract::SequenceShape>(base_shape_ptr);
  if (tuple_shape_ptr != nullptr) {
    auto return_flag = GenerateShapeForTupleShapeNode(tuple_shape_ptr, &shapes);
    if (return_flag) {
      return shapes;
    }
  } else if (base_shape_ptr->isa<abstract::DynamicSequenceShape>()) {
    std::vector<int64_t> shape_value = {-1};
    shapes.emplace_back(std::make_shared<ShapeValue>(shape_value));
  } else if (base_shape_ptr->isa<abstract::Shape>()) {
    auto shape_ptr = dyn_cast<abstract::Shape>(base_shape_ptr);
    MS_EXCEPTION_IF_NULL(shape_ptr);
    std::vector<int64_t> shape_value = shape_ptr->shape();
    shapes.emplace_back(std::make_shared<ShapeValue>(shape_value));
  } else if (base_shape_ptr->isa<abstract::NoShape>()) {
    std::vector<int64_t> shape_value = {};
    shapes.emplace_back(std::make_shared<ShapeValue>(shape_value));
  } else {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "GetNodeShape: " << node->ToString()
                                      << " should be Tuple/List/Tensor/Scalar, but got " << base_shape_ptr->ToString()
                                      << "full name is " << node->fullname_with_scope();
  }
  return shapes;
}

RankList FindCommonMirrorGroup(const FuncGraphPtr &root) {
  auto parameters = root->parameters();
  for (auto &parameter : parameters) {
    auto param_ptr = parameter->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (!(param_ptr->has_default() && ParameterRequireGrad(param_ptr))) {
      continue;
    }
    size_t allow_repeat_num = 1;
    if (ParallelContext::GetInstance()->enable_parallel_optimizer() &&
        (!param_ptr->param_info() || param_ptr->param_info()->parallel_optimizer())) {
      if (ParallelContext::GetInstance()->optimizer_weight_shard_size() == -1) {
        MS_LOG(INFO) << "The parameter :" << param_ptr->fullname_with_scope()
                     << " is fully shard by optimizer parallel,"
                        " thus cannot find common data parallel group for this rank";
        return {g_device_manager->global_rank()};
      }
      allow_repeat_num = size_t(ParallelContext::GetInstance()->optimizer_weight_shard_size());
    }
    if (IsFullySplitParameter(param_ptr, allow_repeat_num)) {
      MS_LOG(INFO) << "The parameter :" << param_ptr->fullname_with_scope()
                   << " is fully shard, thus cannot find common data parallel group for this rank";
      return {g_device_manager->global_rank()};
    }
  }
  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<int64_t> common_group_list;
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  bool is_first_group = true;
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!IsPrimitiveCNode(node, prim::kPrimMirror) && !IsPrimitiveCNode(node, prim::kPrimMirrorMicroStep) &&
        !IsPrimitiveCNode(node, prim::kPrimMirrorMiniStep)) {
      continue;
    }
    auto prim = GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(prim);
    if (!prim->HasAttr(GROUP)) {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "The mirror operator does not have group attr : " << node->DebugString();
    }
    std::string group_name = GetValue<std::string>(prim->GetAttr(GROUP));
    std::vector<int64_t> group_list = g_device_manager->FindRankListByHashName(group_name);
    if (is_first_group) {
      common_group_list = group_list;
      is_first_group = false;
    } else {
      std::vector<int64_t> new_comm_group_list;
      (void)std::set_intersection(common_group_list.begin(), common_group_list.end(), group_list.begin(),
                                  group_list.end(), std::back_inserter(new_comm_group_list));
      common_group_list = new_comm_group_list;
    }
  }
  MS_LOG(INFO) << "The common mirror group is:" << common_group_list;
  return common_group_list;
}

std::string CreateInstanceName(const CNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsValueNode<Primitive>(node->input(0))) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "CreateInstanceName: " << node->ToString() << " doesn't have primitive";
  }
  std::string name_base = node->fullname_with_scope();
  std::string name = name_base + "_" + std::to_string(index);
  std::string instance_name = HashInstanceName(name);
  return instance_name;
}

void SetCommunicationOpGroupLabel(std::vector<AnfNodePtr> new_node_input) {
  if (new_node_input.empty()) {
    return;
  }

  auto prim_anf_node = new_node_input[0]->cast<ValueNodePtr>();
  auto prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  MS_EXCEPTION_IF_NULL(prim);

  auto attrs = prim->attrs();
  auto iter = attrs.find(GROUP);
  if (iter != attrs.end()) {
    auto value = iter->second;
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<StringImm>()) {
      std::string hash_name = value->cast<StringImmPtr>()->value();
      MS_EXCEPTION_IF_NULL(g_device_manager);
      std::string rank_list_name = g_device_manager->FindRankListNameByHashName(hash_name);
      (void)prim->AddAttr(GROUP_RANKS, MakeValue(rank_list_name));
    }
  }
}

void SetStridedSliceSplitStrategy(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!IsPrimitiveCNode(cnode, prim::kPrimStridedSlice)) {
      continue;
    }
    auto slice_prim = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(slice_prim);
    if (slice_prim->HasAttr(FUNC_GRAPH_FLAG_STRIDED_SLICE)) {
      SetStridedSliceStrategy(cnode);
    }
  }
}

// Check the given tensor, return nullptr if the given type is not an TensorType
bool CheckTensorType(const TypePtr &node_type) {
  MS_EXCEPTION_IF_NULL(node_type);
  if (!node_type->isa<mindspore::TensorType>()) {
    return false;
  }
  return true;
}

void FindReturnUser(const CNodePtr &cnode, const std::vector<AnfNodePtr> &all_nodes,
                    std::pair<std::shared_ptr<AnfNode>, int> *queue_node) {
  auto graph = cnode->func_graph();
  auto is_target = [&graph](const AnfNodePtr &ele) {
    if (ele->isa<CNode>()) {
      auto parent_cnode = ele->cast<CNodePtr>();
      return IsValueNode<FuncGraph>(parent_cnode->input(0)) &&
             GetValueNode<FuncGraphPtr>(parent_cnode->input(0)) == graph;
    }
    return false;
  };
  auto it = std::find_if(all_nodes.begin(), all_nodes.end(), is_target);
  if (it == all_nodes.end()) {
    return;
  }
  *queue_node = {*it, 0};
}

void AddVisitedNode(std::queue<std::pair<std::shared_ptr<AnfNode>, int>> *visited, const NodeUsersMap &node_users_map,
                    const AnfNodePtr &key_node) {
  if (IsPrimitiveCNode(key_node, prim::kPrimReturn)) {
    return;
  }
  auto node_users = node_users_map.at(key_node);
  for (auto &node_user : node_users) {
    MS_EXCEPTION_IF_NULL(node_user.first);
    auto cnode = node_user.first->cast<CNodePtr>();
    if (!cnode || IsSomePrimitiveList(cnode, {MAKE_TUPLE, UPDATESTATE})) {
      continue;
    }
    if (node_user.first) {
      visited->push(node_user);
    }
  }
}

std::pair<std::shared_ptr<AnfNode>, int> BFSParallelCareNode(const AnfNodePtr &node_ptr,
                                                             const NodeUsersMap &node_users_map, const int index,
                                                             const std::vector<AnfNodePtr> &all_nodes) {
  std::queue<std::pair<std::shared_ptr<AnfNode>, int>> visited;
  CNodePtr cnode = nullptr;
  AnfNodePtr node = nullptr;
  if (!node_ptr) {
    return std::make_pair(nullptr, 0);
  }
  AddVisitedNode(&visited, node_users_map, node_ptr);
  while (!visited.empty()) {
    auto queue_node = visited.front();
    visited.pop();
    cnode = queue_node.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsParallelCareNode(cnode) || IsAutoParallelCareNode(cnode)) {
      return queue_node;
    } else if (IsValueNode<FuncGraph>(cnode->input(0))) {
      auto graph = GetValueNode<FuncGraphPtr>(cnode->input(0));
      auto params = graph->parameters();
      auto target_param = params[queue_node.second - 1];
      auto node_set = node_users_map.at(target_param);
      for (auto &node_user : node_set) {
        cnode = node_user.first->cast<CNodePtr>();
        if (IsParallelCareNode(cnode) || IsAutoParallelCareNode(cnode)) {
          return node_user;
        } else if (IsSomePrimitiveList(cnode, {MAKE_TUPLE, UPDATESTATE})) {
          continue;
        }
        visited.push(node_user);
      }
    } else {
      if (IsSomePrimitive(cnode, RETURN)) {
        FindReturnUser(cnode, all_nodes, &queue_node);
      } else if (IsSomePrimitive(cnode, kTupleGetItemOpName)) {
        auto tuple_index = LongToSize(GetValue<int64_t>(GetValueNode(cnode->input(2))));
        if (tuple_index != IntToSize(index - 1)) {
          continue;
        }
      }
      AddVisitedNode(&visited, node_users_map, queue_node.first);
    }
  }
  return std::make_pair(nullptr, 0);
}

// For the weight used by cast and matmul at the same time, like the followings
// weight1->mirror->cast1-> matmul1;
// weight1->add
// we will not insert the cast(FP32->FP16), as it will cause the input of the operator add to be changed to fp16.
AnfNodePtr GetChildCastNode(const AnfNodePtr &node_ptr, const NodeUsersMap &node_users_map) {
  std::queue<std::pair<AnfNodePtr, int>> visited;
  AnfNodePtr queue_node = nullptr;
  CNodePtr cnode = nullptr;
  AnfNodePtr node = nullptr;
  if (!node_ptr) {
    return nullptr;
  }
  auto users = node_users_map.at(node_ptr);
  for (auto &node_user : users) {
    MS_EXCEPTION_IF_NULL(node_user.first);
    cnode = node_user.first->cast<CNodePtr>();
    if (!cnode || !cnode->in_forward_flag()) {
      continue;
    }
    if (node_user.first) {
      visited.push(node_user);
    }
  }
  while (!visited.empty()) {
    const auto &queue_pair = visited.front();
    queue_node = queue_pair.first;
    visited.pop();
    cnode = queue_node->cast<CNodePtr>();
    // MAKE_TUPLE will not appear after the load in the forward graph
    if (cnode == nullptr || IsOneOfPrimitiveCNode(cnode, {prim::kPrimMakeTuple, prim::kPrimUpdateState})) {
      continue;
    } else if (IsInAllGatherNodeList(cnode) || IsSomePrimitiveList(cnode, {LOAD, RESHAPE})) {
      auto node_set = node_users_map.at(queue_node);
      for (auto &node_user : node_set) {
        visited.push(node_user);
      }
    } else if (IsValueNode<FuncGraph>(cnode->input(kIndex0))) {
      const auto &sub_graph = GetValueNode<FuncGraphPtr>(cnode->input(kIndex0));
      MS_EXCEPTION_IF_NULL(sub_graph);
      const auto &parameters = sub_graph->parameters();
      const auto &parameter_sub = parameters[IntToSize(queue_pair.second - 1)];
      node = GetChildCastNode(parameter_sub, node_users_map);
    } else if (!IsSomePrimitive(cnode, CAST)) {
      MS_LOG(INFO) << "The weight's users including the non cast node So "
                   << "will not insert cast for this parameter " << node_ptr->DebugString();
      return nullptr;
    } else if (!node) {
      node = queue_node;
    }
  }
  return node;
}

// Given the cnode ptr, find its users until we find the computation node, then return the type of the
// computation node. This function is used to find the target type for CreateFP16Cast. Only returns the target type if
// it is float16, and the source node is float32. If the situation is not matched, then return the nullptr.
TypePtr FindChildCastWithFP32ToFP16(const std::pair<AnfNodePtr, int> &res, const NodeUsersMap &node_users_map) {
  if (ParallelContext::GetInstance()->pipeline_stage_split_num() <= 1 &&
      ParallelContext::GetInstance()->grad_accumulation_step() <= 1) {
    return nullptr;
  }
  auto cnode_ptr = res.first->cast<CNodePtr>();
  if (!cnode_ptr) {
    return nullptr;
  }
  auto cnode_inputs = cnode_ptr->inputs();
  if (cnode_inputs.size() < TWO_INPUT_SIZE) {
    return nullptr;
  }

  AnfNodePtr node = nullptr;
  if (IsValueNode<FuncGraph>(cnode_ptr->input(kIndex0))) {
    auto graph_sub = GetValueNode<FuncGraphPtr>(cnode_ptr->input(0));
    auto parameters = graph_sub->parameters();
    auto parameter_sub = parameters[IntToSize(res.second - 1)];
    node = GetChildCastNode(parameter_sub, node_users_map);
  } else {
    // As we execute the function IsWeightValidUsed when we start to insert the mirror, so the second parameter
    // is always the parameter.
    auto weight = cnode_inputs[1];
    if (!weight->isa<Parameter>()) {
      return nullptr;
    }
    MS_LOG(INFO) << "Start to search the weight params:" << weight->DebugString();
    node = GetChildCastNode(weight, node_users_map);
  }

  if (!node) {
    return nullptr;
  }
  // get the output dtype of the operator
  auto node_type = node->Type();
  if (!CheckTensorType(node_type)) {
    return nullptr;
  }
  auto input_element_type = node_type->cast<mindspore::TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(input_element_type);
  if (!IsPrimitiveCNode(node)) {
    return nullptr;
  }
  auto cast_input_cnode = node->cast<CNodePtr>()->input(kIndex1)->cast<CNodePtr>();
  if (!cast_input_cnode) {
    return nullptr;
  }
  auto source_node_type = cast_input_cnode->Type();
  if (!CheckTensorType(source_node_type)) {
    return nullptr;
  }
  auto source_element_type = source_node_type->cast<mindspore::TensorTypePtr>()->element();
  MS_EXCEPTION_IF_NULL(source_element_type);
  // We only add cast operation when the source is fp32 type, and the users is fp16 type.
  if ((source_element_type->type_id() == kNumberTypeFloat32 && input_element_type->type_id() == kNumberTypeFloat16) ||
      (source_element_type->type_id() == kNumberTypeFloat32 && input_element_type->type_id() == kNumberTypeBFloat16)) {
    return input_element_type;
  }
  return nullptr;
}

// Create a cast node given the current node and the previous node. The target type of the the cast is from the
// compute_node_type.
// Return the new cast node with pre_node as the inputs.
AnfNodePtr CreateFP16Cast(const CNodePtr &node, const AnfNodePtr &pre_node, const TypePtr &compute_node_type) {
  const char kOpsFunctionModelName[] = "mindspore.ops.functional";
  static py::object cast_prim = python_adapter::GetPyFn(kOpsFunctionModelName, "_cast");
  const auto &adapter = py::cast<PrimitivePyAdapterPtr>(cast_prim);
  MS_EXCEPTION_IF_NULL(adapter);
  MS_EXCEPTION_IF_NULL(compute_node_type);
  auto prim = adapter->attached_primitive();
  if (prim == nullptr) {
    prim = std::make_shared<PrimitivePy>(cast_prim);
  }
  // Insert cast.
  auto type_node = NewValueNode(compute_node_type);
  type_node->set_abstract(compute_node_type->ToAbstract());
  auto new_node = node->func_graph()->NewCNode({NewValueNode(prim), pre_node, type_node});
  new_node->set_abstract(node->abstract());
  new_node->set_scope(node->scope());
  new_node->set_in_forward_flag(true);
  return new_node;
}

void LabelGenMaskMicro(const FuncGraphPtr &root) {
  AnfNodePtr ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimDropoutDoMask)) {
      auto gen_mask_node = RealInputNode(node->cast<CNodePtr>(), 2);
      if (gen_mask_node->isa<CNode>()) {
        gen_mask_node->cast<CNodePtr>()->set_primal_attrs(node->cast<CNodePtr>()->primal_attrs());
      }
    }
  }
}

void SetCastForParamNotRecompute(const std::vector<AnfNodePtr> &all_nodes) {
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto cnode_prim = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(cnode_prim);
    if (cnode_prim->HasAttr("DISABLE_MERGE_ASSIGN_ADD")) {
      cnode->AddPrimalAttr("DISABLE_MERGE_ASSIGN_ADD", cnode_prim->GetAttr("DISABLE_MERGE_ASSIGN_ADD"));
    }
    if (!IsPrimitiveCNode(node, prim::kPrimCast)) {
      continue;
    }
    auto cast_input = RealInputNode(cnode, 1);
    MS_EXCEPTION_IF_NULL(cast_input);
    if (cast_input->isa<Parameter>() && cast_input->cast<ParameterPtr>()->has_default()) {
      MS_LOG(INFO) << "Cast for parameter no needs recompute to avoid redundant trans_data operator";
      PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0)->cast<ValueNodePtr>());
      MS_EXCEPTION_IF_NULL(prim);
      (void)prim->AddAttr("recompute", MakeValue(false));
    }
  }
}

std::shared_ptr<Value> GetAttrsFromAnfNode(const std::shared_ptr<AnfNode> &node, const string &key) {
  if (!node) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  auto prim = GetCNodePrimitive(cnode);
  if (prim && prim->HasAttr(key)) {
    return prim->GetAttr(key);
  }
  return nullptr;
}

bool IsSplittableOperator(const std::string &op_name) {
  // clang-format off
  static const std::set<std::string> splittable_op =
    {MATMUL, TRANSPOSE, GELU, FAST_GELU, TANH, SOFTMAX, SUB, MUL, DIV, RESHAPE, GREATER, LOG_SOFTMAX, ACTIVATION, PRELU,
     BATCH_MATMUL_EXT, MATMUL_EXT, LAYER_NORM_V3, FLOORDIV, L2_NORMALIZE, ADD, MAX, MAXDIM, MAXPOOL, OUTER,
     AVGPOOL, MAXPOOLV2, INDEX_INFO, INDEX_ADD_EXT, VIRTUAL_DATA_SET, RELU, ONEHOT, DROPOUT_DO_MASK,
     REDUCE_MAX, REDUCE_MIN, ARGMAXWITHVALUE, ARGMINWITHVALUE, REDUCE_SUM, SUM_EXT,
     CONV2D, FUSE_BATCH_NORM, POOLING, STACK_EXT, LEAKYRELU_EXT, SOFTPLUS_EXT,
     MAX_POOL_WITH_ARGMAX, SIMPLE_MEAN, FLATTEN, BATCH_NORM, LAYER_NORM, BIAS_ADD, ASSIGN_SUB, COS, ACOS, EXP, STACK,
     LOG, REDUCE_MEAN, REAL_DIV, SIGMOID, POW, MAXIMUM, MINIMUM, EQUAL, NOT_EQUAL, LOGICALNOT, GATHERV2, SQRT, CONCAT,
     STRIDEDSLICE, GET_NEXT, CAST, NEG, SQUARE, BATCH_MATMUL, EXPAND_DIMS, SQUEEZE, SPARSE_GATHERV2, TILE, DROPOUT,
     SOFTMAX_CROSS_ENTROPY_WITH_LOGITS, SIGMOID_CROSS_ENTROPY_WITH_LOGITS, SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS,
     EMBEDDING_LOOKUP, FUSE_BATCH_NORM_EX, SPLIT, BROADCAST_TO, ABS, ACOSH, ASIN, ASINH, ATAN, ATANH, CEIL, COSH,
     EXPM1, LOG1P, SIN, SINH, TAN, RSQRT, INV, RECIPROCAL, ROUND, FLOOR, SIGN, ERF, ERFC, ZEROSLIKE, ONESLIKE,
     BESSELI0E, BESSELI1E, FLOORMOD, ASSIGN, ASSIGN_ADD, ATAN2, DIVNONAN, LOGICALAND, LOGICALOR, ELU, ELU_EXT,
     RELU6, SOFTPLUS, SOFTSIGN, GREATEREQUAL, LESSEQUAL, LESS, APPROXIMATEEQUAL, MOD, UNIQUE, UNSORTED_SEGMENT_SUM,
     UNSORTED_SEGMENT_MIN, REPEAT_ELEMENTS, TENSOR_DOT, RANGE, UNIFORM_CANDIDATE_SAMPLER, SLICE, SLICE_EXT, SELECT,
     GATHERD, UNSORTED_SEGMENT_MAX, GATHER_ND, TOPK, SCATTER_UPDATE, SCATTER_ND_UPDATE, SCATTER_ND_ADD, SCATTER_ND_SUB,
     TENSOR_SCATTER_UPDATE, TENSOR_SCATTER_ADD, TENSOR_SCATTER_SUB, TENSOR_SCATTER_MAX, TENSOR_SCATTER_MIN, WKV,
     TENSOR_SCATTER_MUL, TENSOR_SCATTER_DIV, VIRTUAL_OUTPUT, CONV2D_BACK_PROP_INPUT, CONV2D_TRANSPOSE, SORT, SORT_EXT,
     MATMUL_DDS, DSD_MATMUL, UNIFORMREAL, STANDARD_NORMAL, RESIZE_BILINEAR_V2, RESIZE_NEAREST_NEIGHBOR, FAST_GELU, IOU,
     BOUNDING_BOX_ENCODE, UNSORTED_SEGMENT_PROD, SQUARE_SUM_ALL, UNIQUE_CONSECUTIVE, SILU, INDEX_SELECT, CLAMP_SCALAR,
     RANDOM_CHOICE_WITH_MASK, CROP_AND_RESIZE, ROI_ALIGN, REDUCE_PROD, REDUCE_ANY, REDUCE_ALL, ARGMAX, ARGMIN, ARGMINV2,
     RESIZE_NEAREST_NEIGHBOR, CUM_SUM, FAST_GELU, IOU, BOUNDING_BOX_ENCODE, RANDOM_CHOICE_WITH_MASK, CROP_AND_RESIZE,
     ROI_ALIGN, IS_FINITE, RINT, HSHRINK, HSIGMOID, MISH, SELU, SOFT_SHRINK, XLOGY, XDIVY, CUM_PROD, BITWISE_AND,
     BITWISE_OR, BITWISE_XOR, MUL_NO_NAN, TRUNCATE_DIV, TRUNCATE_MOD, INPLACE_ADD, INPLACE_SUB, INPLACE_UPDATE, PAD_V3,
     L2_LOSS, LERP, ADDN, CDIST, SQUARED_DIFFERENCE, ERFINV, MASKED_FILL, SPLITV, GAMMA, KLDIV_LOSS, LIN_SPACE,
     CHECK_VALID, INVERT, SCATTER_ADD, SCATTER_DIV, SCATTER_MUL, SCATTER_MAX, SCATTER_MIN, SCATTER_SUB, UNIQUE_WITH_PAD,
     POPULATION_COUNT, IDENTITY, BESSELI0, BESSELI1, BESSELJ0, BESSELJ1, CUM_MAX, CUM_MIN, HYPOT, IGAMMA, IGAMMAC,
     LEFT_SHIFT, RIGHT_SHIFT, NEXT_AFTER, ZETA, REVERSEV2, LGAMMA, TRUNC, BETAINC, GCD, CHOLESKY, CONV3D, MAXPOOL_3D,
     AVGPOOL_3D, FILLV2, FAKE_QUANT_PER_LAYER, FAKE_QUANT_PER_CHANNEL, MIN_MAX_UPDATE_PER_LAYER, ASCEND_QUANTV2,
     MIN_MAX_UPDATE_PER_CHANNEL, FFN, FLASH_ATTENTION_SCORE, ASCEND_QUANT, ASCEND_DEQUANT, GRID_SAMPLER_2D, ANTI_QUANT,
     CONVOLUTION, LIN_SPACE_EXT, ONEHOTEXT, FFTSHIFT, IFFTSHIFT, FFT, IFFT, FFT2, IFFT2, FFTN, IFFTN, CUM_SUM_EXT,
     RFFT, IRFFT, RFFT2, IRFFT2, RFFTN, IRFFTN, HFFT, IHFFT, HFFT2, IHFFT2, HFFTN, IHFFTN, DCT, IDCT, DCTN, IDCTN,
     NANTONUM, ZEROS, ISCLOSE, POLAR, REMAINDER_TENSOR_TENSOR, REMAINDER_TENSOR_SCALAR, REMAINDER_SCALAR_TENSOR,
     SOLVE_TRIANGULAR, TRACEV2, LSTSQV2, GENERATEEODMASKV2, MEANEXT, REPEAT_INTERLEAVE_INT, REPEAT_INTERLEAVE_TENSOR,
     FMOD_TENSOR, INPLACE_COPY};
  // clang-format on

  auto iter = splittable_op.find(op_name);
  return (iter != splittable_op.end());
}

bool IsAutoParallelCareNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  ValueNodePtr prim_node = cnode->input(0)->cast<ValueNodePtr>();
  if (prim_node == nullptr) {
    return false;
  }
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_node);
  if (prim == nullptr) {
    return false;
  }
  if (IsSomePrimitiveList(cnode, {SEND, RECEIVE, MAKE_TUPLE, MAKE_LIST})) {
    return false;
  }
  bool bool_result = IsParallelCareNode(cnode) && !IsSplittableOperator(prim->name());
  if (bool_result) {
    MS_LOG(INFO) << "For 'auto_parallel', missing the splitable implementation of OperatorInfo for: " << prim->name()
                 << ", default strategy will be assigned. Network training may deteriorate or malfunction";
  } else if (prim->name() == CAST) {
    if (cnode->fullname_with_scope().find(OPTIMIZER_SUB_STRING) != std::string::npos) {
      // Do not care CASTs from optimizer
      return false;
    }
    return cnode->in_forward_flag();
  }
  return IsParallelCareNode(cnode);
}

void UpdateMicroBatchInterleavedStatus(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!IsPrimitiveCNode(cnode, prim::kPrimStridedSlice)) {
      continue;
    }
    auto slice_prim = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(slice_prim);
    if (!slice_prim->HasAttr(FUNC_GRAPH_FLAG_STRIDED_SLICE)) {
      continue;
    }
    if (!slice_prim->HasAttr(INTERLEAVED_NUM)) {
      continue;
    }
    if (GetValue<int64_t>(slice_prim->GetAttr(INTERLEAVED_NUM)) == MICRO_INTERLEAVED_SIZE) {
      ParallelContext::GetInstance()->set_enable_micro_interleaved(true);
      cnode->AddAttr(INTERLEAVED_NUM, slice_prim->GetAttr(INTERLEAVED_NUM));
    }
  }
}

std::string GetDisOpName(const std::string &prim_name) {
  std::string op_name = prim_name;
  if (!prim_name.empty() && (prim_name[0] == '_')) {
    op_name = prim_name.substr(1);
  }
  return op_name + "Info";
}

OperatorInfoPtr OperatorInstanceByName(const std::string &name, const PrimitiveAttrs &attrs,
                                       const std::vector<Shapes> &shape_list) {
  if (shape_list.size() != 2) {
    MS_LOG(ERROR) << "The size of shape list is not 2";
    return nullptr;
  }
  if (name.length() == 0) {
    MS_LOG(EXCEPTION) << "Length of name is zero!";
  }

  std::string distribute_opname = GetDisOpName(name);
  OperatorInfoPtr op_info =
    (OperatorInfoPtr)DynCreator::Instance().Create(distribute_opname, shape_list[0], shape_list[1], attrs, TOTAL_OPS);
  if (op_info == nullptr) {
    MS_LOG(INFO) << "Create " << name << " failed";
    return nullptr;
  }
  std::string origin_name = op_info->name();
  op_info->set_name(origin_name + std::to_string(TOTAL_OPS));
  if (origin_name.find("VirtualDataset") != std::string::npos) {
    op_info->set_topo_index(TOTAL_OPS_MAX);
  } else {
    op_info->set_topo_index(TOTAL_OPS);
  }
  MS_LOG(INFO) << "Successfully created operator " << origin_name;
  ++TOTAL_OPS;
  return op_info;
}

OperatorInfoPtr OperatorInstance(const PrimitivePtr &prim, const PrimitiveAttrs &attrs,
                                 const std::vector<Shapes> &shape_list) {
  MS_EXCEPTION_IF_NULL(prim);
  OperatorInfoPtr op_info;
  if (prim->HasAttr(SELF_DEFINE_SHARD)) {
    auto self_define_shard_attr = prim->GetAttr(SELF_DEFINE_SHARD);
    MS_EXCEPTION_IF_NULL(self_define_shard_attr);
    if (self_define_shard_attr->cast_ptr<BoolImm>() == nullptr) {
      MS_LOG(EXCEPTION) << "SELF_DEFINE_SHARD attribute is not a bool";
    }
    if (GetValue<bool>(self_define_shard_attr)) {
      op_info = OperatorInstanceByName(SELF_DEFINE_SHARD_OP, attrs, shape_list);
      MS_LOG(INFO) << "Operator " << prim->name() << " has self_define_shard attribute. Create SelfDefineShardInfo";
      return op_info;
    }
  }
  op_info = OperatorInstanceByName(prim->name(), attrs, shape_list);
  if (op_info) {
    return op_info;
  }
  if (IsInBatchParallelBlackList(prim)) {
    op_info = OperatorInstanceByName(STAND_ALONE, attrs, shape_list);
    prim->AddAttr(STAND_ALONE, MakeValue<bool>(true));
    MS_LOG(INFO) << "Operator " << prim->name() << " is not supported yet in auto parallel mode. Use Stand Alone";
    return op_info;
  }
  auto input_shape = shape_list[0];
  auto output_shape = shape_list[1];
  MS_EXCEPTION_IF_NULL(g_device_manager);
  auto device_num = g_device_manager->stage_device_num();
  MS_EXCEPTION_IF_ZERO("device_num", device_num);
  if (input_shape.empty() || input_shape[0].empty() || input_shape[0][0] % device_num != 0 || output_shape[0].empty() ||
      output_shape[0][0] % device_num != 0) {
    MS_LOG(INFO) << "Operator " << prim->name() << " use Stand Alone, the input shape is " << input_shape
                 << ", the output shape is " << output_shape;
    op_info = OperatorInstanceByName(STAND_ALONE, attrs, shape_list);
    prim->AddAttr(STAND_ALONE, MakeValue<bool>(true));
    return op_info;
  }
  MS_LOG(INFO) << "Operator " << prim->name() << " use Batch Parallel";
  op_info = OperatorInstanceByName(BATCH_PARALLEL, attrs, shape_list);
  prim->AddAttr(BATCH_PARALLEL, MakeValue<bool>(true));
  return op_info;
}

static Shapes GetRefKeyNodeShape(const AnfNodePtr &node, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);

  std::vector<AnfNodePtr> parameters = FindParameterByRefKeyNode(node, func_graph);
  if (parameters.size() != 1) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Find parameter by ref key node failed";
  }

  Shapes input_shapes = GetNodeShape(parameters[0]);
  if (input_shapes.size() != 1) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Get input shape failed";
  }

  MS_LOG(INFO) << "The parameter shape is " << ShapeToString(input_shapes[0]);
  return input_shapes;
}

void ObtainRealShape(const AnfNodePtr &input, const NewShapes &local_new_shapes, ShapeBasePtr *input_new_shapes) {
  if ((input->abstract()->isa<abstract::AbstractSequence>() || IsValueSequence(input))) {
    *input_new_shapes = std::make_shared<ShapeList>(local_new_shapes);
  } else {
    *input_new_shapes = local_new_shapes[0];
  }
}

std::pair<std::vector<NewShapes>, std::vector<Symbols>> ExtractNewShapeAndSymbol(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  NewShapes shape_inputs;
  NewShapes shape_outputs;
  Symbols symbol_inputs;
  Symbols symbol_outputs;
  std::vector<NewShapes> shape_all;
  std::vector<Symbols> symbol_all;
  std::vector<AnfNodePtr> all_inputs = node->inputs();
  const int min_size = 2;
  size_t inputs_size = all_inputs.size();
  for (size_t i = 1; i < inputs_size; ++i) {
    ShapeBasePtr input_new_shapes;
    Shapes input_shapes;
    Symbols input_symbols;
    AnfNodePtr input = all_inputs[i];
    if (input->Shape() == nullptr) {
      MS_LOG(DEBUG) << "The " << i << "th input shape of " << node->DebugString() << " is null";
      continue;
    }
    MS_LOG(DEBUG) << "The " << i << "th input shape of " << node->DebugString() << " is " << input->Shape()->ToString();
    if (HasAbstractMonad(input) || IsPrimitiveCNode(input, prim::kPrimTensorToScalar)) {
      continue;
    }
    if (IsValueNode<RefKey>(input)) {
      auto func_graph = node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      std::vector<AnfNodePtr> parameters = FindParameterByRefKeyNode(input, func_graph);
      if (parameters.size() != 1) {
        MS_LOG_WITH_NODE(EXCEPTION, input) << "Find parameter by ref key node failed";
      }
      std::pair<AnfNodePtr, int64_t> node_pair = std::make_pair(node, SizeToLong(i));
      g_RefMap[parameters[0]] = node_pair;
      MS_LOG(INFO) << "Find parameter by ref key node" << node_pair.first;
      input_shapes = GetRefKeyNodeShape(input, func_graph);
      input_new_shapes = std::make_shared<ShapeValue>(input_shapes[0]);
      input_symbols = StaticShapesToSymbols(input_shapes);  // now the parameter can only be static shape
    } else if (input->isa<CNode>() || IsValueNode<Tensor>(input) || input->isa<Parameter>() ||
               (IsValueSequence(input) &&
                (inputs_size == min_size || IsSomePrimitiveList(node, INPUT_IS_TUPLE_OR_LIST_OPS)))) {
      if (IsDynamicShapeInput(node, input)) {
        MS_LOG(INFO) << "may be dynamic shape, no need to get input's shape, the node is " << node->ToString();
        continue;
      }
      auto anode = IsPrimitiveCNode(input, prim::kPrimShape) ? input->cast<CNodePtr>()->input(1) : input;
      const auto &local_new_shapes = GetNodeNewShape(anode);
      input_symbols = GetNodeSymbol(anode);
      ObtainRealShape(input, local_new_shapes, &input_new_shapes);
    } else if (IsValueSequence(input)) {
      // Not in INPUT_IS_TUPLE_OR_LIST_OPS but has tuple input, like virtual data set
      const auto &local_new_shapes = GetNodeNewShape(input);
      input_symbols = GetNodeSymbol(input);
      ObtainRealShape(input, local_new_shapes, &input_new_shapes);
    } else if (input->isa<ValueNode>() && input->Shape()->isa<abstract::NoShape>()) {
      std::vector<int64_t> shape_value = {};
      input_new_shapes = std::make_shared<ShapeValue>(shape_value);
      input_symbols = StaticShapesToSymbols({shape_value});
    } else {
      MS_LOG(INFO) << "The " << i << "th input shape of " << node->DebugString()
                   << " does not transfer to new shape base";
      continue;
    }
    MS_LOG(DEBUG) << "The " << i << "th input shape of " << node->DebugString() << " is "
                  << input_new_shapes->ToString();
    // For normal shape
    shape_inputs.emplace_back(input_new_shapes);
    symbol_inputs.push_back(input_symbols[0]);
  }
  shape_all.push_back(shape_inputs);
  symbol_all.push_back(symbol_inputs);
  // extract out shape
  shape_outputs = GetNodeNewShape(node);
  symbol_outputs = GetNodeSymbol(node);
  shape_all.push_back(shape_outputs);
  symbol_all.push_back(symbol_outputs);

  return std::make_pair(shape_all, symbol_all);
}

std::pair<std::vector<Shapes>, std::vector<Symbols>> ExtractShapeAndSymbol(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  Shapes shape_inputs;
  Shapes shape_outputs;
  Symbols symbol_inputs;
  Symbols symbol_outputs;
  std::vector<Shapes> shape_all;
  std::vector<Symbols> symbol_all;
  std::vector<AnfNodePtr> all_inputs = node->inputs();

  const int min_size = 2;
  size_t inputs_size = all_inputs.size();
  for (size_t i = 1; i < inputs_size; ++i) {
    MS_LOG(DEBUG) << "extract " << i << "th shape for node " << node->DebugString();
    Shapes input_shapes;
    Symbols input_symbols;
    AnfNodePtr input = all_inputs[i];
    if (HasAbstractMonad(input) || IsPrimitiveCNode(input, prim::kPrimTensorToScalar)) {
      continue;
    }
    if (IsValueNode<RefKey>(input)) {
      auto func_graph = node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      std::vector<AnfNodePtr> parameters = FindParameterByRefKeyNode(input, func_graph);
      if (parameters.size() != 1) {
        MS_LOG_WITH_NODE(EXCEPTION, input) << "Find parameter by ref key node failed";
      }
      std::pair<AnfNodePtr, int64_t> node_pair = std::make_pair(node, SizeToLong(i));
      g_RefMap[parameters[0]] = node_pair;
      MS_LOG(INFO) << "Find parameter by ref key node" << node_pair.first;
      input_shapes = GetRefKeyNodeShape(input, func_graph);
      input_symbols = StaticShapesToSymbols(input_shapes);  // now the parameter can only be static shape
    } else if (input->isa<CNode>() || IsValueNode<Tensor>(input) || input->isa<Parameter>() ||
               (IsValueSequence(input) &&
                (inputs_size == min_size || IsSomePrimitiveList(node, INPUT_IS_TUPLE_OR_LIST_OPS)))) {
      if (IsDynamicShapeInput(node, input)) {
        MS_LOG(INFO) << "may be dynamic shape, no need to get input's shape, the node is " << node->ToString();
        continue;
      }

      if (IsPrimitiveCNode(input, prim::kPrimShape)) {
        input_shapes = GetNodeShape(input->cast<CNodePtr>()->input(1));
        input_symbols = GetNodeSymbol(input->cast<CNodePtr>()->input(1));
      } else {
        input_shapes = GetNodeShape(input);
        input_symbols = GetNodeSymbol(input);
      }
    } else {
      MS_LOG(DEBUG) << "Shape is not extracted";
      continue;
    }
    if (input_shapes.size() != 1) {
      if (inputs_size == min_size || IsSomePrimitiveList(node, INPUT_IS_TUPLE_OR_LIST_OPS)) {
        shape_inputs = input_shapes;
        symbol_inputs = input_symbols;
        break;
      } else {
        MS_LOG_WITH_NODE(EXCEPTION, input) << "ExtractShape: Get input shape failed";
      }
    }
    MS_LOG(DEBUG) << "Shape is " << input_shapes[0];
    shape_inputs.push_back(input_shapes[0]);
    symbol_inputs.push_back(input_symbols[0]);
  }
  shape_all.push_back(shape_inputs);
  symbol_all.push_back(symbol_inputs);
  // extract out shape
  shape_outputs = GetNodeShape(node);
  symbol_outputs = GetNodeSymbol(node);
  shape_all.push_back(shape_outputs);
  symbol_all.push_back(symbol_outputs);

  return std::make_pair(shape_all, symbol_all);
}

std::vector<Shapes> ExtractShape(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto shapes_and_symbols = ExtractShapeAndSymbol(node);
  return shapes_and_symbols.first;
}

std::vector<NewShapes> ExtractNewShape(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto shapes_and_symbols = ExtractNewShapeAndSymbol(node);
  return shapes_and_symbols.first;
}

std::vector<Shapes> ExtractRealDivisor(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto shapes_and_symbols = ExtractShapeAndSymbol(node);
  std::vector<Shapes> shapes = shapes_and_symbols.first;
  std::vector<Symbols> symbols = shapes_and_symbols.second;
  if (shapes.size() != INPUT_OUTPUT_SYMBOLS_SIZE || symbols.size() != INPUT_OUTPUT_SYMBOLS_SIZE) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "the size of shapes or symbols must be " << INPUT_OUTPUT_SYMBOLS_SIZE
                                      << ", but the size of shapes is " << shapes.size() << ", the size of symbols is "
                                      << symbols.size();
  }

  auto inputs_shape = shapes[0];
  auto outputs_shape = shapes[1];
  auto inputs_symbol = symbols[0];
  auto outputs_symbol = symbols[1];

  Shapes in_divisor_symbols;
  Shapes out_divisor_symbols;
  MS_LOG(DEBUG) << "the node is " << node->ToString() << ", the divisor of inputs is "
                << DivisorOfSymbolsToString(inputs_symbol) << ", the inputs shape is " << ShapesToString(inputs_shape);
  in_divisor_symbols = GetRealDivisorSymbols(inputs_shape, inputs_symbol);
  out_divisor_symbols = GetRealDivisorSymbols(outputs_shape, outputs_symbol);

  MS_LOG(DEBUG) << "the node is " << node->ToString() << ", the inputs shape is " << ShapesToString(inputs_shape)
                << ", the inputs divisor is " << ShapesToString(in_divisor_symbols);
  MS_LOG(DEBUG) << "the node is " << node->ToString() << ", the outputs shape is " << ShapesToString(outputs_shape)
                << ", the outputs divisor is " << ShapesToString(out_divisor_symbols);
  return {in_divisor_symbols, out_divisor_symbols};
}

AnfNodePtr GetInputNodeWithFilter(const AnfNodePtr &node,
                                  std::function<std::pair<bool, size_t>(const CNodePtr &)> filter) {
  std::queue<AnfNodePtr> anf_queue;
  anf_queue.push(node);
  while (!anf_queue.empty()) {
    auto queue_end = anf_queue.front();
    anf_queue.pop();
    if (!queue_end->isa<CNode>()) {
      return queue_end;
    }
    auto cnode_queue_end = queue_end->cast<CNodePtr>();
    auto filter_res = filter(cnode_queue_end);
    if (!filter_res.first) {
      return queue_end;
    }
    anf_queue.push(cnode_queue_end->input(filter_res.second));
  }
  return node;
}

std::vector<std::pair<AnfNodePtr, int>> GetOutputNodesWithFilter(const AnfNodePtr &node,
                                                                 std::function<bool(const AnfNodePtr &)> filter,
                                                                 bool get_all_nodes_according_search) {
  MS_EXCEPTION_IF_NULL(node);
  auto func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<std::pair<AnfNodePtr, int>> res;
  std::queue<AnfNodePtr> anf_queue;
  anf_queue.push(node);
  while (!anf_queue.empty()) {
    auto queue_end = anf_queue.front();
    anf_queue.pop();
    auto user_set = manager->node_users()[queue_end];
    for (auto &pair : user_set) {
      if (filter(pair.first)) {
        anf_queue.push(pair.first);
        if (get_all_nodes_according_search) {
          res.push_back(pair);
        }
        continue;
      }
      res.push_back(pair);
    }
  }
  return res;
}

std::vector<std::pair<AnfNodePtr, int>> GetOutputNodesSkipDepend(const AnfNodePtr &node) {
  auto func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<std::pair<AnfNodePtr, int>> res;
  std::queue<AnfNodePtr> anf_queue;
  anf_queue.push(node);
  while (!anf_queue.empty()) {
    auto queue_end = anf_queue.front();
    anf_queue.pop();
    auto user_set = manager->node_users()[queue_end];
    for (auto &pair : user_set) {
      if (IsPrimitiveCNode(pair.first, prim::kPrimDepend)) {
        if (pair.second == 1) {
          anf_queue.push(pair.first);
        }
        continue;
      }
      res.push_back(pair);
    }
  }
  return res;
}

std::pair<bool, size_t> CanMergeConcatSlice(const std::pair<std::shared_ptr<AnfNode>, int> &pair,
                                            const CNodePtr &concat_cnode,
                                            const ShapeVector &concat_output_shape_element, int64_t concat_axis) {
  if (!IsPrimitiveCNode(pair.first, prim::kPrimStridedSlice)) {
    return {false, 0};
  }
  auto slice_cnode = pair.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(slice_cnode);
  MS_LOG(INFO) << "concat slice cnode:" << slice_cnode->fullname_with_scope();
  auto begin_value = GetValueNode(slice_cnode->input(2));
  auto end_value = GetValueNode(slice_cnode->input(3));
  auto strided_value = GetValueNode(slice_cnode->input(4));
  if (!begin_value || !end_value || !strided_value) {
    return {false, 0};
  }
  auto begin = GetValue<std::vector<int64_t>>(begin_value);
  auto end = GetValue<std::vector<int64_t>>(end_value);
  auto strided = GetValue<std::vector<int64_t>>(strided_value);
  if (!std::all_of(strided.begin(), strided.end(), [](auto s) { return s == 1; })) {
    return {false, 0};
  }
  MS_EXCEPTION_IF_NULL(concat_cnode);
  MS_EXCEPTION_IF_NULL(concat_cnode->input(1));
  if (!IsPrimitiveCNode(concat_cnode->input(1), prim::kPrimMakeTuple)) {
    return {false, 0};
  }
  auto concat_input_node = concat_cnode->input(1)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(concat_input_node);
  auto concat_input_size = concat_input_node->size();
  bool can_merge = false;
  size_t concat_input_index = 0;
  for (size_t i = 0; i < begin.size(); ++i) {
    int64_t slice_len = (end[i] - begin[i]);
    if (i == size_t(concat_axis)) {
      int64_t slice_index = begin[i] / slice_len;
      if (slice_len == concat_output_shape_element[i] || size_t(slice_index + 1) >= concat_input_size) {
        can_merge = false;
        break;
      }
      concat_input_index = size_t(slice_index + 1);
      can_merge = true;
    } else if (slice_len != concat_output_shape_element[i]) {
      can_merge = false;
      break;
    }
  }
  return {can_merge, concat_input_index};
}

void UpdateUpdateStateForMergeConcatSlice(const FuncGraphManagerPtr &manager,
                                          const std::vector<std::pair<AnfNodePtr, int>> &update_list,
                                          const CNodePtr &tuple_get_item_node) {
  for (const auto &ups_pair : update_list) {
    (void)manager->SetEdge(ups_pair.first, ups_pair.second, tuple_get_item_node);
  }
}

std::pair<AnfNodePtrList, AnfNodePtrList> GetParamsAndInputs(const std::pair<AnfNodePtr, int> &fg_users,
                                                             const FuncGraphPtr &user_func_graph) {
  //   call -> tuplegetitem -> call
  auto user_graph_parameters = user_func_graph->parameters();
  auto new_user_graph_parameters(user_graph_parameters);
  new_user_graph_parameters.erase(new_user_graph_parameters.begin() + fg_users.second - 1);
  auto fg_users_inputs_all(fg_users.first->cast<CNodePtr>()->inputs());
  fg_users_inputs_all.erase(fg_users_inputs_all.begin() + fg_users.second);
  return std::make_pair(new_user_graph_parameters, fg_users_inputs_all);
}

struct ConcatTransformContext {
  std::vector<AnfNodePtr> new_concat_maketuple_inputs;
  std::vector<AbstractBasePtr> new_maketuple_abstracts;
  AnfNodePtrList new_user_graph_parameters;
  std::vector<std::pair<bool, size_t>> input_index;
  std::vector<std::pair<AnfNodePtr, int>> func_node_users;
  std::vector<std::pair<AnfNodePtr, int>> update_list;
  std::vector<AnfNodePtr> fg_users_inputs_all;
  FuncGraphPtr user_func_graph;
};

std::optional<std::pair<AnfNodePtr, int>> HandleMerge(const CNodePtr &call_cnode, const FuncGraphManagerPtr &manager,
                                                      const CNodePtr &concat_cnode,
                                                      const ShapeVector &concat_output_shape_element,
                                                      int64_t concat_axis, ConcatTransformContext *ctx) {
  auto func_users = manager->node_users()[call_cnode];
  size_t func_users_size = 0;
  std::pair<AnfNodePtr, int> fg_users;

  for (const auto &cur_fg_users : func_users) {
    if (IsPrimitiveCNode(cur_fg_users.first, prim::kPrimUpdateState)) {
      ctx->update_list.push_back(cur_fg_users);
      continue;
    }
    ++func_users_size;
    fg_users = cur_fg_users;
  }

  if (func_users_size > 1) {
    return std::nullopt;
  }

  ctx->func_node_users = FuncGraphNodeUsers(fg_users);
  if (ctx->func_node_users.empty()) {
    return std::nullopt;
  }

  bool have_can_merge = false;
  for (const auto &new_pair : ctx->func_node_users) {
    auto can_merge = CanMergeConcatSlice(new_pair, concat_cnode, concat_output_shape_element, concat_axis);
    ctx->input_index.push_back(can_merge);
    if (can_merge.first) {
      have_can_merge = true;
    }
  }

  if (!have_can_merge) {
    return std::nullopt;
  }

  return fg_users;
}

void HandleTupleReturn(const CNodePtr &call_cnode, const CNodePtr &concat_cnode, const std::pair<AnfNodePtr, int> &pair,
                       const std::pair<AnfNodePtr, int> &fg_users, const FuncGraphManagerPtr &manager,
                       ConcatTransformContext *ctx) {
  // maketuple->Return
  auto concat_input_node = concat_cnode->input(1)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(concat_input_node);
  (void)manager->SetEdge(pair.first, pair.second, concat_input_node);
  MS_EXCEPTION_IF_NULL(fg_users.first->cast<CNodePtr>());
  ctx->user_func_graph = GetValueNode<FuncGraphPtr>(fg_users.first->cast<CNodePtr>()->input(0));
  MS_EXCEPTION_IF_NULL(ctx->user_func_graph);
  std::tie(ctx->new_user_graph_parameters, ctx->fg_users_inputs_all) =
    GetParamsAndInputs(fg_users, ctx->user_func_graph);

  // New concat CNode in user_func_graph
  ctx->new_concat_maketuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  std::vector<AbstractBasePtr> new_maketuple_abstracts;
  bool updated_update_state = false;
  for (size_t i = 0; i < concat_input_node->size() - 1; ++i) {
    std::vector<AnfNodePtr> tuple_get_item_inputs{NewValueNode(prim::kPrimTupleGetItem), call_cnode,
                                                  ValuePtrToAnfNodePtr(MakeValue<int64_t>(i))};
    auto tuple_get_item_node = call_cnode->func_graph()->NewCNode(tuple_get_item_inputs);
    if (!updated_update_state) {
      UpdateUpdateStateForMergeConcatSlice(manager, ctx->update_list, tuple_get_item_node);
      updated_update_state = true;
    }

    // replace fg_users->inputs(fg_users.second) to a list fg_users->inputs(fg_users.second+i)
    ctx->fg_users_inputs_all.insert(ctx->fg_users_inputs_all.begin() + fg_users.second + i, tuple_get_item_node);
    auto new_parameter = ctx->user_func_graph->add_parameter();
    auto cloned_abstract = concat_input_node->input(i + 1)->abstract()->Clone();
    new_parameter->set_abstract(cloned_abstract);
    new_maketuple_abstracts.push_back(cloned_abstract);
    tuple_get_item_node->set_abstract(cloned_abstract);
    ctx->new_user_graph_parameters.insert(ctx->new_user_graph_parameters.begin() + fg_users.second - 1 + i,
                                          new_parameter);
    ctx->new_concat_maketuple_inputs.push_back(new_parameter);
  }
  ctx->new_maketuple_abstracts = new_maketuple_abstracts;
}

void UpdateUserGraphAndReplaceCall(const std::pair<AnfNodePtr, int> &fg_users, const FuncGraphManagerPtr &manager,
                                   const CNodePtr &concat_cnode, int64_t concat_axis, ConcatTransformContext *ctx) {
  ctx->user_func_graph->set_parameters(ctx->new_user_graph_parameters);
  auto user_func_graph_return_cnode = ctx->user_func_graph->get_return();
  auto return_input_cnode = user_func_graph_return_cnode->input(kIndex1);
  auto new_call_cnode = fg_users.first->func_graph()->NewCNode(ctx->fg_users_inputs_all);
  new_call_cnode->set_abstract(return_input_cnode->abstract()->Clone());
  (void)manager->Replace(fg_users.first, new_call_cnode);

  for (size_t j = 0; j < ctx->func_node_users.size(); ++j) {
    auto new_pair = ctx->func_node_users[j];
    if (!ctx->input_index[j].first) {
      auto new_maketuple_cnode = ctx->user_func_graph->NewCNode(ctx->new_concat_maketuple_inputs);
      new_maketuple_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(ctx->new_maketuple_abstracts));

      auto old_concat_prim = GetCNodePrimitive(concat_cnode);
      MS_EXCEPTION_IF_NULL(old_concat_prim);
      std::vector<AnfNodePtr> new_concat_inputs{NewValueNode(old_concat_prim->Clone()), new_maketuple_cnode,
                                                NewValueNode(MakeValue<int64_t>(concat_axis))};
      auto new_concat = ctx->user_func_graph->NewCNode(new_concat_inputs);
      new_concat->set_abstract(concat_cnode->abstract()->Clone());
      auto new_concat_prim = GetCNodePrimitive(new_concat);
      MS_EXCEPTION_IF_NULL(new_concat_prim);
      if (new_concat_prim->HasAttr("fine_grained_interleaved_index")) {
        new_concat_prim->EraseAttr("fine_grained_interleaved_index");
      }
      (void)manager->SetEdge(new_pair.first, new_pair.second, new_concat);
      continue;
    }
    auto old_parameter = ctx->user_func_graph->parameters()[fg_users.second - kIndex2 + ctx->input_index[j].second];
    (void)manager->Replace(new_pair.first, old_parameter);
  }
}

bool HandleFuncConcatSlice(const FuncGraphManagerPtr &manager, const std::pair<AnfNodePtr, int> &pair,
                           const CNodePtr &concat_cnode, const ShapeVector &concat_output_shape_element,
                           int64_t concat_axis) {
  MS_EXCEPTION_IF_NULL(pair.first);
  auto fg = pair.first->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto fg_map = fg->func_graph_cnodes_index();
  if (fg_map.size() > 1) {
    return false;
  }

  for (const auto &fg_use : fg_map) {
    if (!fg_use.first->first->isa<CNode>() || fg_use.first->second > 0) {
      continue;
    }

    // handle call_cnode
    CNodePtr call_cnode = fg_use.first->first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(call_cnode);
    MS_EXCEPTION_IF_NULL(concat_cnode);
    ConcatTransformContext ctx;
    auto ret = HandleMerge(call_cnode, manager, concat_cnode, concat_output_shape_element, concat_axis, &ctx);
    if (!ret.has_value()) {
      continue;
    }
    auto fg_users = ret.value();

    // TupleGetItem
    HandleTupleReturn(call_cnode, concat_cnode, pair, fg_users, manager, &ctx);

    // update parameters
    UpdateUserGraphAndReplaceCall(fg_users, manager, concat_cnode, concat_axis, &ctx);
  }
  return true;
}

bool MergeConcatSlice(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphManagerPtr &manager) {
  bool merged = false;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimConcat)) {
      continue;
    }
    auto concat_cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(concat_cnode->abstract());
    auto concat_output_shape = concat_cnode->abstract()->BuildShape();
    MS_EXCEPTION_IF_NULL(concat_output_shape);
    MS_EXCEPTION_IF_NULL(concat_output_shape->cast<abstract::ShapePtr>());
    auto concat_output_shape_element = concat_output_shape->cast<abstract::ShapePtr>()->shape();
    auto axis_value_node = concat_cnode->input(kIndex2);
    auto axis_value = GetValueNode(axis_value_node);
    auto concat_axis = GetValue<int64_t>(axis_value);
    auto next_nodes = GetOutputNodesSkipDepend(node);
    for (const auto &pair : next_nodes) {
      if (IsPrimitiveCNode(pair.first, prim::kPrimReturn) && next_nodes.size() == 1) {
        merged = HandleFuncConcatSlice(manager, pair, concat_cnode, concat_output_shape_element, concat_axis);
        continue;
      }
      auto can_merge = CanMergeConcatSlice(pair, concat_cnode, concat_output_shape_element, concat_axis);
      if (!can_merge.first) {
        continue;
      }
      auto concat_input_node = concat_cnode->input(1)->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(concat_input_node);
      auto concat_real_input_node = concat_input_node->input(can_merge.second);
      (void)manager->Replace(pair.first->cast<CNodePtr>(), concat_real_input_node);
      merged = true;
    }
  }
  return merged;
}

AnfNodePtr NewMicroMirrorPrimByMicroMirror(const FuncGraphPtr &func_graph, const CNodePtr &micro_mirror,
                                           const AnfNodePtr &micro_mirror_new_input) {
  auto prim_origin = GetCNodePrimitive(micro_mirror);
  MS_EXCEPTION_IF_NULL(prim_origin);
  Attr attr0 = std::make_pair(GROUP, prim_origin->GetAttr(GROUP));
  Attr attr1 = std::make_pair(DEV_NUM, prim_origin->GetAttr(DEV_NUM));
  Attr attr2 = std::make_pair(MEAN_FLAG, prim_origin->GetAttr(MEAN_FLAG));
  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);
  operator_attrs.push_back(attr1);
  operator_attrs.push_back(attr2);
  ValuePtr pyop_instance = CreateOpInstance(operator_attrs, MIRROR_MICRO_STEP_OPERATOR, prim_origin->instance_name());
  MS_EXCEPTION_IF_NULL(pyop_instance);
  std::vector<AnfNodePtr> mirror_inputs{NewValueNode(pyop_instance), micro_mirror_new_input,
                                        micro_mirror->input(kIndex2)};
  auto new_mirror_node = func_graph->NewCNode(mirror_inputs);
  auto prim = GetCNodePrimitive(new_mirror_node);
  MS_EXCEPTION_IF_NULL(prim);
  (void)prim->SetAttrs(prim_origin->attrs());
  new_mirror_node->set_attrs(micro_mirror->attrs());
  new_mirror_node->set_primal_attrs(micro_mirror->primal_attrs());
  new_mirror_node->set_in_forward_flag(true);
  return new_mirror_node;
}

void AddNodeFusionInfo(const CNodePtr &node, const CNodePtr &comm_node, const std::string &backward_comm_name,
                       const std::string &param_name, int32_t fusion_id) {
  auto comm_id = MakeValue<std::string>(param_name);
  comm_node->AddPrimalAttr(kPrimalAttrMirrorUserId, comm_id);
  auto prim = GetCNodePrimitive(comm_node);
  MS_EXCEPTION_IF_NULL(prim);
  if (prim) {
    prim->AddAttr(kPrimalAttrMirrorUserId, comm_id);
  }
  if (GetValueNode<PrimitivePtr>(comm_node->input(0))->HasAttr(GROUP)) {
    auto comm_group = GetValue<std::string>(GetValueNode<PrimitivePtr>(comm_node->input(0))->GetAttr(GROUP));
    std::string fusion_key = backward_comm_name + "_" + comm_group + "_" + std::to_string(fusion_id);
    if (!IsPrimitiveCNode(node, prim::kPrimLoad) && !IsPrimitiveCNode(node, prim::kPrimCast)) {
      if (fusion_id > 0) {
        node->AddPrimalAttr(kRelatedFusionKey, MakeValue<std::string>(fusion_key));
        node->AddPrimalAttr(kRelatedNodeId, MakeValue<std::string>(node->UniqueId()));
        node->AddAttr(kRelatedCommNodeId, MakeValue<std::string>(comm_node->UniqueId()));
      }
      node->AddPrimalAttr(kPrimalAttrMirrorUserId, comm_id);
      prim = GetCNodePrimitive(node);
      if (prim) {
        prim->AddAttr(kPrimalAttrMirrorUserId, comm_id);
      }
      return;
    }
    auto next_nodes = GetOutputNodesWithFilter(node, [&](const AnfNodePtr &anode) {
      return IsPrimitiveCNode(anode, prim::kPrimLoad) || IsPrimitiveCNode(anode, prim::kPrimCast) ||
             IsPrimitiveCNode(anode, prim::kPrimAllGather) || IsPrimitiveCNode(anode, prim::kPrimMirror) ||
             IsPrimitiveCNode(anode, prim::kPrimMicroStepAllGather) ||
             IsPrimitiveCNode(anode, prim::kPrimMirrorMicroStep) || IsPrimitiveCNode(anode, prim::kPrimMakeTuple);
    });
    for (auto &pair : next_nodes) {
      if (!IsPrimitiveCNode(pair.first)) {
        continue;
      }
      auto next_cnode = pair.first->cast<CNodePtr>();
      if (fusion_id > 0) {
        next_cnode->AddPrimalAttr(kRelatedFusionKey, MakeValue<std::string>(fusion_key));
        next_cnode->AddPrimalAttr(kRelatedNodeId, MakeValue<std::string>(node->UniqueId()));
        next_cnode->AddAttr(kRelatedCommNodeId, MakeValue<std::string>(comm_node->UniqueId()));
      }
      next_cnode->AddPrimalAttr(kPrimalAttrMirrorUserId, comm_id);
      prim = GetCNodePrimitive(next_cnode);
      if (prim) {
        prim->AddAttr(kPrimalAttrMirrorUserId, comm_id);
      }
    }
  }
}

void AddNodeMirrorInfo(const CNodePtr &cnode, const std::string &param_name) {
  auto comm_id = MakeValue<std::string>(param_name);
  if (IsParallelCareNode(cnode)) {
    cnode->AddPrimalAttr(kPrimalAttrMirrorUserId, comm_id);
    auto prim = GetCNodePrimitive(cnode);
    if (prim) {
      prim->AddAttr(kPrimalAttrMirrorUserId, comm_id);
    }
    return;
  }
  auto next_nodes = GetOutputNodesWithFilter(cnode, [&](const AnfNodePtr &anode) {
    return IsPrimitiveCNode(anode, prim::kPrimLoad) || IsPrimitiveCNode(anode, prim::kPrimCast) ||
           IsPrimitiveCNode(anode, prim::kPrimAllGather) || IsPrimitiveCNode(anode, prim::kPrimMirror) ||
           IsPrimitiveCNode(anode, prim::kPrimMicroStepAllGather) ||
           IsPrimitiveCNode(anode, prim::kPrimMirrorMicroStep) || IsPrimitiveCNode(anode, prim::kPrimMakeTuple);
  });
  for (auto &pair : next_nodes) {
    if (!IsPrimitiveCNode(pair.first)) {
      continue;
    }
    auto next_node = pair.first->cast<CNodePtr>();
    next_node->AddPrimalAttr(kPrimalAttrMirrorUserId, comm_id);
    auto prim = GetCNodePrimitive(next_node);
    if (prim) {
      prim->AddAttr(kPrimalAttrMirrorUserId, comm_id);
    }
  }
}

static ValuePtr GetMakeTupleValue(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto &inputs = cnode->inputs();

  std::vector<int64_t> value_list;
  for (size_t index = 1; index < inputs.size(); ++index) {
    if (inputs[index]->isa<ValueNode>()) {
      auto element = GetValueNode(inputs[index]);
      MS_EXCEPTION_IF_NULL(element);
      if (element->isa<Int64Imm>()) {
        int64_t value = element->cast<Int64ImmPtr>()->value();
        value_list.push_back(value);
        continue;
      }
    }
    value_list.push_back(-1);  // dynamic shape
  }

  MS_LOG(INFO) << "the make tuple value is " << value_list;
  return MakeValue(value_list);
}

bool IsSupportNewShapeBaseNode(const CNodePtr &node) {
  const auto &all_inputs = node->inputs();
  auto has_value_seq = std::any_of(all_inputs.begin() + 1, all_inputs.end(), [&node](const AnfNodePtr &input) {
    bool is_abs_seq = false;
    auto abs = input->abstract();
    if (abs != nullptr) {
      is_abs_seq = abs->isa<abstract::AbstractSequence>();
    }
    return ((is_abs_seq || IsValueSequence(input)) && IsSomePrimitiveList(node, SUPPORT_NEW_SHAPEBASE_OPS));
  });
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(node->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  auto has_self_define_shard = prim->HasAttr(SELF_DEFINE_SHARD);
  auto is_support_new_shape_base_node =
    has_value_seq || IsSomePrimitiveList(node, ALWAYS_NEW_SHAPEBASE_OPS) || has_self_define_shard;
  if (is_support_new_shape_base_node) {
    MS_LOG(INFO) << "The node is " << node->DebugString() << " supports new shape base.";
  }
  return is_support_new_shape_base_node;
}

OperatorInfoPtr CreateOperatorInfoForNewShape(const CNodePtr &cnode) {
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  MS_LOG(INFO) << prim->name() << ": supports new shape base, enter new shape logic.";
  std::pair<std::vector<NewShapes>, std::vector<Symbols>> shapes_and_symbols = ExtractNewShapeAndSymbol(cnode);
  auto shape_list = shapes_and_symbols.first;
  auto symbol_list = shapes_and_symbols.second;
  if (shape_list.size() != INPUT_OUTPUT_SYMBOLS_SIZE || symbol_list.size() != INPUT_OUTPUT_SYMBOLS_SIZE) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "the size of shapes or symbols must be " << INPUT_OUTPUT_SYMBOLS_SIZE
                                       << ", but the size of shapes is " << shape_list.size()
                                       << ", the size of symbols is " << symbol_list.size();
  }
  auto attrs = prim->attrs();
  std::vector<Shapes> temp_shape_list = {{}, {}};
  OperatorInfoPtr op_info = OperatorInstance(prim, attrs, temp_shape_list);
  MS_EXCEPTION_IF_NULL(op_info);

  // When the 'inputs' contains numerical values for some operators, these values should be extracted from
  // ANF graph
  auto &inputs = cnode->inputs();
  std::vector<ValuePtr> input_value;
  for (size_t index = 1; index < inputs.size(); ++index) {
    if (inputs[index]->isa<ValueNode>() || inputs[index]->isa<tensor::Tensor>()) {
      (void)input_value.emplace_back(GetValueNode(inputs[index]));
      continue;
    } else if (IsPrimitiveCNode(inputs[index], prim::kPrimMakeTuple)) {
      auto make_tuple_value = GetMakeTupleValue(inputs[index]);
      (void)input_value.emplace_back(make_tuple_value);
      continue;
    } else if (IsPrimitiveCNode(inputs[index], prim::kPrimShape)) {
      auto shape_op_cnode = dyn_cast_ptr<CNode>(inputs[index]);
      MS_EXCEPTION_IF_NULL(shape_op_cnode);
      auto dst_shape = GetNodeShape(shape_op_cnode->input(1));
      (void)input_value.emplace_back(MakeValue(dst_shape[0]));
      MS_LOG(INFO) << "The prim is " << prim->name() << ", the input index is " << index - 1
                   << ", is Shape op, dst shape is " << dst_shape;
      continue;
    }
    (void)input_value.emplace_back(nullptr);
  }
  (*op_info).set_input_value(input_value);
  (*op_info).set_outputs_dtype(cnode->Type());
  (*op_info).set_cnode(cnode);
  (*op_info).set_new_shape(shape_list);
  return op_info;
}

OperatorInfoPtr CreateOperatorInfo(const CNodePtr &cnode) {
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  if (IsSupportNewShapeBaseNode(cnode)) {
    return CreateOperatorInfoForNewShape(cnode);
  }
  std::pair<std::vector<Shapes>, std::vector<Symbols>> shapes_and_symbols = ExtractShapeAndSymbol(cnode);
  auto shape_list = shapes_and_symbols.first;
  auto symbol_list = shapes_and_symbols.second;
  if (shape_list.size() != INPUT_OUTPUT_SYMBOLS_SIZE || symbol_list.size() != INPUT_OUTPUT_SYMBOLS_SIZE) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "the size of shapes or symbols must be " << INPUT_OUTPUT_SYMBOLS_SIZE
                                       << ", but the size of shapes is " << shape_list.size()
                                       << ", the size of symbols is " << symbol_list.size();
  }

  auto attrs = prim->attrs();
  OperatorInfoPtr op_info = OperatorInstance(prim, attrs, shape_list);
  MS_EXCEPTION_IF_NULL(op_info);
  MS_LOG(INFO) << "shape_list.size(): " << shape_list.size();

  // When the 'inputs' contains numerical values for some operators, these values should be extracted from
  // ANF graph
  auto &inputs = cnode->inputs();
  std::vector<ValuePtr> input_value;
  for (size_t index = 1; index < inputs.size(); ++index) {
    if (inputs[index]->isa<ValueNode>() || inputs[index]->isa<tensor::Tensor>()) {
      (void)input_value.emplace_back(GetValueNode(inputs[index]));
      continue;
    } else if (IsPrimitiveCNode(inputs[index], prim::kPrimMakeTuple)) {
      auto make_tuple_value = GetMakeTupleValue(inputs[index]);
      (void)input_value.emplace_back(make_tuple_value);
      continue;
    } else if (IsPrimitiveCNode(inputs[index], prim::kPrimShape)) {
      auto shape_op_cnode = dyn_cast_ptr<CNode>(inputs[index]);
      auto dst_shape = GetNodeShape(shape_op_cnode->input(1));
      (void)input_value.emplace_back(MakeValue(dst_shape[0]));
      MS_LOG(INFO) << "The prim is " << prim->name() << ", the input index is " << index - 1
                   << ", is Shape op, dst shape is " << dst_shape;
      continue;
    }
    (void)input_value.emplace_back(nullptr);
  }

  (*op_info).set_input_value(input_value);
  (*op_info).set_outputs_dtype(cnode->Type());
  (*op_info).set_cnode(cnode);
  if (IsForwardDynamicShape() && IsDynamicShapesList(shape_list)) {
    Shapes in_real_divisors;
    Shapes out_real_divisors;
    in_real_divisors = GetRealDivisorSymbols(shape_list[INPUT_SYMBOLS_INDEX], symbol_list[INPUT_SYMBOLS_INDEX]);
    out_real_divisors = GetRealDivisorSymbols(shape_list[OUTPUT_SYMBOLS_INDEX], symbol_list[OUTPUT_SYMBOLS_INDEX]);
    (*op_info).set_dynamic_shape_flag(True);
    (*op_info).set_inputs_divisor(in_real_divisors);
    (*op_info).set_outputs_divisor(out_real_divisors);
    MS_LOG(DEBUG) << (*op_info).name() << ": inputs-shape: " << ShapesToString(shape_list[0])
                  << ", inputs_d_symbol: " << ShapesToString(in_real_divisors);
    MS_LOG(DEBUG) << (*op_info).name() << ": outputs-shape: " << ShapesToString(shape_list[1])
                  << ", outputs_d_symbol: " << ShapesToString(out_real_divisors);
  }
  return op_info;
}

void ExtendInputArgsAbstractShape(const AbstractBasePtr &args_abstract_item, size_t index) {
  auto args_abstract_item_shape = args_abstract_item->BuildShape();
  auto shape_ptr = dyn_cast<abstract::Shape>(args_abstract_item_shape);
  if (shape_ptr == nullptr) {
    MS_LOG(WARNING) << "The input " << index << " is not a tensor.";
    return;
  }
  auto shape_value = parallel::ToFullShape(shape_ptr->shape(), index);
  auto new_shape_item = std::make_shared<abstract::Shape>(shape_value);
  args_abstract_item->set_shape(new_shape_item);
}

ShapeVector ToFullShape(const ShapeVector &input_shape, size_t index) {
  if (input_shape.empty()) {
    return input_shape;
  }
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  auto has_devmat = !ParallelContext::GetInstance()->dataset_strategy_devmat().empty();
  auto has_tensor_map = !ParallelContext::GetInstance()->dataset_strategy_tensormap().empty();
  std::vector<int64_t> dataset_strategy_item;
  if (has_devmat && has_tensor_map) {
    MS_LOG(INFO) << "Use layout dataset strategy, device matrix is " << has_devmat << ", tensor map is "
                 << has_tensor_map;
    // (2, 2, 2)
    auto corresponding_devmat = ParallelContext::GetInstance()->dataset_strategy_devmat().at(index);
    // ((2), (1), (0))
    auto corresponding_tensormap = ParallelContext::GetInstance()->dataset_strategy_tensormap().at(index);
    for (const auto &tensormap : corresponding_tensormap) {
      int64_t stra = 1;
      for (const auto &value : tensormap) {
        if (value != -1) {
          auto dev_mat_idx = corresponding_devmat.size() - value - 1;
          stra *= corresponding_devmat.at(dev_mat_idx);
        }
      }
      dataset_strategy_item.push_back(stra);
    }
    MS_LOG(INFO) << "The dataset strategy is " << dataset_strategy_item;
  } else {
    if (ParallelContext::GetInstance()->dataset_strategy().empty()) {
      auto shape_value = input_shape;
      if (!parallel::ParallelContext::GetInstance()->full_batch()) {
        auto comm_info = parallel::GetCommInfo();
        auto world_rank_size = comm_info.device_num / ParallelContext::GetInstance()->pipeline_stage_split_num();
        if (shape_value[0] > 0) {
          shape_value[0] = shape_value[0] * SizeToLong(world_rank_size);  // only for static shape
        }
      }
      return shape_value;
    }
    auto dataset_strategy = ParallelContext::GetInstance()->dataset_strategy();
    if (index >= dataset_strategy.size()) {
      MS_LOG(EXCEPTION) << "The input shapes size is not equal to dataset strategy size " << dataset_strategy.size();
    }
    dataset_strategy_item = dataset_strategy[index];
    if (input_shape.size() != dataset_strategy_item.size()) {
      MS_LOG(EXCEPTION) << "The input_shapes[" << index << "]'s size" << input_shape.size()
                        << " is not equal to dataset_strategy[" << index << "]'s size " << dataset_strategy_item.size();
    }
  }

  ShapeVector shape_value;
  for (size_t i = 0; i < dataset_strategy_item.size(); ++i) {
    if (input_shape[i] > 0) {
      shape_value.push_back(input_shape[i] * dataset_strategy_item[i]);
    } else {
      shape_value.push_back(input_shape[i]);  // dynamic shape, shape is still -1
    }
  }
  return shape_value;
}

CommInfo GetCommInfo() {
  int64_t device_num = ParallelContext::GetInstance()->device_num();
  int64_t global_rank = ParallelContext::GetInstance()->global_rank();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  std::string world_group;
  std::string communication_backend;
  if (backend == kAscendDevice || backend == kDavinciDevice) {
    world_group = HCCL_WORLD_GROUP;
    communication_backend = HCCL_BACKEND;
  } else if (backend == kGPUDevice) {
    world_group = NCCL_WORLD_GROUP;
    communication_backend = NCCL_BACKEND;
  } else {
    MS_LOG(EXCEPTION) << "Invalid communication backend: " << backend
                      << " for semi_auto_parallel/auto_parallel mode,"
                         " currently only support Ascend/GPU backend.";
  }
  uint32_t world_rank_size = 0;
  if (!CommManager::GetInstance().GetRankSize(world_group, &world_rank_size)) {
    MS_LOG(EXCEPTION) << "Get rank size failed";
  }

  if (!ParallelContext::GetInstance()->device_num_is_set()) {
    device_num = UintToInt(world_rank_size);
    MS_LOG(INFO) << "Get device num from communication model, the device num is  " << device_num;
  }
#if (!defined(_WIN32) && !defined(__APPLE__) || defined(ENABLE_TEST))
  if (ParallelContext::GetInstance()->device_num_is_set() && world_rank_size != device_num &&
      !ParallelContext::GetInstance()->hccl_test_available()) {
    // hccl_test_available is used when we compile graphs in real ascend card environment, but with hccl_test.
    MS_LOG(EXCEPTION) << "The device_num " << device_num << " set in the context is not consist with "
                      << world_rank_size << " devices you have"
                      << ". Please check your rank_table file(for Ascend) or host file(for GPU).";
  }
#endif
  uint32_t rank_id = 0;
  if (!ParallelContext::GetInstance()->global_rank_is_set()) {
    if (!CommManager::GetInstance().GetRankID(world_group, &rank_id)) {
      MS_LOG(EXCEPTION) << "Get rank id failed";
    }
    global_rank = UintToInt(rank_id);
    ParallelContext::GetInstance()->set_global_rank(global_rank);
    MS_LOG(INFO) << "Get global rank from communication model, the global rank is  " << global_rank;
  }
  CommInfo comm_info{device_num, global_rank, world_group, communication_backend};
  return comm_info;
}

bool IsAutoParallelCareGraph(const FuncGraphPtr &func_graph) {
  // compile graph order:
  // 1, ParallelParameterContextRestoreShape
  // 2, PipelineSplit: insert virtual dataset
  // 3, StepAutoParallel
  // 4, StepParallel
  // if IsParallel() is true, it maybe has some graphs that we now care, so need to check
  // 'sharded' or 'has_shard' flag
  MS_EXCEPTION_IF_NULL(func_graph);
  if (func_graph->has_flag(kSkipAutoParallelCompile)) {
    return false;
  }

  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (parallel_mode != kAutoParallel && parallel_mode != kSemiAutoParallel) {
    return false;
  }
  return true;
}

void FindPreNodeCrossFuncGraph(CNodePtr *cnode, int64_t out_index) {
  if (IsValueNode<FuncGraph>((*cnode)->input(0))) {
    auto graph = GetValueNode<FuncGraphPtr>((*cnode)->input(0));
    MS_EXCEPTION_IF_NULL(graph);
    auto output = graph->output();
    MS_EXCEPTION_IF_NULL(output);
    while (IsPrimitiveCNode(output, prim::kPrimDepend)) {
      auto output_cnode = output->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(output_cnode);
      output = output_cnode->input(1);
    }
    while (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
      auto make_tuple_cnode = output->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(make_tuple_cnode);
      output = make_tuple_cnode->input(out_index + 1);
    }
    *cnode = output->cast<CNodePtr>();
  }
}

AnfNodePtr FindRealInputByFormalParameter(const CNodePtr &node, const AnfNodePtr &input,
                                          const std::vector<AnfNodePtr> &all_nodes) {
  auto prev_node = input;
  auto graph = node->func_graph();
  auto params = graph->parameters();
  int64_t param_index = -1;
  for (size_t j = 0; j < params.size(); ++j) {
    if (params[j] == input) {
      param_index = SizeToLong(j);
    }
  }
  if (param_index == -1) {
    return prev_node;
  }
  for (auto &ele : all_nodes) {
    if (!ele->isa<CNode>()) {
      continue;
    }
    auto parent_node = ele->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(parent_node);
    if (IsValueNode<FuncGraph>(parent_node->input(0)) && GetValueNode<FuncGraphPtr>(parent_node->input(0)) == graph) {
      return parent_node->input(param_index + 1);
    }
  }
  return prev_node;
}

bool CrossInterNode(CNodePtr *prev_cnode, ValueNodePtr *prev_prim_anf_node, PrimitivePtr *prev_prim) {
  if ((*prev_cnode == nullptr) ||
      !(IsValueNode<Primitive>((*prev_cnode)->input(0)) || IsValueNode<FuncGraph>((*prev_cnode)->input(0)))) {
    return true;
  }
  if (!IsValueNode<FuncGraph>((*prev_cnode)->input(0))) {
    *prev_prim_anf_node = (*prev_cnode)->input(0)->cast<ValueNodePtr>();
    *prev_prim = (*prev_prim_anf_node)->value()->cast<PrimitivePtr>();
  }
  return false;
}

bool IsCarePrevCNode(const CNodePtr &prev_cnode, const PrimitivePtr &prev_prim) {
  return (IsValueNode<FuncGraph>(prev_cnode->input(0))) || (prev_prim->name() == kTupleGetItemOpName) ||
         (prev_prim->name() == kDependOpName) || (prev_prim->name() == kMakeListOpName) ||
         (prev_prim->name() == kLoadOpName) || (prev_prim->name() == kMakeTupleOpName) ||
         (prev_prim->name() == kShapeOpName) || IsAutoParallelCareNode(prev_cnode);
}

bool IsCrossedCNode(std::string prev_prim_name) {
  const std::set<std::string> crossed_cnode_list = {kDependOpName, kLoadOpName, kShapeOpName};
  return crossed_cnode_list.find(prev_prim_name) != crossed_cnode_list.end();
}

// Needed by rec_parser
std::vector<std::string> ExtractInputsTensorName(const CNodePtr &node, const std::vector<AnfNodePtr> &all_nodes) {
  std::vector<std::string> name_inputs;
  std::vector<AnfNodePtr> all_inputs = node->inputs();
  std::vector<AnfNodePtr> node_inputs{all_inputs.begin() + 1, all_inputs.end()};

  std::string node_id = node->UniqueId();
  name_inputs.push_back(node_id);
  for (auto &input : node_inputs) {
    AnfNodePtr prev_node = input;
    if (input->isa<Parameter>()) {
      prev_node = FindRealInputByFormalParameter(node, input, all_nodes);
      if (prev_node->UniqueId() == input->UniqueId()) {
        name_inputs.push_back(input->UniqueId());
        continue;
      }
    }
    auto prev_cnode = prev_node->cast<CNodePtr>();
    PrimitivePtr prev_prim;
    ValueNodePtr prev_prim_anf_node;

    bool is_cross = CrossInterNode(&prev_cnode, &prev_prim_anf_node, &prev_prim);
    if (is_cross) {
      name_inputs.push_back(input->UniqueId());
      continue;
    }

    size_t output_index = 0;
    while (IsCarePrevCNode(prev_cnode, prev_prim)) {
      if (IsValueNode<FuncGraph>(prev_cnode->input(0))) {
        auto graph = GetValueNode<FuncGraphPtr>(prev_cnode->input(0));
        auto output = graph->output();
        MS_EXCEPTION_IF_NULL(output);
        prev_cnode = output->cast<CNodePtr>();
        (void)CrossInterNode(&prev_cnode, &prev_prim_anf_node, &prev_prim);
      } else if (IsAutoParallelCareNode(prev_cnode)) {
        name_inputs.push_back(prev_cnode->UniqueId());
        break;
      } else if (prev_prim->name() == kTupleGetItemOpName) {
        // In this case, 'prev_anf_node' is 'tuple_getitem', the actual precursor node is node before
        // this 'tuple_getitem'
        output_index = LongToSize(GetValue<int64_t>(GetValueNode(prev_cnode->input(INDEX_TWO))));
        prev_node = prev_cnode->input(1);
        prev_cnode = prev_node->cast<CNodePtr>();

        if (prev_cnode != nullptr && common::AnfAlgo::GetCNodeName(prev_cnode) == kTupleGetItemOpName) {
          continue;
        }

        is_cross = CrossInterNode(&prev_cnode, &prev_prim_anf_node, &prev_prim);
        if (is_cross) {
          name_inputs.push_back(prev_node->UniqueId());
          break;
        }

        // In dynamic shape scenarios, the situation op1->Shape->TupleGetItem->op2 will occur.
        // The incoming operator of op2 should be op1 instead of Shape,
        // so the Shape operator is skipped when looking for the incoming operator.
        if (prev_prim->name() == kShapeOpName) {
          continue;
        }

        if (!IsAutoParallelCareNode(prev_cnode) && !IsValueNode<FuncGraph>(prev_cnode->input(0))) {
          MS_LOG_WITH_NODE(EXCEPTION, prev_cnode) << "Did not create OperatorInfo for : " << prev_prim->name();
        }
      } else if (prev_prim->name() == kMakeTupleOpName) {
        prev_node = prev_cnode->input(output_index + 1);
        prev_cnode = prev_node->cast<CNodePtr>();
        output_index = 0;
        is_cross = CrossInterNode(&prev_cnode, &prev_prim_anf_node, &prev_prim);
        if (is_cross) {
          name_inputs.push_back(prev_node->UniqueId());
          break;
        }
      } else if (IsCrossedCNode(prev_prim->name())) {
        // In this case, 'prev_anf_node' is 'depend', the actual precursor node is node before
        // this 'depend'
        prev_node = prev_cnode->input(1);
        prev_cnode = prev_node->cast<CNodePtr>();
        is_cross = CrossInterNode(&prev_cnode, &prev_prim_anf_node, &prev_prim);
        if (is_cross) {
          name_inputs.push_back(prev_node->UniqueId());
          break;
        }
      }
    }
  }

  return name_inputs;
}

OperatorInfoPtr GetDistributeOperator(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!IsParallelCareNode(node)) {
    return nullptr;
  }
  OperatorInfoPtr distribute_operator = node->user_data<OperatorInfo>();
  return distribute_operator;
}

bool StrategyFound(const mindspore::HashMap<std::string, ValuePtr> &attrs) {
  auto iter = attrs.find(IN_STRATEGY);
  return !((iter == attrs.end()) || (iter->second->type_name() == NONE));
}

bool LayoutFound(const mindspore::HashMap<std::string, ValuePtr> &attrs) {
  auto iter = attrs.find(IN_LAYOUT);
  return !((iter == attrs.end()) || (iter->second->type_name() == NONE));
}

bool OutLayoutFound(const mindspore::HashMap<std::string, ValuePtr> &attrs) {
  auto iter = attrs.find(OUT_LAYOUT);
  return !((iter == attrs.end()) || (iter->second->type_name() == NONE));
}

bool OutStrategyFound(const mindspore::HashMap<std::string, ValuePtr> &attrs) {
  auto iter = attrs.find(OUT_STRATEGY);
  return !((iter == attrs.end()) || (iter->second->type_name() == NONE));
}

bool AttrFound(const mindspore::HashMap<std::string, ValuePtr> &attrs, const std::string &target) {
  auto iter = attrs.find(target);
  return !((iter == attrs.end()) || (iter->second->type_name() == NONE));
}

bool IsCommunicationOp(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  return (COMMUNICATION_OPS.find(prim->name()) != COMMUNICATION_OPS.end());
}

void ExceptionIfHasCommunicationOp(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    ValueNodePtr prim_value_node = cnode->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_value_node);
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_value_node);
    MS_EXCEPTION_IF_NULL(prim);

    if (IsCommunicationOp(prim) && cnode->in_forward_flag()) {
      MS_EXCEPTION_IF_NULL(prim_value_node->scope());
      MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
      std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
      MS_LOG_WITH_NODE(EXCEPTION, cnode)
        << "If the parallel mode is semi_auto_parallel or auto_parallel, the graph can not contain "
           "communication op, the parallel mode is "
        << parallel_mode << ", and the graph has communication op : " << prim->name() << ", scope name is "
        << prim_value_node->scope()->name();
    }
  }
}

std::string MirrorOpName() {
  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  int64_t split_stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();
  std::string mirror_op_name;
  if (split_stage_num > 1 || grad_accumulation_step > 1) {
    mirror_op_name = MIRROR_MICRO_STEP_OPERATOR;
  } else {
    mirror_op_name = MIRROR_OPERATOR;
  }
  return mirror_op_name;
}

bool CheckStrategyWithTupleInTuple(const std::vector<ValuePtr> &elements) {
  bool has_tuple_in_tuple = false;
  for (size_t i = 0; i < elements.size(); ++i) {
    if (elements[i]->isa<ValueSequence>()) {
      auto value_tuple = elements[i]->cast<ValueTuplePtr>();
      MS_EXCEPTION_IF_NULL(value_tuple);
      std::vector<ValuePtr> value_vector = value_tuple->value();
      auto local_tuple_in_tuple = std::any_of(value_vector.begin(), value_vector.end(),
                                              [](const ValuePtr &value) { return value->isa<ValueSequence>(); });
      has_tuple_in_tuple = has_tuple_in_tuple || local_tuple_in_tuple;
    } else {
      MS_LOG(EXCEPTION) << "Failure: Strategy's format is wrong! Need ValueSequence";
    }
  }
  MS_LOG(INFO) << "CheckStrategyWithTupleInTuple: has_tuple_in_tuple = " << has_tuple_in_tuple << ".";
  return has_tuple_in_tuple;
}

NewDimensions ExtractDimensions(const ValuePtr &stra) {
  auto value_tuple = stra->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple);
  std::vector<ValuePtr> value_vector = value_tuple->value();
  bool has_tuple_in_tuple = std::any_of(value_vector.begin(), value_vector.end(),
                                        [](const ValuePtr &value) { return value->isa<ValueSequence>(); });
  if (has_tuple_in_tuple) {
    std::vector<NewDimensions> dim;
    (void)std::transform(value_vector.begin(), value_vector.end(), std::back_inserter(dim),
                         [](const ValuePtr &value) { return ExtractDimensions(value); });
    return std::make_shared<ShapeList>(dim);
  }
  Dimensions dim;
  (void)std::transform(value_vector.begin(), value_vector.end(), std::back_inserter(dim),
                       [](const ValuePtr &value) { return static_cast<int64_t>(GetValue<int64_t>(value)); });
  return std::make_shared<ShapeValue>(dim);
}

StrategyPtr ExtractNewStrategy(const std::vector<ValuePtr> &elements, const int64_t &stage_id) {
  NewStrategies strategy;
  for (uint64_t index = 0; index < elements.size(); ++index) {
    if (elements[index]->isa<ValueSequence>()) {
      auto dim = ExtractDimensions(elements[index]);
      strategy.emplace_back(dim);
    } else {
      MS_LOG(EXCEPTION) << "Failure: Strategy's format is wrong! Need ValueSequence";
    }
  }
  if (strategy.empty()) {
    MS_LOG(EXCEPTION) << "ExtractStrategy: failed to extract strategy";
  }
  StrategyPtr strategyPtr = NewStrategy(stage_id, strategy);
  return strategyPtr;
}

StrategyPtr ExtractStrategy(const ValuePtr &stra, const bool use_shape_base) {
  if (stra == nullptr) {
    return nullptr;
  }

  auto var = stra->cast<ValueTuplePtr>();
  if (var == nullptr) {
    return nullptr;
  }

  StrategyPtr strategyPtr;
  int64_t stage_id = g_device_manager->stage_id();

  MS_LOG(INFO) << "Extract information: strategy " << stra->ToString();
  if (var->size() > 0) {
    std::vector<ValuePtr> elements = var->value();
    if (CheckStrategyWithTupleInTuple(elements) || use_shape_base) {
      return ExtractNewStrategy(elements, stage_id);
    }
    Strategies strategy;
    for (uint64_t index = 0; index < elements.size(); ++index) {
      Dimensions dim;
      if (elements[index]->isa<ValueSequence>()) {
        auto value_tuple = elements[index]->cast<ValueTuplePtr>();
        std::vector<ValuePtr> value_vector = value_tuple->value();
        (void)std::transform(value_vector.begin(), value_vector.end(), std::back_inserter(dim),
                             [](const ValuePtr &value) { return static_cast<int64_t>(GetValue<int64_t>(value)); });
        strategy.push_back(dim);
      } else {
        MS_LOG(EXCEPTION) << "Failure: Strategy's format is wrong! Need ValueSequence";
      }
    }
    if (strategy.empty()) {
      MS_LOG(EXCEPTION) << "ExtractStrategy: failed to extract strategy";
    }
    strategyPtr = NewStrategy(stage_id, strategy);
  }
  return strategyPtr;
}

Status GetLayoutFromAttrValue(const ValuePtr &layout_item, std::vector<std::string> *alias_name,
                              std::vector<int64_t> *device_matrix_vector,
                              std::vector<std::vector<int64_t>> *tensor_map_vector, bool *interleaved_parallel) {
  auto layout_dict_value = layout_item->cast<ValueDictionaryPtr>();
  if (!layout_dict_value) {
    MS_LOG(ERROR) << "The layout item configured for node is unreasonable";
    return FAILED;
  }
  auto layout_dict = layout_dict_value->value();
  ValuePtr alias_name_value = nullptr;
  ValuePtr device_matrix_value = nullptr;
  ValuePtr tensor_map_value = nullptr;
  ValuePtr interleaved_parallel_value = nullptr;
  for (const auto &value_pair : layout_dict) {
    if ((*value_pair.first) == (*MakeValue<std::string>(ALIAS_NAME))) {
      alias_name_value = value_pair.second;
    }
    if ((*value_pair.first) == (*MakeValue<std::string>(DEVICE_MATRIX))) {
      device_matrix_value = value_pair.second;
    }
    if ((*value_pair.first) == (*MakeValue<std::string>(TENSOR_MAP))) {
      tensor_map_value = value_pair.second;
    }
    if ((*value_pair.first) == (*MakeValue<std::string>(INTERLEAVED_PARALLEL))) {
      interleaved_parallel_value = value_pair.second;
    }
  }
  if (!device_matrix_value || !tensor_map_value || !interleaved_parallel_value) {
    MS_LOG(ERROR) << "The layout item configured for node is unreasonable";
    return FAILED;
  }

  if (alias_name_value != nullptr) {
    *alias_name = GetValue<std::vector<std::string>>(alias_name_value);
  }
  *device_matrix_vector = GetValue<std::vector<int64_t>>(device_matrix_value);
  auto tensor_map_value_tuple = tensor_map_value->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(tensor_map_value_tuple);
  *interleaved_parallel = GetValue<bool>(interleaved_parallel_value);
  std::vector<ValuePtr> tensor_map_value_tuple_vector = tensor_map_value_tuple->value();
  for (const auto &tensor_map_item : tensor_map_value_tuple_vector) {
    if (tensor_map_item->isa<ValueSequence>()) {
      auto tensor_map_item_v = GetValue<std::vector<int64_t>>(tensor_map_item);
      tensor_map_vector->push_back(tensor_map_item_v);
      continue;
    }
    auto tensor_map_item_i = GetValue<int64_t>(tensor_map_item);
    tensor_map_vector->push_back({tensor_map_item_i});
  }
  return SUCCESS;
}

void ConvertLayoutToStrategy(Dimensions *sub_strategy_vec, const std::vector<std::string> &alias_name,
                             const std::vector<int64_t> &device_matrix,
                             const std::vector<std::vector<int64_t>> &tensor_map) {
  const int64_t alias_len = (int64_t)alias_name.size() - 1;
  for (const auto &sub_tensor_map : tensor_map) {
    auto dim_strategy = 1;
    for (const int64_t tensor_ind : sub_tensor_map) {
      if (tensor_ind == -1) {
        continue;
      }
      int64_t actual_ind = std::abs(tensor_ind - alias_len);
      if (alias_name[actual_ind] != INTERLEAVED_PARALLEL) {
        dim_strategy *= device_matrix[actual_ind];
      }
    }
    sub_strategy_vec->emplace_back(dim_strategy);
  }
}

bool CheckShardingPropagation() {
  bool is_auto_parallel = ParallelContext::GetInstance()->parallel_mode() == kAutoParallel;
  bool is_sharding_propagation = (ParallelContext::GetInstance()->strategy_search_mode() == kShardingPropagation) ||
                                 (ParallelContext::GetInstance()->sharding_propagation());
  return is_auto_parallel && is_sharding_propagation;
}

Status ExtractUserConfigLayout(const mindspore::HashMap<std::string, ValuePtr> &prim_attrs, const Shapes &inputs_shape,
                               const Shapes &outputs_shape,
                               std::vector<std::shared_ptr<TensorLayout>> *in_tensor_layouts,
                               std::vector<std::shared_ptr<TensorLayout>> *out_tensor_layouts) {
  bool use_sp = CheckShardingPropagation();
  if (prim_attrs.count(IN_LAYOUT) > 0) {
    auto layout_value = prim_attrs.at(IN_LAYOUT);
    if (!layout_value->isa<ValueSequence>()) {
      MS_LOG(ERROR) << "The in_layout configured for node is not a tuple";
      return FAILED;
    }
    auto layout_value_tuple = layout_value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(layout_value_tuple);
    std::vector<ValuePtr> layout_value_vector = layout_value_tuple->value();
    if (inputs_shape.size() != layout_value_vector.size()) {
      MS_LOG(ERROR) << "The number of in_layout configured for the node must be equal to its input's number but got"
                    << layout_value_vector.size() << " and " << inputs_shape.size();
      return FAILED;
    }

    for (size_t i = 0; i < layout_value_vector.size(); ++i) {
      auto layout_item = layout_value_vector[i];
      std::vector<std::string> alias_name;
      std::vector<int64_t> device_matrix_vector;
      std::vector<std::vector<int64_t>> tensor_map_vector;
      bool interleaved_parallel;
      if (GetLayoutFromAttrValue(layout_item, &alias_name, &device_matrix_vector, &tensor_map_vector,
                                 &interleaved_parallel) != SUCCESS) {
        return FAILED;
      }
      auto in_layout = std::make_shared<TensorLayout>();
      if (use_sp && !alias_name.empty()) {
        // Convert tensor layout input to tuple-like strategy.
        Dimensions sub_strategy_vec;
        ConvertLayoutToStrategy(&sub_strategy_vec, alias_name, device_matrix_vector, tensor_map_vector);
        in_layout->set_in_layout_strategy(sub_strategy_vec);
      }
      if (in_layout->InitFromExtendVector(device_matrix_vector, tensor_map_vector, inputs_shape[i],
                                          interleaved_parallel) != SUCCESS) {
        MS_LOG(ERROR) << "The in_layout configured incorrect, device_matrix:" << device_matrix_vector
                      << ", tensor_map:" << tensor_map_vector;
        return FAILED;
      }
      in_tensor_layouts->push_back(in_layout);
    }
  }
  if (prim_attrs.count(OUT_LAYOUT) > 0) {
    auto layout_value = prim_attrs.at(OUT_LAYOUT);
    if (!layout_value->isa<ValueSequence>()) {
      MS_LOG(EXCEPTION) << "The in_layout configured for node is not a tuple";
    }
    auto layout_value_tuple = layout_value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(layout_value_tuple);
    std::vector<ValuePtr> layout_value_vector = layout_value_tuple->value();
    if (outputs_shape.size() != layout_value_vector.size()) {
      MS_LOG(EXCEPTION) << "The out_layout configured for node is not equal to its output nums";
    }
    for (size_t i = 0; i < layout_value_vector.size(); ++i) {
      auto layout_item = layout_value_vector[i];
      std::vector<std::string> alias_name;
      std::vector<int64_t> device_matrix_vector;
      std::vector<std::vector<int64_t>> tensor_map_vector;
      bool interleaved_parallel;
      if (GetLayoutFromAttrValue(layout_item, &alias_name, &device_matrix_vector, &tensor_map_vector,
                                 &interleaved_parallel) != SUCCESS) {
        return FAILED;
      }
      auto out_layout = std::make_shared<TensorLayout>();
      if (use_sp && !alias_name.empty()) {
        // Convert tensor layout input to tuple-like strategy.
        Dimensions sub_strategy_vec;
        ConvertLayoutToStrategy(&sub_strategy_vec, alias_name, device_matrix_vector, tensor_map_vector);
        out_layout->set_out_layout_strategy(sub_strategy_vec);
      }
      if (out_layout->InitFromExtendVector(device_matrix_vector, tensor_map_vector, outputs_shape[i],
                                           interleaved_parallel) != SUCCESS) {
        MS_LOG(ERROR) << "The out_layout configured incorrect, device_matrix:" << device_matrix_vector
                      << ", tensor_map:" << tensor_map_vector;
        return FAILED;
      }
      out_tensor_layouts->push_back(out_layout);
    }
  }
  return SUCCESS;
}

Status ConvertValueToTensorLayout(const ValuePtr &layout_value, const ShapeBasePtr &shape,
                                  TensorLayoutBasePtr *out_tensor_layout) {
  if (layout_value->isa<ValueSequence>()) {
    if (!shape->is_list()) {
      MS_LOG(ERROR) << "layout is tuple in tuple, but shape is not. Please check the input shape or layout";
      return FAILED;
    }
    const auto &layout_value_tuple = layout_value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(layout_value_tuple);
    auto layout_value_vector = layout_value_tuple->value();
    std::vector<TensorLayoutBasePtr> layout_list;
    for (size_t i = 0; i < layout_value_vector.size(); ++i) {
      TensorLayoutBasePtr local_tensor_layout;
      if (ConvertValueToTensorLayout(layout_value_vector[i], shape->GetElement(SizeToLong(i)), &local_tensor_layout) !=
          SUCCESS) {
        return FAILED;
      }
      layout_list.push_back(local_tensor_layout);
    }
    *out_tensor_layout = std::make_shared<TensorLayoutList>(layout_list);
  } else {
    if (shape->is_list()) {
      MS_LOG(ERROR) << "layout is not a tuple, but shape is tuple. Please check the input shape or layout";
      return FAILED;
    }
    std::vector<int64_t> device_matrix_vector;
    std::vector<std::vector<int64_t>> tensor_map_vector;
    std::vector<std::string> alias_name;
    bool interleaved_parallel;
    if (GetLayoutFromAttrValue(layout_value, &alias_name, &device_matrix_vector, &tensor_map_vector,
                               &interleaved_parallel) != SUCCESS) {
      return FAILED;
    }
    auto in_layout = std::make_shared<TensorLayout>();
    if (in_layout->InitFromExtendVector(device_matrix_vector, tensor_map_vector, shape->GetValue(),
                                        interleaved_parallel) != SUCCESS) {
      MS_LOG(ERROR) << "The in_layout configured incorrect, device_matrix:" << device_matrix_vector
                    << ", tensor_map:" << tensor_map_vector;
      return FAILED;
    }
    *out_tensor_layout = std::make_shared<TensorLayoutValue>(in_layout);
  }
  return SUCCESS;
}

bool CheckNoShape(const ShapeBasePtr &shape) {
  if (shape->is_list()) {
    for (size_t i = 0; i < shape->size(); ++i) {
      if (shape->GetElement(SizeToLong(i))->size() == 0) {
        return true;
      }
    }
  }
  return shape->size() == 0;
}

Status ConvertValueTupleToTensorLayoutVector(const ValueTuplePtr &in_layout_value, const NewShapes &inputs_shape,
                                             std::vector<TensorLayoutBasePtr> *out_tensor_layouts) {
  MS_EXCEPTION_IF_NULL(in_layout_value);
  std::vector<ValuePtr> layout_value_vector = in_layout_value->value();
  if (inputs_shape.size() < layout_value_vector.size()) {
    MS_LOG(ERROR) << "The in_layout configured for node is smaller than its input nums";
    return FAILED;
  }

  for (size_t i = 0; i < layout_value_vector.size(); ++i) {
    auto layout_item = layout_value_vector[i];
    if (CheckNoShape(inputs_shape[i])) {
      MS_LOG(INFO) << "The input " << i << " is a value tuple or scalar, set a dummy layout. is_list is "
                   << inputs_shape[i]->is_list();
      if (inputs_shape[i]->is_list()) {
        out_tensor_layouts->push_back(std::make_shared<TensorLayoutList>(inputs_shape[i]->size()));
      } else {
        out_tensor_layouts->push_back(std::make_shared<TensorLayoutValue>());
      }
      continue;
    }
    TensorLayoutBasePtr tensor_layouts;
    if (ConvertValueToTensorLayout(layout_item, inputs_shape[i], &tensor_layouts) != SUCCESS) {
      return FAILED;
    }
    out_tensor_layouts->push_back(tensor_layouts);
  }
  // Clip the layout to the size of inputs_shape, the remained input will set to no shard.
  if (layout_value_vector.size() < inputs_shape.size()) {
    MS_LOG(INFO) << "The in_layout configured for node number is " << layout_value_vector.size()
                 << ", which is smaller than its input nums " << inputs_shape.size() << ", set a dummy layout.";
  }
  for (size_t i = layout_value_vector.size(); i < inputs_shape.size(); ++i) {
    if (inputs_shape[i]->is_list()) {
      out_tensor_layouts->push_back(std::make_shared<TensorLayoutList>(inputs_shape[i]->size()));
    } else {
      out_tensor_layouts->push_back(std::make_shared<TensorLayoutValue>());
    }
  }
  return SUCCESS;
}

Status ExtractUserConfigLayoutForNewShape(const mindspore::HashMap<std::string, ValuePtr> &prim_attrs,
                                          const NewShapes &inputs_shape, const NewShapes &outputs_shape,
                                          std::vector<TensorLayoutBasePtr> *in_tensor_layouts,
                                          std::vector<TensorLayoutBasePtr> *out_tensor_layouts) {
  if (prim_attrs.count(IN_LAYOUT) > 0) {
    auto layout_value = prim_attrs.at(IN_LAYOUT);
    if (!layout_value->isa<ValueSequence>()) {
      MS_LOG(ERROR) << "The in_layout configured for node is not a tuple";
      return FAILED;
    }
    auto layout_value_tuple = layout_value->cast<ValueTuplePtr>();
    if (ConvertValueTupleToTensorLayoutVector(layout_value_tuple, inputs_shape, in_tensor_layouts) != SUCCESS) {
      return FAILED;
    }
  }
  if (prim_attrs.count(OUT_LAYOUT) > 0) {
    auto layout_value = prim_attrs.at(OUT_LAYOUT);
    if (!layout_value->isa<ValueSequence>()) {
      MS_LOG(EXCEPTION) << "The out_layout configured for node is not a tuple";
    }
    auto layout_value_tuple = layout_value->cast<ValueTuplePtr>();
    if (ConvertValueTupleToTensorLayoutVector(layout_value_tuple, outputs_shape, out_tensor_layouts) != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

static bool IsCohesiveNode(const CNodePtr &cnode) {
  return IsPrimitiveCNode(cnode, prim::kPrimCast) || IsPrimitiveCNode(cnode, prim::kPrimLoad) ||
         IsPrimitiveCNode(cnode, prim::kPrimDepend) || IsPrimitiveCNode(cnode, prim::kPrimAllGather) ||
         IsPrimitiveCNode(cnode, prim::kPrimMiniStepAllGather) || IsPrimitiveCNode(cnode, prim::kPrimMirrorMicroStep) ||
         IsPrimitiveCNode(cnode, prim::kPrimMicroStepAllGather) || IsPrimitiveCNode(cnode, prim::kPrimMirror) ||
         IsPrimitiveCNode(cnode, prim::kPrimMirrorMiniStep) || IsPrimitiveCNode(cnode, prim::kPrimVirtualDiv) ||
         IsPrimitiveCNode(cnode, prim::kPrimInsertGradientOf) || IsPrimitiveCNode(cnode, prim::kPrimDumpGradient);
}

ParameterMap NodeParameterName(const CNodePtr &node, int64_t index, size_t curr_depth) {
  if (curr_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(WARNING) << "When finding the parameters' name of a operator, exceeded the maximum depth: "
                    << MAX_RECURSIVE_DEPTH;
    return {};
  }
  bool only_trainable_params = ParallelContext::GetInstance()->stra_file_only_trainable_params();
  std::vector<AnfNodePtr> node_inputs{node->inputs()};
  ParameterMap param_names;
  for (int64_t i = 0; i < UlongToLong(node_inputs.size()); ++i) {
    int64_t idx = index > i ? index : i;
    auto input = node_inputs[LongToSize(i)];
    if (input->isa<Parameter>()) {
      auto input_parameter = input->cast<ParameterPtr>();
      if (input_parameter->has_default() && (!only_trainable_params || ParameterRequireGrad(input_parameter))) {
        (void)param_names.emplace_back(std::make_pair(input_parameter->name(), input_parameter));
        continue;
      }
      auto actual_param_node = RefParameterToActualParameter(input_parameter);
      if (!actual_param_node) {
        continue;
      }
      auto actual_param = actual_param_node->cast<ParameterPtr>();
      if (!only_trainable_params || ParameterRequireGrad(actual_param)) {
        (void)param_names.emplace_back(std::make_pair(actual_param->name(), actual_param));
      }
    } else if (input->isa<CNode>()) {
      CNodePtr cnode = input->cast<CNodePtr>();
      if (!IsValueNode<Primitive>(cnode->input(0))) {
        continue;
      }
      if (IsCohesiveNode(cnode) && cnode->size() >= 1) {
        auto input_param_names = NodeParameterName(cnode, idx, 0);
        (void)param_names.insert(param_names.cend(), input_param_names.cbegin(), input_param_names.cend());
      }
    }
  }
  return param_names;
}

Status ParallelInit(size_t rank_id, const size_t devices) {
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  int32_t split_stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();

  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  if (split_stage_num <= 0) {
    MS_LOG(ERROR) << "The parameter 'split_stage_num' must be a positive number, but got the value : "
                  << split_stage_num;
    return FAILED;
  }
  int64_t device_num;
  int64_t global_rank;
  std::string backend;
  if (devices == 0) {
    auto comm_info = GetCommInfo();
    device_num = comm_info.device_num;
    global_rank = comm_info.global_rank;
    backend = comm_info.communication_backend;
  } else {
    device_num = devices;
    global_rank = rank_id;
    backend = HCCL_BACKEND;
  }

  if ((device_num <= 0) || (device_num > MAX_DEVICE_NUM)) {
    MS_LOG(ERROR) << "The context configuration parameter 'device_num' must be positive, "
                     "but got the value of device_num: "
                  << device_num;
    return FAILED;
  }

  // the device_num maybe get from communication interface
  if (device_num % split_stage_num != 0) {
    MS_LOG(ERROR) << "The parameter 'device_num' must be divided by 'split_stage_num', but got the device_num : "
                  << device_num << "and the split_stage_num : " << split_stage_num;
    return FAILED;
  }

  int64_t optimizer_weight_shard_size = ParallelContext::GetInstance()->optimizer_weight_shard_size();
  if (ParallelContext::GetInstance()->enable_parallel_optimizer() && optimizer_weight_shard_size > 0 &&
      device_num < optimizer_weight_shard_size) {
    MS_LOG(ERROR) << "When parallel_optimizer is enabled, the optimizer_weight_shard_size "
                  << optimizer_weight_shard_size << " should not exceed the device num " << device_num << ".";
    return FAILED;
  }

  if ((global_rank < 0) || (global_rank >= device_num)) {
    MS_LOG(ERROR) << "The parameter 'global_rank' must be  greater than 0 and less equal 'device num', "
                     "but got the global_rank : "
                  << global_rank << "and the device_num : " << device_num;
    return FAILED;
  }

  std::vector<int64_t> stages;
  for (int i = 0; i < split_stage_num; i++) {
    stages.push_back(device_num / split_stage_num);
  }

  bool use_rec = (ParallelContext::GetInstance()->strategy_search_mode() == kRecursiveProgramming);
  bool use_sp = (ParallelContext::GetInstance()->strategy_search_mode() == kShardingPropagation) ||
                (ParallelContext::GetInstance()->sharding_propagation());
  if ((split_stage_num > 1) && (parallel_mode == kAutoParallel) && !(use_sp || use_rec)) {
    MS_LOG(ERROR) << "To enable the pipeline parallel, please set the parallel mode to " << kSemiAutoParallel << " or "
                  << kAutoParallel << " with " << kShardingPropagation << " or " << kRecursiveProgramming;
    return FAILED;
  }

  if (!InitDevice(device_num, global_rank, backend, stages)) {
    MS_LOG(ERROR) << "Init device failed";
    return FAILED;
  }

  MS_LOG(INFO) << "The parallel context: device_num: " << device_num << ", global_rank: "
               << global_rank
               //               << ", communication_backend: " << comm_info.communication_backend
               << ", communication_backend: " << HCCL_BACKEND
               << ", gradients_mean: " << ParallelContext::GetInstance()->gradients_mean()
               << ", gradient_fp32_sync: " << ParallelContext::GetInstance()->gradient_fp32_sync();
  return SUCCESS;
}

// only used for FindCNode
static CNodePtr SkipTrivialNodesMoveDown(const FuncGraphManagerPtr &manager, CNodePtr node) {
  MS_EXCEPTION_IF_NULL(node);
  while (IsInTrivialNodeList(node) || IsSomePrimitive(node, LOAD)) {
    node = manager->node_users()[node].begin()->first->cast<CNodePtr>();
  }
  return node;
}

std::pair<bool, CNodePtr> FindCNode(const AnfNodePtr &anode, const std::string &name, const FuncGraphPtr &func_graph,
                                    size_t max_depth) {
  MS_EXCEPTION_IF_NULL(anode);
  MS_EXCEPTION_IF_NULL(anode->func_graph());
  FuncGraphManagerPtr manager = anode->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (max_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG_WITH_NODE(EXCEPTION, anode) << "Recursive call is larger than 100000.";
  }
  AnfNodeIndexSet node_set = manager->node_users()[anode];
  bool result = false;
  CNodePtr cnode_return = nullptr;
  for (auto &node_pair : node_set) {
    CNodePtr use_apply = node_pair.first->cast<CNodePtr>();
    if (use_apply == nullptr || !IsValueNode<Primitive>(use_apply->input(0))) {
      continue;
    }
    use_apply = SkipTrivialNodesMoveDown(manager, use_apply);
    if (use_apply == nullptr || !IsValueNode<Primitive>(use_apply->input(0))) {
      continue;
    }
    ValueNodePtr prim_anf_node = use_apply->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_anf_node);
    PrimitivePtr node_prim = prim_anf_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(node_prim);
    if (node_prim->name() == name && node_pair.second == 1) {
      if (use_apply->func_graph() == func_graph) {
        result = true;
        cnode_return = use_apply;
        MS_LOG(INFO) << "Find Primitive " << name << " in the same func_graph";
        continue;
      }
      MS_LOG(INFO) << "Find Primitive " << name << " in different func_graph";
    }
    if (ParallelContext::GetInstance()->enable_parallel_optimizer() && IsInAllGatherNodeList(use_apply)) {
      return FindCNode(node_pair.first, name, func_graph, max_depth + 1);
    }
  }
  return std::make_pair(result, cnode_return);
}

void SetSharedParameterFlag(const FuncGraphPtr &root, const AnfNodePtr &parameter) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(parameter);
  FuncGraphManagerPtr manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  ParameterPtr parameter_ptr = parameter->cast<ParameterPtr>();
  if (parameter_ptr == nullptr) {
    MS_LOG(INFO) << parameter->ToString() << ": cast to ptr failed. it may not be a parameter";
    return;
  }
  auto user_set = manager->node_users()[parameter];
  int32_t user_count = 0;
  for (auto &param_pair : user_set) {
    CNodePtr cnode = param_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->in_forward_flag()) {
      user_count++;
    }
  }
  if (user_count > 1) {
    auto tensor_layout = parameter_ptr->user_data<TensorLayout>();
    tensor_layout->set_is_shared_param(true);
  }
}

StrategyPtr GenerateBatchParallelStrategy(const OperatorInfoPtr operator_, const PrimitivePtr prim) {
  MS_EXCEPTION_IF_NULL(operator_);
  MS_EXCEPTION_IF_NULL(prim);
  if (!operator_->inputs_shape_new().empty()) {
    MS_LOG(EXCEPTION) << "Currently, tuple in tuple input does not support GenerateBatchParallelStrategy, please set "
                         "strategy in python side";
  }
  StrategyPtr strategyPtr;
  std::shared_ptr<Strategies> strategy_v_ptr = operator_->GenerateBatchStrategiesWithCheck();
  MS_EXCEPTION_IF_NULL(strategy_v_ptr);
  auto stage_id = g_device_manager->stage_id();
  strategyPtr = NewStrategy(stage_id, *strategy_v_ptr);
  std::vector<ValuePtr> elements;
  for (size_t i = 0; i < strategy_v_ptr->size(); i++) {
    elements.push_back(MakeValue((*strategy_v_ptr)[i]));
  }
  ValueTuplePtr strategy = std::make_shared<ValueTuple>(elements);
  // display the strategy generated by batch parallel
  auto attrs = prim->attrs();
  attrs[GEN_STRATEGY] = strategy;
  (void)prim->SetAttrs(attrs);
  MS_LOG(INFO) << "prim " << prim->name() << " batch parallel strategy is " << attrs[GEN_STRATEGY]->ToString();
  return strategyPtr;
}

StrategyPtr GenerateStandAloneStrategy(const Shapes &inputs_shape) {
  Strategies strategy_v;
  for (size_t i = 0; i != inputs_shape.size(); i++) {
    if (inputs_shape[i].empty()) {
      MS_LOG(INFO) << "Elements of shapes is empty.";
      Dimensions empty_element;
      strategy_v.push_back(empty_element);
    } else {
      Dimensions element(inputs_shape[i].size(), 1);
      strategy_v.push_back(element);
    }
  }
  auto stage_id = g_device_manager->stage_id();
  auto stra_ptr = NewStrategy(stage_id, strategy_v);
  return stra_ptr;
}

ShapeBasePtr GenerateStra(const ShapeBasePtr &shape) {
  ShapeBasePtr out_shape;
  if (shape->is_list()) {
    std::vector<ShapeBasePtr> list_stra;
    for (size_t i = 0; i < shape->size(); ++i) {
      auto recursive_stra = GenerateStra(shape->GetElement(SizeToLong(i)));
      list_stra.emplace_back(recursive_stra);
    }
    out_shape = std::make_shared<ShapeList>(list_stra);
  } else {
    if (shape->empty()) {
      MS_LOG(INFO) << "Elements of shapes is empty.";
      Dimensions empty_element;
      out_shape = std::make_shared<ShapeValue>(empty_element);
    } else {
      Dimensions element(shape->size(), 1);
      out_shape = std::make_shared<ShapeValue>(element);
    }
  }
  return out_shape;
}

StrategyPtr GenerateStandAloneStrategyForNewShapes(const NewShapes &inputs_shape) {
  NewStrategies strategy_v;
  for (size_t i = 0; i != inputs_shape.size(); i++) {
    strategy_v.emplace_back(GenerateStra(inputs_shape[i]));
  }
  auto stage_id = g_device_manager->stage_id();
  auto stra_ptr = NewStrategy(stage_id, strategy_v);
  return stra_ptr;
}

bool IsInsertVirtualOutput(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  auto comm_info = GetCommInfo();
  int64_t split_stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();
  int64_t per_stage_device_num = comm_info.device_num / split_stage_num;
  int64_t current_stage = comm_info.global_rank / per_stage_device_num;
  MS_LOG(INFO) << "The current stage is: " << current_stage;
  if (!root->has_flag(kTraining) && !ParallelContext::GetInstance()->dataset_strategy().empty()) {
    MS_LOG(WARNING) << "In eval/predict net, the output parallel strategy would not follow "
                       "the input parallel strategy when using context.set_auto_parallel_context(dataset_strategy)"
                       " to configure the input strategy.";
  }
  return ((!root->has_flag(kTraining) && ParallelContext::GetInstance()->dataset_strategy().empty() &&
           current_stage == split_stage_num - 1));
}

TensorLayout GetInputLayoutFromCNode(const std::pair<AnfNodePtr, int64_t> &node_pair, const int &make_tuple_index) {
  CNodePtr cnode = node_pair.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  OperatorInfoPtr distribute_operator = GetDistributeOperator(cnode);
  MS_EXCEPTION_IF_NULL(distribute_operator);
  int64_t index = node_pair.second;
  TensorLayout tensorlayout_in;
  if (distribute_operator->inputs_tensor_info_new().empty()) {
    if (index > SizeToLong(distribute_operator->inputs_tensor_info().size())) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "The index is out of range, the node_pair.second is  " << (index - 1)
                                         << ", the vector size is  "
                                         << distribute_operator->inputs_tensor_info().size();
    }
    TensorInfo tensorinfo_in = distribute_operator->inputs_tensor_info()[LongToSize(index - 1)];
    tensorlayout_in = tensorinfo_in.tensor_layout();
  } else {
    if (index > SizeToLong(distribute_operator->inputs_tensor_info_new().size())) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "The index is out of range, the node_pair.second is  " << (index - 1)
                                         << ", the vector size is  "
                                         << distribute_operator->inputs_tensor_info_new().size();
    }
    auto tensorinfo_in = distribute_operator->inputs_tensor_info_new()[LongToSize(index - 1)];
    if (tensorinfo_in->is_list() && make_tuple_index != -1) {
      auto new_tensorinfo_in = tensorinfo_in->GetElement(make_tuple_index - 1);
      tensorlayout_in = new_tensorinfo_in->GetValue().tensor_layout();
    } else if (!tensorinfo_in->is_list() && make_tuple_index == -1) {
      tensorlayout_in = tensorinfo_in->GetValue().tensor_layout();
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "tensorinfo_in does not match with make_tuple_index: make_tuple_index is "
                                         << make_tuple_index << ", node is " << node_pair.first->DebugString();
    }
  }
  return tensorlayout_in;
}

bool IsCellReuseForwardGraph(const FuncGraphPtr &graph) { return graph->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE); }

FuncGraphPtr GetCellReuseBackwardGraph(const FuncGraphPtr &forward_graph) {
  AnfNodePtr node = forward_graph->get_return();
  std::vector<std::pair<PrimitivePtr, int64_t>> patterns = {
    {prim::kPrimReturn, kIndex1}, {prim::kPrimMakeTuple, kIndex2}, {prim::kPrimPartial, kIndex1}};
  for (const auto &pattern : patterns) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || !IsPrimitiveCNode(cnode, pattern.first)) {
      return nullptr;
    }
    auto prev_node_index = pattern.second;
    if (prev_node_index >= SizeToLong(cnode->size())) {
      return nullptr;
    }
    node = cnode->input(prev_node_index);
  }
  return GetValueNode<FuncGraphPtr>(node);
}

Shape mirror_group_list(const TensorLayoutPtr &layout) {
  int64_t rank = g_device_manager->global_rank();
  auto stage_dev_list = g_device_manager->GetDeviceListInThisStage();
  DeviceMatrix dev_matrix(rank, stage_dev_list, layout->device_arrangement().array());
  RankList group_devices;
  if (dev_matrix.GetDevicesByTensorMap(layout->tensor_map().array(), &group_devices) != SUCCESS) {
    MS_LOG(EXCEPTION) << "For layout:" << layout->ToString() << ", infer mirror failed";
  }
  return group_devices;
}

std::string GetSerialNumberString(size_t number) {
  std::string suffix = "th";
  if (number == kSizeOne) {
    suffix = "st";
  } else if (number == kSizeTwo) {
    suffix = "nd";
  } else if (number == kSizeThree) {
    suffix = "rd";
  }
  std::ostringstream oss;
  oss << number << suffix;
  return oss.str();
}

// Get single device capacity in Go
size_t GetDeviceCapacity() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  size_t size_from_context;
  auto max_device_memory = runtime::RuntimeConf::GetInstance()->mem_max_size();
  float total_device_memory = 32.0f;
  if (context->ascend_soc_version() == kAscendVersion910b || context->ascend_soc_version() == kAscendVersion910_93) {
    total_device_memory = 64.0f;
  }
  if (max_device_memory <= total_device_memory) {
    MS_LOG(DEBUG) << "context max_device_memory:" << max_device_memory;
    size_from_context = FloatToSize(max_device_memory * kGBToByte);
  } else {
    auto variable_memory_max_size = context->get_param<std::string>(MS_CTX_VARIABLE_MEMORY_MAX_SIZE);
    if (variable_memory_max_size == "0") {
      return 0;
    }
    MS_LOG(DEBUG) << "context variable_memory_max_size:" << variable_memory_max_size;
    auto pos = variable_memory_max_size.find('*');
    if (pos == std::string::npos) {
      MS_LOG(EXCEPTION) << "Invalid variable_memory_max_size";
    }
    auto gb_str = variable_memory_max_size.substr(0, pos);
    auto gb_var = std::stoull(gb_str);
    MS_LOG(DEBUG) << "variable_memory_max_size(GB):" << gb_var;
    size_from_context = gb_var * kGBToByte;
  }
  return size_from_context;
}

abstract::AbstractBasePtr GenerateAbsByOpInfer(const PrimitivePtr &primitive, const AnfNodePtrList &input_list) {
  MS_EXCEPTION_IF_NULL(primitive);
  std::vector<AbstractBasePtr> input_args;
  (void)std::for_each(input_list.begin(), input_list.end(),
                      [&input_args](const auto &input) { (void)input_args.emplace_back(input->abstract()); });
  auto abs_opt = abstract::TryInferAbstract(primitive, input_args);
  if (!abs_opt.has_value()) {
    MS_LOG(EXCEPTION) << primitive->name() << " infer is not registered.";
  }
  auto abs = abs_opt.value();
  MS_EXCEPTION_IF_NULL(abs);
  MS_LOG(DEBUG) << "Abstract for " << primitive->name() << " is " << abs->ToString();
  return abs;
}

StrategyPtr GenerateStandAloneStra(const OperatorInfoPtr &op_info, const bool use_new_shape_base) {
  MS_EXCEPTION_IF_NULL(op_info);
  StrategyPtr in_strategy;
  if (!op_info->inputs_shape_new().empty() || use_new_shape_base) {
    in_strategy = GenerateStandAloneStrategyForNewShapes(op_info->inputs_shape_new());
  } else {
    in_strategy = GenerateStandAloneStrategy(op_info->inputs_shape());
  }
  return in_strategy;
}

void CheckStrategyAndShape(const StrategyPtr &in_strategy, const OperatorInfoPtr &op_info) {
  MS_EXCEPTION_IF_NULL(op_info);
  MS_EXCEPTION_IF_NULL(in_strategy);
  auto has_tuple_stra = in_strategy->HasTupleInTupleStrategy();
  auto has_new_shape = !op_info->inputs_shape_new().empty();
  if (has_tuple_stra != has_new_shape) {
    MS_LOG(EXCEPTION)
      << "For " << op_info->name()
      << ": One of the strategy or input shape have tuple in tuple input, but the other does not; in_strategy is "
      << has_tuple_stra << ", input shape is " << has_new_shape;
  }
}

void PaddingStrategy(const OperatorInfoPtr &op_info, const bool is_new_shape_base_node, StrategyPtr *in_strategy) {
  MS_EXCEPTION_IF_NULL(op_info);
  if (is_new_shape_base_node) {
    auto standalone_stra = GenerateStandAloneStra(op_info, is_new_shape_base_node)->GetInputNewDim();
    auto stra = (*in_strategy)->GetInputNewDim();
    std::vector<NewDimensions> new_stra;
    if (stra.size() <= op_info->inputs_shape_new().size()) {
      MS_LOG(INFO) << "The strategy size " << stra.size() << " is smaller than actual input size(contain attr) "
                   << op_info->inputs_shape_new().size() << ", the remained input will set to no split.";
      // For no shape like scalar / None, set a standalone stra
      MS_LOG(DEBUG) << "Before padding, the strategy is " << (*in_strategy)->ToString() << ", size is "
                    << (*in_strategy)->GetInputNewDim().size();
      size_t real_idx = 0;
      for (size_t i = 0; i < op_info->inputs_shape_new().size(); ++i) {
        if (op_info->inputs_shape_new()[i]->size() == 0) {
          MS_LOG(DEBUG) << "The " << i << "th input shape is no shape set to standalone strategy";
          new_stra.push_back(standalone_stra[i]);
        } else {
          if (real_idx >= stra.size()) {
            break;
          }
          new_stra.push_back(stra[real_idx]);
          ++real_idx;
        }
      }
      if (new_stra.size() < op_info->inputs_shape_new().size()) {
        (void)new_stra.insert(new_stra.end(), standalone_stra.begin() + SizeToLong(real_idx), standalone_stra.end());
      }
      (*in_strategy)->ResetInputs(new_stra);
      MS_LOG(DEBUG) << "After padding, the new strategy is " << (*in_strategy)->ToString() << ", size is "
                    << (*in_strategy)->GetInputNewDim().size();
    } else {
      MS_LOG(EXCEPTION) << "For op " << op_info->name()
                        << ": its strategy size is larger than input shape size. The strategy size is " << stra.size()
                        << ", but the input shape size is " << op_info->inputs_shape_new().size();
    }
  }
}

void ObtainInOutStrategy(const StrategyMap &stra_map, const PrimitivePtr &prim, const std::string &strategy_key_name,
                         const CNodePtr &cnode, const OperatorInfoPtr &op_info, const bool is_new_shape_base_node,
                         StrategyPtr *in_strategy, StrategyPtr *out_strategy) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(op_info);
  auto attrs = prim->attrs();
  bool load_strategy_from_ckpt =
    StrategyCheckpoint::GetInstance().LoadCheckPointOn() && stra_map.find(strategy_key_name) != stra_map.end();
  if (!prim->HasAttr(STAND_ALONE)) {
    if (((!StrategyFound(attrs) && !load_strategy_from_ckpt) && !cnode->HasPrimalAttr(IN_STRATEGY)) ||
        prim->HasAttr(BATCH_PARALLEL)) {
      MS_LOG(INFO) << "ExtractInformation: the strategy of node " << cnode->ToString() << " prim " << prim->name()
                   << " is empty, using batch parallel";
      *in_strategy = GenerateBatchParallelStrategy(op_info, prim);
    } else if (cnode->HasPrimalAttr(IN_STRATEGY)) {
      *in_strategy = ExtractStrategy(cnode->GetPrimalAttr(IN_STRATEGY), is_new_shape_base_node);
      *out_strategy = ExtractStrategy(cnode->GetPrimalAttr(OUT_STRATEGY), is_new_shape_base_node);
    } else if (StrategyFound(attrs)) {
      *in_strategy = ExtractStrategy(attrs[IN_STRATEGY], is_new_shape_base_node);
      *out_strategy = ExtractStrategy(attrs[OUT_STRATEGY], is_new_shape_base_node);
    } else {
      *in_strategy = stra_map.at(strategy_key_name);
    }
  } else {
    *in_strategy = GenerateStandAloneStra(op_info, is_new_shape_base_node);
  }
  CheckStrategyAndShape(*in_strategy, op_info);
}

TensorLayouts GetLossNodeGradOutputLayout(const LossNodeInfo &node_info) {
  TensorLayouts ret;
  auto loss_cnode = node_info.loss_node;
  MS_EXCEPTION_IF_NULL(loss_cnode);

  ValueNodePtr prim_anf_node = loss_cnode->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(prim_anf_node);
  PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(prim);

  if (INVALID_LOSS_OPS.find(prim->name()) != INVALID_LOSS_OPS.end()) {
    MS_LOG(WARNING) << "The loss name is: " << prim->name() << ", do nothing for split sens now";
    return ret;
  }

  OperatorInfoPtr operator_info = loss_cnode->user_data<OperatorInfo>();
  if (!operator_info) {
    return ret;
  }

  auto op_output_size = operator_info->outputs_tensor_info_new().empty()
                          ? operator_info->outputs_tensor_info().size()
                          : operator_info->outputs_tensor_info_new().size();
  MS_LOG(INFO) << "The loss name is " << operator_info->name() << ", the has tuple item is  "
               << node_info.has_tuple_getitem << ", the output size is  " << op_output_size << ", the dout_index is  "
               << node_info.dout_index;

  if ((op_output_size == 0) || (op_output_size <= LongToSize(node_info.dout_index))) {
    MS_LOG_WITH_NODE(EXCEPTION, loss_cnode)
      << "The index is  " << node_info.dout_index << ", but the size of outputs is  " << op_output_size;
  }

  if (!node_info.has_tuple_getitem && (op_output_size > 1)) {
    MS_LOG_WITH_NODE(EXCEPTION, loss_cnode) << "Currently, it is not supported that the sens is a tuple.";
  }

  if (operator_info->outputs_tensor_info_new().empty()) {
    auto loss_grad_tensor_info = operator_info->outputs_tensor_info()[LongToSize(node_info.dout_index)];
    ret.push_back(loss_grad_tensor_info.tensor_layout());
  } else {
    auto loss_grad_tensor_info = operator_info->outputs_tensor_info_new()[LongToSize(node_info.dout_index)];
    if (loss_grad_tensor_info->is_list()) {
      MS_LOG_WITH_NODE(EXCEPTION, loss_cnode) << "For" << operator_info->name() << ": the" << node_info.dout_index
                                              << " out tensorinfo is a list, which does not support yet";
    }
    ret.push_back(loss_grad_tensor_info->GetValue().tensor_layout());
  }
  return ret;
}

LossNodeInfo FindLossCNode(const FuncGraphPtr &func_graph) {
  LossNodeInfo loss_node_info;
  MS_EXCEPTION_IF_NULL(func_graph);
  CNodePtr return_node = func_graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  if (return_node->size() <= 1) {
    MS_LOG_WITH_NODE(EXCEPTION, return_node) << "Failure: " << return_node->DebugString() << " size is smaller than 2";
  }
  auto pre_node_pair = GetRealKernelNode(return_node->input(1), -1, nullptr);
  auto pre_node = pre_node_pair.first;
  MS_EXCEPTION_IF_NULL(pre_node);
  auto pre_cnode = pre_node->cast<CNodePtr>();
  if (pre_cnode == nullptr || !IsValueNode<Primitive>(pre_cnode->input(0))) {
    return loss_node_info;
  }
  if (!IsValueNode<Primitive>(pre_cnode->input(0))) {
    MS_LOG(DEBUG) << "pre_cnode:" << pre_cnode->ToString();
    return loss_node_info;
  }
  auto current_prim = GetValueNode<PrimitivePtr>(pre_cnode->input(0));
  // notice: the GetNext op has not input
  MS_EXCEPTION_IF_NULL(current_prim);
  if (INVALID_LOSS_OPS.find(current_prim->name()) != INVALID_LOSS_OPS.end()) {
    MS_LOG(INFO) << "The loss is: " << current_prim->name();
    loss_node_info.loss_node = pre_cnode;
    return loss_node_info;
  }

  // return -> tuple_getitem -> loss
  if (pre_node_pair.second != -1) {
    loss_node_info.has_tuple_getitem = true;
    loss_node_info.dout_index = pre_node_pair.second;
    loss_node_info.loss_node = pre_cnode;
    return loss_node_info;
  }

  // return -> make_tuple
  if (current_prim->name() == MAKE_TUPLE) {
    // functional parallelism return multiple result, including label, which donot need grad
    if (std::find_if(pre_cnode->inputs().begin(), pre_cnode->inputs().end(), [](const auto &input) {
          return IsPrimitiveCNode(input, prim::kPrimStopGradient);
        }) != pre_cnode->inputs().end()) {
      return loss_node_info;
    }
    loss_node_info.has_make_tuple = true;
    loss_node_info.dout_index = -1;
    loss_node_info.loss_node = pre_cnode;
    return loss_node_info;
  }

  // return -> loss
  loss_node_info.loss_node = pre_cnode;
  MS_LOG(DEBUG) << "The loss name is " << current_prim->name();
  loss_node_info.loss_node->AddPrimalAttr(FLASH_LOSS_NODE, MakeValue<std::string>("True"));
  return loss_node_info;
}

static std::vector<AnfNodePtr> FindRootForwardCNode(const FuncGraphPtr &graph,
                                                    const std::vector<AnfNodePtr> &all_nodes) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> root_forward_nodes;
  auto loss_cnode = FindLossCNode(graph).loss_node;
  if (loss_cnode == nullptr) {
    return root_forward_nodes;
  }

  auto loss_cnode_id = loss_cnode->UniqueIdThroughCopy();
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto root_node_id = node->UniqueIdThroughCopy();
    if (loss_cnode_id == root_node_id) {
      root_forward_nodes = DeepLinkedGraphSearch(cnode);
      break;
    }
  }
  return root_forward_nodes;
}

static void SetForwardFlag(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsValueNode<FuncGraph>(cnode->input(0))) {
      cnode->set_in_forward_flag(true);
      continue;
    }
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }

    // CNode is globally unique.
    MS_LOG(DEBUG) << "Set forward flag " << cnode->DebugString() << ".";
    cnode->set_in_forward_flag(true);
  }
}

static void SetForwardFlag(const AnfNodeSet &all_nodes) {
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }

    // CNode is globally unique.
    cnode->set_in_forward_flag(true);
  }
}

void MarkForwardCNode(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto ret = root->get_return();
  MS_EXCEPTION_IF_NULL(ret);
  auto all_nodes = TopoSort(ret, SuccDeeperSimple);
  auto graph_set = FindForwardGraphByRootNodes(all_nodes);
  if (graph_set.empty()) {
    MS_LOG(INFO) << "Can not find the forward graph, so mark the ops in root graph";
    auto fgs = root->manager()->func_graphs();
    for (auto fg = fgs.cbegin(); fg != fgs.cend(); ++fg) {
      SetForwardFlag((*fg)->nodes());
    }
  } else {
    for (auto func_graph = graph_set.cbegin(); func_graph != graph_set.cend(); ++func_graph) {
      MS_LOG(INFO) << "The sub graph size of root is " << root->func_graphs_used().size();
      auto return_node = (*func_graph)->get_return();
      MS_EXCEPTION_IF_NULL(return_node);
      auto all_dfs_nodes = DeepLinkedGraphSearch(return_node);
      SetForwardFlag(all_dfs_nodes);
      auto root_forward_nodes = FindRootForwardCNode(*func_graph, all_nodes);
      if (root_forward_nodes.empty()) {
        continue;
      }
      // Mark forward flag for the nodes in root graph.
      SetForwardFlag(root_forward_nodes);
    }
  }
}

static FuncGraphPtr PynativeParallelGraph(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  FuncGraphPtr real_graph = root;
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    auto expect_shard_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(expect_shard_prim);
    if (expect_shard_prim->name() != SHARD) {
      continue;
    }
    real_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
  }
  return real_graph;
}

Shapes ConvertDatasetLayoutToStrategy() {
  Shapes dataset_strategy;
  auto all_tensor_map = ParallelContext::GetInstance()->dataset_strategy_tensormap();
  auto all_dev_mat = ParallelContext::GetInstance()->dataset_strategy_devmat();
  for (size_t i = 0; i < all_tensor_map.size(); ++i) {
    Shape local_stra;
    auto cur_tensor_map = all_tensor_map.at(i);
    auto cur_dev_mat = all_dev_mat.at(i);
    for (size_t j = 0; j < cur_tensor_map.size(); ++j) {
      size_t shard_size = 1;
      for (size_t k = 0; k < cur_tensor_map.at(j).size(); ++k) {
        auto val = cur_tensor_map.at(j).at(k);
        if (val != -1) {
          auto real_idx = cur_dev_mat.size() - LongToSize(val) - 1;
          shard_size *= LongToSize(cur_dev_mat.at(real_idx));
        }
      }
      local_stra.push_back(shard_size);
    }
    dataset_strategy.push_back(local_stra);
  }
  return dataset_strategy;
}

void InsertVirtualOutput(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  auto real_graph = PynativeParallelGraph(root, all_nodes);
  MS_EXCEPTION_IF_NULL(real_graph);
  auto out_pair = GetRealKernelNode(real_graph->output(), -1, nullptr, false);
  auto out_node = out_pair.first;
  MS_EXCEPTION_IF_NULL(out_node);
  OperatorParams params;
  OperatorAttrs attrs;
  OperatorArgs args = std::make_pair(attrs, params);
  Operator op = std::make_pair(VIRTUAL_OUTPUT, args);
  bool full_batch = ParallelContext::GetInstance()->full_batch();
  int64_t dev_num = 1;
  if (!full_batch && ParallelContext::GetInstance()->dataset_strategy().empty()) {
    dev_num = g_device_manager->stage_device_num();
  }
  if (IsPrimitiveCNode(out_node, prim::kPrimMakeTuple)) {
    auto tuple = out_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple);
    for (size_t i = 1; i < tuple->size(); ++i) {
      auto cur_input = tuple->input(i);
      Shapes shape_outputs = GetNodeShape(cur_input);
      if (shape_outputs[0].empty() || shape_outputs[0][0] % dev_num != 0) {
        continue;
      }
      InsertNode(op, tuple, i, cur_input, tuple->func_graph(), VIRTUAL_OUTPUT);
      auto virtual_output_abstract = cur_input->abstract()->Clone();
      std::shared_ptr<abstract::BaseShape> virtual_output_shape = std::make_shared<abstract::Shape>(shape_outputs[0]);
      virtual_output_abstract->set_shape(virtual_output_shape);
      auto virtual_output_node = tuple->input(i);
      virtual_output_node->set_abstract(virtual_output_abstract);
    }
  } else {
    Shapes shape_outputs = GetNodeShape(out_node);
    if (shape_outputs[0].empty() || shape_outputs[0][0] % dev_num != 0 || out_node->isa<Parameter>()) {
      return;
    }
    auto node_input = CreateInput(op, out_node, VIRTUAL_OUTPUT);
    const auto out_cnode = out_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(out_cnode);
    auto cur_graph = out_cnode->func_graph();
    MS_EXCEPTION_IF_NULL(cur_graph);
    auto new_node = cur_graph->NewCNode(node_input);
    auto manager = cur_graph->manager();
    (void)manager->Replace(out_node, new_node);
    auto virtual_output_abstract = out_node->abstract()->Clone();
    std::shared_ptr<abstract::BaseShape> virtual_output_shape = std::make_shared<abstract::Shape>(shape_outputs[0]);
    virtual_output_abstract->set_shape(virtual_output_shape);
    new_node->set_abstract(virtual_output_abstract);
  }
}

AnfNodePtr ParamNodeForMoveMirror(const CNodePtr &micro_mirror) {
  return GetInputNodeWithFilter(micro_mirror, [&](const CNodePtr &cnode) {
    bool filter = IsPrimitiveCNode(cnode, prim::kPrimMirrorMicroStep) || IsPrimitiveCNode(cnode, prim::kPrimLoad) ||
                  IsPrimitiveCNode(cnode, prim::kPrimDepend) || IsPrimitiveCNode(cnode, prim::kPrimMicroStepAllGather);
    return std::make_pair(filter, 1);
  });
}

AnfNodePtr MoveSingeMirrorOutCallFunc(const CNodePtr &micro_mirror) {
  auto param_anf_node = ParamNodeForMoveMirror(micro_mirror);
  MS_EXCEPTION_IF_NULL(param_anf_node);
  if (!param_anf_node->isa<Parameter>()) {
    return nullptr;
  }
  auto param = param_anf_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param);
  if (param->has_default()) {
    return nullptr;
  }
  auto sub_func_graph = param_anf_node->func_graph();
  auto call_cnodes_map = sub_func_graph->func_graph_cnodes_index();
  auto sub_graph_parameters = sub_func_graph->parameters();
  auto curr_param_iter = std::find(sub_graph_parameters.begin(), sub_graph_parameters.end(), param_anf_node);
  if (curr_param_iter == sub_graph_parameters.end()) {
    MS_LOG_WITH_NODE(EXCEPTION, param_anf_node)
      << "Cannot find param " << param_anf_node->DebugString() << " in current sub_graph";
  }
  size_t curr_param_index = static_cast<size_t>(curr_param_iter - sub_graph_parameters.begin());
  AnfNodePtr call_nodes_common_param_input = nullptr;
  FuncGraphPtr call_nodes_func_graph = nullptr;
  for (const auto &node_pair : call_cnodes_map) {
    if (!node_pair.first->first->isa<CNode>() || node_pair.first->second > 0) {
      continue;
    }
    auto cnode = node_pair.first->first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    call_nodes_func_graph = cnode->func_graph();
    auto cnode_input = cnode->input(curr_param_index + 1);
    if (!call_nodes_common_param_input) {
      call_nodes_common_param_input = cnode_input;
    }
    if (call_nodes_common_param_input != cnode_input) {
      call_nodes_common_param_input = nullptr;
      break;
    }
  }
  if (!call_nodes_common_param_input || !call_nodes_func_graph) {
    return nullptr;
  }
  auto manager = call_nodes_func_graph->manager();
  // Insert new MicroMirror in root func
  if (!IsPrimitiveCNode(call_nodes_common_param_input, prim::kPrimMirrorMicroStep)) {
    auto new_mirror_node =
      NewMicroMirrorPrimByMicroMirror(call_nodes_func_graph, micro_mirror, call_nodes_common_param_input);
    for (const auto &node_pair : call_cnodes_map) {
      if (!node_pair.first->first->isa<CNode>() || node_pair.first->second > 0) {
        continue;
      }
      (void)manager->SetEdge(node_pair.first->first, curr_param_index + 1, new_mirror_node);
    }
    // Remove MicroMirror in call_func
    (void)manager->Replace(micro_mirror, micro_mirror->input(kIndex1));
    return new_mirror_node;
  }
  (void)manager->Replace(micro_mirror, micro_mirror->input(kIndex1));
  return nullptr;
}

int64_t LongAdd(int64_t base, int64_t shift) {
  int64_t result;
  if (shift > 0) {
    if (base > INT_MAX - shift) {
      result = INT_MAX;
    } else {
      result = base + shift;
    }
  } else {
    if (base < INT_MIN - shift) {
      result = INT_MIN;
    } else {
      result = base + shift;
    }
  }
  return result;
}

bool NeededHandleShardParam() {
  return ParallelContext::GetInstance()->enable_parallel_optimizer() &&
         ParallelContext::GetInstance()->pipeline_interleave() && !ParallelContext::GetInstance()->zero3() &&
         !ParallelContext::GetInstance()->grad_accumulation_shard();
}

bool IsCommunicateNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  static const PrimitiveSet comm_op_type = {prim::kPrimAllReduce, prim::kPrimReduceScatter, prim::kPrimAllGather,
                                            prim::kPrimBroadcast, prim::kPrimAlltoAll,      prim::kPrimAlltoAllV,
                                            prim::kPrimSend,      prim::kPrimReceive};
  return IsOneOfPrimitiveCNode(node, comm_op_type);
}

bool StringToInt(std::string *str, int64_t *value) {
  try {
    *value = stoi(*str);
  } catch (std::invalid_argument &) {
    MS_LOG(ERROR) << "Catch invalid argument, invalid of digit string:" << *str;
    return false;
  }
  return true;
}

AnfNodePtr GetInputNodeBySkipDependAndUpdateState(const CNodePtr &cnode, size_t index) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto input_size = cnode->size();
  if (index >= cnode->size()) {
    MS_LOG(EXCEPTION) << "The index( " << index << ") exceeds the input size of cnode(" << input_size
                      << "). The node is " << cnode->DebugString();
  }
  AnfNodePtr ret_node = cnode->input(index);
  while (ret_node != nullptr && IsOneOfPrimitiveCNode(ret_node, {prim::kPrimDepend, prim::kPrimUpdateState})) {
    auto ret_cnode = ret_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(ret_cnode);
    ret_node = ret_cnode->input(kIndex1);
  }
  return ret_node;
}
}  // namespace parallel
}  // namespace mindspore
