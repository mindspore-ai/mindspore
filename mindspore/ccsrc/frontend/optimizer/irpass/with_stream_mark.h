/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_WITH_STREAM_MARK_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_WITH_STREAM_MARK_H_

#include "ir/graph_utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
constexpr auto kInvalidStreamId = -1;
constexpr auto kInvalidStreamLimitId = -1;
constexpr auto kInvalidStreamCtxAfterId = -1;
constexpr auto kInvalidStreamLimitAfterId = -1;
int64_t GetStreamId(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto value = func_graph->get_attr(kFuncGraphFlagStreamId);
  return (value != nullptr) ? static_cast<int64_t>(GetValue<size_t>(value)) : kInvalidStreamId;
}

int64_t GetStreamLimitId(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto value = func_graph->get_attr(kFuncGraphFlagStreamLimitId);
  return (value != nullptr) ? static_cast<int64_t>(GetValue<size_t>(value)) : kInvalidStreamLimitId;
}

int64_t GetStreamCtxAfterId(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto value = func_graph->get_attr(kFuncGraphFlagStreamCtxAfter);
  return (value != nullptr) ? static_cast<int64_t>(GetValue<size_t>(value)) : kInvalidStreamCtxAfterId;
}

void TransferFlagForSubFunc(const FuncGraphPtr &func) {
  if (func->has_attr("marked_stream_id")) {
    return;
  }
  MS_LOG(DEBUG) << "Transfer stream_id for func: " << func->ToString();
  func->set_flag("marked_stream_id", true);

  int64_t stream_id = GetStreamId(func);
  if (stream_id == kInvalidStreamId) {
    return;
  }
  const auto &sub_graphs = func->func_graphs_used();
  for (auto &used : sub_graphs) {
    auto sub_graph = used.first;
    int64_t stream_after_id = GetStreamCtxAfterId(sub_graph);
    if (stream_after_id == stream_id) {
      MS_LOG(DEBUG) << "The sub_graph : " << sub_graph->ToString() << " has stream_after_id: " << stream_after_id
                    << ", do not mark stream_id: " << stream_id;
      continue;
    }
    const int64_t cur_stream_id = GetStreamId(sub_graph);
    if (cur_stream_id != kInvalidStreamId) {
      MS_LOG(DEBUG) << "The sub_graph : " << sub_graph->ToString() << " has cur_stream_id: " << cur_stream_id
                    << ", do not mark stream_id: " << stream_id;
      continue;
    }
    MS_LOG(DEBUG) << "Transfer func: " << sub_graph->ToString() << " stream_id: " << stream_id;
    sub_graph->set_attr(kFuncGraphFlagStreamId, MakeValue(static_cast<size_t>(stream_id)));
  }
}

void MarkStreamIdForNodes(const FuncGraphPtr &func) {
  // If has stream_id, and has not stream_ctx_after flag or the
  // stream_ctx_after is not equal stream_id, need mark.
  const int64_t stream_id = GetStreamId(func);
  if (stream_id == kInvalidStreamId) {
    return;
  }
  MS_LOG(DEBUG) << "The func: " << func->ToString() << " stream_id: " << stream_id;
  int64_t stream_limit_after_id = GetStreamCtxAfterId(func);
  if (stream_limit_after_id == kInvalidStreamCtxAfterId || stream_id != stream_limit_after_id) {
    // Need mark stream_id
    const auto &all_nodes = TopoSort(func->return_node(), SuccDeeperSimple, AlwaysInclude);
    MS_LOG(DEBUG) << "all_nodes size: " << all_nodes.size();
    for (auto &node : all_nodes) {
      if (!node->isa<CNode>()) {
        continue;
      }
      // Need consider free variable
      auto need_check_node = node;
      auto cnode = node->cast<CNodePtr>();
      if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
        need_check_node = cnode->input(1);
      }
      if (!need_check_node->isa<CNode>()) {
        continue;
      }
      const auto &cur_func = need_check_node->func_graph();
      const auto real_stream_id = GetStreamId(cur_func);
      if (real_stream_id == kInvalidStreamId) {
        continue;
      }
      MS_LOG(DEBUG) << "The node: " << node->DebugString() << " mark stream_id: " << real_stream_id;
      cnode->AddAttr(kFuncGraphFlagStreamId, MakeValue(static_cast<int64_t>(real_stream_id)));
    }
  }
}

void MarkStreamLimitCtxForNodes(const FuncGraphPtr &func) {
  // If has stream_id, and has stream_limit_id, stream_id is equal
  // stream_limit_id, need mark.
  int64_t stream_id = GetStreamId(func);
  if (stream_id == kInvalidStreamId) {
    return;
  }
  int64_t stream_limit_id = GetStreamLimitId(func);
  if (stream_limit_id == kInvalidStreamLimitId) {
    return;
  }
  if (stream_id != stream_limit_id) {
    return;
  }
  MS_LOG(DEBUG) << "The func: " << func->ToString() << " stream_id: " << stream_id
                << " stream_limit_id: " << stream_limit_id;

  // Need mark stream_limit_ctx: stream_id, cube_num, vector_num
  auto cube_num_value = func->get_attr(kFuncGraphFlagCubeNum);
  MS_EXCEPTION_IF_NULL(cube_num_value);
  int64_t cube_num = GetValue<int64_t>(cube_num_value);
  auto vector_num_value = func->get_attr(kFuncGraphFlagVectorNum);
  MS_EXCEPTION_IF_NULL(vector_num_value);
  int64_t vector_num = GetValue<int64_t>(vector_num_value);

  const auto &all_nodes = TopoSort(func->return_node(), SuccDeeperSimple, AlwaysInclude);
  MS_LOG(DEBUG) << "The all_nodes size is: " << all_nodes.size();
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    const auto &cur_func = cnode->func_graph();
    const auto real_stream_limit_id = GetStreamLimitId(cur_func);
    if (real_stream_limit_id == stream_limit_id) {
      MS_LOG(DEBUG) << "Mark StreamLimitCtx for node: " << node->DebugString();
      cnode->AddAttr(kFuncGraphFlagStreamId, MakeValue(static_cast<int64_t>(stream_id)));
      cnode->AddAttr(kFuncGraphFlagCubeNum, MakeValue(static_cast<int64_t>(cube_num)));
      cnode->AddAttr(kFuncGraphFlagVectorNum, MakeValue(static_cast<int64_t>(vector_num)));
    }
  }
}

int64_t ExtractStreamId(const std::string &text) {
  std::string keyword = "stream id:";
  size_t pos = text.find(keyword);
  if (pos == std::string::npos) {
    return -1;
  }
  pos += keyword.length();
  while (pos < text.length() && std::isspace(text[pos])) {
    pos++;
  }
  int64_t result = 0;
  int64_t decimal_num = 10;
  bool found_digit = false;

  while (pos < text.length() && std::isdigit(text[pos])) {
    found_digit = true;
    result = result * decimal_num + (text[pos] - '0');
    pos++;
  }
  return found_digit ? result : -1;
}

size_t GetStreamId(const ValuePtr &value) {
  auto stream_id = ExtractStreamId(value->ToString());
  if (stream_id == -1) {
    MS_LOG(EXCEPTION) << "GetStreamID node is wrong.";
  }
  return static_cast<size_t>(stream_id);
}

void GetFuncAttrFromGetStreamInfoNode(const FuncGraphPtr &func) {
  auto topo_nodes = TopoSort(func->get_return());
  auto mgr = func->manager();
  MS_EXCEPTION_IF_NULL(mgr);
  for (const auto &node : topo_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    if (!IsPrimitiveCNode(node, prim::kPrimGetStreamInfo)) {
      continue;
    }

    const size_t args_min_size = 3;
    const size_t args_max_size = 5;

    // GetStreamInfo(kFuncGraphFlagStreamId, stream_id_node)
    // GetStreamInfo(kFuncGraphFlagStreamCtxAfter, stream_id_node)
    // GetStreamInfo(kFuncGraphFlagStreamLimitId, stream_id_node, cube_num, vector_num)
    // GetStreamInfo(kFuncGraphFlagStreamLimitCtxAfter, stream_id_node)

    auto get_stream_info = node->cast<CNodePtr>();
    size_t arg_length = get_stream_info->inputs().size();
    if (arg_length != args_min_size && arg_length != args_max_size) {
      MS_LOG(INTERNAL_EXCEPTION) << "The GetStreamInfo operator requires 3 or 5 arguments, but got " << arg_length
                                 << ".";
    }
    MS_LOG(DEBUG) << "get_stream_info: " << get_stream_info->DebugString();
    constexpr auto kFlagIndex = 1;
    constexpr auto kStreamIdIndex = 2;

    auto flag_arg = get_stream_info->input(kFlagIndex)->abstract();
    auto flag_str = GetValue<string>(flag_arg->BuildValue());
    MS_LOG(DEBUG) << "flag_str: " << flag_str;
    auto stream_id_node = get_stream_info->input(kStreamIdIndex);
    MS_LOG(DEBUG) << "stream_id_node: " << stream_id_node->DebugString();
    auto stream_id_abs = stream_id_node->abstract();
    MS_EXCEPTION_IF_NULL(stream_id_abs);
    ValuePtr value_track = stream_id_abs->GetValueTrack();
    MS_EXCEPTION_IF_NULL(value_track);
    size_t stream_id = GetStreamId(value_track);

    if (arg_length == args_min_size) {
      MS_LOG(DEBUG) << "set flag_str: " << flag_str << " for func:" << func->ToString();
      func->set_attr(flag_str, MakeValue(static_cast<size_t>(stream_id)));
    } else {
      constexpr auto kCubeNumIndex = 3;
      constexpr auto kVectorNumIndex = 4;
      auto cube_num_node = get_stream_info->input(kCubeNumIndex);
      auto cube_num_abs = cube_num_node->abstract();
      MS_EXCEPTION_IF_NULL(cube_num_abs);
      auto cube_value = cube_num_abs->BuildValue();
      auto cube_num = GetValue<int64_t>(cube_value);

      auto vector_num_node = get_stream_info->input(kVectorNumIndex);
      auto vector_num_abs = vector_num_node->abstract();
      MS_EXCEPTION_IF_NULL(vector_num_abs);
      auto vector_value = vector_num_abs->BuildValue();
      auto vector_num = GetValue<int64_t>(vector_value);
      MS_LOG(DEBUG) << "set kFuncGraphFlagStreamLimitId: " << stream_id << " for func:" << func->ToString();
      MS_LOG(DEBUG) << "cube_num: " << cube_num << " vector_num: " << vector_num;
      func->set_attr(kFuncGraphFlagStreamLimitId, MakeValue(static_cast<size_t>(stream_id)));
      func->set_attr(kFuncGraphFlagCubeNum, MakeValue(static_cast<int64_t>(cube_num)));
      func->set_attr(kFuncGraphFlagVectorNum, MakeValue(static_cast<int64_t>(vector_num)));
    }

    auto scalar_abs = std::make_shared<abstract::AbstractScalar>(0);
    ValuePtr val = scalar_abs->BuildValue();
    MS_EXCEPTION_IF_NULL(val);
    AnfNodePtr value_node = NewValueNode(val);
    value_node->set_abstract(scalar_abs);
    mgr->Replace(node, value_node);
  }
}

bool WithStreamMark(const FuncGraphPtr &root, const opt::OptimizerPtr &) {
  MS_EXCEPTION_IF_NULL(root);
  MS_LOG(DEBUG) << "The root fg: " << root->ToString();
  GetFuncAttrFromGetStreamInfoNode(root);
  const auto &all_func_graphs = root->func_graphs_used_total();
  for (auto &fg : all_func_graphs) {
    MS_EXCEPTION_IF_NULL(fg);
    GetFuncAttrFromGetStreamInfoNode(fg);
  }
  // For root func_graph, only need transfer stream_id flag.
  TransferFlagForSubFunc(root);
  for (auto &fg : all_func_graphs) {
    MS_EXCEPTION_IF_NULL(fg);
    TransferFlagForSubFunc(fg);
  }

  root->erase_flag("marked_stream_id");
  for (auto &fg : all_func_graphs) {
    MS_EXCEPTION_IF_NULL(fg);
    fg->erase_flag("marked_stream_id");
  }
  MarkStreamIdForNodes(root);
  for (auto &fg : all_func_graphs) {
    MarkStreamIdForNodes(fg);
  }
  for (auto &fg : all_func_graphs) {
    MarkStreamLimitCtxForNodes(fg);
  }

  for (auto &fg : all_func_graphs) {
    MS_EXCEPTION_IF_NULL(fg);
    bool need_clear = fg->has_attr(kFuncGraphFlagStreamId) || fg->has_attr(kFuncGraphFlagStreamLimitId) ||
                      fg->has_attr(kFuncGraphFlagStreamCtxAfter) || fg->has_attr(kFuncGraphFlagStreamLimitCtxAfter);
    if (need_clear) {
      fg->erase_flag(FUNC_GRAPH_FLAG_NO_INLINE);
    }
    fg->erase_flag(kFuncGraphFlagStreamId);
    fg->erase_flag(kFuncGraphFlagStreamLimitId);
    fg->erase_flag(kFuncGraphFlagCubeNum);
    fg->erase_flag(kFuncGraphFlagVectorNum);
    fg->erase_flag(kFuncGraphFlagStreamCtxAfter);
    fg->erase_flag(kFuncGraphFlagStreamLimitCtxAfter);
  }
  return false;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_WITH_STREAM_MARK_H_
