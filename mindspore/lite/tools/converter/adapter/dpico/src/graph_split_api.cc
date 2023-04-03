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

#include "src/graph_split_api.h"
#include <unordered_set>
#include <string>
#include <algorithm>
#include "ops/return.h"
#include "ops/transpose.h"
#include "ops/make_tuple.h"
#include "ops/tuple_get_item.h"
#include "common/op_attr.h"
#include "include/errorcode.h"
#include "include/api/format.h"
#include "src/graph_split_info.h"
#include "common/check_base.h"
#include "common/anf_util.h"
#include "common/op_enum.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
namespace mindspore {
namespace dpico {
namespace {
constexpr auto kFormat = "format";
const std::unordered_set<std::string> kRoiOpList = {"DetectionOutput", "DecBBox"};
const std::unordered_set<std::string> kRecurrentOpList = {"Lstm", "Rnn", "Gru", "BiLstm"};
const size_t kMaximumNumbOfSegments = 64;
struct SegmentInfo {
  size_t left_border;
  size_t right_border;
  bool is_supported;
};
api::CNodePtrList GetFuncGraphTotalCNodes(const api::FuncGraphPtr &func_graph) {
  api::CNodePtrList graph_total_cnodes;
  auto node_list = api::FuncGraph::TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    auto cnode = node->cast<api::CNodePtr>();
    if (cnode != nullptr &&
        !CheckPrimitiveType(cnode, api::MakeShared<ops::TupleGetItem>())) {  // tuple_get_item may affect split graph
      graph_total_cnodes.push_back(cnode);
    }
  }
  return graph_total_cnodes;
}

bool IsTupleGetItemNeeded(const api::CNodePtr &cnode, const api::CNodePtr &linked_cnode,
                          const api::CNodePtrList &total_cnodes) {
  return CheckPrimitiveType(cnode, api::MakeShared<ops::TupleGetItem>()) &&
         GetBoolAttr(cnode, kIsMapperSupported) == GetBoolAttr(linked_cnode, kIsMapperSupported) &&
         std::find(total_cnodes.begin(), total_cnodes.end(), cnode) == total_cnodes.end();
}

api::CNodePtrList GetSubgraphTotalCNodes(const api::FuncGraphPtr &func_graph, const api::CNodePtrList &cnode_list,
                                         const SegmentInfo &segment_info) {
  api::CNodePtrList total_cnodes{};
  for (size_t i = segment_info.left_border; i <= segment_info.right_border; i++) {
    auto &cur_cnode = cnode_list[i];
    total_cnodes.push_back(cur_cnode);
    for (const auto &input_node : cur_cnode->inputs()) {
      auto input_cnode = input_node->cast<api::CNodePtr>();
      if (input_cnode == nullptr) {
        continue;
      }
      if (IsTupleGetItemNeeded(input_cnode, cur_cnode, total_cnodes)) {
        total_cnodes.push_back(input_cnode);
      }
    }

    auto manager = api::FuncGraphManager::Manage(func_graph, true);
    MS_CHECK_TRUE_MSG(manager != nullptr, {}, "manager is nullptr.");
    auto node_users = manager->GetUsers(cur_cnode);
    for (const auto &node_user : node_users) {
      auto output_cnode = node_user.first->cast<api::CNodePtr>();
      if (IsTupleGetItemNeeded(output_cnode, cur_cnode, total_cnodes)) {
        total_cnodes.push_back(output_cnode);
      }
    }
  }
  return total_cnodes;
}

STATUS GetSubgraphNetType(const api::CNodePtrList &cnodes, OmNetType *om_net_type) {
  for (const auto &cnode : cnodes) {
    std::string op_type_name;
    if (GetPrimitiveType(cnode, &op_type_name) != RET_OK) {
      MS_LOG(ERROR) << "get cnode primitive type failed:" << cnode->fullname_with_scope();
      return RET_ERROR;
    }
    if (kRoiOpList.find(op_type_name) != kRoiOpList.end()) {
      if (*om_net_type == OmNetType::kRecurrent) {
        MS_LOG(ERROR) << "subgraph has been marked as Recurrent net type. " << op_type_name;
        return RET_ERROR;
      }
      *om_net_type = OmNetType::kRoi;
    } else if (kRecurrentOpList.find(op_type_name) != kRecurrentOpList.end()) {
      if (*om_net_type == OmNetType::kRoi) {
        MS_LOG(ERROR) << "subgraph has been marked as ROI net type. " << op_type_name;
        return RET_ERROR;
      }
      *om_net_type = OmNetType::kRecurrent;
    }
  }
  return RET_OK;
}

STATUS GenerateSegmentInfos(const api::CNodePtrList &graph_total_cnodes, std::vector<SegmentInfo> *segment_infos,
                            bool *last_op_is_custom) {
  MS_CHECK_TRUE_MSG(segment_infos != nullptr, RET_ERROR, "segment_infos are nullptr");
  size_t start = 0;
  for (size_t pos = 0; pos < graph_total_cnodes.size(); pos++) {
    if (pos == graph_total_cnodes.size() - 1) {
      if (!CheckPrimitiveType(graph_total_cnodes[pos], api::MakeShared<ops::Return>())) {
        MS_LOG(ERROR) << "last cnode should be return node.";
        return RET_ERROR;
      }
      (void)segment_infos->emplace_back(SegmentInfo{start, pos, false});
      auto is_supported = GetBoolAttr(graph_total_cnodes[pos - 1], kIsMapperSupported);
      if (is_supported) {
        *last_op_is_custom = true;
      } else {
        *last_op_is_custom = false;
      }
    } else {
      auto is_supported = GetBoolAttr(graph_total_cnodes[pos], kIsMapperSupported);
      if (is_supported != GetBoolAttr(graph_total_cnodes[pos + 1], kIsMapperSupported)) {
        (void)segment_infos->emplace_back(SegmentInfo{start, pos, is_supported});
        start = pos + 1;
      }
    }
  }
  return RET_OK;
}

STATUS ComputeNetworkSegments(const std::vector<SegmentInfo> &segment_infos, const bool &last_op_is_custom,
                              GraphSplitInfo *graph_split_info) {
  MS_CHECK_TRUE_MSG(graph_split_info != nullptr, RET_ERROR, "graph split info is nullptr.");
  MS_CHECK_TRUE_MSG(segment_infos.size() > 0, RET_ERROR, "segment_infos shouldn't be smaller than 1.");
  for (auto segment_info : segment_infos) {
    if (!segment_info.is_supported) {
      graph_split_info->num_of_segments++;
    } else {
      graph_split_info->num_of_custom_op++;
    }
  }
  if (!segment_infos.back().is_supported) {  // remove last unsupported subgraph which contains Return cnode
    graph_split_info->num_of_segments--;
  }
  if (segment_infos.front().is_supported && last_op_is_custom) {
    graph_split_info->head_tail_op_is_custom = 1;
  }
  MS_LOG(DEBUG) << "graph_split_info->head_tail_op_is_custom = " << graph_split_info->head_tail_op_is_custom;
  if (graph_split_info->num_of_segments > kMaximumNumbOfSegments) {
    MS_LOG(ERROR) << "There are over " << kMaximumNumbOfSegments
                  << " segments in this network, so this process will stop. It's recommended that you don't set "
                     "\"configFile\" when converting, and all operators will be running on CPU.";
    return RET_ERROR;
  }
  return RET_OK;
}

std::vector<Subgraph> GenerateSubgraphs(const api::FuncGraphPtr &func_graph,
                                        const api::CNodePtrList &graph_total_cnodes,
                                        const std::vector<SegmentInfo> &segment_infos, size_t *subgraph_cnt) {
  MS_CHECK_TRUE_MSG(subgraph_cnt != nullptr, {}, "subgraph_cnt is nullptr.");
  std::vector<Subgraph> subgraphs;
  for (const auto &segment_info : segment_infos) {
    auto subgraph_total_cnodes = GetSubgraphTotalCNodes(func_graph, graph_total_cnodes, segment_info);
    MS_CHECK_TRUE_MSG(!subgraph_total_cnodes.empty(), {}, "get subgraph cnodes failed.");
    OmNetType net_type = OmNetType::kCnn;
    if (GetSubgraphNetType(subgraph_total_cnodes, &net_type) != RET_OK) {
      MS_LOG(ERROR) << "get subgraph net type failed.";
      return {};
    }
    (void)subgraphs.emplace_back(*subgraph_cnt, segment_info.is_supported, net_type, subgraph_total_cnodes);
    *subgraph_cnt += 1;
  }
  return subgraphs;
}

bool FilterMakeTuple(const api::FuncGraphManagerPtr &manager, const Subgraph &subgraph, const api::CNodePtr &cnode) {
  auto node_users = manager->GetUsers(cnode);
  bool is_subgraph_output = true;
  for (const auto &node_user : node_users) {
    auto output_cnode = node_user.first->cast<api::CNodePtr>();
    if (output_cnode == nullptr) {
      continue;
    }
    if (std::find(subgraph.cnodes.begin(), subgraph.cnodes.end(), output_cnode) != subgraph.cnodes.end()) {
      is_subgraph_output = false;
    }
  }
  return is_subgraph_output;
}

bool IsSubgraphParamInput(const api::AnfNodePtr &front_node, const api::AnfNodePtrList &subgraph_param_inputs) {
  auto param = front_node->cast<api::ParameterPtr>();
  return !param->has_default() &&
         std::find(subgraph_param_inputs.begin(), subgraph_param_inputs.end(), param) == subgraph_param_inputs.end();
}

bool IsSubgraphCNodeInput(const api::AnfNodePtr &front_node, const Subgraph &subgraph,
                          const api::AnfNodePtrList &subgraph_cnode_inputs) {
  return std::find(subgraph.cnodes.begin(), subgraph.cnodes.end(), front_node) == subgraph.cnodes.end() &&
         std::find(subgraph_cnode_inputs.begin(), subgraph_cnode_inputs.end(), front_node) ==
           subgraph_cnode_inputs.end();
}

int DetermineOutputFormat(const api::AnfNodePtr &output_node, Format *format) {
  MS_CHECK_TRUE_MSG(output_node != nullptr && output_node->isa<api::CNode>(), RET_ERROR, "output node is invalid.");
  auto output_cnode = output_node->cast<api::CNodePtr>();
  int64_t local_format = static_cast<int64_t>(NCHW);
  auto search_cnode = output_cnode;
  const int max_search_depth = 10;
  int loop = 0;
  // current node may has no format, which can be obtain by transitivity of format.
  while (loop < max_search_depth) {
    auto primitive = api::GetValueNode<api::PrimitivePtr>(search_cnode->input(0));
    if (primitive == nullptr) {
      break;
    }
    if (primitive->GetAttr(kFormat) != nullptr) {
      local_format = api::GetValue<int64_t>(primitive->GetAttr(kFormat));
      break;
    }
    auto input_node = search_cnode->input(1);
    if (!api::utils::isa<api::CNode>(input_node)) {
      break;
    }
    search_cnode = input_node->cast<api::CNodePtr>();
    loop++;
  }
  if (local_format < static_cast<int64_t>(NCHW) || local_format > static_cast<int64_t>(NCW)) {
    MS_LOG(ERROR) << "obtained format is out of range, which is invalid.";
    return RET_ERROR;
  }
  *format = static_cast<Format>(local_format);
  if (CheckPrimitiveType(output_cnode, api::MakeShared<ops::Transpose>())) {
    auto abstract = GetCNodeInputAbstract(output_cnode, 1);
    MS_CHECK_TRUE_MSG(abstract != nullptr, RET_ERROR, "input's abstract is nullptr.");
    ShapeVector input_shape;
    auto ret = FetchShapeFromAbstract(abstract, &input_shape);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "obtain input shape failed.");
    abstract = output_cnode->abstract();
    MS_CHECK_TRUE_MSG(abstract != nullptr, RET_ERROR, "current cnode's abstract is nullptr.");
    ShapeVector output_shape;
    ret = FetchShapeFromAbstract(abstract, &output_shape);
    MS_CHECK_TRUE_MSG(ret == RET_OK, RET_ERROR, "obtain output shape failed.");
    if (input_shape.size() != kDims4) {
      return RET_OK;
    }
    ShapeVector nc2nh = {input_shape[0], input_shape[kDims2], input_shape[kDims3], input_shape[kDims1]};
    if (nc2nh == output_shape && *format == NCHW) {
      *format = NHWC;
    }
    ShapeVector nh2nc = {input_shape[0], input_shape[kDims3], input_shape[kDims1], input_shape[kDims2]};
    if (nh2nc == output_shape && *format == NHWC) {
      *format = NCHW;
    }
  }
  return RET_OK;
}
}  // namespace

int GraphSplit(const std::vector<api::FuncGraphPtr> &func_graphs, GraphSplitInfo *graph_split_info) {
  MS_CHECK_TRUE_MSG(graph_split_info != nullptr, RET_NULL_PTR, "graph_split_info is nullptr.");
  size_t subgraph_cnt = 0;
  for (const auto &func_graph : func_graphs) {
    MS_CHECK_TRUE_MSG(func_graph != nullptr, RET_NULL_PTR, "func_graph is nullptr.");
    auto graph_total_cnodes = GetFuncGraphTotalCNodes(func_graph);
    MS_CHECK_TRUE_MSG(graph_total_cnodes.size() > 1, {}, "func graph should have 2 cnode at least.");
    std::vector<SegmentInfo> segment_infos;
    bool last_op_is_custom = false;
    if (GenerateSegmentInfos(graph_total_cnodes, &segment_infos, &last_op_is_custom) != RET_OK) {
      MS_LOG(ERROR) << "generate segment infos failed.";
      return RET_ERROR;
    }

    if (ComputeNetworkSegments(segment_infos, last_op_is_custom, graph_split_info) != RET_OK) {
      MS_LOG(ERROR) << "compute network segments failed.";
      return RET_ERROR;
    }

    auto subgraphs = GenerateSubgraphs(func_graph, graph_total_cnodes, segment_infos, &subgraph_cnt);
    if (subgraphs.empty()) {
      MS_LOG(ERROR) << "generate subgraphs failed.";
      return RET_ERROR;
    }
    graph_split_info->subgraphs_map[func_graph] = subgraphs;
  }
  return RET_OK;
}

api::AnfNodePtrList GetSubgraphInputs(const Subgraph &subgraph, const api::FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, {}, "func_graph is nullptr.");
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, {}, "funcgraph manager is nullptr.");
  api::AnfNodePtrList subgraph_param_inputs;
  api::AnfNodePtrList subgraph_cnode_inputs;
  for (const auto &cnode : subgraph.cnodes) {
    if (CheckPrimitiveType(cnode, api::MakeShared<ops::MakeTuple>())) {
      if (FilterMakeTuple(manager, subgraph, cnode)) {
        continue;
      }
    }
    for (size_t i = 1; i < cnode->inputs().size(); i++) {
      auto front_node = cnode->input(i);
      MS_CHECK_TRUE_MSG(front_node != nullptr, {}, "input node is nullptr.");
      if (api::utils::isa<api::Parameter>(front_node)) {
        if (IsSubgraphParamInput(front_node, subgraph_param_inputs)) {
          subgraph_param_inputs.push_back(front_node);
        }
      } else if (api::utils::isa<api::CNode>(front_node)) {
        if (IsSubgraphCNodeInput(front_node, subgraph, subgraph_cnode_inputs)) {
          subgraph_cnode_inputs.push_back(front_node);
        }
      }
    }
  }
  // keep subgraph input as origin graph input order
  api::AnfNodePtrList subgraph_inputs;
  auto graph_inputs = func_graph->get_inputs();
  for (auto &graph_input : graph_inputs) {
    if (std::find(subgraph_param_inputs.begin(), subgraph_param_inputs.end(), graph_input) ==
        subgraph_param_inputs.end()) {
      continue;
    }
    subgraph_inputs.push_back(graph_input);
  }
  (void)subgraph_inputs.insert(subgraph_inputs.end(), subgraph_cnode_inputs.begin(), subgraph_cnode_inputs.end());
  return subgraph_inputs;
}

api::AnfNodePtrList GetSubgraphOutputs(const Subgraph &subgraph, const api::FuncGraphManagerPtr &manager) {
  api::AnfNodePtrList subgraph_outputs;
  for (const auto &cnode : subgraph.cnodes) {
    auto node_users = manager->GetUsers(cnode);
    for (const auto &node_user : node_users) {
      auto output_cnode = node_user.first->cast<api::CNodePtr>();
      if (output_cnode == nullptr) {
        continue;
      }
      if (std::find(subgraph.cnodes.begin(), subgraph.cnodes.end(), output_cnode) != subgraph.cnodes.end()) {
        continue;
      }
      if (!CheckPrimitiveType(cnode, api::MakeShared<ops::MakeTuple>())) {
        subgraph_outputs.push_back(cnode);
        break;
      }
      for (size_t i = 1; i < cnode->inputs().size(); i++) {
        auto input_node = cnode->input(i);
        if (api::utils::isa<api::CNodePtr>(input_node) &&
            std::find(subgraph.cnodes.begin(), subgraph.cnodes.end(), input_node) != subgraph.cnodes.end() &&
            std::find(subgraph_outputs.begin(), subgraph_outputs.end(), input_node) == subgraph_outputs.end()) {
          subgraph_outputs.push_back(input_node);
        }
      }
      break;
    }
  }
  return subgraph_outputs;
}

int FillSubgraphOutputsFormat(Subgraph *subgraph, const api::FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(subgraph != nullptr && func_graph != nullptr, RET_ERROR, "output node is invalid.");
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_ERROR, "func_graph's manager is a nullptr.");
  auto subgraph_outputs = GetSubgraphOutputs(*subgraph, manager);
  for (auto &output_node : subgraph_outputs) {
    Format output_format;
    if (DetermineOutputFormat(output_node, &output_format) != RET_OK) {
      MS_LOG(ERROR) << "obtain output format failed.";
      return RET_ERROR;
    }
    subgraph->outputs_format.push_back(static_cast<int>(output_format));
  }
  return RET_OK;
}
}  // namespace dpico
}  // namespace mindspore
