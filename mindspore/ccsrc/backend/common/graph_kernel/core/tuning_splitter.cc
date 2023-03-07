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
#include "backend/common/graph_kernel/core/tuning_splitter.h"
#include <fstream>
#include <vector>
#include "utils/ms_utils.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {
bool TuningSplitSchemer::ReadCache(const std::string &filename, nlohmann::json *result) const {
  std::ifstream json_reader(filename);
  if (!json_reader.is_open()) {
    MS_LOG(ERROR) << "Read json file(" << filename << ") error.";
    return false;
  }
  json_reader >> (*result);
  return true;
}

bool TuningSplitSchemer::ParseResult(const AnfNodePtrList &nodes, const nlohmann::json &result) {
  size_t group_num = result["group_num"];
  std::vector<size_t> split_result = result["split_result"];
  std::vector<std::string> graph_mode = result["graph_mode"];
  split_plan_.resize(group_num);
  need_inline_.reserve(group_num);
  if (nodes.size() != split_result.size()) {
    MS_LOG(EXCEPTION) << "The node num is " << nodes.size() << ", but got split_result size " << split_result.size();
  }
  if (group_num != graph_mode.size()) {
    MS_LOG(EXCEPTION) << "The group num is " << group_num << ", but got graph_mode size " << graph_mode.size();
  }
  for (size_t i = 0; i < nodes.size(); i++) {
    auto group_id = split_result[i];
    if (group_id >= group_num) {
      MS_LOG(EXCEPTION) << "The group_id should be in range [0, " << group_num << "), but got " << group_id;
    }
    node_group_[nodes[i]] = group_id;
    (void)split_plan_[group_id].emplace_back(nodes[i]);
  }
  for (size_t i = 0; i < graph_mode.size(); i++) {
    (void)need_inline_.emplace_back(graph_mode[i] == "basic" ? 1 : 0);
  }
  return split_plan_.size() > 1 || (split_plan_.size() == 1 && NeedInline(0));
}

bool TuningSplitSchemer::Split(const FuncGraphPtr &func_graph) {
  if (!func_graph->has_attr(kAttrNodeName)) {
    MS_LOG(WARNING) << "The func_graph has not attr \"node_name\".";
    return false;
  }
  std::string node_name = GetValue<std::string>(func_graph->get_attr(kAttrNodeName));
  AnfNodePtrList nodes;
  GkUtils::GetValidKernelNodes(func_graph, &nodes, nullptr, nullptr);
  // the input json has postfix ".info", and the result file has postfix ".json"
  auto result_file = tuning_path_ + "/" + node_name + ".json";
  nlohmann::json tuning_result;
  if (!ReadCache(result_file, &tuning_result)) {
    return false;
  }
  if (!ParseResult(nodes, tuning_result)) {
    return false;
  }
  GroupReturnNode(func_graph);
  return true;
}
}  // namespace mindspore::graphkernel
