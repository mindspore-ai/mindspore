/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "backend/common/pass/custom_defined_depend.h"
#include <algorithm>
#include <list>
#include <map>
#include <fstream>
#include <iostream>
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "utils/file_utils.h"
#include "include/backend/distributed/collective/collective_manager.h"

namespace mindspore {
namespace opt {

namespace {
void InsertDepend(const AnfNodePtr &prior_node, const AnfNodePtr &post_node, const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(prior_node);
  MS_EXCEPTION_IF_NULL(post_node);
  auto post_cnode = post_node->cast<CNodePtr>();
  if (IsPrimitiveCNode(post_cnode->input(1))) {
    auto cnode = post_cnode->input(1)->cast<CNodePtr>();
    if (GetCNodePrimitive(cnode)->name() == prim::kPrimDepend->name() && post_cnode->inputs().size() >= 2 &&
        post_cnode->input(kIndex2) == prior_node) {
      return;
    }
  }
  auto manager = root->manager();
  std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), post_cnode->input(1), prior_node};
  auto depend_node = root->NewCNode(depend_input);
  depend_node->set_abstract(post_cnode->input(1)->abstract()->Clone());
  manager->SetEdge(post_node, 1, depend_node);
}

bool FileExists(const string &filename) {
  std::ifstream f(filename.c_str());
  return f.good();
}

std::string GetRankID() {
  uint32_t rank_id = 0;
#if !defined(BUILD_LITE)
  if (distributed::collective::CollectiveManager::instance()->initialized()) {
    rank_id = CommManager::GetInstance().GetRank();
  } else {
    rank_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  }
#endif
  return std::to_string(rank_id);
}

using json = nlohmann::ordered_json;

void GetDeleteDependList(CNodePtrList *cnode_list, std::map<string, CNodePtr> name_node_map,
                         const std::vector<string> &delete_depend_list, string stage_name, int64_t current_graph_id) {
  for (auto delete_name : delete_depend_list) {
    if (name_node_map.count(delete_name) != 0) {
      (void)cnode_list->emplace_back(name_node_map[delete_name]);
    } else {
      MS_LOG(WARNING) << delete_name << " is not in graph " << current_graph_id << " of " << stage_name;
      break;
    }
  }
}

bool GetGraphDependList(CNodePtrList *cnode_list, std::map<string, CNodePtr> name_node_map,
                        const std::vector<string> &depend_src_list, string stage_name, int64_t current_graph_id) {
  for (auto src_name : depend_src_list) {
    if (name_node_map.count(src_name) != 0) {
      (void)cnode_list->emplace_back(name_node_map[src_name]);
    } else {
      MS_LOG(WARNING) << src_name << " is not in graph " << current_graph_id << " of " << stage_name;
      return false;
    }
  }
  return true;
}

bool GetStageDependList(std::string depend_config_path, bool *get_full_op_name_list, CNodePtrList *src_cnode_list,
                        CNodePtrList *dest_cnode_list, int64_t current_graph_id,
                        const std::map<string, CNodePtr> &name_node_map, CNodePtrList *delete_depend_node_list) {
  std::ifstream ifs(depend_config_path);
  json args;
  ifs >> args;
  for (auto it = args.begin(); it != args.end(); ++it) {
    if (it.key() == "get_full_op_name_list") {
      *get_full_op_name_list = it.value();
      continue;
    }
    for (auto graph_arg : it.value()) {
      (void)src_cnode_list->clear();
      (void)dest_cnode_list->clear();
      (void)delete_depend_node_list->clear();
      if (!graph_arg.contains("graph_id")) {
        MS_LOG_EXCEPTION << "Key 'graph_id' does not exist, please check!";
      }
      // Find depend pair in current graph id.
      if (current_graph_id != graph_arg["graph_id"]) {
        continue;
      }
      if (!graph_arg.contains("depend_src_list")) {
        MS_LOG_EXCEPTION << "Key 'depend_src_list' does not exist, please check!";
      }
      std::vector<string> depend_src_list = graph_arg["depend_src_list"];
      if (!GetGraphDependList(src_cnode_list, name_node_map, depend_src_list, it.key(), current_graph_id)) {
        break;
      }
      if (!graph_arg.contains("depend_dest_list")) {
        MS_LOG_EXCEPTION << "Key 'depend_dest_list' does not exist, please check!";
      }
      std::vector<string> depend_dest_list = graph_arg["depend_dest_list"];
      if (!GetGraphDependList(dest_cnode_list, name_node_map, depend_dest_list, it.key(), current_graph_id)) {
        break;
      }
      if (dest_cnode_list->size() != src_cnode_list->size()) {
        MS_LOG(WARNING) << "depend_dest_list's size " << dest_cnode_list->size()
                        << " is not equal to depend_src_list's size " << dest_cnode_list->size();
        return false;
      }
      if (graph_arg.contains("delete_depend_list")) {
        std::vector<string> delete_depend_list = graph_arg["delete_depend_list"];
        GetDeleteDependList(delete_depend_node_list, name_node_map, delete_depend_list, it.key(), current_graph_id);
      }
      return true;
    }
  }
  return false;
}

void MergeCsv(std::vector<string> csv_path_list, string csv_full_path) {
  std::ofstream file(csv_full_path, std::ios::out | std::ios::trunc);
  file << "name"
       << ","
       << "graph_id" << std::endl;
  for (auto csv_path : csv_path_list) {
    std::ifstream part_scv(csv_path);
    string line;
    // Skip the header of the csv file
    getline(part_scv, line);
    while (getline(part_scv, line)) {
      file << line << std::endl;
    }
    part_scv.close();
  }
  file.close();
}

}  // namespace

bool CustomDefinedDepend::Run(const FuncGraphPtr &graph) {
  string depend_config_path = common::GetEnv("MS_CUSTOM_DEPEND_CONFIG_PATH");
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_ge = !ms_context->IsKByKExecutorMode();
  if (depend_config_path.empty() || is_ge != is_ge_) {
    return false;
  }
  if (!FileExists(depend_config_path)) {
    MS_LOG_EXCEPTION << depend_config_path << " does not exist, please check!";
  }
  bool generate_bool_list = false;
  CNodePtrList src_list;
  CNodePtrList dest_list;
  CNodePtrList delete_list;
  std::map<string, CNodePtr> name_node_map;
  auto manager = graph->manager();
  std::list<CNodePtr> graph_orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
  for (const auto &node : origin_nodes_topological) {
    name_node_map[node->fullname_with_scope()] = node;
  }
  bool formatted_input = GetStageDependList(depend_config_path, &generate_bool_list, &src_list, &dest_list, graph_id_,
                                            name_node_map, &delete_list);

  std::optional<std::string> json_dir = "";
  std::optional<std::string> json_path = "";
  FileUtils::SplitDirAndFileName(depend_config_path.c_str(), &json_dir, &json_path);
  if (!json_path.has_value()) {
    MS_LOG_EXCEPTION << "Failed to get real path: " << depend_config_path;
  }
  auto rank_id = GetRankID();
  if (generate_bool_list) {
    std::string csv_path =
      json_dir.value() + "/rank_id" + rank_id + "/custom_depend_graph_" + std::to_string(graph_id_) + ".csv";
    Common::CreatePrefixPath(csv_path);
    std::ofstream file(csv_path, std::ios::out | std::ios::trunc);
    file << "name"
         << ","
         << "graph_id" << std::endl;
    for (const auto &node : origin_nodes_topological) {
      file << node->fullname_with_scope() << "," << graph_id_ << std::endl;
    }
    file.close();
    std::vector<string> csv_path_list;
    for (int64_t i = 0; i <= graph_id_; i++) {
      std::string csv_path_part =
        json_dir.value() + "/rank_id" + rank_id + "/custom_depend_graph_" + std::to_string(i) + ".csv";
      if (FileExists(csv_path_part)) {
        (void)csv_path_list.emplace_back(csv_path_part);
      }
    }
    MergeCsv(csv_path_list, json_dir.value() + "/custom_depend_rank_id_" + rank_id + ".csv");
  }

  if (!formatted_input) {
    return false;
  }
  for (size_t i = 0; i < src_list.size(); i++) {
    InsertDepend(src_list[i], dest_list[i], graph);
  }
  for (auto const &cnode : delete_list) {
    manager->Replace(cnode, cnode->input(1));
  }
  return true;
}

}  // namespace opt
}  // namespace mindspore
