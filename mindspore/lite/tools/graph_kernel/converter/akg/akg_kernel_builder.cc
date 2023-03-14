/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "tools/graph_kernel/converter/akg/akg_kernel_builder.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <vector>
#include <algorithm>

#include "ir/anf.h"
#include "ir/func_graph.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {
bool SaveJsonInfo(const std::string &json_name, const std::string &info) {
  std::string path = json_name + ".info";
  std::ofstream filewrite(path);
  if (!filewrite.is_open()) {
    MS_LOG(ERROR) << "Open file '" << path << "' failed!";
    return false;
  }
  filewrite << info << std::endl;
  filewrite.close();
  return true;
}

std::string SaveNodesInfo(const AnfNodePtrList &nodes, const std::string &dir, const DumpOption &option,
                          std::map<AnfNodePtr, std::string> *node_kernel, std::set<std::string> *kernel_names) {
  auto dir_path = FileUtils::CreateNotExistDirs(dir);
  if (!dir_path.has_value()) {
    MS_LOG(ERROR) << "Failed to CreateNotExistDirs: " << dir;
    return "";
  }
  std::set<std::string> unique_kernel_name;
  for (const auto &node : nodes) {
    graphkernel::AkgKernelJsonGenerator akg_kernel_json_generator(option);
    auto fg = GetCNodeFuncGraph(node);
    MS_EXCEPTION_IF_NULL(fg);
    auto mng = fg->manager();
    if (mng == nullptr) {
      mng = Manage(fg, true);
      fg->set_manager(mng);
    }
    std::vector<AnfNodePtr> node_list, input_list, output_list;
    GkUtils::GetValidKernelNodes(fg, &node_list, &input_list, &output_list);
    akg_kernel_json_generator.CollectFusedJson(node_list, input_list, output_list);
    auto json_kernel_name = akg_kernel_json_generator.kernel_name();
    if (node_kernel != nullptr) {
      (*node_kernel)[node] = json_kernel_name;
    }
    if (!unique_kernel_name.insert(json_kernel_name).second) {
      continue;
    }
    if (!SaveJsonInfo(dir_path.value() + "/" + json_kernel_name, akg_kernel_json_generator.kernel_json_str())) {
      return "";
    }
  }
  if (kernel_names != nullptr) {
    *kernel_names = std::move(unique_kernel_name);
  }
  return dir_path.value();
}
}  // namespace mindspore::graphkernel
