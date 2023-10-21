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
#include "backend/common/graph_kernel/convert_custom_for_ge.h"
#include <set>
#include <vector>
#include <string>
#include <sstream>
#include <memory>
#include "mindspore/core/ops/custom.h"
#include "utils/anf_utils.h"
#include "utils/file_utils.h"
#include "kernel/framework_utils.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/graph_kernel/graph_kernel_json_generator.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {
AnfNodePtr ConvertCustomForGE::CreateCustomOp(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  auto op = std::make_shared<ops::Custom>();
  op->set_type("GraphKernel");
  auto custom_prim = op->GetPrim();
  auto inputs = cnode->inputs();
  inputs[0] = NewValueNode(custom_prim);
  auto custom_cnode = func_graph->NewCNode(inputs);
  auto json_name = node_json_name_[cnode->cast<AnfNodePtr>()];
  auto input_num = AnfUtils::GetInputTensorNum(cnode);
  auto output_num = AnfUtils::GetOutputTensorNum(cnode);
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  for (size_t i = 0; i < input_num; ++i) {
    input_names.push_back("x" + std::to_string(i));
  }
  for (size_t i = 0; i < output_num; ++i) {
    output_names.push_back("y" + std::to_string(i));
  }

  std::ostringstream oss;
  oss << "Fused_x" << input_num << "_y" << output_num;
  std::string op_tye = oss.str();
  custom_prim->set_attr("reg_op_name", MakeValue(op_tye));
  custom_prim->set_attr("info_path", MakeValue(info_dir_ + "/" + json_name + ".info"));
  custom_prim->set_attr("input_names", MakeValue(input_names));
  custom_prim->set_attr("output_names", MakeValue(output_names));
  custom_cnode->set_fullname_with_scope(cnode->fullname_with_scope());
  custom_cnode->set_abstract(cnode->abstract()->Clone());
  return custom_cnode;
}

void ConvertCustomForGE::CreateInfoDir() {
  static std::string rank_id = common::GetEnv("RANK_ID");
  std::string dir;
  if (rank_id.empty()) {
    dir = "./akg_kernel_meta";
  } else {
    dir = "./rank_" + rank_id + "/akg_kernel_meta";
  }
  auto dir_path = FileUtils::CreateNotExistDirs(dir);
  if (!dir_path.has_value()) {
    MS_LOG(EXCEPTION) << "Failed to create directory: '" << dir << "'";
  }
  info_dir_ = dir_path.value();
}

void ConvertCustomForGE::SaveNodesInfo(const AnfNodePtrList &nodes) {
  CreateInfoDir();
  DumpOption option;
  option.get_target_info = true;
  std::set<std::string> unique_kernel_name;
  for (const auto &node : nodes) {
    graphkernel::GraphKernelJsonGenerator graph_kernel_json_generator(option);
    FuncGraphPtr sub_func_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(node);
    MS_EXCEPTION_IF_NULL(sub_func_graph);
    auto mng = sub_func_graph->manager();
    if (mng == nullptr) {
      mng = Manage(sub_func_graph, true);
      sub_func_graph->set_manager(mng);
    }
    std::vector<AnfNodePtr> node_list, input_list, output_list;
    GkUtils::GetValidKernelNodes(sub_func_graph, &node_list, &input_list, &output_list);
    graph_kernel_json_generator.CollectFusedJson(node_list, input_list, output_list);
    auto kernel_name = graph_kernel_json_generator.kernel_name();
    node_json_name_[node] = kernel_name;
    if (!unique_kernel_name.insert(kernel_name).second) {
      continue;
    }
    kernel::SaveJsonInfo(kernel_name, graph_kernel_json_generator.kernel_json_str(), info_dir_ + "/");
  }
}

bool ConvertCustomForGE::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  auto node_list = GkUtils::GetGraphKernelNodes(func_graph);
  // 1. generate node info file
  SaveNodesInfo(node_list);
  // 2. convert fused node to Custom op
  for (const auto &node : node_list) {
    auto cnode = node->cast<CNodePtr>();
    auto custom_cnode = CreateCustomOp(func_graph, cnode);
    if (custom_cnode == nullptr) {
      MS_LOG(EXCEPTION) << "Create custom op failed for " << cnode->fullname_with_scope();
    }
    mng->Replace(node, custom_cnode);
  }
  return !node_list.empty();
}
}  // namespace mindspore::graphkernel
