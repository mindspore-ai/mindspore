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
#include "tools/graph_kernel/converter/graph_kernel_splitter_lite.h"
#include <map>
#include <memory>
#include <cstdio>
#include <vector>
#include "utils/system/env.h"
#include "utils/file_utils.h"
#include "utils/anf_utils.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/core/tuning_splitter.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "tools/graph_kernel/converter/akg/akg_kernel_builder.h"

namespace mindspore::graphkernel {
SplitSchemerPtr GraphKernelSplitterWithTuning::GetSplitSchema(const std::string &processor) {
  if (tuning_flag_) {
    return std::make_shared<TuningSplitSchemer>(tuning_path_);
  }
  return GraphKernelSplitter::GetSplitSchema(processor);
}

bool GraphKernelSplitterWithTuning::StartTuning(const std::string &dir_path) const {
  std::ostringstream attrs;
  attrs << "{";
  attrs << "\'repository_path\':\'" << dir_path << "\'";
  if (common::GetEnv("MS_DEV_GRAPH_KERNEL_SPLIT_DEBUG_TUNING") != "on") {
    attrs << ",\'online_tuning\':" << GraphKernelFlags::GetInstance().online_tuning;
  }
  attrs << "}";
  std::ostringstream py_cmd;
  std::string tune_interface = "poly_graph_split_with_json_dir";
  py_cmd << kAddAkgPath;
  py_cmd << "from akg.ms import " << tune_interface << "\n";
  py_cmd << "if not " << tune_interface << "(\'" << dir_path << "\', " << attrs.str() << "):\n";
  py_cmd << "    raise RuntimeError(\'Tune fail. info path: " << dir_path << "\')";
  std::string cmd = "python -c \"" + py_cmd.str() + "\"";
  MS_LOG(INFO) << "GraphKernel split tuning content: \n" << cmd;
  auto ret = std::system(cmd.c_str());
  if (!WIFEXITED(ret)) {
    MS_LOG(ERROR) << "Python process start fail! process content is as follows:\n" << cmd;
    return false;
  }
  if (WEXITSTATUS(ret) != 0) {
    MS_LOG(ERROR) << "Failed to tune kernel: " << dir_path;
    return false;
  }
  return true;
}

void SignTunedGraphs(const FuncGraphPtr &func_graph) {
  auto kernel_meta = FileUtils::GetRealPath("./akg_kernel_meta/");
  if (!kernel_meta.has_value()) {
    return;
  }
  auto fs = system::Env::GetFileSystem();
  MS_EXCEPTION_IF_NULL(fs);
  DumpOption option = AkgKernelBuilder::json_option();
  option.gen_kernel_name_only = true;

  auto todos = TopoSort(func_graph->get_return());
  for (const auto &node : todos) {
    if (!AnfUtils::IsGraphKernel(node)) {
      continue;
    }
    auto fg = GetCNodeFuncGraph(node);
    if (!fg->has_attr(kAttrNodeName)) {
      continue;
    }
    auto node_name = GetValue<std::string>(fg->get_attr(kAttrNodeName));
    auto kernel_obj = kernel_meta.value() + "/best_split_" + node_name + ".o";
    if (fs->FileExist(kernel_obj)) {
      // sign the funcgraph with its current kernel name, the tuned result can be used if
      // its kernel name is the same as the signature when building kernels.
      AkgKernelJsonGenerator json_generator(option);
      std::vector<AnfNodePtr> node_list, input_list, output_list;
      GkUtils::GetValidKernelNodes(fg, &node_list, &input_list, &output_list);
      (void)json_generator.CollectFusedJson(node_list, input_list, output_list);
      fg->set_attr(kTunedSign, MakeValue(json_generator.kernel_name()));
      MS_LOG(INFO) << "The " << kernel_obj << " is the tuning result of " << json_generator.kernel_name();
    }
  }
}

bool GraphKernelSplitterWithTuning::Run(const FuncGraphPtr &func_graph) {
  if (Callback::Instance()->GetTargetFromContext() == kAscendDevice ||
      GraphKernelFlags::GetInstance().online_tuning == 0) {
    tuning_flag_ = false;
    return GraphKernelSplitter::Run(func_graph);
  }
  auto todos = TopoSort(func_graph->get_return());
  AnfNodePtrList gknodes;
  std::copy_if(todos.cbegin(), todos.cend(), std::back_inserter(gknodes), AnfUtils::IsGraphKernel);
  if (gknodes.empty()) {
    return false;
  }
  std::map<AnfNodePtr, std::string> node_name;
  tuning_path_ = SaveNodesInfo(gknodes, "./split_tuning", AkgKernelBuilder::json_option(), &node_name, nullptr);
  if (tuning_path_.empty()) {
    tuning_flag_ = false;
  } else {
    tuning_flag_ = StartTuning(tuning_path_);
  }
  for (const auto &iter : node_name) {
    AnfUtils::SetNodeAttr(kAttrNodeName, MakeValue(iter.second), iter.first);
  }
  auto changed = GraphKernelSplitter::Run(func_graph);
  if (tuning_flag_) {
    SignTunedGraphs(func_graph);
  }
  return changed;
}
}  // namespace mindspore::graphkernel
