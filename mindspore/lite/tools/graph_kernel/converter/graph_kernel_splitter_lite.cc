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
#include "utils/anf_utils.h"
#include "common/graph_kernel/graph_kernel_flags.h"
#include "common/graph_kernel/core/tuning_splitter.h"
#include "tools/graph_kernel/converter/akg/akg_build.h"

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
  py_cmd << "from mindspore._extends.parallel_compile.akg_compiler.get_file_path import get_akg_path\n";
  py_cmd << "import sys; sys.path.insert(0, get_akg_path())\n";
  py_cmd << "from akg.ms import " << tune_interface << "\n";
  py_cmd << "if not " << tune_interface << "(\'" << dir_path << "\', " << attrs.str() << "):\n";
  py_cmd << "    raise RuntimeError(\'Tune fail for json: " << dir_path << "\')";
  std::string cmd = "unset LD_LIBRARY_PATH;python -c \"" + py_cmd.str() + "\"";
  MS_LOG(INFO) << "GraphKernel online tuning content: \n" << cmd;
  auto ret = system(cmd.c_str());
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

bool GraphKernelSplitterWithTuning::Run(const FuncGraphPtr &func_graph) {
  if (GraphKernelFlags::GetInstance().online_tuning == 0) {
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
  tuning_path_ = SaveNodesInfo(gknodes, "./split_tuning", &node_name, nullptr);
  if (tuning_path_.empty()) {
    tuning_flag_ = false;
  } else {
    tuning_flag_ = StartTuning(tuning_path_);
  }
  for (const auto &iter : node_name) {
    AnfUtils::SetNodeAttr("kernel_name", MakeValue(iter.second), iter.first);
  }
  return GraphKernelSplitter::Run(func_graph);
}
}  // namespace mindspore::graphkernel
