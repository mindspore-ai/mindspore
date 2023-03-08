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
#include "debug/rdr/graph_exec_order_recorder.h"
#include <fstream>
#include <utility>
#include "mindspore/core/ir/anf.h"
#include "mindspore/core/utils/log_adapter.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/rdr/recorder_manager.h"
#include "mindspore/core/utils/file_utils.h"

namespace mindspore {
namespace {
bool DumpGraphExeOrder(const std::string &filename, const std::vector<CNodePtr> &execution_order) {
  ChangeFileMode(filename, S_IRWXU);
  std::ofstream fout(filename, std::ofstream::app);
  if (!fout.is_open()) {
    MS_LOG(WARNING) << "Open file for saving graph exec order failed.";
    return false;
  }
  fout << "================== execution order ==================\n";
  fout << "execution_order size: " << execution_order.size() << "\n";
  int i = 0;
  for (auto &cnode : execution_order) {
    MS_EXCEPTION_IF_NULL(cnode);
    fout << i << ":\n";
    fout << "\t" << cnode->DebugString() << "\n";
    fout << "\t" << AnfAlgo::GetStreamDistinctionLabel(cnode.get()) << "\n";
    fout << "\t" << AnfAlgo::GetGraphId(cnode.get()) << "\n";
    i++;
  }
  fout << "================== execution order ==================\n";
  fout.close();
  ChangeFileMode(filename, S_IRUSR);
  return true;
}
}  // namespace

void GraphExecOrderRecorder::Export() {
  auto realpath = GetFileRealPath();
  if (!realpath.has_value()) {
    return;
  }
  std::string real_file_path = realpath.value() + ".txt";
  DumpGraphExeOrder(real_file_path, exec_order_);
}

namespace RDR {
bool RecordGraphExecOrder(const SubModuleId module, const std::string &name,
                          const std::vector<CNodePtr> &final_exec_order) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!mindspore::RecorderManager::Instance().RdrEnable() ||
      ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  GraphExecOrderRecorderPtr graph_exec_order_recorder =
    std::make_shared<GraphExecOrderRecorder>(submodule_name, name, final_exec_order);
  bool ans = mindspore::RecorderManager::Instance().RecordObject(std::move(graph_exec_order_recorder));
  return ans;
}
}  // namespace RDR
}  // namespace mindspore
