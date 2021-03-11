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
#include <sstream>
#include <fstream>
#include "mindspore/core/ir/anf.h"
#include "mindspore/core/utils/log_adapter.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/utils.h"

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
}  // namespace mindspore
