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

#include "src/common/draw/drawer.h"
#ifdef ENABLE_DRAW
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include "src/common/file_utils.h"
#include "src/common/draw/adapter_graphs/sub_graph_kernel_adapter_graph.h"
#include "src/common/draw/adapter_graphs/compile_result_adapter_graph.h"
#endif

namespace mindspore::lite {
constexpr char kDefaultDrawDIR[] = "./graphs";
#ifdef ENABLE_DRAW
inline void Drawer::Reset() { count_ = 0; }

void Drawer::Init() {
  auto ret = CreateDir(kDefaultDrawDIR);
  if (ret != RET_OK) {
    MS_LOG(WARNING) << "Create draw directory failed, disable draw.";
    enabled_ = false;
  }
  if (enabled_) {
    base_dir_ = RealPath(kDefaultDrawDIR);
    if (base_dir_.empty()) {
      MS_LOG(WARNING) << kDefaultDrawDIR << " is invalid: " << base_dir_ << ", disable draw.";
      enabled_ = false;
    }
  }
  Reset();
}

std::string Drawer::GetNextFileName(const std::string &name) {
  std::ostringstream oss;
  oss << std::setw(3) << std::setfill('0') << count_++ << '-' << name << ".dot";
  return oss.str();
}

inline bool Drawer::SaveDotFile(const std::string &dot_name, const std::string &dot_content) {
  auto fname = GetNextFileName(dot_name);
  auto write_path = lite::WriteStrToFile(this->base_dir_, fname, dot_content);
  if (write_path.empty()) {
    MS_LOG(ERROR) << "Save dot-file failed, path: " << this->base_dir_ << ", fname: " << fname;
    return false;
  } else {
    MS_LOG(INFO) << "Save dot-file successfully, path: " << write_path;
    return true;
  }
}

void Drawer::Draw(const kernel::SubGraphKernel *graph, const std::string &name) {
  if (!enabled_) {
    return;
  }
  auto gv_graph = lite::CreateGVGraph(graph);
  if (gv_graph == nullptr) {
    MS_LOG(ERROR) << "Create gv_graph failed.";
    return;
  }
  (void)SaveDotFile(name, gv_graph->Code());
}
#ifdef ENABLE_CLOUD_INFERENCE
void Drawer::Draw(const CompileResult *graph, const std::string &name) {
  if (!enabled_) {
    return;
  }
  auto gv_graph = lite::CreateGVGraph(graph);
  if (gv_graph == nullptr) {
    MS_LOG(ERROR) << "Create gv_graph failed.";
    return;
  }
  (void)SaveDotFile(name, gv_graph->Code());
}
#endif
#else
#define WARNLOG                                                                                           \
  MS_LOG(WARNING) << "Drawer is not enabled, please set env 'export MSLITE_EXPORT_COMPUTE_IR=on; export " \
                  << kDrawDIREnvKey << "=/path/to/draw_dir' to enable drawer."

inline void Drawer::Reset() { WARNLOG; }

void Drawer::Init() { WARNLOG; }

inline bool Drawer::SaveDotFile(const std::string &dot_name, const std::string &dot_content) {
  WARNLOG;
  return false;
}

void Drawer::Draw(const kernel::SubGraphKernel *graph, const std::string &name) { WARNLOG; }
#ifdef ENABLE_CLOUD_INFERENCE
void Drawer::Draw(const CompileResult *graph, const std::string &name) { WARNLOG; }
#endif
#endif
}  // namespace mindspore::lite
