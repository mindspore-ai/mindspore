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
#include "debug/rdr/graph_recorder.h"
#include "mindspore/core/base/base.h"
#include "mindspore/core/ir/func_graph.h"
#include "mindspore/core/utils/log_adapter.h"
#include "debug/anf_ir_dump.h"
#include "debug/anf_ir_utils.h"
#include "debug/dump_proto.h"

namespace mindspore {
namespace dumper {
#ifdef ENABLE_DUMP_IR
void DumpIRProto(const std::string &filename, const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Func graph is nullptr";
    return;
  }

  if (filename.size() > PATH_MAX) {
    MS_LOG(ERROR) << "File path " << filename << " is too long.";
    return;
  }
  char real_path[PATH_MAX] = {0};
  char *real_path_ret = nullptr;
#if defined(_WIN32) || defined(_WIN64)
  real_path_ret = _fullpath(real_path, filename.c_str(), PATH_MAX);
#else
  real_path_ret = realpath(filename.c_str(), real_path);
#endif
  if (nullptr == real_path_ret) {
    MS_LOG(DEBUG) << "dir " << filename << " does not exit.";
  } else {
    std::string path_string = real_path;
    if (chmod(common::SafeCStr(path_string), S_IRUSR | S_IWUSR) == -1) {
      MS_LOG(ERROR) << "Modify file:" << real_path << " to rw fail.";
      return;
    }
  }

  // write to pb file
  std::ofstream ofs(real_path);
  if (!ofs.is_open()) {
    MS_LOG(ERROR) << "Open file '" << real_path << "' failed!";
    return;
  }
  ofs << GetFuncGraphProtoString(func_graph);
  ofs.close();
  // set file mode to read only by user
  ChangeFileMode(real_path, S_IRUSR);
}
#else
void DumpIRProto(const FuncGraphPtr &, const std::string &) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping function graph IR in protobuf format is disabled, "
                  << "please recompile source to enable it. See help of building script.";
}
#endif
}  // namespace dumper

void GraphRecorder::Export() {
  bool save_flag = false;
  if (filename_.empty()) {
    filename_ = directory_ + module_ + "_" + tag_ + "_" + timestamp_;
  }
  if (graph_type_.find(".dat") != std::string::npos) {
    save_flag = true;
    AnfExporter exporter(std::to_string(id_));
    std::string filename = filename_ + ".dat";
    ChangeFileMode(filename, S_IRWXU);
    exporter.ExportFuncGraph(filename, func_graph_);
    ChangeFileMode(filename, S_IRUSR);
  }
  if (graph_type_.find(".ir") != std::string::npos) {
    save_flag = true;
    if (full_name_) {
      DumpIRForRDR(filename_ + ".ir", func_graph_, true, kTopStack);
    } else {
      DumpIRForRDR(filename_ + ".ir", func_graph_, false, kOff);
    }
  }
  if (graph_type_.find(".pb") != std::string::npos) {
    save_flag = true;
    dumper::DumpIRProto(filename_ + ".pb", func_graph_);  // save *.pb file
  }
  if (!save_flag) {
    MS_LOG(WARNING) << "Unknown save graph type: " << graph_type_;
  }
}
}  // namespace mindspore
