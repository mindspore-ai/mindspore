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
#include "backend/session/kernel_graph.h"
#include "mindspore/core/utils/log_adapter.h"
#include "debug/anf_ir_dump.h"
#include "debug/anf_ir_utils.h"
#include "debug/dump_proto.h"
#include "debug/common.h"

namespace mindspore {
namespace protobuf {
#ifdef ENABLE_DUMP_IR
void DumpIRProto(const std::string &real_path, const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Func graph is nullptr.";
    return;
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
void DumpIRProto(const std::string &, const FuncGraphPtr &) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping function graph IR in protobuf format is disabled, "
                  << "please recompile source to enable it. See help of building script.";
}
#endif
}  // namespace protobuf

void GraphRecorder::Export() {
  bool save_flag = false;
  auto tmp_realpath = GetFileRealPath();
  if (!tmp_realpath.has_value()) {
    return;
  }
  std::string realpath = tmp_realpath.value();
  if (graph_type_.find(".dat") != std::string::npos) {
    save_flag = true;
    AnfExporter exporter("");
    std::string realpath_dat = realpath + ".dat";
    ChangeFileMode(realpath_dat, S_IRWXU);
    exporter.ExportFuncGraph(realpath_dat, func_graph_);
    ChangeFileMode(realpath_dat, S_IRUSR);
  }
  if (graph_type_.find(".ir") != std::string::npos) {
    save_flag = true;
    std::string realpath_ir = realpath + ".ir";
    if (dump_graph_info_.dump_mode <= static_cast<int>(kWholeStack) &&
        dump_graph_info_.dump_mode >= static_cast<int>(kOff)) {
      LocDumpMode dump_mode = LocDumpMode(dump_graph_info_.dump_mode);
      DumpIRForRDR(realpath_ir, func_graph_, dump_graph_info_.dump_full_name, dump_mode);
    } else {
      MS_LOG(WARNING) << "Unknown save graph LocDumpMode: " << dump_graph_info_.dump_mode
                      << ", it must be in the range [0,2].";
    }
  }
  if (graph_type_.find(".pb") != std::string::npos) {
    save_flag = true;
    protobuf::DumpIRProto(realpath + ".pb", func_graph_);  // save *.pb file
  }
  if (!save_flag) {
    MS_LOG(WARNING) << "Unknown save graph type: " << graph_type_;
  }
}
}  // namespace mindspore
