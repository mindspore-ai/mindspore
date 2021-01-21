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
void GraphRecorder::Export() {
  bool save_flag = false;
  if (filename_.empty()) {
    filename_ = directory_ + module_ + "_" + tag_ + "_" + timestamp_;
  }
  if (graph_type_.find(".dat") != std::string::npos) {
    save_flag = true;
    ExportIR(filename_ + ".dat", std::to_string(id_), func_graph_);  // saving  *.dat file
  }
  if (graph_type_.find(".ir") != std::string::npos) {
    save_flag = true;
    DumpIR(filename_ + ".ir", func_graph_);  // saving *.ir file
  }
  if (graph_type_.find(".pb") != std::string::npos) {
    save_flag = true;
    DumpIRProto(func_graph_, filename_);  // save *.pb file
  }
  if (!save_flag) {
    MS_LOG(WARNING) << "Unknown save graph type: " << graph_type_;
  }
}
}  // namespace mindspore
