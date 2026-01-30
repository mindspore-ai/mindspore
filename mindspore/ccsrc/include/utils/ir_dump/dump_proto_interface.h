/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_UTILS_IR_DUMP_PROTO_INTERFACE_H_
#define MINDSPORE_CCSRC_UTILS_IR_DUMP_PROTO_INTERFACE_H_

#include <fstream>
#include <string>
#include <vector>

#include "include/utils/common.h"
#include "include/utils/visible.h"
#include "ir/func_graph.h"
#include "proto/mind_ir.pb.h"

namespace mindspore {
using ModelProtoPtr = std::shared_ptr<mind_ir::ModelProto>;

COMMON_EXPORT std::string GetFuncGraphProtoString(const FuncGraphPtr &func_graph);

COMMON_EXPORT std::string GetBinaryProtoString(const FuncGraphPtr &func_graph, const bool &incremental = false);

COMMON_EXPORT ModelProtoPtr GenBinaryProto(const FuncGraphPtr &func_graph);

COMMON_EXPORT bool DumpBinaryProto(const FuncGraphPtr &func_graph, const std::string &file_path);
COMMON_EXPORT bool DumpBinaryProto(const FuncGraphPtr &root_graph, const std::vector<FuncGraphPtr> &child_graphs,
                                   const std::vector<AnfNodePtr> &isolated_nodes, const std::string &file_path);
COMMON_EXPORT void DumpIRProto(const FuncGraphPtr &func_graph, const std::string &suffix);

COMMON_EXPORT std::string GetFuncGraphProtoJsonString(const FuncGraphPtr &func_graph);

class COMMON_EXPORT MindIRExporter {
 public:
  MindIRExporter() {}
  explicit MindIRExporter(bool is_export_model) { is_export_model_ = is_export_model; }
  virtual ~MindIRExporter() {
    if (data_fs_ != nullptr) {
      data_fs_->close();
      delete data_fs_;
      data_fs_ = nullptr;
    }
  }

  bool ExportProto(const FuncGraphPtr &func_graph, const std::string &file_path,
                   const FuncGraphPtr &param_layout_fg = nullptr);
  bool IsSystemLittleEndidan() const;
  bool PreProcSaveTogether(const FuncGraphPtr &func_graph);
  bool SaveProtoToFile(mind_ir::ModelProto *model_proto, const std::string &output_file);

 private:
  bool ParserPath(const std::string &output_path);
  bool IfSaveTogether(bool *save_together);
  bool SaveMindIRTogether();
  bool SplitSave();
  bool UpdateParamCount(const FuncGraphPtr &func_graph);

 private:
  bool ParamDict(const FuncGraphPtr &func_graph);
  bool CreateParameterDir();
  std::shared_ptr<Parameter> GetFgParaAccordingToProtoName(const std::string &proto_name);
  bool ChangeParaDataFile(const std::string &file);
  std::string CreateExternalPath(const std::string &external_file);

 private:
  std::string model_name_;
  std::string save_path_;
  std::string save_model_path_;
  std::string dir_name_;
  std::string dir_path_;
  bool save_together_ = true;
  mind_ir::ModelProto model_proto_;
  std::unordered_map<std::string, ParameterPtr> param_dict_{};
  std::unordered_map<tensor::TensorPtr, mind_ir::TensorProto *> para_proto_dict_{};
  std::fstream *data_fs_ = nullptr;
  std::shared_ptr<system::FileSystem> fs_{};
  bool is_export_model_ = true;
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_IR_DUMP_PROTO_INTERFACE_H_
