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

#ifndef MINDSPORE_LITE_TOOLS_MINDIR_EXPORTER_MINDIR_SERIALIZER_H_
#define MINDSPORE_LITE_TOOLS_MINDIR_EXPORTER_MINDIR_SERIALIZER_H_

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <set>
#include "mindspore/core/ir/func_graph.h"
#include "tools/converter/converter_context.h"
#include "proto/mind_ir.pb.h"
#include "mindspore/core/utils/system/env.h"
#include "tools/converter/cxx_api/converter_para.h"

namespace mindspore::lite {
class MindIRSerializer {
 public:
  MindIRSerializer() {}
  explicit MindIRSerializer(bool is_export_model) { is_export_model_ = is_export_model; }
  virtual ~MindIRSerializer() {
    if (data_fs_ != nullptr) {
      data_fs_->close();
      delete data_fs_;
      data_fs_ = nullptr;
    }
  }
  int Save(const std::shared_ptr<ConverterPara> &param, const FuncGraphPtr &func_graph);
  int GetBuffAndSize(void **buff, size_t *size);
  int PreProcSaveTogether(const FuncGraphPtr &func_graph);

 private:
  int ParserPath(const std::string &output_path);
  int IfSaveTogether(bool *save_together);
  int SaveMindIRTogether(const std::shared_ptr<ConverterPara> &param);
  int SplitSave(const std::shared_ptr<ConverterPara> &param);
  int SaveProtoToFile(mind_ir::ModelProto *model_proto, const std::string &output_file,
                      const std::shared_ptr<ConverterPara> &param);
  int ConvertQuantHolderToQuantizationParam(const FuncGraphPtr &func_graph);
  std::shared_ptr<mindspore::QuantizationParam> ConvertQuantParamTToQuantizationParam(
    std::vector<schema::QuantParamT> quant_param);
  int UpdateParamCount(const FuncGraphPtr &func_graph);

 private:
  int ParamDict(const FuncGraphPtr &func_graph);
  int CreateParameterDir();
  std::shared_ptr<Parameter> GetFgParaAccordingToProtoName(const std::string &proto_name);
  int ChangeParaDataFile(const std::string &file);
  bool IsSystemLittleEndidan() const;
  int GetDataFile(const std::string &data_file_name, std::ofstream *fout, int64_t *parameter_size, int64_t *offset);
  std::string CreateExternalPath(const std::string &external_file);
  int RemoveQuantParameterHolder(FuncGraphPtr func_graph);

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
// export func_graph
int MindIRSerialize(const std::shared_ptr<ConverterPara> &param, const FuncGraphPtr &func_graph, bool need_buff,
                    void **buff, size_t *size);
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_TOOLS_MINDIR_EXPORTER_MINDIR_SERIALIZER_H_
