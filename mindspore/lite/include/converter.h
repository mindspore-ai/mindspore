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
#ifndef MINDSPORE_LITE_INCLUDE_CONVERTER_H_
#define MINDSPORE_LITE_INCLUDE_CONVERTER_H_

#include <map>
#include <string>
#include <vector>
#include <memory>
#include "include/api/format.h"
#include "include/api/status.h"
#include "include/registry/converter_context.h"

namespace mindspore {
struct ConverterPara;
class MS_API Converter {
 public:
  Converter(converter::FmkType fmk_type, const std::string &model_file, const std::string &output_file,
            const std::string &weight_file = "");
  ~Converter() = default;

  void SetConfigFile(const std::string &config_file);
  std::string GetConfigFile() const;

  void SetWeightFp16(bool weight_fp16);
  bool GetWeightFp16() const;

  void SetInputShape(const std::map<std::string, std::vector<int64_t>> &input_shape);
  std::map<std::string, std::vector<int64_t>> GetInputShape() const;

  void SetInputFormat(Format format);
  Format GetInputFormat() const;

  void SetInputDataType(DataType data_type);
  DataType GetInputDataType();

  void SetOutputDataType(DataType data_type);
  DataType GetOutputDataType();

  void SetExportMindIR(bool export_mindir);
  bool GetExportMindIR() const;

  void SetDecryptKey(const std::string &key);
  std::string GetDecryptKey() const;

  void SetDecryptMode(const std::string &mode);
  std::string GetDecryptMode() const;

  void SetEnableEncryption(bool encryption);
  bool GetEnableEncryption() const;

  void SetEncryptKey(const std::string &key);
  std::string GetEncryptKey() const;

  void SetInfer(bool infer);
  bool GetInfer() const;

  void SetTrainModel(bool train_model);
  bool GetTrainModel() const;

  void SetNoFusion(bool no_fusion);
  bool GetNoFusion();

  Status Convert();

 private:
  std::shared_ptr<ConverterPara> data_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_CONVERTER_H_
