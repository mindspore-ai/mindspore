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
#include "include/api/dual_abi_helper.h"

namespace mindspore {
struct ConverterPara;
/// \brief Converter provides C++ API for user to integrate model conversion into user application.
///
/// \note Converter C++ API cannot be used in Converter main process.
///
/// \note Converter C++ API doesn't support calling with multi-threads in a single process.
class MS_API Converter {
 public:
  inline Converter(converter::FmkType fmk_type, const std::string &model_file, const std::string &output_file = "",
                   const std::string &weight_file = "");
  ~Converter() = default;

  inline void SetConfigFile(const std::string &config_file);
  inline std::string GetConfigFile() const;

  inline void SetConfigInfo(const std::string &section, const std::map<std::string, std::string> &config);
  inline std::map<std::string, std::map<std::string, std::string>> GetConfigInfo() const;

  void SetWeightFp16(bool weight_fp16);
  bool GetWeightFp16() const;

  inline void SetInputShape(const std::map<std::string, std::vector<int64_t>> &input_shape);
  inline std::map<std::string, std::vector<int64_t>> GetInputShape() const;

  void SetInputFormat(Format format);
  Format GetInputFormat() const;

  void SetInputDataType(DataType data_type);
  DataType GetInputDataType();

  void SetOutputDataType(DataType data_type);
  DataType GetOutputDataType();

  void SetSaveType(ModelType save_type);
  ModelType GetSaveType() const;

  inline void SetDecryptKey(const std::string &key);
  inline std::string GetDecryptKey() const;

  inline void SetDecryptMode(const std::string &mode);
  inline std::string GetDecryptMode() const;

  void SetEnableEncryption(bool encryption);
  bool GetEnableEncryption() const;

  inline void SetEncryptKey(const std::string &key);
  inline std::string GetEncryptKey() const;

  void SetInfer(bool infer);
  bool GetInfer() const;

  void SetTrainModel(bool train_model);
  bool GetTrainModel() const;

  void SetNoFusion(bool no_fusion);
  bool GetNoFusion();

  void SetOptimizeTransformer(bool optimize_transformer);
  bool GetOptimizeTransformer();

  inline void SetDevice(const std::string &device);
  inline std::string GetDevice();

  /// \brief Convert model and save .ms format model into `output_file` that passed in constructor.
  Status Convert();

  /// \brief Convert model and return converted FlatBuffer model binary buffer.
  ///
  /// \param[in] data_size Converted FlatBuffer model's buffer size.
  ///
  /// \return A pointer to converted FlatBuffer model buffer.
  void *Convert(size_t *data_size);

 private:
  Converter(converter::FmkType fmk_type, const std::vector<char> &model_file, const std::vector<char> &output_file,
            const std::vector<char> &weight_file);
  void SetConfigFile(const std::vector<char> &config_file);
  std::vector<char> GetConfigFileChar() const;
  void SetConfigInfo(const std::vector<char> &section, const std::map<std::vector<char>, std::vector<char>> &config);
  std::map<std::vector<char>, std::map<std::vector<char>, std::vector<char>>> GetConfigInfoChar() const;
  void SetInputShape(const std::map<std::vector<char>, std::vector<int64_t>> &input_shape);
  std::map<std::vector<char>, std::vector<int64_t>> GetInputShapeChar() const;
  void SetDecryptKey(const std::vector<char> &key);
  std::vector<char> GetDecryptKeyChar() const;
  void SetDecryptMode(const std::vector<char> &mode);
  std::vector<char> GetDecryptModeChar() const;
  void SetEncryptKey(const std::vector<char> &key);
  std::vector<char> GetEncryptKeyChar() const;
  void SetDevice(const std::vector<char> &device);
  std::vector<char> GetDeviceChar();
  std::shared_ptr<ConverterPara> data_;
};

Converter::Converter(converter::FmkType fmk_type, const std::string &model_file, const std::string &output_file,
                     const std::string &weight_file)
    : Converter(fmk_type, StringToChar(model_file), StringToChar(output_file), StringToChar(weight_file)) {}

void Converter::SetConfigFile(const std::string &config_file) { SetConfigFile(StringToChar(config_file)); }

std::string Converter::GetConfigFile() const { return CharToString(GetConfigFileChar()); }

void Converter::SetConfigInfo(const std::string &section, const std::map<std::string, std::string> &config) {
  SetConfigInfo(StringToChar(section), MapStringToVectorChar(config));
}

std::map<std::string, std::map<std::string, std::string>> Converter::GetConfigInfo() const {
  return MapMapCharToString(GetConfigInfoChar());
}

void Converter::SetInputShape(const std::map<std::string, std::vector<int64_t>> &input_shape) {
  SetInputShape(MapStringToChar(input_shape));
}

std::map<std::string, std::vector<int64_t>> Converter::GetInputShape() const {
  return MapCharToString(GetInputShapeChar());
}

void Converter::SetDecryptKey(const std::string &key) { SetDecryptKey(StringToChar(key)); }

std::string Converter::GetDecryptKey() const { return CharToString(GetDecryptKeyChar()); }

void Converter::SetDecryptMode(const std::string &mode) { SetDecryptMode(StringToChar(mode)); }

std::string Converter::GetDecryptMode() const { return CharToString(GetDecryptModeChar()); }

void Converter::SetEncryptKey(const std::string &key) { SetEncryptKey(StringToChar(key)); }

std::string Converter::GetEncryptKey() const { return CharToString(GetEncryptKeyChar()); }

void Converter::SetDevice(const std::string &device) { SetDevice(StringToChar(device)); }

std::string Converter::GetDevice() { return CharToString(GetDeviceChar()); }
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_CONVERTER_H_
