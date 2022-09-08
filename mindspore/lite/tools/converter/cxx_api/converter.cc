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
#include "include/converter.h"
#include "include/api/data_type.h"
#include "tools/converter/cxx_api/converter_para.h"
#include "tools/converter/converter_context.h"
#include "tools/converter/converter.h"
#include "src/common/log_adapter.h"

namespace mindspore {
namespace {
constexpr size_t kMaxSectionNum = 100;
constexpr size_t kMaxConfigNumPerSection = 1000;
}  // namespace
namespace lite {
int RunConverter(const std::shared_ptr<ConverterPara> &data_);
}
Converter::Converter(converter::FmkType fmk_type, const std::string &model_file, const std::string &output_file,
                     const std::string &weight_file) {
  data_ = std::make_shared<ConverterPara>();
  if (data_ != nullptr) {
    data_->fmk_type = fmk_type;
    data_->model_file = model_file;
    data_->output_file = output_file;
    data_->weight_file = weight_file;
  } else {
    MS_LOG(ERROR) << "Create ConverterPara failed";
  }
}

void Converter::SetConfigFile(const std::string &config_file) {
  if (data_ != nullptr) {
    data_->config_file = config_file;
  }
}

std::string Converter::GetConfigFile() const {
  if (data_ != nullptr) {
    return data_->config_file;
  } else {
    return "";
  }
}

void Converter::SetConfigInfo(const std::string &section, const std::map<std::string, std::string> &config) {
  if (data_ != nullptr) {
    if (data_->config_param.size() > kMaxSectionNum) {
      MS_LOG(ERROR) << "Section num " << data_->config_param.size() << "exceeds max num " << kMaxSectionNum;
      return;
    }
    if (data_->config_param.find(section) != data_->config_param.end()) {
      MS_LOG(WARNING) << "Section " << section << "already exists, "
                      << "value will be overwrite.";
    }
    if (config.size() > kMaxConfigNumPerSection) {
      MS_LOG(ERROR) << "Config num " << config.size() << " exceeds max num " << kMaxConfigNumPerSection << " in "
                    << section;
      return;
    }
    data_->config_param[section] = config;
  }
}

std::map<std::string, std::map<std::string, std::string>> Converter::GetConfigInfo() const {
  return data_->config_param;
}

void Converter::SetWeightFp16(bool weight_fp16) {
  if (data_ != nullptr) {
    data_->weight_fp16 = weight_fp16;
  }
}

bool Converter::GetWeightFp16() const {
  if (data_ != nullptr) {
    return data_->weight_fp16;
  } else {
    return false;
  }
}

void Converter::SetInputShape(const std::map<std::string, std::vector<int64_t>> &input_shape) {
  if (data_ != nullptr) {
    for (auto &it : input_shape) {
      lite::ConverterInnerContext::GetInstance()->UpdateGraphInputTensorShape(it.first, it.second);
    }
    data_->input_shape = input_shape;
  }
}

std::map<std::string, std::vector<int64_t>> Converter::GetInputShape() const {
  if (data_ != nullptr) {
    return data_->input_shape;
  } else {
    return {};
  }
}

void Converter::SetInputFormat(Format format) {
  if (data_ != nullptr) {
    if (format != DEFAULT_FORMAT) {
      data_->input_format = format;
    }
    data_->spec_input_format = format;
  }
}

Format Converter::GetInputFormat() const {
  if (data_ != nullptr) {
    return data_->input_format;
  } else {
    return DEFAULT_FORMAT;
  }
}

void Converter::SetInputDataType(DataType data_type) {
  if (data_ != nullptr) {
    data_->input_data_type = data_type;
  }
}

DataType Converter::GetInputDataType() {
  if (data_ != nullptr) {
    return data_->input_data_type;
  } else {
    return DataType::kTypeUnknown;
  }
}

void Converter::SetOutputDataType(DataType data_type) {
  if (data_ != nullptr) {
    data_->output_data_type = data_type;
  }
}

DataType Converter::GetOutputDataType() {
  if (data_ != nullptr) {
    return data_->output_data_type;
  } else {
    return DataType::kTypeUnknown;
  }
}

void Converter::SetExportMindIR(ModelType export_mindir) {
  if (data_ != nullptr) {
    data_->export_mindir = export_mindir;
  }
}

ModelType Converter::GetExportMindIR() const {
  if (data_ != nullptr) {
    return data_->export_mindir;
  } else {
    return kMindIR_Lite;
  }
}

void Converter::SetDecryptKey(const std::string &key) {
  if (data_ != nullptr) {
    data_->decrypt_key = key;
  }
}

std::string Converter::GetDecryptKey() const {
  if (data_ != nullptr) {
    return data_->decrypt_key;
  } else {
    return "";
  }
}

void Converter::SetDecryptMode(const std::string &mode) {
  if (data_ != nullptr) {
    data_->decrypt_mode = mode;
  }
}

std::string Converter::GetDecryptMode() const {
  if (data_ != nullptr) {
    return data_->decrypt_mode;
  } else {
    return "";
  }
}

void Converter::SetEnableEncryption(bool encryption) {
  if (data_ != nullptr) {
    data_->enable_encryption = encryption;
  }
}

bool Converter::GetEnableEncryption() const {
  if (data_ != nullptr) {
    return data_->enable_encryption;
  } else {
    return false;
  }
}

void Converter::SetEncryptKey(const std::string &key) {
  if (data_ != nullptr) {
    data_->encrypt_key = key;
  }
}

std::string Converter::GetEncryptKey() const {
  if (data_ != nullptr) {
    return data_->encrypt_key;
  } else {
    return "";
  }
}

void Converter::SetInfer(bool infer) {
  if (data_ != nullptr) {
    data_->pre_infer = infer;
  }
}

bool Converter::GetInfer() const {
  if (data_ != nullptr) {
    return data_->pre_infer;
  } else {
    return false;
  }
}

void Converter::SetTrainModel(bool train_model) {
  if (data_ != nullptr) {
    data_->train_model = train_model;
  }
}

bool Converter::GetTrainModel() const {
  if (data_ != nullptr) {
    return data_->train_model;
  } else {
    return false;
  }
}

void Converter::SetNoFusion(bool no_fusion) {
  if (data_ != nullptr) {
    data_->no_fusion = no_fusion;
  }
}

bool Converter::GetNoFusion() {
  if (data_ != nullptr) {
    return data_->no_fusion;
  } else {
    return false;
  }
}

void Converter::SetDevice(const std::string &device) {
  if (data_ != nullptr) {
    data_->device = device;
  }
}

std::string Converter::GetDevice() {
  if (data_ != nullptr) {
    return data_->device;
  } else {
    return "";
  }
}

Status Converter::Convert() {
  if (data_ != nullptr) {
    Status ret = Status(static_cast<StatusCode>(lite::RunConverter(data_, nullptr, nullptr, false)));
    data_->decrypt_key.clear();  // clear key
    data_->encrypt_key.clear();  // clear key
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Convert model failed, ret=" << ret;
    }
    return ret;
  } else {
    return kLiteError;
  }
}

void *Converter::Convert(size_t *data_size) {
  void *model_data = nullptr;
  if (data_ != nullptr) {
    Status ret = Status(static_cast<StatusCode>(lite::RunConverter(data_, &model_data, data_size, true)));
    data_->decrypt_key.clear();  // clear key
    data_->encrypt_key.clear();  // clear key
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Convert model failed, ret=" << ret;
    }
  } else {
    MS_LOG(ERROR) << "Convert model failed, data is null.";
  }
  return model_data;
}
}  // namespace mindspore
