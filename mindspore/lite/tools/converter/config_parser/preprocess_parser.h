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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_PREPROCESS_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_PREPROCESS_PARSER_H
#include <string>
#include <map>
#include <vector>
#include "tools/converter/preprocess/preprocess_param.h"
#include "tools/converter/config_parser/config_file_parser.h"

namespace mindspore {
namespace lite {
class PreprocessParser {
 public:
  int ParsePreprocess(const DataPreProcessString &data_pre_process_str,
                      preprocess::DataPreProcessParam *data_pre_process);

 private:
  int ParseInputType(const std::string &input_type_str, preprocess::InputType *input_type);

  int ParseImagePreProcess(const DataPreProcessString &data_pre_process_str,
                           preprocess::ImagePreProcessParam *image_pre_process);

  int ParseImageNormalize(const DataPreProcessString &data_pre_process_str,
                          preprocess::ImagePreProcessParam *image_pre_process);

  int ParseImageResize(const DataPreProcessString &data_pre_process_str,
                       preprocess::ImagePreProcessParam *image_pre_process);

  int ParseImageCenterCrop(const DataPreProcessString &data_pre_process_str,
                           preprocess::ImagePreProcessParam *image_pre_process);

  int ParseImageToFormat(const std::string &image_to_format_str, preprocess::ImageToFormat *image_to_format);

  int ParseCalibratePath(const std::string &str, std::map<std::string, std::string> *value);

  int CollectCalibInputs(const std::map<std::string, std::string> &calibrate_data_path, size_t limited_count,
                         std::map<std::string, std::vector<std::string>> *inputs);
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_PREPROCESS_PARSER_H
