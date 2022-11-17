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

#include "src/mapper_config_parser.h"
#include <algorithm>
#include <stack>
#include <unordered_set>
#include "common/op_enum.h"
#include "common/string_util.h"
#include "common/file_util.h"
#include "mindapi/base/logging.h"

namespace mindspore {
namespace dpico {
namespace {
constexpr size_t kPairSize = 2;
constexpr size_t kMaxPixelVal = 255;
constexpr int kMaxVarReciChn = 65504;
const std::unordered_set<std::string> kSupportedInputFormat = {"YUV400", "BGR_PLANAR", "RGB_PLANAR"};
const std::unordered_set<std::string> kSupportedModelFormat = {"RGB", "BGR"};
const std::unordered_map<std::string, TypeId> kDpicoDataTypeMap = {
  {"FP16", kNumberTypeFloat16},  {"FP32", kNumberTypeFloat32}, {"INT16", kNumberTypeInt16}, {"INT8", kNumberTypeInt8},
  {"S16", kNumberTypeInt16},     {"S8", kNumberTypeInt8},      {"U16", kNumberTypeUInt16},  {"U8", kNumberTypeUInt8},
  {"UINT16", kNumberTypeUInt16}, {"UINT8", kNumberTypeUInt8}};
int ParseInputFormat(AippModule *aipp_module, const std::string &content) {
  if (aipp_module == nullptr) {
    MS_LOG(ERROR) << "input aipp_module is nullptr.";
    return RET_ERROR;
  }
  if (kSupportedInputFormat.find(content) == kSupportedInputFormat.end()) {
    MS_LOG(ERROR) << "input_format is invalid. " << content;
    return RET_ERROR;
  }
  if (!aipp_module->input_format.empty()) {
    MS_LOG(ERROR) << "there is redundant \"input_format\" param in your config.";
    return RET_ERROR;
  }
  aipp_module->input_format = content;
  return RET_OK;
}
int ParseModelFormat(AippModule *aipp_module, const std::string &content) {
  if (aipp_module == nullptr) {
    MS_LOG(ERROR) << "input aipp_module is nullptr.";
    return RET_ERROR;
  }
  if (kSupportedModelFormat.find(content) == kSupportedModelFormat.end()) {
    MS_LOG(ERROR) << "model_format is invalid. " << content;
    return RET_ERROR;
  }
  if (!aipp_module->model_format.empty()) {
    MS_LOG(ERROR) << "there is redundant \"model_format\" param in your config.";
    return RET_ERROR;
  }
  aipp_module->model_format = content;
  return RET_OK;
}
template <int32_t n>
int ParseMeanChn(AippModule *aipp_module, const std::string &content) {
  if (aipp_module == nullptr) {
    MS_LOG(ERROR) << "input aipp_module is nullptr.";
    return RET_ERROR;
  }
  if (!IsValidDoubleNum(content)) {
    MS_LOG(ERROR) << "input mecn_chn_" << n << " is invalid. " << content;
    return RET_ERROR;
  }
  auto mean_val = std::stod(content);
  if (mean_val < 0 || mean_val > kMaxPixelVal) {
    MS_LOG(ERROR) << "input mecn_chn_" << n << "should be in range [0, 255]";
    return RET_ERROR;
  }
  if (aipp_module->mean_map.find(n) != aipp_module->mean_map.end()) {
    MS_LOG(ERROR) << "there is redundant \"mecn_chn_" << n << "\" param in your config.";
    return RET_ERROR;
  }
  aipp_module->mean_map[n] = mean_val;
  return RET_OK;
}
template <int32_t n>
int ParseVarReciChn(AippModule *aipp_module, const std::string &content) {
  if (aipp_module == nullptr) {
    MS_LOG(ERROR) << "input aipp_module is nullptr.";
    return RET_ERROR;
  }
  if (!IsValidDoubleNum(content)) {
    MS_LOG(ERROR) << "input var_reci_chn_" << n << " is invalid. " << content;
    return RET_ERROR;
  }
  auto var_reci_val = std::stod(content);
  if (var_reci_val < -kMaxVarReciChn || var_reci_val > kMaxVarReciChn) {
    MS_LOG(ERROR) << "input var_reci_chn_" << n << "should be in range [-66504, 65504]";
    return RET_ERROR;
  }
  if (aipp_module->val_map.find(n) != aipp_module->val_map.end()) {
    MS_LOG(ERROR) << "there is redundant \"var_reci_chn_" << n << "\" param in your config.";
    return RET_ERROR;
  }
  aipp_module->val_map[n] = var_reci_val;
  return RET_OK;
}
using AippCfgFunc = int (*)(AippModule *aipp_module, const std::string &content);
const std::unordered_map<std::string, AippCfgFunc> kParseAippFuncMap = {
  {kInputFormat, &ParseInputFormat},   {kModelFormat, &ParseModelFormat},   {kMeanChn0, &ParseMeanChn<0>},
  {kMeanChn1, &ParseMeanChn<1>},       {kMeanChn2, &ParseMeanChn<2>},       {kMeanChn3, &ParseMeanChn<3>},
  {kVarReciChn0, &ParseVarReciChn<0>}, {kVarReciChn1, &ParseVarReciChn<1>}, {kVarReciChn2, &ParseVarReciChn<2>},
  {kVarReciChn3, &ParseVarReciChn<3>}};
}  // namespace

MapperConfigParser *MapperConfigParser::GetInstance() {
  static MapperConfigParser instance;
  return &instance;
}

int MapperConfigParser::ParseInputType(const std::string &input_type_str,
                                       const std::vector<std::string> &graph_input_names) {
  auto input_types = SplitString(input_type_str, ';');
  for (const auto &input_type : input_types) {
    auto pos = input_type.rfind(':');
    if (pos == std::string::npos) {
      MS_LOG(ERROR) << "can't find \":\" in " << kInputType << ", please config it like: op_name:data_type";
      return RET_ERROR;
    }
    auto op_name = input_type.substr(0, pos);
    if (std::find(graph_input_names.begin(), graph_input_names.end(), op_name) == graph_input_names.end()) {
      MS_LOG(ERROR) << "can't find " << op_name << " of " << kInputType << " in origin graph inputs.";
      return RET_ERROR;
    }
    auto data_type = input_type.substr(pos + 1);
    if (kDpicoDataTypeMap.find(data_type) == kDpicoDataTypeMap.end()) {
      MS_LOG(WARNING) << data_type << " is unsupported, will be set to FP32.";
    }
  }
  return RET_OK;
}

int MapperConfigParser::ParseImageList(const std::string &image_list_str,
                                       const std::vector<std::string> &graph_input_names) {
  auto image_lists = SplitString(image_list_str, ';');
  if (image_lists.size() != graph_input_names.size()) {
    MS_LOG(ERROR) << "image_list calib data path size " << image_lists.size() << " is not equal to graph input size "
                  << graph_input_names.size();
    return RET_ERROR;
  }
  if (image_lists.size() == 1) {
    auto pos = image_lists.at(0).rfind(':');
    if (pos != std::string::npos) {
      auto op_name = image_lists.at(0).substr(0, pos);
      auto calib_data_path = image_lists.at(0).substr(pos + 1);
      image_lists_[op_name] = calib_data_path;
    } else {
      image_lists_[graph_input_names.at(0)] = image_lists.at(0);
    }
  } else {
    for (const auto &image_list : image_lists) {
      auto pos = image_list.rfind(':');
      if (pos == std::string::npos) {
        MS_LOG(ERROR) << "multi-input model should specify node name in [image_list] config.";
        return RET_ERROR;
      }
      auto op_name = image_list.substr(0, pos);
      auto calib_data_path = image_list.substr(pos + 1);
      image_lists_[op_name] = calib_data_path;
    }
  }
  return RET_OK;
}

int MapperConfigParser::ParseRawLine(const std::string &raw_line, const std::vector<std::string> &graph_input_names,
                                     size_t *graph_input_idx) {
  auto aipp_param = SplitString(raw_line, ':');
  if (aipp_param.size() != kPairSize) {
    MS_LOG(ERROR) << "input pair is invalid. " << raw_line;
    return RET_ERROR;
  }
  if (aipp_param.at(0) == kRelatedInputRank) {
    if (IsValidUnsignedNum(aipp_param.at(1))) {
      size_t idx = std::stoi(aipp_param.at(1));
      if (idx > graph_input_names.size() - 1) {
        MS_LOG(ERROR) << kRelatedInputRank << " index " << idx << " is greater than graph input size "
                      << graph_input_names.size();
        return RET_ERROR;
      }
      *graph_input_idx = idx;
    } else {
      MS_LOG(ERROR) << kRelatedInputRank << " index is invalid.";
      return RET_ERROR;
    }
  } else {
    if (kParseAippFuncMap.find(aipp_param.at(0)) == kParseAippFuncMap.end()) {
      MS_LOG(ERROR) << aipp_param.at(0) << " is unrecognized.";
      return RET_ERROR;
    } else {
      auto parse_aipp_func = kParseAippFuncMap.at(aipp_param.at(0));
      if (parse_aipp_func == nullptr) {
        MS_LOG(ERROR) << "parse aipp func is nullptr. " << aipp_param.at(0);
        return RET_ERROR;
      }
      if (parse_aipp_func(&aipp_[graph_input_names.at(*graph_input_idx)], aipp_param.at(1)) != RET_OK) {
        MS_LOG(ERROR) << "run " << aipp_param.at(1) << " parse func failed.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}
int MapperConfigParser::ParseAippModule(const std::string &aipp_cfg,
                                        const std::vector<std::string> &graph_input_names) {
  std::ifstream ifs;
  if (ReadFileToIfstream(aipp_cfg, &ifs) != RET_OK) {
    MS_LOG(ERROR) << "read file to ifstream failed.";
    return RET_ERROR;
  }
  size_t graph_input_idx = 0;
  size_t num_of_line = 0;
  std::string raw_line;
  int aipp_op_num = 0;
  while (getline(ifs, raw_line)) {
    if (num_of_line > kMaxLineCount) {
      MS_LOG(ERROR) << "the line count is exceeds the maximum range 9999.";
      return RET_ERROR;
    }
    num_of_line++;
    if (EraseBlankSpace(&raw_line) != RET_OK) {
      MS_LOG(ERROR) << "erase blank space failed. " << raw_line;
      return RET_ERROR;
    }
    if (raw_line.empty() || raw_line.at(0) == '#') {
      continue;
    }
    if (raw_line[raw_line.size() - 1] == '}') {
      aipp_op_num--;
      if (aipp_op_num != 0) {
        MS_LOG(ERROR) << "brackets are not a pair! \"}\" is mismatched.";
        return RET_ERROR;
      }
      continue;
    } else if (raw_line[raw_line.size() - 1] == '{') {
      aipp_op_num++;
      if (aipp_op_num == 0) {
        MS_LOG(ERROR) << "brackets are not a pair! \"{\" is mismatched.";
        return RET_ERROR;
      }
      continue;
    } else {
      if (ParseRawLine(raw_line, graph_input_names, &graph_input_idx) != RET_OK) {
        MS_LOG(ERROR) << "parse raw line failed. " << raw_line;
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int MapperConfigParser::AddImageList(const std::string &op_name, const std::string &calib_data_path) {
  if (op_name.empty()) {
    MS_LOG(ERROR) << "op_name is nullptr.";
    return RET_ERROR;
  }
  if (calib_data_path.empty()) {
    MS_LOG(ERROR) << "calib_data_path is nullptr.";
    return RET_ERROR;
  }
  image_lists_[op_name] = calib_data_path;
  return RET_OK;
}

int MapperConfigParser::Parse(const std::string &cfg_file, const std::vector<std::string> &graph_input_names) {
  std::ifstream ifs;
  if (ReadFileToIfstream(cfg_file, &ifs) != RET_OK) {
    MS_LOG(ERROR) << "read file to ifstream failed.";
    return RET_ERROR;
  }
  origin_config_file_path_ = cfg_file;
  auto dir_pos = cfg_file.find_last_of('/');
  tmp_generated_file_dir_ = cfg_file.substr(0, dir_pos + 1) + "tmp/";
  if (AccessFile(tmp_generated_file_dir_, F_OK) == 0 && RemoveDir(tmp_generated_file_dir_) != RET_OK) {
    MS_LOG(ERROR) << "rm dir failed. " << tmp_generated_file_dir_;
    return RET_ERROR;
  }
  if (CreateDir(&tmp_generated_file_dir_) != RET_OK) {
    MS_LOG(ERROR) << "create dir failed. " << tmp_generated_file_dir_;
    return RET_ERROR;
  }

  size_t num_of_line = 0;
  std::string raw_line;
  while (getline(ifs, raw_line)) {
    if (num_of_line > kMaxLineCount) {
      MS_LOG(ERROR) << "the line count is exceeds the maximum range 9999.";
      return RET_ERROR;
    }
    num_of_line++;
    std::istringstream iss(raw_line);
    if (iss.fail()) {
      MS_LOG(ERROR) << "read input line to istringstream failed.";
      return RET_ERROR;
    }
    std::string key;
    std::string value;
    if (iss >> key >> value) {
      mapper_config_[key] = value;
    } else {
      if (key.empty() && value.empty()) {
        MS_LOG(INFO) << "blank line is filtered out.";
        continue;
      } else {
        MS_LOG(ERROR) << "invalid input line: " << raw_line;
        return RET_ERROR;
      }
    }
  }
  if (mapper_config_.find(kInputType) != mapper_config_.end()) {
    if (ParseInputType(mapper_config_.at(kInputType), graph_input_names) != RET_OK) {
      MS_LOG(ERROR) << "parse input type failed.";
      return RET_ERROR;
    }
  }

  if (mapper_config_.find(kImageList) != mapper_config_.end()) {
    if (ParseImageList(mapper_config_.at(kImageList), graph_input_names) != RET_OK) {
      MS_LOG(ERROR) << "parse image list failed.";
      return RET_OK;
    }
  }

  if (mapper_config_.find(kInsertOpConf) != mapper_config_.end()) {
    if (ParseAippModule(mapper_config_.at(kInsertOpConf), graph_input_names) != RET_OK) {
      MS_LOG(ERROR) << "parse aipp config file failed.";
      return RET_ERROR;
    }
  }

  return RET_OK;
}
void MapperConfigParser::SetOriginConfigFilePath(const std::string &origin_config_file_path) {
  origin_config_file_path_ = origin_config_file_path;
}
}  // namespace dpico
}  // namespace mindspore
