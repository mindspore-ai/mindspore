/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "tools/converter/config_parser/config_file_parser.h"
#include "tools/common/parse_config_utils.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "tools/converter/converter_context.h"

#include "tools/common/string_util.h"
#include "src/common/config_infos.h"
#include "src/common/common.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kCommonQuantParam = "common_quant_param";
constexpr auto kFullQuantParam = "full_quant_param";
constexpr auto kWeightQuantParam = "weight_quant_param";
constexpr auto kMixedBitWeightQuantParam = "mixed_bit_weight_quant_param";
constexpr auto kDataPreprocessParam = "data_preprocess_param";
constexpr auto kRegistry = "registry";
constexpr auto kMicroParam = "micro_param";
constexpr auto kCpuOptionParam = "cpu_option_cfg_param";
constexpr auto kCustomOppPath = "custom_opp_path";
constexpr auto kTransformQuantParam = "transform_quant_param";
constexpr auto kDynamicQuantParam = "dynamic_quant_param";
constexpr auto kGraphKernelParam = "graph_kernel_param";
constexpr int kNumSize3 = 3;
constexpr int kNumSize2 = 2;
}  // namespace
using ShapeVector = std::vector<int64_t>;
const int kBatchDim = 0;
const int kDynImgSize = 0;
const int kDynBatchSize = 1;
bool CheckBatchStringSupport(const std::vector<std::string> &batch_str_vec) {
  if (batch_str_vec.empty()) {
    return false;
  }
  std::string only_batch = batch_str_vec[0];
  for (size_t i = 1; i < batch_str_vec.size(); ++i) {
    if (batch_str_vec[i] != only_batch) {
      return false;
    }
  }
  return true;
}

const size_t kIndex0 = 0;
const size_t kIndex1 = 1;
const size_t kIndex2 = 2;
const size_t kIndex3 = 3;
const int64_t kdynDim = -1;
int DynBatchOrDynImage(const mindspore::ProfileConfigs &profile, size_t dynamic_input_index) {
  int dynamic_type = -1;
  for (auto &info : profile.input_infos) {
    if (!info.is_dynamic_shape) {
      continue;
    }
    const auto &shape = info.input_shape;
    if (shape.size() != kNCHWDimNumber) {
      MS_LOG(ERROR) << "Dynamic input whose shape is not 4-dimensional is not supported, input shape: " << shape;
      return -1;
    }
    auto dynamic_dims_count = std::count_if(shape.begin(), shape.end(), [](int64_t dim) { return dim == kdynDim; });
    if (shape[kIndex0] != kdynDim && dynamic_dims_count == kHWDimNumber) {
      if (dynamic_type != -1 && dynamic_type != kDynImgSize) {
        MS_LOG(ERROR) << "Only dynamic batch or dynamic image size is supported, hybrid scenarios are not supported";
        return -1;
      }
      dynamic_type = kDynImgSize;
    } else if (shape[kIndex0] == kdynDim && dynamic_dims_count == 1) {
      if (dynamic_type != -1 && dynamic_type != kDynBatchSize) {
        MS_LOG(ERROR) << "Only dynamic batch or dynamic image size is supported, hybrid scenarios are not supported";
        return -1;
      }
      dynamic_type = kDynBatchSize;
    } else {
      MS_LOG(ERROR) << "Only dynamic batch or dynamic image size is supported, input shape: " << shape;
      return -1;
    }
  }
  return dynamic_type;
}

std::string CombineDynamicImageString(const struct mindspore::ProfileConfigs &profile, size_t dynamic_input) {
  ShapeVector shape = profile.input_infos[dynamic_input].input_shape;
  std::string ret = "";
  size_t first_dim = kIndex0, second_dim = kIndex0;
  if (shape[kIndex1] == kdynDim && shape[kIndex2] == kdynDim) {
    first_dim = kIndex1;
    second_dim = kIndex2;
  } else if (shape[kIndex1] == kdynDim && shape[kIndex3] == kdynDim) {
    first_dim = kIndex1;
    second_dim = kIndex3;
  } else if (shape[kIndex2] == kdynDim && shape[kIndex3] == kdynDim) {
    first_dim = kIndex2;
    second_dim = kIndex3;
  }
  for (size_t dim_idx = 0; dim_idx < profile.profiles.size(); ++dim_idx) {
    auto &dynamic_item = profile.profiles[dim_idx].inputs[dynamic_input];
    int64_t min_first = dynamic_item.min_dims[first_dim];
    int64_t max_first = dynamic_item.max_dims[first_dim];
    int64_t min_second = dynamic_item.min_dims[second_dim];
    int64_t max_second = dynamic_item.max_dims[second_dim];
    for (int64_t i = min_first; i <= max_first; ++i) {
      for (int64_t j = min_second; j <= max_second; ++j) {
        ret += std::to_string(i) + "," + std::to_string(j) + ";";
      }
    }
  }
  ret = ret.substr(0, ret.size() - 1);  // discard the final ";"
  return ret;
}

std::vector<size_t> CombineDynamicBatchList(const struct mindspore::ProfileConfigs &profile, size_t dynamic_input) {
  std::vector<size_t> ret;
  size_t batch_dim = 0;
  for (size_t dim_idx = 0; dim_idx < profile.profiles.size(); ++dim_idx) {
    auto &dynamic_item = profile.profiles[dim_idx].inputs[dynamic_input];
    int64_t min = dynamic_item.min_dims[batch_dim];
    int64_t max = dynamic_item.max_dims[batch_dim];
    for (int64_t i = min; i <= max; ++i) {
      ret.push_back(LongToSize(i));
    }
  }
  return ret;
}

std::string RemoveInputShapeBrackets(const std::string &input_shape_str) {
  std::string ret = "";
  for (size_t i = 0; i < input_shape_str.size(); ++i) {
    if (input_shape_str[i] == '[' || input_shape_str[i] == ']') {
      continue;
    }
    ret += input_shape_str[i];
  }
  return ret;
}

std::string FindInAscendMap(const std::string &key, const std::map<std::string, std::string> &ascend_map) {
  auto it = ascend_map.find(key);
  if (it != ascend_map.end()) {
    return it->second;
  }
  return "";
}

bool SetDynParams(const std::shared_ptr<mindspore::ConverterPara> &param,
                  const std::map<std::string, std::string> &ascend_map) {
  struct mindspore::ProfileConfigs profile_configs;
  if (!mindspore::ProfileParser::Parse(ascend_map, false, &profile_configs)) {
    MS_LOG(ERROR) << "Parse input_shape and dynamic_dims failed";
    return false;
  }
  const auto &input_infos = profile_configs.input_infos;
  auto it = ascend_map.find("dynamic_dims");
  if (it == ascend_map.end()) {
    MS_LOG(INFO) << "Inputs are not dynamic";
    return true;
  }
  std::vector<std::string> dynamic_dims_strs = mindspore::lite::SplitStringToVector(it->second, ';');
  if (dynamic_dims_strs.size() != input_infos.size()) {
    MS_LOG(ERROR) << "Invalid dynamic_dims, size " << dynamic_dims_strs.size() << " != input size "
                  << input_infos.size();
    return false;
  }
  std::string one_dym_dims;
  size_t dynamic_input_index = 0;
  for (size_t i = 0; i < input_infos.size(); i++) {
    auto &info = input_infos[i];
    if (!info.is_dynamic_shape) {
      continue;
    }
    if (one_dym_dims.empty()) {
      one_dym_dims = dynamic_dims_strs[i];
      dynamic_input_index = i;
    } else if (one_dym_dims != dynamic_dims_strs[i]) {
      MS_LOG(ERROR) << "Do not support different dynamic_dims, one " << one_dym_dims << ", other "
                    << dynamic_dims_strs[i];
      return false;
    }
  }
  int dynamic_type = DynBatchOrDynImage(profile_configs, dynamic_input_index);
  switch (dynamic_type) {
    case kDynImgSize:
      param->aclModelOptionCfgParam.dynamic_image_size =
        CombineDynamicImageString(profile_configs, dynamic_input_index);
      break;
    case kDynBatchSize:
      param->aclModelOptionCfgParam.dynamic_batch_size = CombineDynamicBatchList(profile_configs, dynamic_input_index);
      break;
    default:
      MS_LOG(ERROR) << "Do not support input shape";
      return false;
  }
  return true;
}

void ConfigFileParser::SetVariableParams(const std::shared_ptr<mindspore::ConverterPara> &param,
                                         const std::map<std::string, std::string> &ascend_map) {
  auto it = ascend_map.find("inputs_to_variable");
  if (it != ascend_map.end()) {
    std::vector<std::string> inputs_to_variables = mindspore::lite::SplitStringToVector(it->second, ',');
    ProcessVariableParam(inputs_to_variables, inputs_variable_index_);
    if (CheckVariableParm(inputs_variable_index_) != RET_OK) {
      MS_LOG(ERROR) << "Check input variable param failed";
      return;
    }
  }
  auto output_it = ascend_map.find("outputs_to_variable");
  if (output_it != ascend_map.end()) {
    std::vector<std::string> outputs_to_variables = mindspore::lite::SplitStringToVector(output_it->second, ',');
    ProcessVariableParam(outputs_to_variables, outputs_variable_index_);
    if (CheckVariableParm(outputs_variable_index_) != RET_OK) {
      MS_LOG(ERROR) << "Check output variable param failed";
      return;
    }
  }
  if (!inputs_variable_index_.empty() && !outputs_variable_index_.empty() &&
      inputs_variable_index_.size() != outputs_variable_index_.size()) {
    MS_LOG(ERROR) << "Input variable number is not equal output variable number";
    return;
  }
  param->ascendGeOptionCfg.inputs_to_variable = inputs_variable_index_;
  param->ascendGeOptionCfg.outputs_to_variable = outputs_variable_index_;
}

int ConfigFileParser::ProcessVariableParam(const std::vector<std::string> &variable_param,
                                           std::vector<int64_t> &variable_index) {
  for (auto &it : variable_param) {
    auto remove_str = RemoveInputShapeBrackets(it);
    int64_t min_index;
    int64_t max_index;
    if (!ProfileParser::ParseRangeStr(remove_str, &min_index, &max_index)) {
      MS_LOG(ERROR) << "Parser range string " << remove_str << " failed";
      return RET_ERROR;
    }
    if (max_index < min_index) {
      MS_LOG(ERROR) << "The variable param in not valid" << max_index << "is not larger than" << min_index;
      return RET_ERROR;
    }
    for (int64_t i = min_index; i <= max_index; ++i) {
      variable_index.emplace_back(i);
    }
  }
  return RET_OK;
}

int ConfigFileParser::CheckVariableParm(const std::vector<int64_t> &variable_index) {
  for (size_t i = 1; i < variable_index.size(); ++i) {
    if (variable_index[i] < variable_index[i - 1]) {
      MS_LOG(ERROR) << "variable index is not valid" << variable_index[i] << " is less than " << variable_index[i - 1];
      return RET_ERROR;
    }
  }
  return RET_OK;
}

bool ConfigFileParser::CheckPluginCustomOps(const std::vector<std::string> &plugin_custom_ops) {
  if (find(plugin_custom_ops.begin(), plugin_custom_ops.end(), "All") != plugin_custom_ops.end() &&
      plugin_custom_ops.size() != 1) {
    MS_LOG(ERROR) << "plugin_custom_ops include All, can not include other param.";
    return false;
  }
  if (find(plugin_custom_ops.begin(), plugin_custom_ops.end(), "None") != plugin_custom_ops.end() &&
      plugin_custom_ops.size() != 1) {
    MS_LOG(ERROR) << "plugin_custom_ops include None, can not include other param.";
    return false;
  }
  return true;
}

STATUS ConfigFileParser::ParseCustomPattern(const std::shared_ptr<mindspore::ConverterPara> &param,
                                            std::string custom_pattern_str) {
  std::vector<std::string> custom_pattern_strs = mindspore::lite::SplitStringToVector(custom_pattern_str, ";");
  for (auto custom_pattern : custom_pattern_strs) {
    std::vector<std::string> item = mindspore::lite::SplitStringToVector(custom_pattern, ":");
    if (item.size() != kNumSize3) {
      return RET_ERROR;
    }
    std::string op_type = item[0];
    auto names_list = mindspore::lite::SplitStringToVector(item[1], ",");
    std::string status = item[kNumSize2];
    if (status == "enable") {
      if (param->aclModelOptionCfgParam.enable_custom_fusion_pattern.find(op_type) !=
          param->aclModelOptionCfgParam.enable_custom_fusion_pattern.end()) {
        MS_LOG(ERROR) << op_type << " has define, can not defined repeat.";
        return RET_ERROR;
      }
      param->aclModelOptionCfgParam.enable_custom_fusion_pattern[op_type] = names_list;
    } else if (status == "disable") {
      if (param->aclModelOptionCfgParam.disable_custom_fusion_pattern.find(op_type) !=
          param->aclModelOptionCfgParam.disable_custom_fusion_pattern.end()) {
        MS_LOG(ERROR) << op_type << " has define, can not defined repeat.";
        return RET_ERROR;
      }
      param->aclModelOptionCfgParam.disable_custom_fusion_pattern[op_type] = names_list;
    } else {
      MS_LOG(ERROR) << "status only support enable or disable";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

bool ConfigFileParser::SetParamByConfigfile(const std::shared_ptr<mindspore::ConverterPara> &param,
                                            const std::map<std::string, std::string> &ascend_map) {
  std::string ascend_string = "";
  auto set_option = [&ascend_map](const std::string &key, std::string *option) {
    auto it = ascend_map.find(key);
    if (it != ascend_map.end() && !it->second.empty()) {
      *option = it->second;
    }
  };
  set_option("input_format", &param->aclModelOptionCfgParam.input_format);
  set_option("precision_mode", &param->aclModelOptionCfgParam.precision_mode);
  set_option("op_select_impl_mode", &param->aclModelOptionCfgParam.op_select_impl_mode);
  set_option("fusion_switch_config_file_path", &param->aclModelOptionCfgParam.fusion_switch_config_file_path);
  set_option("buffer_optimize", &param->aclModelOptionCfgParam.buffer_optimize);
  set_option("insert_op_config_file_path", &param->aclModelOptionCfgParam.insert_op_config_file_path);
  set_option("om_file_path", &param->aclModelOptionCfgParam.om_file_path);
  set_option("aoe_mode", &param->aclModelOptionCfgParam.aoe_mode);
  set_option(kDumpModelNameKey, &param->aclModelOptionCfgParam.dump_model_name);
  set_option("provider", &param->provider);

  auto plugin_custom_ops_str = FindInAscendMap(kPluginCustomOps, ascend_map);
  std::vector<std::string> plugin_custom_ops_vec = {};
  if (!plugin_custom_ops_str.empty()) {
    MS_LOG(INFO) << "plugin_custom_ops: " << plugin_custom_ops_str;
    plugin_custom_ops_vec = mindspore::lite::SplitStringToVector(plugin_custom_ops_str, ",");
    if (!CheckPluginCustomOps(plugin_custom_ops_vec)) {
      return false;
    }
  }
  if (!plugin_custom_ops_vec.empty()) {
    param->ascendGeOptionCfg.plugin_custom_ops = plugin_custom_ops_vec;
  } else if (!(ascend_string = FindInAscendMap(kEnableCustomOp, ascend_map)).empty()) {
    param->ascendGeOptionCfg.plugin_custom_ops = {"All"};
  }
  // parse for ascend custom fusion op
  if (!plugin_custom_ops_vec.empty()) {
    param->aclModelOptionCfgParam.plugin_custom_ops = plugin_custom_ops_vec;
  }
  auto custom_fusion_pattern_str = FindInAscendMap("custom_fusion_pattern", ascend_map);
  if (!custom_fusion_pattern_str.empty()) {
    auto status = ParseCustomPattern(param, custom_fusion_pattern_str);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "custom fusion pattern wrong, eg:\n"
                       "custom_fusion_pattern=Fusion_op_type:node_name_1,node_name_2:enable\n"
                       "or: "
                       "custom_fusion_pattern=Fusion_op_type:node_name_1,node_name_2:disable";
      return false;
    }
  }
  auto it = ascend_map.find("input_shape");
  if (it != ascend_map.end()) {
    param->aclModelOptionCfgParam.input_shape = RemoveInputShapeBrackets(it->second);
  }

  it = ascend_map.find("device_id");
  if (it != ascend_map.end()) {
    int32_t val;
    if (mindspore::lite::ConvertIntNum(it->second, &val)) {
      param->aclModelOptionCfgParam.device_id = val;
    } else {
      MS_LOG(ERROR) << "Convert device id failed";
      return false;
    }
  }

  it = ascend_map.find("output_type");
  if (it != ascend_map.end()) {
    auto dtype_str = it->second;
    if (dtype_str == "FP16") {
      param->aclModelOptionCfgParam.output_type = DataType::kNumberTypeFloat16;
    } else if (dtype_str == "FP32") {
      param->aclModelOptionCfgParam.output_type = DataType::kNumberTypeFloat32;
    } else if (dtype_str == "UINT8") {
      param->aclModelOptionCfgParam.output_type = DataType::kNumberTypeUInt8;
    } else {
      MS_LOG(WARNING) << "Unsupported or invalid output_type, using default type";
    }
  }
  SetVariableParams(param, ascend_map);
  return SetDynParams(param, ascend_map);
}

int ConfigFileParser::ParseConfigFile(const std::string &config_file_path,
                                      std::map<int, std::map<std::string, std::string>> *model_param_infos) {
  std::map<std::string, std::map<std::string, std::string>> maps;
  auto ret = mindspore::lite::ParseConfigFile(config_file_path, &maps, model_param_infos);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse config file failed.";
    return ret;
  }
  ret = ParseConfigParam(&maps);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Parse config param failed.";
    return ret;
  }
  return RET_OK;
}

int ConfigFileParser::ParseConfigParam(std::map<std::string, std::map<std::string, std::string>> *maps) {
  if (maps == nullptr) {
    MS_LOG(ERROR) << "Maps is nullptr.";
    return RET_ERROR;
  }
  for (const auto &config_info : *maps) {
    ConverterInnerContext::GetInstance()->SetExternalUsedConfigInfos(config_info.first, config_info.second);
  }
  auto ret = ParseDataPreProcessString(*maps);
  (void)maps->erase(kDataPreprocessParam);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParseDataPreProcessString failed.";
    return ret;
  }
  ret = ParseCommonQuantString(*maps);
  (void)maps->erase(kCommonQuantParam);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParseCommonQuantString failed.";
    return ret;
  }
  ret = ParseMixedBitQuantString(*maps);
  (void)maps->erase(kMixedBitWeightQuantParam);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParseMixedBitQuantString failed.";
    return ret;
  }
  ret = ParseFullQuantString(*maps);
  (void)maps->erase(kFullQuantParam);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParseFullQuantString failed.";
    return ret;
  }
  ret = ParseRegistryInfoString(*maps);
  (void)maps->erase(kRegistry);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParseExtendedintegrationString failed.";
    return ret;
  }
  ret = ParseAclOptionCfgString(*maps);
  (void)maps->erase(kAclOptionParam);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParseAclOptionCfgString failed.";
    return ret;
  }
  ret = ParseMicroParamString(*maps);
  (void)maps->erase(kMicroParam);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParseMicroParamString failed.";
    return ret;
  }
  ret = ParseWeightQuantString(*maps);
  (void)maps->erase(kWeightQuantParam);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParseWeightQuantString failed.";
    return ret;
  }
  ret = ParseCpuOptionCfgString(*maps);
  (void)maps->erase(kCpuOptionParam);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParseCpuOptionCfgString failed.";
    return ret;
  }
  ret = ParseTransformQuantString(*maps);
  (void)maps->erase(kTransformQuantParam);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParseTransformQuantString failed.";
    return ret;
  }
  ret = ParseDynamicQuantString(*maps);
  (void)maps->erase(kDynamicQuantParam);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParseDynamicQuantString failed.";
    return ret;
  }
  (void)ParseGraphKernelString(*maps);
  (void)maps->erase(kGraphKernelParam);
  ret = ParseOMConverterString(*maps);
  (void)maps->erase(kOMConverterOptionsSection);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParseOMConverterString failed.";
    return ret;
  }
  return RET_OK;
}

int ConfigFileParser::SetMapData(const std::map<std::string, std::string> &input_map,
                                 const std::map<std::string, std::string &> &parse_map, const std::string &section) {
  for (const auto &map : input_map) {
    if (parse_map.find(map.first) == parse_map.end()) {
      MS_LOG(ERROR) << "INPUT ILLEGAL: `" << map.first << "` is not supported in "
                    << "[" << section << "]";
      return RET_INPUT_PARAM_INVALID;
    } else {
      parse_map.at(map.first) = map.second;
    }
  }
  return RET_OK;
}

int ConfigFileParser::ParseDataPreProcessString(const std::map<std::string, std::map<std::string, std::string>> &maps) {
  if (maps.find(kDataPreprocessParam) != maps.end()) {
    const auto &map = maps.at(kDataPreprocessParam);
    std::map<std::string, std::string &> parse_map{
      {"calibrate_path", data_pre_process_string_.calibrate_path},
      {"calibrate_size", data_pre_process_string_.calibrate_size},
      {"input_type", data_pre_process_string_.input_type},
      {"image_to_format", data_pre_process_string_.image_to_format},
      {"normalize_mean", data_pre_process_string_.normalize_mean},
      {"normalize_std", data_pre_process_string_.normalize_std},
      {"resize_width", data_pre_process_string_.resize_width},
      {"resize_height", data_pre_process_string_.resize_height},
      {"resize_method", data_pre_process_string_.resize_method},
      {"center_crop_width", data_pre_process_string_.center_crop_width},
      {"center_crop_height", data_pre_process_string_.center_crop_height},
    };
    return SetMapData(map, parse_map, kDataPreprocessParam);
  }
  return RET_OK;
}

int ConfigFileParser::ParseCommonQuantString(const std::map<std::string, std::map<std::string, std::string>> &maps) {
  if (maps.find(kCommonQuantParam) != maps.end()) {
    const auto &map = maps.at(kCommonQuantParam);
    std::map<std::string, std::string &> parse_map{
      {"quant_type", common_quant_string_.quant_type},
      {"bit_num", common_quant_string_.bit_num},
      {"min_quant_weight_size", common_quant_string_.min_quant_weight_size},
      {"min_quant_weight_channel", common_quant_string_.min_quant_weight_channel},
      {"skip_quant_node", common_quant_string_.skip_quant_node},
      {"debug_info_save_path", common_quant_string_.debug_info_save_path},
      {"enable_encode", common_quant_string_.enable_encode},
      {"workspace", common_quant_string_.workspace},
    };
    return SetMapData(map, parse_map, kCommonQuantParam);
  }
  return RET_OK;
}

int ConfigFileParser::ParseMixedBitQuantString(const std::map<std::string, std::map<std::string, std::string>> &maps) {
  if (maps.find(kMixedBitWeightQuantParam) != maps.end()) {
    const auto &map = maps.at(kMixedBitWeightQuantParam);
    std::map<std::string, std::string &> parse_map{
      {"init_scale", mixed_bit_quant_string_.init_scale},
      {"auto_tune", mixed_bit_quant_string_.auto_tune},
      {"use_cv_data", mixed_bit_quant_string_.use_cv_data},
      {"max_iterations", mixed_bit_quant_string_.max_iterations},
    };
    return SetMapData(map, parse_map, kMixedBitWeightQuantParam);
  }
  return RET_OK;
}

int ConfigFileParser::ParseFullQuantString(const std::map<std::string, std::map<std::string, std::string>> &maps) {
  if (maps.find(kFullQuantParam) != maps.end()) {
    const auto &map = maps.at(kFullQuantParam);
    std::map<std::string, std::string &> parse_map{
      {"activation_quant_method", full_quant_string_.activation_quant_method},
      {"bias_correction", full_quant_string_.bias_correction},
      {"target_device", full_quant_string_.target_device},
      {"per_channel", full_quant_string_.per_channel},
      {"smooth_alpha", full_quant_string_.smooth_alpha},
      {"enable_smooth_shift", full_quant_string_.enable_smooth_shift},
    };
    return SetMapData(map, parse_map, kFullQuantParam);
  }
  return RET_OK;
}

int ConfigFileParser::ParseRegistryInfoString(const std::map<std::string, std::map<std::string, std::string>> &maps) {
  if (maps.find(kRegistry) != maps.end()) {
    const auto &map = maps.at(kRegistry);
    std::map<std::string, std::string &> parse_map{
      {"plugin_path", registry_info_string_.plugin_path},
      {"disable_fusion", registry_info_string_.disable_fusion},
      {"fusion_blacklists", registry_info_string_.fusion_blacklists},
    };
    return SetMapData(map, parse_map, kRegistry);
  }
  return RET_OK;
}

int ConfigFileParser::ParseAclOptionCfgString(const std::map<std::string, std::map<std::string, std::string>> &maps) {
  if (maps.find(kAclOptionParam) != maps.end()) {
    const auto &map = maps.at(kAclOptionParam);
    std::map<std::string, std::string &> parse_map{
      {"device_id", acl_option_cfg_string_.device_id},
      {"input_format", acl_option_cfg_string_.input_format},
      {"input_shape_vector", acl_option_cfg_string_.input_shape_vector},
      {"input_shape", acl_option_cfg_string_.input_shape},
      {"output_type", acl_option_cfg_string_.output_type},
      {"precision_mode", acl_option_cfg_string_.precision_mode},
      {"op_select_impl_mode", acl_option_cfg_string_.op_select_impl_mode},
      {"fusion_switch_config_file_path", acl_option_cfg_string_.fusion_switch_config_file_path},
      {"dynamic_batch_size", acl_option_cfg_string_.dynamic_batch_size},
      {"buffer_optimize", acl_option_cfg_string_.buffer_optimize},
      {"insert_op_config_file_path", acl_option_cfg_string_.insert_op_config_file_path},
      {"dynamic_image_size", acl_option_cfg_string_.dynamic_image_size},
      {"dynamic_dims", acl_option_cfg_string_.dynamic_dims},
      {"aoe_mode", acl_option_cfg_string_.aoe_mode},
      {"custom_opp_path", acl_option_cfg_string_.custom_opp_path}};
    auto ret = SetMapData(map, parse_map, kAclOptionParam);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "set map data failed.";
      return ret;
    }
  }
  if (maps.find(kAclInitOptionParam) != maps.end()) {
    const auto &map = maps.at(kAclInitOptionParam);
    for (const auto &item : map) {
      (void)acl_option_cfg_string_.init_options_map.emplace(item.first, item.second);
    }
  }
  if (maps.find(kAclBuildOptionParam) != maps.end()) {
    const auto &map = maps.at(kAclBuildOptionParam);
    for (const auto &item : map) {
      (void)acl_option_cfg_string_.build_options_map.emplace(item.first, item.second);
    }
  }
  if (maps.find(kAoeGlobalOptionsSection) != maps.end()) {
    const auto &map = maps.at(kAoeGlobalOptionsSection);
    for (const auto &item : map) {
      (void)acl_option_cfg_string_.aoe_global_options_map.emplace(item.first, item.second);
    }
  }
  if (maps.find(kAoeTuningOptionsSection) != maps.end()) {
    const auto &map = maps.at(kAoeTuningOptionsSection);
    for (const auto &item : map) {
      (void)acl_option_cfg_string_.aoe_tuning_options_map.emplace(item.first, item.second);
    }
  }
  return RET_OK;
}

int ConfigFileParser::ParseMicroParamString(const std::map<std::string, std::map<std::string, std::string>> &maps) {
  if (maps.find(kMicroParam) != maps.end()) {
    const auto &map = maps.at(kMicroParam);
    std::map<std::string, std::string &> parse_map{
      {"target", micro_param_string_.target},
      {"codegen_mode", micro_param_string_.codegen_mode},
      {"debug_mode", micro_param_string_.debug_mode},
      {"support_parallel", micro_param_string_.support_parallel},
      {"enable_micro", micro_param_string_.enable_micro},
      {"save_path", micro_param_string_.save_path},
      {"project_name", micro_param_string_.project_name},
      {"keep_original_weight", micro_param_string_.keep_original_weight},
      {"changeable_weights_name", micro_param_string_.changeable_weights_name}};
    return SetMapData(map, parse_map, kMicroParam);
  }
  return RET_OK;
}

int ConfigFileParser::ParseWeightQuantString(const std::map<std::string, std::map<std::string, std::string>> &maps) {
  if (maps.find(kWeightQuantParam) != maps.end()) {
    const auto &map = maps.at(kWeightQuantParam);
    std::map<std::string, std::string &> parse_map{{"dequant_strategy", weight_quant_string_.dequant_strategy},
                                                   {"update_mindir", weight_quant_string_.update_mindir},
                                                   {"max_segments", weight_quant_string_.max_segments},
                                                   {"per_channel", weight_quant_string_.per_channel},
                                                   {"bias_correction", weight_quant_string_.bias_correction},
                                                   {"quant_strategy", weight_quant_string_.quant_strategy}};
    return SetMapData(map, parse_map, kWeightQuantParam);
  }
  return RET_OK;
}

int ConfigFileParser::ParseCpuOptionCfgString(const std::map<std::string, std::map<std::string, std::string>> &maps) {
  if (maps.find(kCpuOptionParam) != maps.end()) {
    const auto &map = maps.at(kCpuOptionParam);
    std::map<std::string, std::string &> parse_map{{"architecture", cpu_option_cfg_string_.architecture},
                                                   {"instruction", cpu_option_cfg_string_.instruction}};
    auto ret = SetMapData(map, parse_map, kCpuOptionParam);
    if (cpu_option_cfg_string_.architecture.empty() || cpu_option_cfg_string_.instruction.empty()) {
      MS_LOG(WARNING) << "[cpu_option_cfg_param] set incompletely, the model won't do optimize for cpu, please "
                         "check the parameter architecture and instruction are correct.";
    }
    return ret;
  }
  return RET_OK;
}

int ConfigFileParser::ParseTransformQuantString(const std::map<std::string, std::map<std::string, std::string>> &maps) {
  if (maps.find(kTransformQuantParam) != maps.end()) {
    const auto &map = maps.at(kTransformQuantParam);
    std::map<std::string, std::string &> parse_map{
      {"export_precision_mode", transform_quant_string_.export_precision_mode},
    };
    return SetMapData(map, parse_map, kTransformQuantParam);
  }
  return RET_OK;
}

int ConfigFileParser::ParseDynamicQuantString(const std::map<std::string, std::map<std::string, std::string>> &maps) {
  if (maps.find(kDynamicQuantParam) != maps.end()) {
    const auto &map = maps.at(kDynamicQuantParam);
    std::map<std::string, std::string &> parse_map{
      {"quant_strategy", dynamic_quant_string_.quant_strategy},
    };
    return SetMapData(map, parse_map, kDynamicQuantParam);
  }
  return RET_OK;
}

int ConfigFileParser::ParseGraphKernelString(const std::map<std::string, std::map<std::string, std::string>> &maps) {
  if (maps.find(kGraphKernelParam) != maps.end()) {
    const auto &map = maps.at(kGraphKernelParam);
    for (const auto &item : map) {
      std::stringstream oss;
      oss << "--" << item.first << "=" << item.second;
      (void)graph_kernel_string_.emplace_back(oss.str());
    }
  }
  return RET_OK;
}

int ConfigFileParser::ParseOMConverterString(const std::map<std::string, std::map<std::string, std::string>> &maps) {
  if (maps.find(kOMConverterOptionsSection) != maps.end()) {
    const auto &map = maps.at(kOMConverterOptionsSection);
    std::map<std::string, std::string &> parse_map{
      {"input_name_vector", om_converter_string_.input_name_vector},
      {"input_shape_vector", om_converter_string_.input_shape_vector},
      {"input_data_type_vector", om_converter_string_.input_data_type_vector},
      {"output_name_vector", om_converter_string_.output_name_vector},
      {"output_shape_vector", om_converter_string_.output_shape_vector},
      {"output_data_type_vector", om_converter_string_.output_data_type_vector}};
    auto ret = SetMapData(map, parse_map, kOMConverterOptionsSection);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set map data failed.";
      return ret;
    }
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
