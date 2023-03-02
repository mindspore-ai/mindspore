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

#include "tools/converter/config_parser/acl_option_param_parser.h"
#include <map>
#include <vector>
#include "tools/common/string_util.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
static const std::map<std::string, DataType> kSupportedDtypeOptionMap = {{"FP16", DataType::kNumberTypeFloat16},
                                                                         {"FP32", DataType::kNumberTypeFloat32},
                                                                         {"UINT8", DataType::kNumberTypeUInt8}};

STATUS AclOptionParamParser::ParseAclOptionCfg(const AclOptionCfgString &acl_option_string,
                                               acl::AclModelOptionCfg *acl_option_cfg) {
  CHECK_NULL_RETURN(acl_option_cfg);

  if (ParseCommon(acl_option_string, acl_option_cfg) != RET_OK) {
    MS_LOG(ERROR) << "Parse common failed.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (!acl_option_string.device_id.empty()) {
    if (ParseDeviceId(acl_option_string.device_id, acl_option_cfg) != RET_OK) {
      MS_LOG(ERROR) << "Parse device id failed, val: " << acl_option_string.device_id;
      return RET_INPUT_PARAM_INVALID;
    }
  }
  if (!acl_option_string.output_type.empty()) {
    if (ParseOutputType(acl_option_string.output_type, acl_option_cfg) != RET_OK) {
      MS_LOG(ERROR) << "Parse output type failed, val； " << acl_option_string.output_type;
      return RET_INPUT_PARAM_INVALID;
    }
  }
  if (!acl_option_string.dynamic_batch_size.empty()) {
    if (ParseDynamicBatchSize(acl_option_string.dynamic_batch_size, acl_option_cfg) != RET_OK) {
      MS_LOG(ERROR) << "Parse dynamic batch size failed, val: " << acl_option_string.dynamic_batch_size;
      return RET_INPUT_PARAM_INVALID;
    }
  }
  if (!acl_option_string.input_shape_vector.empty()) {
    if (ParseInputShapeVector(acl_option_string.input_shape_vector, acl_option_cfg) != RET_OK) {
      MS_LOG(ERROR) << "Parse input shape vector failed, val: " << acl_option_string.input_shape_vector;
      return RET_INPUT_PARAM_INVALID;
    }
  }
  return RET_OK;
}

STATUS AclOptionParamParser::ParseCommon(const AclOptionCfgString &acl_option_string,
                                         acl::AclModelOptionCfg *acl_option_cfg) {
  MS_LOG(DEBUG) << "Input format: " << acl_option_string.input_format
                << ", Input shape: " << acl_option_string.input_shape
                << ", Precision_mode: " << acl_option_string.precision_mode
                << ", Op_select_impl_mode: " << acl_option_string.op_select_impl_mode
                << ", Fusion_switch_config_file_path" << acl_option_string.fusion_switch_config_file_path
                << ", Buffer_optimize: " << acl_option_string.buffer_optimize
                << ", Insert_op_config_file_path: " << acl_option_string.insert_op_config_file_path
                << ", Dynamic image size: " << acl_option_string.dynamic_image_size
                << ", Aoe mode: " << acl_option_string.aoe_mode;
  acl_option_cfg->input_format = acl_option_string.input_format;
  acl_option_cfg->input_shape = acl_option_string.input_shape;
  acl_option_cfg->precision_mode = acl_option_string.precision_mode;
  acl_option_cfg->op_select_impl_mode = acl_option_string.op_select_impl_mode;
  acl_option_cfg->fusion_switch_config_file_path = acl_option_string.fusion_switch_config_file_path;
  acl_option_cfg->buffer_optimize = acl_option_string.buffer_optimize;
  acl_option_cfg->insert_op_config_file_path = acl_option_string.insert_op_config_file_path;
  acl_option_cfg->dynamic_image_size = acl_option_string.dynamic_image_size;
  acl_option_cfg->aoe_mode = acl_option_string.aoe_mode;
  acl_option_cfg->custom_opp_path = acl_option_string.custom_opp_path;
  return RET_OK;
}

STATUS AclOptionParamParser::ParseDeviceId(const std::string &device_id, acl::AclModelOptionCfg *acl_option_cfg) {
  MS_LOG(DEBUG) << "Acl option device id: " << device_id;
  acl_option_cfg->device_id = 0;  // default
  int32_t device_id_num;
  if (ConvertIntNum(device_id, &device_id_num)) {
    acl_option_cfg->device_id = device_id_num;
  }
  return RET_OK;
}

STATUS AclOptionParamParser::ParseOutputType(const std::string &output_type, acl::AclModelOptionCfg *acl_option_cfg) {
  MS_LOG(DEBUG) << "Acl option output type: " << output_type;
  acl_option_cfg->output_type = DataType::kInvalidType;
  if (kSupportedDtypeOptionMap.find(output_type) != kSupportedDtypeOptionMap.end()) {
    acl_option_cfg->output_type = kSupportedDtypeOptionMap.at(output_type);
  }
  return RET_OK;
}

// dynamic_batch_size="1,2,4,8"
STATUS AclOptionParamParser::ParseDynamicBatchSize(const std::string &dynamic_batch_size,
                                                   acl::AclModelOptionCfg *acl_option_cfg) {
  MS_LOG(DEBUG) << "Acl option dynamic batch size: " << dynamic_batch_size;
  std::vector<std::string> batch_size_string = SplitStringToVector(dynamic_batch_size, ',');
  for (const auto &item : batch_size_string) {
    int32_t val;
    if (ConvertIntNum(item, &val)) {
      size_t tmp_val = static_cast<size_t>(val);
      acl_option_cfg->dynamic_batch_size.push_back(tmp_val);
    }
  }
  return RET_OK;
}

// input shape vector=[1,2,3];[4,5,6]
STATUS AclOptionParamParser::ParseInputShapeVector(const std::string &input_shape_vector,
                                                   acl::AclModelOptionCfg *acl_option_cfg) {
  MS_LOG(DEBUG) << "Acl option input shape vector: " << input_shape_vector;
  std::vector<std::string> intput_shape_str = SplitStringToVector(input_shape_vector, ';');
  int32_t idx = 0;
  std::map<int32_t, std::vector<int32_t>> input_shape_map;
  for (auto &item : intput_shape_str) {
    if (item.size() < DIMENSION_2D || item[0] != '[' || item[item.size() - 1] != ']') {
      MS_LOG(ERROR) << "Input param is invalid, val: " << item << ", the format should be [a, b].";
      return RET_ERROR;
    }
    std::string tmp = item.substr(1, item.size() - DIMENSION_2D);
    if (tmp.find("[") != std::string::npos) {
      MS_LOG(ERROR) << "Input param is invalid, value: " << item << ", multi input shape should be split by ;.";
      return RET_ERROR;
    }
    std::vector<std::string> intput_shape = SplitStringToVector(tmp, ',');
    std::vector<int32_t> input_shape_int;
    for (auto &shape : intput_shape) {
      int32_t val;
      if (ConvertIntNum(shape, &val)) {
        input_shape_int.push_back(val);
      }
    }
    input_shape_map[idx] = input_shape_int;
    idx++;
  }

  acl_option_cfg->input_shape_map = input_shape_map;
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
