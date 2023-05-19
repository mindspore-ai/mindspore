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

#include "tools/converter/config_parser/micro_param_parser.h"
#include "tools/converter/micro/coder/config.h"
#include "tools/common/string_util.h"
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
STATUS MicroParamParser::ParseTarget(const std::string &target, micro::MicroParam *micro_param) {
  MS_LOG(DEBUG) << "Micro HW target: " << target;
  if (!target.empty()) {
    micro_param->target = target;
  }
  return RET_OK;
}
STATUS MicroParamParser::ParseCodeGenMode(const std::string &codegen_mode, micro::MicroParam *micro_param) {
  MS_LOG(DEBUG) << "Micro codegen mode: " << codegen_mode;
  if (!codegen_mode.empty()) {
    micro_param->codegen_mode = codegen_mode;
  }
  return RET_OK;
}
STATUS MicroParamParser::ParseSupportParallel(const std::string &support_parallel, micro::MicroParam *micro_param) {
  MS_LOG(DEBUG) << "Micro supports parallel: " << support_parallel;
  if (support_parallel.empty()) {
    return RET_OK;
  }
  micro_param->support_parallel = false;  // default
  bool is_parallel;
  if (ConvertBool(support_parallel, &is_parallel)) {
    micro_param->support_parallel = is_parallel;
  }
  return RET_OK;
}
STATUS MicroParamParser::ParseDebugMode(const std::string &debug_mode, micro::MicroParam *micro_param) {
  MS_LOG(DEBUG) << "Micro enables debug mode: " << debug_mode;
  if (debug_mode.empty()) {
    return RET_OK;
  }
  micro_param->debug_mode = false;  // default
  bool is_debug_mode;
  if (ConvertBool(debug_mode, &is_debug_mode)) {
    micro_param->debug_mode = is_debug_mode;
  }
  return RET_OK;
}

STATUS MicroParamParser::ParseEnableMicro(const std::string &enable_micro, micro::MicroParam *micro_param) {
  MS_LOG(DEBUG) << "Micro enables : " << enable_micro;
  if (enable_micro.empty()) {
    return RET_OK;
  }
  micro_param->enable_micro = false;  // default
  bool is_enable_micro;
  if (ConvertBool(enable_micro, &is_enable_micro)) {
    micro_param->enable_micro = is_enable_micro;
  }
  return RET_OK;
}

STATUS MicroParamParser::ParseSavePath(const std::string &save_path, micro::MicroParam *micro_param) {
  MS_LOG(DEBUG) << "Micro save path : " << save_path;
  if (!save_path.empty()) {
    micro_param->save_path = save_path;
  }
  return RET_OK;
}

STATUS MicroParamParser::ParseProjName(const std::string &project_name, micro::MicroParam *micro_param) {
  MS_LOG(DEBUG) << "Micro project name : " << project_name;
  if (!project_name.empty()) {
    micro_param->project_name = project_name;
  }
  return RET_OK;
}

STATUS MicroParamParser::ParseKeepOriginalWeight(const std::string &keep_weight, micro::MicroParam *micro_param) {
  MS_LOG(DEBUG) << "Micro enables : " << keep_weight;
  if (keep_weight.empty()) {
    return RET_OK;
  }
  micro_param->keep_original_weight = false;  // default
  bool is_keep_original_weight;
  if (ConvertBool(keep_weight, &is_keep_original_weight)) {
    micro_param->keep_original_weight = is_keep_original_weight;
  } else {
    MS_LOG(ERROR) << "Micro param invalid, keep_original_weight can only be set as true or false.";
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}

STATUS MicroParamParser::ParseChangeableWeightsName(const std::string &changeable_weights_name,
                                                    micro::MicroParam *micro_param) {
  MS_LOG(DEBUG) << "Micro record changeable weights name: " << changeable_weights_name;
  if (!changeable_weights_name.empty()) {
    micro_param->changeable_weights_name = changeable_weights_name;
  }
  return RET_OK;
}

STATUS MicroParamParser::ParseMicroParam(const MicroParamString &micro_param_string, micro::MicroParam *micro_param) {
  CHECK_NULL_RETURN(micro_param);
  if (ParseTarget(micro_param_string.target, micro_param) != RET_OK) {
    MS_LOG(ERROR) << "Parse HW target val: " << micro_param_string.target;
    return RET_INPUT_PARAM_INVALID;
  }
  if (ParseCodeGenMode(micro_param_string.codegen_mode, micro_param) != RET_OK) {
    MS_LOG(ERROR) << "Parse codegen_mode val； " << micro_param_string.codegen_mode;
    return RET_INPUT_PARAM_INVALID;
  }
  if (ParseSupportParallel(micro_param_string.support_parallel, micro_param) != RET_OK) {
    MS_LOG(ERROR) << "Parse support_parallel val； " << micro_param_string.support_parallel;
    return RET_INPUT_PARAM_INVALID;
  }
  if (ParseDebugMode(micro_param_string.debug_mode, micro_param) != RET_OK) {
    MS_LOG(ERROR) << "Parse debug mode val； " << micro_param_string.debug_mode;
    return RET_INPUT_PARAM_INVALID;
  }
  if (ParseEnableMicro(micro_param_string.enable_micro, micro_param) != RET_OK) {
    MS_LOG(ERROR) << "Parse enable micro val； " << micro_param_string.enable_micro;
    return RET_INPUT_PARAM_INVALID;
  }
  if (ParseSavePath(micro_param_string.save_path, micro_param) != RET_OK) {
    MS_LOG(ERROR) << "Parse save path val failed: " << micro_param_string.save_path;
    return RET_INPUT_PARAM_INVALID;
  }
  if (ParseProjName(micro_param_string.project_name, micro_param) != RET_OK) {
    MS_LOG(ERROR) << "Parse project name val failed: " << micro_param_string.project_name;
    return RET_INPUT_PARAM_INVALID;
  }
  if (ParseKeepOriginalWeight(micro_param_string.keep_original_weight, micro_param) != RET_OK) {
    MS_LOG(ERROR) << "Parse keep_original_weight failed, the val: " << micro_param_string.keep_original_weight;
    return RET_INPUT_PARAM_INVALID;
  }
  if (!micro_param_string.changeable_weights_name.empty() && !micro_param->keep_original_weight) {
    MS_LOG(ERROR) << "When changeable_weights_name is set, the keep_original_weight must be true.";
    return RET_INPUT_PARAM_INVALID;
  }
  if (ParseChangeableWeightsName(micro_param_string.changeable_weights_name, micro_param) != RET_OK) {
    MS_LOG(ERROR) << "Parse changeable_weights_name failed, the val: " << micro_param_string.changeable_weights_name;
    return RET_INPUT_PARAM_INVALID;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
