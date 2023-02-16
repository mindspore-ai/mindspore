/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "cxx_api/model/acl/acl_model_options.h"
#include <set>
#include <memory>
#include "utils/log_adapter.h"
#include "external/ge/ge_api_types.h"
#include "acl/acl_base.h"

namespace mindspore {
static const std::map<enum DataType, std::string> kSupportedDtypeOptionMap = {{DataType::kNumberTypeFloat16, "FP16"},
                                                                              {DataType::kNumberTypeFloat32, "FP32"},
                                                                              {DataType::kNumberTypeUInt8, "UINT8"}};

std::string TransforPrecisionToAcl(std::string getPrecisionMode) {
  if (getPrecisionMode == "enforce_fp32") {
    return "force_fp32";
  } else if (getPrecisionMode == "preferred_fp32") {
    return "allow_fp32_to_fp16";
  } else if (getPrecisionMode == "enforce_fp16") {
    return "force_fp16";
  } else if (getPrecisionMode == "enforce_origin") {
    return "must_keep_origin_dtype";
  } else if (getPrecisionMode == "preferred_optimal") {
    return "allow_mix_precision";
  }
  return getPrecisionMode;
}

AclModelOptions::AclModelOptions(const std::shared_ptr<Context> &context) {
  if (context == nullptr) {
    return;
  }
  auto &device_infos = context->MutableDeviceInfo();
  if (device_infos.size() != 1) {
    return;
  }
  auto ascend_info = device_infos[0]->Cast<AscendDeviceInfo>();
  if (ascend_info == nullptr) {
    return;
  }

  insert_op_cfg_path_ = ascend_info->GetInsertOpConfigPath();
  input_format_ = ascend_info->GetInputFormat();
  input_shape_map_ = ascend_info->GetInputShapeMap();
  auto out_type = ascend_info->GetOutputType();
  auto iter = kSupportedDtypeOptionMap.find(out_type);
  if (out_type == DataType::kTypeUnknown) {
    // do nothing
  } else if (iter == kSupportedDtypeOptionMap.end()) {
    MS_LOG(INFO) << "Unsupported output type " << out_type << ", use FP32 as default.";
  } else {
    output_type_ = iter->second;
  }
  dynamic_batch_size_ = ascend_info->GetDynamicBatchSize();
  dynamic_image_size_ = ascend_info->GetDynamicImageSize();
  precision_mode_ = TransforPrecisionToAcl(ascend_info->GetPrecisionMode());
  op_select_impl_mode_ = ascend_info->GetOpSelectImplMode();
  fusion_switch_cfg_path_ = ascend_info->GetFusionSwitchConfigPath();
  device_id_ = ascend_info->GetDeviceID();
  buffer_optimize_mode_ = ascend_info->GetBufferOptimizeMode();
  if (!ascend_info->GetInputShape().empty()) {
    input_shape_ = ascend_info->GetInputShape();
  }
  const char *soc_name = aclrtGetSocName();
  if (soc_name == nullptr) {
    MS_LOG(WARNING) << "Get soc version failed.";
    return;
  }
  soc_version_ = soc_name;
}

void AclModelOptions::RenameInput(const std::vector<std::string> &input_names) {
  if (input_names.size() != input_shape_map_.size()) {
    MS_LOG(INFO) << "Inputs count not match";
    return;
  }
  input_shape_ = "";
  for (size_t i = 0; i < input_shape_map_.size(); i++) {
    if (input_shape_map_.find(i) == input_shape_map_.end()) {
      MS_LOG(WARNING) << "Not find the key: " << i;
      return;
    }
    std::string s;
    for (size_t j = 0; j < input_shape_map_[i].size(); j++) {
      s += std::to_string(input_shape_map_[i][j]) + ",";
    }
    input_shape_ += input_names[i] + ":" + s.substr(0, s.size() - 1) + ";";
  }
  input_shape_ = input_shape_.substr(0, input_shape_.size() - 1);
  MS_LOG(INFO) << "input name is " << input_shape_;
}

std::tuple<std::map<std::string, std::string>, std::map<std::string, std::string>> AclModelOptions::GenAclOptions()
  const {
  const std::map<std::string const *, std::string> init_options_map = {
    {&op_select_impl_mode_, ge::ir_option::OP_SELECT_IMPL_MODE},
    {&soc_version_, ge::ir_option::SOC_VERSION},
    {&fusion_switch_cfg_path_, ge::ir_option::FUSION_SWITCH_FILE},
    {&buffer_optimize_mode_, ge::ir_option::BUFFER_OPTIMIZE}};

  const std::map<std::string const *, std::string> build_options_map = {
    {&insert_op_cfg_path_, ge::ir_option::INSERT_OP_FILE},
    {&input_format_, ge::ir_option::INPUT_FORMAT},
    {&input_shape_, ge::ir_option::INPUT_SHAPE},
    {&output_type_, ge::ir_option::OUTPUT_TYPE},
    {&precision_mode_, ge::ir_option::PRECISION_MODE},
    {&dynamic_batch_size_, ge::ir_option::DYNAMIC_BATCH_SIZE},
    {&dynamic_image_size_, ge::ir_option::DYNAMIC_IMAGE_SIZE}};

  const std::set<std::string> first_graph_options = {
    ge::ir_option::INSERT_OP_FILE,
    ge::ir_option::INPUT_FORMAT,
    ge::ir_option::INPUT_SHAPE,
  };

  const std::set<std::string> multi_graph_unsupported_options = {ge::ir_option::OUTPUT_TYPE};

  std::map<std::string, std::string> init_options;
  std::map<std::string, std::string> build_options;
  for (auto [ms_option, acl_option_key] : init_options_map) {
    if (ms_option == nullptr || ms_option->empty()) {
      continue;
    }
    MS_LOG(INFO) << "Option " << acl_option_key << " : " << *ms_option;
    init_options.emplace(acl_option_key, *ms_option);
  }

  for (auto [ms_option, acl_option_key] : build_options_map) {
    if (ms_option == nullptr || ms_option->empty()) {
      continue;
    }
    MS_LOG(INFO) << "Option " << acl_option_key << " : " << *ms_option;
    build_options.emplace(acl_option_key, *ms_option);
  }

  // first_graph_flag has value means being multi graph mode
  if (first_graph_flag_.has_value()) {
    for (const auto &option : multi_graph_unsupported_options) {
      build_options.erase(option);
    }
    // non-input graph
    if (!first_graph_flag_) {
      for (const auto &option : first_graph_options) {
        build_options.erase(option);
      }
    }
  }

  return {init_options, build_options};
}

std::string AclModelOptions::GenAclOptionsKey() const {
  auto [init_options, build_options] = GenAclOptions();
  std::string key_str;
  for (auto &[key, value] : init_options) {
    key_str += key + "^" + value + "^^";
  }
  for (auto &[key, value] : build_options) {
    key_str += key + "^" + value + "^^";
  }
  return key_str;
}
}  // namespace mindspore
