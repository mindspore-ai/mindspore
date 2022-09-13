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
#include <vector>
#include <map>
#include <string>
#include "src/extendrt/convert/runtime_convert.h"
#include "src/common/config_infos.h"
#include "tools/common/string_util.h"
#include "tools/converter/converter.h"
#include "tools/converter/cxx_api/converter_para.h"

const int kBatchDim = 0;
const int kDynImgSize = 0;
const int kDynBatchSize = 1;
int CheckBatchStringSupport(const std::vector<std::string> &batch_str_vec) {
  if (batch_str_vec.empty()) {
    return -1;
  }
  std::string only_batch = batch_str_vec[0];
  for (size_t i = 1; i < batch_str_vec.size(); ++i) {
    if (batch_str_vec[i] != only_batch) {
      return -1;
    }
  }
  return 0;
}

const size_t kIndex0 = 0;
const size_t kIndex1 = 1;
const size_t kIndex2 = 2;
const size_t kIndex3 = 3;
const int64_t kdynDim = -1;
int DynBatchOrDynImage(const ShapeVector &shape) {
  if (shape.size() != 4) {
    MS_LOG(ERROR) << "Do not support input shape which is not equal to 4 (N C H W)";
    return -1;
  }
  if (shape[kIndex0] != -1 && ((shape[kIndex1] == kdynDim && shape[kIndex2] == kdynDim && shape[kIndex3] != kdynDim) ||
                               (shape[kIndex1] == kdynDim && shape[kIndex2] != kdynDim && shape[kIndex3] == kdynDim) ||
                               (shape[kIndex1] != kdynDim && shape[kIndex2] == kdynDim && shape[kIndex3] == kdynDim))) {
    return kDynImgSize;
  }
  if (shape[kIndex0] == kdynDim && shape[kIndex1] != kdynDim && shape[kIndex2] != kdynDim &&
      shape[kIndex3] != kdynDim) {
    return kDynBatchSize;
  }
  return -1;
}

std::string CombineDynImgString(const struct mindspore::ProfileConfigs &profile) {
  ShapeVector shape = profile.input_infos[kIndex0].input_shape;
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
  int64_t min_first = profile.profiles[kIndex0].inputs[kIndex0].min_dims[first_dim];
  int64_t max_first = profile.profiles[kIndex0].inputs[kIndex0].max_dims[first_dim];
  int64_t min_second = profile.profiles[kIndex0].inputs[kIndex0].min_dims[second_dim];
  int64_t max_second = profile.profiles[kIndex0].inputs[kIndex0].max_dims[second_dim];
  for (int64_t i = min_first; i <= max_first; ++i) {
    for (int64_t j = min_second; j <= max_second; ++j) {
      ret += std::to_string(i) + "," + std::to_string(j) + ";";
    }
  }
  ret = ret.substr(0, ret.size() - 1);  // discard the final ";"
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

std::string FindInAscendMap(const std::string &key, const std::map<string, string> &ascend_map) {
  auto it = ascend_map.find(key);
  if (it != ascend_map.end()) {
    return it->second;
  }
  return "";
}

void SetParamByConfigfile(const std::shared_ptr<mindspore::ConverterPara> &param,
                          const std::map<string, string> &ascend_map) {
  param->aclModelOptionCfgParam.input_format = FindInAscendMap("input_format", ascend_map);
  param->aclModelOptionCfgParam.precision_mode = FindInAscendMap("precision_mode", ascend_map);
  param->aclModelOptionCfgParam.op_select_impl_mode = FindInAscendMap("op_select_impl_mode", ascend_map);
  param->aclModelOptionCfgParam.fusion_switch_config_file_path =
    FindInAscendMap("fusion_switch_config_file_path", ascend_map);
  param->aclModelOptionCfgParam.buffer_optimize = FindInAscendMap("buffer_optimize", ascend_map);
  param->aclModelOptionCfgParam.insert_op_config_file_path = FindInAscendMap("insert_op_config_file_path", ascend_map);
  param->aclModelOptionCfgParam.om_file_path = FindInAscendMap("om_file_path", ascend_map);

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
    }
  }

  it = ascend_map.find("output_type");
  if (it != ascend_map.end()) {
    int32_t val;
    if (mindspore::lite::ConvertIntNum(it->second, &val)) {
      param->aclModelOptionCfgParam.output_type = static_cast<mindspore::DataType>(val);
    } else {
      MS_LOG(ERROR) << "Convert output_type failed";
    }
  }

  it = ascend_map.find("dynamic_dims");
  if (it != ascend_map.end()) {
    std::vector<std::string> batch_size_string = mindspore::lite::SplitStringToVector(it->second, ';');
    if (CheckBatchStringSupport(batch_size_string) != 0) {
      MS_LOG(ERROR) << "Do not support different dynamic_dims!";
      return;
    }
    struct mindspore::ProfileConfigs tmp_profile;
    if (!mindspore::ProfileParser::Parse(ascend_map, false, &tmp_profile)) {
      MS_LOG(ERROR) << "Parse dynamic_dims failed";
    }
    ShapeVector input_shape_vec = tmp_profile.input_infos[0].input_shape;
    switch (DynBatchOrDynImage(input_shape_vec)) {
      case kDynImgSize:
        param->aclModelOptionCfgParam.dynamic_image_size = CombineDynImgString(tmp_profile);
        break;
      case kDynBatchSize:
        for (size_t i = 0; i < tmp_profile.profiles.size(); ++i) {  // dynamic batch size
          int64_t min_batch = tmp_profile.profiles[i].inputs[0].min_dims[kBatchDim];
          int64_t max_batch = tmp_profile.profiles[i].inputs[0].max_dims[kBatchDim];
          for (int64_t batch = min_batch; batch <= max_batch; ++batch) {
            param->aclModelOptionCfgParam.dynamic_batch_size.push_back(batch);
          }
        }
        break;
      default:
        MS_LOG(ERROR) << "Do not support input shape which is not equal to 4 (N C H W)";
    }
  }
}

mindspore::api::FuncGraphPtr RuntimeConvert(const char *model_buf, const size_t &buf_size,
                                            const std::shared_ptr<mindspore::Context> &context,
                                            const ConfigInfos &config_info) {
  auto param = std::make_shared<mindspore::ConverterPara>();
  if (param == nullptr) {
    MS_LOG(ERROR) << "New ConverterPara failed";
    return nullptr;
  }
  param->fmk_type = mindspore::converter::kFmkTypeMs;
  param->input_data_type = mindspore::DataType::kTypeUnknown;
  param->output_data_type = mindspore::DataType::kTypeUnknown;
  param->weight_fp16 = false;
  param->train_model = false;
  param->export_mindir = mindspore::kMindIR;
  param->enable_encryption = false;
  param->is_runtime_converter = true;

  auto device_list = context->MutableDeviceInfo();
  for (auto &device : device_list) {
    if (device->GetDeviceType() == mindspore::kAscend) {
      param->aclModelOptionCfgParam.offline = false;
      param->device = "Ascend";
      param->no_fusion = false;
      if (config_info.find("ascend_context") != config_info.end()) {
        std::map<std::string, std::string> ascend_map = config_info.at("ascend_context");
        SetParamByConfigfile(param, ascend_map);
      } else {
        auto ascend_info = device->Cast<mindspore::AscendDeviceInfo>();
        std::string dynamic_batch_size = ascend_info->GetDynamicBatchSize();
        if (!dynamic_batch_size.empty()) {
          std::vector<std::string> batch_size_string = mindspore::lite::SplitStringToVector(dynamic_batch_size, ',');
          for (const auto &item : batch_size_string) {
            int32_t val;
            if (mindspore::lite::ConvertIntNum(item, &val)) {
              size_t tmp_val = static_cast<size_t>(val);
              param->aclModelOptionCfgParam.dynamic_batch_size.push_back(tmp_val);
            }
          }
        }
        param->aclModelOptionCfgParam.device_id = ascend_info->GetDeviceID();
        param->aclModelOptionCfgParam.output_type = ascend_info->GetOutputType();
        param->aclModelOptionCfgParam.input_shape_map = ascend_info->GetInputShapeMap();
        param->aclModelOptionCfgParam.input_format = ascend_info->GetInputFormat();
        param->aclModelOptionCfgParam.input_shape = ascend_info->GetInputShape();
        param->aclModelOptionCfgParam.precision_mode = ascend_info->GetPrecisionMode();
        param->aclModelOptionCfgParam.op_select_impl_mode = ascend_info->GetOpSelectImplMode();
        param->aclModelOptionCfgParam.fusion_switch_config_file_path = ascend_info->GetFusionSwitchConfigPath();
        param->aclModelOptionCfgParam.buffer_optimize = ascend_info->GetBufferOptimizeMode();
        param->aclModelOptionCfgParam.insert_op_config_file_path = ascend_info->GetInsertOpConfigPath();
        param->aclModelOptionCfgParam.dynamic_image_size = ascend_info->GetDynamicImageSize();
      }
    } else {
      continue;
    }
  }

  mindspore::lite::ConverterImpl cvt;
  mindspore::FuncGraphPtr graph = cvt.Convert(param, model_buf, buf_size);
  if (graph == nullptr) {
    MS_LOG(ERROR) << "Convert model failed";
    return nullptr;
  }
  auto api_graph = mindspore::api::MakeShared<mindspore::api::FuncGraph>(graph);
  return api_graph;
}
