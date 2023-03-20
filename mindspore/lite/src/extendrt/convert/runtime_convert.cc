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
#include "tools/common/string_util.h"
#include "tools/converter/converter_funcgraph.h"
#include "tools/converter/cxx_api/converter_para.h"
#include "tools/converter/config_parser/config_file_parser.h"

static int ParseShapeStrToShapeMap(const std::string &input_shape_str,
                                   std::map<std::string, std::vector<int64_t>> *input_shape_map) {
  std::vector<int64_t> shape;
  auto shape_strs = mindspore::lite::StrSplit(input_shape_str, std::string(";"));
  for (const auto &shape_str : shape_strs) {
    if (shape_str.empty()) {
      continue;
    }
    shape.clear();
    auto string_split = mindspore::lite::StrSplit(shape_str, std::string(":"));
    constexpr int kMinShapeSizeInStr = 2;
    if (string_split.size() < kMinShapeSizeInStr) {
      MS_LOG(ERROR) << "shape size must not be less than " << kMinShapeSizeInStr;
      return mindspore::lite::RET_INPUT_PARAM_INVALID;
    }
    auto name = string_split[0];
    for (size_t i = 1; i < string_split.size() - 1; ++i) {
      name += ":" + string_split[i];
    }
    if (name.empty()) {
      MS_LOG(ERROR) << "input tensor name is empty";
      return mindspore::lite::RET_INPUT_PARAM_INVALID;
    }
    auto dim_strs = string_split[string_split.size() - 1];
    if (dim_strs.empty()) {
      MS_LOG(ERROR) << "input tensor dim string is empty";
      return mindspore::lite::RET_INPUT_PARAM_INVALID;
    }
    auto dims = mindspore::lite::StrSplit(dim_strs, std::string(","));
    if (dims.empty()) {
      MS_LOG(ERROR) << "input tensor dim is empty";
      return mindspore::lite::RET_INPUT_PARAM_INVALID;
    }
    for (const auto &dim : dims) {
      int64_t dim_value;
      try {
        dim_value = std::stoi(dim);
      } catch (const std::exception &e) {
        MS_LOG(ERROR) << "Get dim failed: " << e.what();
        return mindspore::lite::RET_INPUT_PARAM_INVALID;
      }
      shape.push_back(dim_value);
    }
    (*input_shape_map)[name] = shape;
  }
  return RET_OK;
}

static int UpdateDynamicInputShape(mindspore::FuncGraphPtr func_graph, const std::string &input_shape_str) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "funcGraph is nullptr";
    return RET_ERROR;
  }
  std::map<std::string, std::vector<int64_t>> input_shape_map;
  auto ret = ParseShapeStrToShapeMap(input_shape_str, &input_shape_map);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "parse shape string to shape map error";
    return RET_ERROR;
  }

  if (input_shape_map.size() != func_graph->get_inputs().size()) {
    MS_LOG(ERROR) << "Number of inputs from the config [" << input_shape_map.size() << "] does not match graph inputs ["
                  << func_graph->get_inputs().size() << "]";
    return RET_ERROR;
  }

  size_t input_index = 0;
  for (auto shape_pair : input_shape_map) {
    std::vector<int64_t> shape_vec = shape_pair.second;
    auto shape_ptr = std::make_shared<mindspore::abstract::Shape>(shape_vec);
    func_graph->get_inputs()[input_index]->abstract()->set_shape(shape_ptr);
    input_index++;
  }

  return RET_OK;
}

void SetParamByAscendInfo(const std::shared_ptr<mindspore::ConverterPara> &param,
                          const std::shared_ptr<mindspore::AscendDeviceInfo> &ascend_info) {
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
  if (ascend_info->GetDeviceID() >= 0) {
    param->aclModelOptionCfgParam.device_id = ascend_info->GetDeviceID();
  }
  if (ascend_info->GetOutputType() != mindspore::DataType::kTypeUnknown) {
    param->aclModelOptionCfgParam.output_type = ascend_info->GetOutputType();
  }
  if (!ascend_info->GetInputShapeMap().empty()) {
    param->aclModelOptionCfgParam.input_shape_map = ascend_info->GetInputShapeMap();
  }
  if (!ascend_info->GetInputFormat().empty()) {
    param->aclModelOptionCfgParam.input_format = ascend_info->GetInputFormat();
  }
  if (!ascend_info->GetInputShape().empty()) {
    param->aclModelOptionCfgParam.input_shape = ascend_info->GetInputShape();
  }
  if (!ascend_info->GetPrecisionMode().empty()) {
    param->aclModelOptionCfgParam.precision_mode = ascend_info->GetPrecisionMode();
  }
  if (!ascend_info->GetOpSelectImplMode().empty()) {
    param->aclModelOptionCfgParam.op_select_impl_mode = ascend_info->GetOpSelectImplMode();
  }
  if (!ascend_info->GetFusionSwitchConfigPath().empty()) {
    param->aclModelOptionCfgParam.fusion_switch_config_file_path = ascend_info->GetFusionSwitchConfigPath();
  }
  if (!ascend_info->GetBufferOptimizeMode().empty()) {
    param->aclModelOptionCfgParam.buffer_optimize = ascend_info->GetBufferOptimizeMode();
  }
  if (!ascend_info->GetInsertOpConfigPath().empty()) {
    param->aclModelOptionCfgParam.insert_op_config_file_path = ascend_info->GetInsertOpConfigPath();
  }
  if (!ascend_info->GetDynamicImageSize().empty()) {
    param->aclModelOptionCfgParam.dynamic_image_size = ascend_info->GetDynamicImageSize();
  }
}

int RuntimeConvert(const mindspore::api::FuncGraphPtr &graph, const std::shared_ptr<mindspore::Context> &context,
                   const ConfigInfos &config_info) {
  auto param = std::make_shared<mindspore::ConverterPara>();
  if (param == nullptr) {
    MS_LOG(ERROR) << "New ConverterPara failed";
    return RET_ERROR;
  }
  param->fmk_type = mindspore::converter::kFmkTypeMs;
  param->input_data_type = mindspore::DataType::kTypeUnknown;
  param->output_data_type = mindspore::DataType::kTypeUnknown;
  param->weight_fp16 = false;
  param->train_model = false;
  param->save_type = mindspore::kMindIR;
  param->enable_encryption = false;
  param->is_runtime_converter = true;

  auto device_list = context->MutableDeviceInfo();
  for (auto &device : device_list) {
    if (device->GetDeviceType() == mindspore::kAscend) {
      param->aclModelOptionCfgParam.offline = false;
      param->device = "Ascend310";
      param->no_fusion = false;
      if (config_info.find("ascend_context") != config_info.end()) {
        std::map<std::string, std::string> ascend_map = config_info.at("ascend_context");
        mindspore::lite::ConfigFileParser config_parser;
        config_parser.SetParamByConfigfile(param, ascend_map);
      }
      auto ascend_info = device->Cast<mindspore::AscendDeviceInfo>();
      SetParamByAscendInfo(param, ascend_info);

      if (!((param->aclModelOptionCfgParam.input_shape).empty())) {
        auto ret = UpdateDynamicInputShape(std::dynamic_pointer_cast<mindspore::FuncGraph>(graph->impl()),
                                           param->aclModelOptionCfgParam.input_shape);
        if (ret != RET_OK) {
          MS_LOG(ERROR) << "Runtime convert update dynamic input shape failed";
          return ret;
        }
      }
    } else {
      continue;
    }
  }
  auto func_graph = std::dynamic_pointer_cast<mindspore::FuncGraph>(graph->impl());
  mindspore::lite::ConverterFuncGraph cvt;
  auto status = cvt.Optimize(param, func_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Convert model failed";
    return status;
  }
  return RET_OK;
}
