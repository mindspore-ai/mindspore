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
