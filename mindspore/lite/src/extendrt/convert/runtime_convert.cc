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
#include "src/extendrt/convert/runtime_convert.h"
#include "tools/common/string_util.h"
#include "tools/converter/converter.h"
#include "tools/converter/cxx_api/converter_para.h"

mindspore::api::FuncGraphPtr RuntimeConvert(const char *model_buf, const size_t &buf_size,
                                            const std::shared_ptr<mindspore::Context> &context) {
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
      param->aclModelOptionCfgParam.offline = false;
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
      param->device = "Ascend";
      param->no_fusion = false;
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
