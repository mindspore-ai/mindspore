/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifdef RUNTIME_CONVERT
#include <vector>
#include "src/litert/runtime_convert.h"
#include "tools/common/string_util.h"
#include "tools/converter/converter.h"
#include "tools/converter/cxx_api/converter_para.h"

namespace mindspore::lite {
char *RuntimeConvert(const char *model_buf, const size_t &buf_size, size_t *size, mindspore::Context *context) {
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "Invalid input model buffer.";
    return nullptr;
  }
  auto param = std::make_shared<ConverterPara>();
  if (param == nullptr) {
    MS_LOG(ERROR) << "New ConverterPara failed";
    return nullptr;
  }
  param->fmk_type = converter::kFmkTypeMs;
  param->input_data_type = DataType::kTypeUnknown;
  param->output_data_type = DataType::kTypeUnknown;
  param->weight_fp16 = false;
  param->train_model = false;

  auto device_list = context->MutableDeviceInfo();
  for (auto &device : device_list) {
    if (device->GetDeviceType() == kAscend) {
      auto ascend_info = device->Cast<AscendDeviceInfo>();
      std::string dynamic_batch_size = ascend_info->GetDynamicBatchSize();
      if (!dynamic_batch_size.empty()) {
        std::vector<std::string> batch_size_string = SplitStringToVector(dynamic_batch_size, ',');
        for (const auto &item : batch_size_string) {
          int32_t val;
          if (ConvertIntNum(item, &val)) {
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
    } else {
      continue;
    }
  }

  ConverterImpl cvt;
  schema::MetaGraphT *meta_graph = nullptr;
  auto status = cvt.Convert(param, &meta_graph, model_buf, buf_size);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Convert model failed.";
    return nullptr;
  }
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "meta graph is nullptr.";
    return nullptr;
  }
  void *lite_buf = nullptr;
  meta_graph->version = Version();
  status = TransferMetaGraph(*meta_graph, &lite_buf, size);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Transfer model failed.";
    delete meta_graph;
    return nullptr;
  }

  delete meta_graph;
  return reinterpret_cast<char *>(lite_buf);
}

char *RuntimeConvert(const std::string &file_path, size_t *size) {
  auto param = std::make_shared<ConverterPara>();
  if (param == nullptr) {
    MS_LOG(ERROR) << "New ConverterPara failed";
    return nullptr;
  }
  param->fmk_type = converter::kFmkTypeMs;
  param->model_file = file_path;
  param->input_data_type = DataType::kTypeUnknown;
  param->output_data_type = DataType::kTypeUnknown;
  param->weight_fp16 = false;
  param->train_model = false;

  ConverterImpl cvt;
  schema::MetaGraphT *meta_graph = nullptr;
  auto status = cvt.Convert(param, &meta_graph);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Convert model failed";
    return nullptr;
  }
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "meta graph is nullptr.";
    return nullptr;
  }

  void *model_buf = nullptr;
  meta_graph->version = Version();
  status = TransferMetaGraph(*meta_graph, &model_buf, size);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Transfer model failed.";
    delete meta_graph;
    return nullptr;
  }

  delete meta_graph;
  return reinterpret_cast<char *>(model_buf);
}
}  // namespace mindspore::lite
#endif
