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

#ifdef RUNTIME_CONVERT
#include <vector>
#include "src/runtime/runtime_convert.h"
#include "tools/common/string_util.h"
#include "include/version.h"
#include "tools/converter/converter.h"
#include "tools/converter/converter_flags.h"
#include "acl/acl_base.h"

namespace mindspore::lite {
char *RuntimeConvert(const char *model_buf, const size_t &buf_size, size_t *size,
                     const std::shared_ptr<mindspore::Context> &context) {
  if (model_buf == nullptr) {
    MS_LOG(ERROR) << "Invalid input model buffer.";
    return nullptr;
  }

  auto flag = std::make_unique<converter::Flags>();
  flag->fmk = converter::kFmkTypeMs;
  flag->inputDataType = kTypeUnknown;
  flag->outputDataType = kTypeUnknown;
  flag->saveFP16 = false;
  flag->trainModel = false;

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
            flag->aclModelOptionCfgParam.dynamic_batch_size.push_back(tmp_val);
          }
        }
      }
      flag->aclModelOptionCfgParam.device_id = ascend_info->GetDeviceID();
      flag->aclModelOptionCfgParam.output_type = ascend_info->GetOutputType();
      flag->aclModelOptionCfgParam.input_shape_map = ascend_info->GetInputShapeMap();
      flag->aclModelOptionCfgParam.input_format = ascend_info->GetInputFormat();
      flag->aclModelOptionCfgParam.input_shape = ascend_info->GetInputShape();
      flag->aclModelOptionCfgParam.precision_mode = ascend_info->GetPrecisionMode();
      flag->aclModelOptionCfgParam.op_select_impl_mode = ascend_info->GetOpSelectImplMode();
      flag->aclModelOptionCfgParam.fusion_switch_config_file_path = ascend_info->GetFusionSwitchConfigPath();
      flag->aclModelOptionCfgParam.buffer_optimize = ascend_info->GetBufferOptimizeMode();
      flag->aclModelOptionCfgParam.insert_op_config_file_path = ascend_info->GetInsertOpConfigPath();
      flag->aclModelOptionCfgParam.dynamic_image_size = ascend_info->GetDynamicImageSize();
    } else {
      continue;
    }
  }

#ifdef ENABLE_LITE_ACL
  const char *soc_name_c = aclrtGetSocName();
  if (soc_name_c != nullptr) {
    std::string soc_name(soc_name_c);
    if (soc_name.find("710") == std::string::npos) {
      flag->device = "Ascend710";
    }
    if (soc_name.find("310") == std::string::npos) {
      flag->device = "Ascend310";
    }
  }
#endif

  Converter cvt;
  auto meta_graph = cvt.Convert(flag, model_buf, buf_size);
  if (meta_graph == nullptr) {
    MS_LOG(ERROR) << "Convert failed.";
    return nullptr;
  }

  void *lite_buf = nullptr;
  meta_graph->version = Version();
  auto status = TransferMetaGraph(*meta_graph, &lite_buf, size);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Transfer model failed.";
    delete meta_graph;
    return nullptr;
  }

  delete meta_graph;
  return reinterpret_cast<char *>(lite_buf);
}

char *RuntimeConvert(const std::string &file_path, size_t *size) {
  auto flag = std::make_unique<converter::Flags>();
  flag->fmk = converter::kFmkTypeMs;
  flag->modelFile = file_path;
  flag->inputDataType = kTypeUnknown;
  flag->outputDataType = kTypeUnknown;
  flag->saveFP16 = false;
  flag->trainModel = false;

  Converter cvt;
  auto meta_graph = cvt.Convert(flag);
  MS_LOG(ERROR) << "Convert failed.";
  if (meta_graph == nullptr) {
    return nullptr;
  }

  void *model_buf = nullptr;
  meta_graph->version = Version();
  auto status = TransferMetaGraph(*meta_graph, &model_buf, size);
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
