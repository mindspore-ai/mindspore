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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_COMMON_ACL_TYPES_UTILS_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_COMMON_ACL_TYPES_UTILS_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "include/api/data_type.h"
#include "include/api/context.h"
#include "mindspore/lite/src/common/log_util.h"
#include "tools/converter/adapter/acl/common/acl_types.h"

namespace mindspore {
namespace lite {
namespace acl {
static void SetAclModelInitOptions(const lite::acl::AclModelOptionCfg &acl_config,
                                   const std::shared_ptr<mindspore::AscendDeviceInfo> &ascend_info) {
  if (!acl_config.fusion_switch_config_file_path.empty()) {
    ascend_info->SetFusionSwitchConfigPath(acl_config.fusion_switch_config_file_path);
  }
  if (!acl_config.op_select_impl_mode.empty()) {
    ascend_info->SetOpSelectImplMode(acl_config.op_select_impl_mode);
  }
  if (!acl_config.buffer_optimize.empty()) {
    ascend_info->SetBufferOptimizeMode(acl_config.buffer_optimize);
  }
}

static void SetAclModelBuildOptions(const lite::acl::AclModelOptionCfg &acl_config,
                                    const std::shared_ptr<mindspore::AscendDeviceInfo> &ascend_info) {
  if (acl_config.output_type != DataType::kInvalidType) {
    ascend_info->SetOutputType(acl_config.output_type);
  }
  if (acl_config.input_shape_map.size() > 0) {
    ascend_info->SetInputShapeMap(acl_config.input_shape_map);
  }
  if (acl_config.dynamic_batch_size.size() > 0) {
    ascend_info->SetDynamicBatchSize(acl_config.dynamic_batch_size);
  }
  if (!acl_config.dynamic_image_size.empty()) {
    ascend_info->SetDynamicImageSize(acl_config.dynamic_image_size);
  }
  if (!acl_config.input_format.empty()) {
    ascend_info->SetInputFormat(acl_config.input_format);
  }
  if (!acl_config.input_shape.empty()) {
    ascend_info->SetInputShape(acl_config.input_shape);
  }
  if (!acl_config.precision_mode.empty()) {
    ascend_info->SetPrecisionMode(acl_config.precision_mode);
  }
  if (!acl_config.insert_op_config_file_path.empty()) {
    ascend_info->SetInsertOpConfigPath(acl_config.insert_op_config_file_path);
  }
}

static std::shared_ptr<mindspore::Context> AsModelContext(const lite::acl::AclModelOptionCfg &acl_config,
                                                          const std::string &provider) {
  auto model_context = std::make_shared<mindspore::Context>();
  MS_CHECK_TRUE_MSG(model_context != nullptr, nullptr, "model_context is nullptr.");
  auto ascend_info = std::make_shared<mindspore::AscendDeviceInfo>();
  MS_CHECK_TRUE_MSG(ascend_info != nullptr, nullptr, "ascend_info is nullptr.");
  if (acl_config.device_id > 0) {
    ascend_info->SetDeviceID(acl_config.device_id);
  }
  ascend_info->SetRankID(acl_config.rank_id);
  ascend_info->SetProvider(provider);
  SetAclModelInitOptions(acl_config, ascend_info);
  SetAclModelBuildOptions(acl_config, ascend_info);

  model_context->MutableDeviceInfo().emplace_back(ascend_info);
  return model_context;
}
}  // namespace acl
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_COMMON_ACL_TYPES_UTILS_H_
