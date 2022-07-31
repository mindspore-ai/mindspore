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

#include "src/litert/kernel/ascend/src/acl_options_parser.h"
#include <utility>
#include <vector>
#include "common/log_adapter.h"
#include "src/common/log_util.h"
#include "src/common/utils.h"
#include "acl/acl_base.h"
#include "acl/acl_rt.h"

namespace mindspore::kernel {
namespace acl {
constexpr auto kImageHwNum = 2;

STATUS AclOptionsParser::ParseAclOptions(const mindspore::Context *ctx, AclModelOptions *acl_options) {
  CHECK_NULL_RETURN(ctx);
  CHECK_NULL_RETURN(acl_options);

  auto context = const_cast<mindspore::Context *>(ctx);
  CHECK_NULL_RETURN(context);
  auto device_infos = context->MutableDeviceInfo();
  if (device_infos.size() < 1) {
    MS_LOG(WARNING) << "Context is not set device info, please check.";
    return lite::RET_OK;
  }
  CHECK_NULL_RETURN(device_infos[0]);
  if (ParseOptions(device_infos[0], acl_options) != lite::RET_OK) {
    MS_LOG(ERROR) << "Parse acl options failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS AclOptionsParser::ParseOptions(const std::shared_ptr<DeviceInfoContext> &device_info,
                                      AclModelOptions *acl_options) {
  auto ascend_info = device_info->Cast<mindspore::AscendDeviceInfo>();
  if (ascend_info == nullptr) {
    MS_LOG(ERROR) << "There is no ascend info.";
    return lite::RET_ERROR;
  }
  int32_t device_id = static_cast<int32_t>(ascend_info->GetDeviceID());
  if (CheckDeviceId(&device_id) != lite::RET_OK) {
    MS_LOG(ERROR) << "Check device id failed, device id = " << device_id;
    return lite::RET_ERROR;
  }
  acl_options->device_id = device_id;
  return lite::RET_OK;
}

STATUS AclOptionsParser::CheckDeviceId(int32_t *device_id) {
  CHECK_NULL_RETURN(device_id);
  uint32_t device_count;
  if (aclrtGetDeviceCount(&device_count) != ACL_ERROR_NONE) {
    MS_LOG(WARNING) << "Get device count failed.";
    return lite::RET_OK;
  }
  if (*device_id >= static_cast<int32_t>(device_count)) {
    MS_LOG(ERROR) << "Current device id " << *device_id << " is larger than max count " << device_count
                  << ",please check the device info of context.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}
}  // namespace acl
}  // namespace mindspore::kernel
