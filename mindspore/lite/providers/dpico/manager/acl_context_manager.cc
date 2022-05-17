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

#include "manager/acl_context_manager.h"
#include <string>
#include "include/svp_acl.h"
#include "include/errorcode.h"
#include "common/check_base.h"
namespace mindspore {
namespace lite {
AclContextManager::~AclContextManager() {
  if (acl_rt_stream_ != nullptr) {
    auto ret = svp_acl_rt_destroy_stream(acl_rt_stream_);
    MS_CHECK_TRUE_MSG_VOID(ret == SVP_ACL_SUCCESS, "acl destroy stream failed.");
    acl_rt_stream_ = nullptr;
  }
  if (acl_rt_context_ != nullptr) {
    auto ret = svp_acl_rt_destroy_context(acl_rt_context_);
    MS_CHECK_TRUE_MSG_VOID(ret == SVP_ACL_SUCCESS, "acl destroy context failed.");
    acl_rt_context_ = nullptr;
    ret = svp_acl_rt_reset_device(acl_device_id_);
    MS_CHECK_TRUE_MSG_VOID(ret == SVP_ACL_SUCCESS, "acl reset device failed.");
    ret = svp_acl_finalize();
    MS_CHECK_TRUE_MSG_VOID(ret == SVP_ACL_SUCCESS, "finalize acl failed.");
  }
}

int AclContextManager::Init(const std::string &acl_config_path) {
  if (acl_rt_context_ != nullptr) {
    MS_LOG(INFO) << "device only needs to init once.";
    return RET_OK;
  }
  int ret;
  if (acl_config_path.empty()) {
    ret = svp_acl_init(nullptr);
  } else {
    ret = svp_acl_init(acl_config_path.c_str());
  }
  MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "svp acl init failed.");
  ret = svp_acl_rt_set_device(acl_device_id_);
  MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "svp acl rt set device failed.");
  ret = svp_acl_rt_create_context(&acl_rt_context_, acl_device_id_);
  MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "svp acl rt create context failed.");
  ret = svp_acl_rt_create_stream(&acl_rt_stream_);
  MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "svp acl rt create stream failed.");
  svp_acl_rt_run_mode acl_run_mode;
  ret = svp_acl_rt_get_run_mode(&acl_run_mode);
  MS_CHECK_TRUE_MSG(ret == SVP_ACL_SUCCESS, RET_ERROR, "svp acl rt get run mode failed.");
  MS_CHECK_TRUE_MSG(acl_run_mode == SVP_ACL_DEVICE, RET_ERROR, "svp acl run mode is invalid.");
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
