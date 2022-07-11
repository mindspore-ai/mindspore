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

#include "tools/converter/adapter/acl/api/acl_pass_api.h"

mindspore::opt::Pass *CreateAclPass(const std::shared_ptr<mindspore::ConverterPara> &param) {
  auto acl_pass_ptr = new (std::nothrow) mindspore::opt::AclPass(param);
  if (acl_pass_ptr == nullptr) {
    MS_LOG(ERROR) << "New acl pass failed.";
    return nullptr;
  }
  return acl_pass_ptr;
}

void DestroyAclPass(mindspore::opt::Pass *acl_pass) {
  if (acl_pass == nullptr) {
    MS_LOG(ERROR) << "Param acl pass is nullptr.";
    return;
  }
  delete acl_pass;
}
