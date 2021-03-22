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
#include "minddata/dataset/engine/tdt/tdt_handle.h"
namespace mindspore {
namespace dataset {

std::vector<acltdtChannelHandle *> TdtHandle::acl_handle = std::vector<acltdtChannelHandle *>();

void TdtHandle::AddHandle(acltdtChannelHandle *handle) {
  if (handle != nullptr) {
    acl_handle.emplace_back(handle);
  }
}

bool TdtHandle::DestroyHandle() {
  bool destroy_all = true;
  for (auto &handle : acl_handle) {
    if (handle != nullptr) {
      if (acltdtDestroyChannel(handle) != ACL_SUCCESS) {
        destroy_all = false;
      } else {
        handle = nullptr;
      }
    }
  }
  return destroy_all;
}

std::vector<acltdtChannelHandle *> TdtHandle::GetHandle() { return acl_handle; }
}  // namespace dataset
}  // namespace mindspore
