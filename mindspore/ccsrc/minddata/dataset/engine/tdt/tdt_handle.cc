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
extern std::set<void **> acl_handle_set;
namespace dataset {

void TdtHandle::AddHandle(acltdtChannelHandle **handle) {
  if (*handle != nullptr) {
    acl_handle_set.insert(reinterpret_cast<void **>(handle));
  }
}

void TdtHandle::DelHandle(acltdtChannelHandle **handle) {
  void **void_handle = reinterpret_cast<void **>(handle);
  acl_handle_set.erase(void_handle);
}

bool TdtHandle::DestroyHandle() {
  bool destroy_all = true;
  for (auto it = acl_handle_set.begin(); it != acl_handle_set.end(); it++) {
    acltdtChannelHandle **handle = reinterpret_cast<acltdtChannelHandle **>(*it);
    if (*handle != nullptr) {
      acltdtStopChannel(*handle);
      if (acltdtDestroyChannel(*handle) != ACL_SUCCESS) {
        destroy_all = false;
      } else {
        *handle = nullptr;
      }
    }
  }
  return destroy_all;
}

}  // namespace dataset
}  // namespace mindspore
