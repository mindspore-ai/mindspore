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

#include <cstdlib>

#include "src/common/helper/external_tensor/memory_helper.h"
#include "src/common/log_adapter.h"

namespace mindspore::infer::helper {
void MemoryExternalTensorHelper::free_data() {
  if (is_copy_data_) {
    // copy data, need free memory
    for (auto it = data_map_.begin(); it != data_map_.end(); it++) {
      if (it->second != nullptr) {
        free(it->second);
        it->second = nullptr;
      }
    }
  }
  data_map_.clear();
}

void *MemoryExternalTensorHelper::GetExternalTensorData(const mindspore::schema::ExternalData *external_info) {
  if (external_info == nullptr) {
    MS_LOG_ERROR << "external_info is nullptr.";
    return nullptr;
  }
  auto data_key = external_info->location()->str() + std::to_string(external_info->offset());
  auto it = this->data_map_.find(data_key);
  if (it == this->data_map_.end()) {
    return nullptr;
  }
  return it->second;
}

void MemoryExternalTensorHelper::SetExternalTensorData(const mindspore::schema::ExternalData *external_info,
                                                       void *data) {
  if (external_info == nullptr) {
    MS_LOG_ERROR << "external_info is nullptr.";
    return;
  }
  auto data_key = external_info->location()->str() + std::to_string(external_info->offset());
  if (is_copy_data_) {
    auto size = external_info->length();
    auto *new_data = malloc(size);
    if (new_data == nullptr) {
      MS_LOG_ERROR << "malloc new data with " << size << " failed.";
      return;
    }
    (void)memcpy(new_data, data, size);
    this->data_map_[data_key] = new_data;
  } else {
    // not copy data, just set the pointer;
    this->data_map_[data_key] = data;
  }
}
}  // namespace mindspore::infer::helper
