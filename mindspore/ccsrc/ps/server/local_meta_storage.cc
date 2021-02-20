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

#include "ps/server/local_meta_storage.h"
#include <string>

namespace mindspore {
namespace ps {
namespace server {
void LocalMetaStorage::remove_value(const std::string &name) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (key_to_meta_.count(name) != 0) {
    key_to_meta_.erase(key_to_meta_.find(name));
  }
}

bool LocalMetaStorage::has_value(const std::string &name) {
  std::unique_lock<std::mutex> lock(mtx_);
  return key_to_meta_.count(name) != 0;
}

void LocalMetaStorage::set_curr_iter_num(size_t num) {
  std::unique_lock<std::mutex> lock(mtx_);
  curr_iter_num_ = num;
}

const size_t LocalMetaStorage::curr_iter_num() {
  std::unique_lock<std::mutex> lock(mtx_);
  return curr_iter_num_;
}
}  // namespace server
}  // namespace ps
}  // namespace mindspore
