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

#include "fl/server/local_meta_store.h"

namespace mindspore {
namespace fl {
namespace server {
void LocalMetaStore::remove_value(const std::string &name) {
  std::unique_lock<std::mutex> lock(mtx_);
  if (key_to_meta_.count(name) != 0) {
    (void)key_to_meta_.erase(key_to_meta_.find(name));
  }
}

bool LocalMetaStore::has_value(const std::string &name) {
  std::unique_lock<std::mutex> lock(mtx_);
  return key_to_meta_.count(name) != 0;
}

void LocalMetaStore::set_curr_iter_num(size_t num) {
  std::unique_lock<std::mutex> lock(mtx_);
  curr_iter_num_ = num;
}

const size_t LocalMetaStore::curr_iter_num() {
  std::unique_lock<std::mutex> lock(mtx_);
  return curr_iter_num_;
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
