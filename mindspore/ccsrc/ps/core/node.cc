/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ps/core/node.h"

namespace mindspore {
namespace ps {
namespace core {
std::string Node::node_id() const { return node_info_.node_id_; }

uint32_t Node::rank_id() const {
  if (!is_ready_.load()) {
    MS_LOG(EXCEPTION) << "The cluster is not ready yet to get rank id!";
  }
  return node_info_.rank_id_;
}

NodeRole Node::role() const { return node_info_.node_role_; }

bool Node::WaitForStart(const uint32_t &timeout) {
  std::unique_lock<std::mutex> lock(wait_start_mutex_);
  bool res = wait_start_cond_.wait_for(lock, std::chrono::seconds(timeout), [&] {
    bool res = is_ready_.load();
    if (res) {
      MS_LOG(INFO) << "The node id:" << node_info_.node_id_ << " is success start!";
    }
    return res;
  });
  return res;
}
}  // namespace core
}  // namespace ps
}  // namespace mindspore
