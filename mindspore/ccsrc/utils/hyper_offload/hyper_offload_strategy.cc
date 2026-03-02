/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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

#include "utils/hyper_offload/hyper_offload_strategy.h"

#include <algorithm>
#include "utils/log_adapter.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {

OffloadInfoList OffloadInfoFilterByNumber::Filter(const OffloadInfoList &offload_info_list) {
  const size_t offload_total_number = HyperOffloadConfig::GetInstance().select_num();
  if (offload_info_list.size() <= offload_total_number) {
    return offload_info_list;
  }
  OffloadInfoList filtered_offload_list{offload_info_list.begin(), offload_info_list.begin() + offload_total_number};
  MS_LOG(INFO) << "OffloadInfoFilterByNumber filtered list size: " << filtered_offload_list.size();
  return filtered_offload_list;
}
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore
