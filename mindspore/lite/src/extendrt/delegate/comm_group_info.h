/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_COMM_GROUP_INFO_H
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_COMM_GROUP_INFO_H

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include "mindspore/core/utils/ms_context.h"
#include "proto/node_strategy.pb.h"

namespace mindspore::lite {
using GroupInfoMap = std::vector<std::pair<std::string, std::vector<uint32_t>>>;

class CommGroupInfo {
 public:
  CommGroupInfo() {}
  ~CommGroupInfo() = default;
  bool LoadGroupInfo(const std::string &file, GroupInfoMap *group_info_map) const;

 private:
  bool CheckPointExit(const std::string path) const;
  bool CheckPath(const std::string path) const;
};
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_COMM_GROUP_INFO_H
