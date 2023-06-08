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

#include <fstream>
#include <vector>
#include <utility>
#include "include/common/utils/utils.h"
#include "utils/ms_utils.h"
#include "src/extendrt/delegate/comm_group_info.h"
#include "src/common/common.h"
#include "include/common/utils/convert_utils.h"
#include "utils/log_adapter.h"
#include "include/common/debug/common.h"
#include "mindspore/core/utils/file_utils.h"
namespace mindspore::lite {
bool CommGroupInfo::CheckPath(const std::string path) const {
  if (path.size() > PATH_MAX) {
    MS_LOG(ERROR) << "The checkpoit path " << path << " is too long";
    return false;
  }
  auto realpath = Common::CreatePrefixPath(path, true);
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Get real path failed, path=" << path;
    return false;
  }
  return true;
}

bool CommGroupInfo::CheckPointExit(const std::string path) const {
  std::ifstream fin(path);
  if (fin) {
    return true;
  }
  return false;
}

bool CommGroupInfo::LoadGroupInfo(const std::string &file, GroupInfoMap *group_info_map) const {
  MS_EXCEPTION_IF_NULL(group_info_map);
  if (!CheckPath(file)) {
    MS_LOG(EXCEPTION) << "CheckPoint file in invalid";
  }
  if (!CheckPointExit(file)) {
    MS_LOG(EXCEPTION) << "CheckPoint file is not found";
  }
  mindspore::straspb::ParallelGroupMap parallel_group_map;
  std::fstream input(file, std::ios::in | std::ios::binary);
  if (!parallel_group_map.ParseFromIstream(&input)) {
    MS_LOG(ERROR) << "Load strategy file failed";
    return false;
  }
  input.close();

  size_t group_num = LongToSize(parallel_group_map.parallel_group_item_size());
  for (size_t i = 0; i < group_num; ++i) {
    mindspore::straspb::ParallelGroupItem parallel_group_item = parallel_group_map.parallel_group_item(SizeToInt(i));
    std::string group_name = parallel_group_item.group_name();

    mindspore::straspb::ParallelGroupRanks parallel_group_ranks = parallel_group_item.parallel_group_ranks();
    size_t rank_num = LongToSize(parallel_group_ranks.dim_size());
    std::vector<uint32_t> ranks;
    for (size_t j = 0; j < rank_num; ++j) {
      uint32_t rank = parallel_group_ranks.dim(SizeToInt(j));
      ranks.push_back(rank);
    }

    std::pair<std::string, std::vector<uint32_t>> group = std::make_pair(group_name, ranks);
    group_info_map->push_back(group);
  }

  return true;
}
}  // namespace mindspore::lite
