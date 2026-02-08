/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_UTILS_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_UTILS_H_

#include <hccl/hccl_types.h>

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "mindspore/core/include/utils/ms_utils.h"

#ifndef EXPORT_WRAPPER
#define EXPORT_WRAPPER __attribute__((visibility("default")))
#endif

namespace mindspore {
namespace device {
namespace ascend {

using GroupsVecBuffleSet = std::set<std::pair<std::vector<uint32_t>, std::string>>;

// hccl buffle size config.
const char kHcclConf[] = "MS_DEV_HCCL_CONF";
const char kHcclEnableConfig[] = "enable_hccl_config";
const char kHcclCustomizedDefault[] = "hccl_customized_default";
const char kHcclListConfig[] = "hccl_list_config";
const char kHcclStrideConfig[] = "hccl_stride_config";

// comm
const char kWhiteSpace[] = " \t";

std::map<std::vector<unsigned int>, uint32_t> GetHcclBuffleConfig();
std::string GetHcclConfigValue(const std::string &hccl_config);
bool IsEnableHcclConfig(const std::string &hccl_config);
bool IsDisableHcclConfig(const std::string &hccl_config);
uint32_t GetHcclBufferSize(const std::string &group_name, const std::vector<unsigned int> &rank_id_list);

class VectorUtils {
 public:
  template <typename T>
  static std::string PrintVector(const std::vector<T> &vec) {
    constexpr int MAX_PRINT_NUM = 100;
    std::stringstream ss;
    ss << "[";
    int size = std::min(static_cast<int>(vec.size()), MAX_PRINT_NUM);
    for (int i = 0; i < size; ++i) {
      ss << std::to_string(vec[i]);
      if (i != size - 1) {
        ss << ", ";
      }
    }
    if (vec.size() > MAX_PRINT_NUM) {
      ss << ", ... to be continue";
    }
    ss << "]";
    return ss.str();
  }
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_UTILS_H_
