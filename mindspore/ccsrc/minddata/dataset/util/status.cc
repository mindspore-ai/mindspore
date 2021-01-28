/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/util/status.h"
#include <sstream>
#include <string>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils/ms_utils.h"
#include "./securec.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/util/task_manager.h"
#else
#include "minddata/dataset/util/log_adapter.h"
#endif

namespace mindspore {
namespace dataset {
#if !defined(_WIN32) && !defined(_WIN64)
float GetMemoryUsage() {
  char buf[128] = {0};

  FILE *fd;
  fd = fopen("/proc/meminfo", "r");
  if (fd == nullptr) {
    MS_LOG(WARNING) << "The meminfo file: /proc/meminfo is opened failed.";
    return 0.0;
  }

  uint32_t status_count = 0;
  uint64_t mem_total = 0L;
  uint64_t mem_available = 0L;
  while (fgets(buf, sizeof(buf), fd)) {
    if (status_count == 2) {  // get MemTotal and MemAvailable yet
      break;
    }

    // get title
    std::string line(buf);
    std::string::size_type position = line.find(":");
    std::string title = line.substr(0, position);

    // get the value when MemTotal or MemAvailable
    if (title == "MemTotal") {
      std::string::size_type pos1 = line.find_last_of(" ");
      std::string::size_type pos2 = line.find_last_of(" ", pos1 - 1);
      mem_total = atol(line.substr(pos2, pos1 - pos2).c_str());
      status_count++;
    } else if (title == "MemAvailable") {
      std::string::size_type pos1 = line.find_last_of(" ");
      std::string::size_type pos2 = line.find_last_of(" ", pos1 - 1);
      mem_available = atol(line.substr(pos2, pos1 - pos2).c_str());
      status_count++;
    }

    (void)memset_s(buf, sizeof(buf), 0, sizeof(buf));
  }
  fclose(fd);

  if (status_count != 2 || mem_total == 0 || mem_available > mem_total) {
    MS_LOG(WARNING) << "Get memory usage failed.";
    return 0.0;
  }

  return (1.0 - static_cast<float>(static_cast<double>(mem_available) / static_cast<double>(mem_total)));
}
#endif
}  // namespace dataset
}  // namespace mindspore
