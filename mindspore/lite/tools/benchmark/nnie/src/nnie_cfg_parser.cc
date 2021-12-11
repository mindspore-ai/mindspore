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
#include "src/nnie_cfg_parser.h"
#include <climits>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "include/errorcode.h"
#include "src/nnie_manager.h"
#include "src/nnie_print.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
namespace mindspore {
namespace nnie {
namespace {
constexpr auto ENV_TIME_STEP = "TIME_STEP";
constexpr auto ENV_MAX_ROI_NUM = "MAX_ROI_NUM";
constexpr auto ENV_CORE_IDS = "CORE_IDS";
constexpr auto DELIM = ",";
constexpr int MAX_CORE_ID = 7;
}  // namespace
void Flags::Init() {
  auto *time_step = std::getenv(ENV_TIME_STEP);
  if (time_step != nullptr) {
    auto iter = std::find_if(time_step, time_step + strlen(time_step), [](char val) { return val < '0' || val > '9'; });
    if (iter != time_step) {
      *iter = '\0';
      this->time_step_ = atoi(time_step);
    } else {
      LOGE("TIME_STEP ENV is invalid, now set to default value %d", this->time_step_);
    }
  } else {
    LOGW("TIME_STEP ENV is not set, now set to default value %d", this->time_step_);
  }
  auto *max_roi_num = std::getenv(ENV_MAX_ROI_NUM);
  if (max_roi_num != nullptr) {
    auto iter =
      std::find_if(max_roi_num, max_roi_num + strlen(max_roi_num), [](char val) { return val < '0' || val > '9'; });
    if (iter != max_roi_num) {
      *iter = '\0';
      this->max_roi_num_ = atoi(max_roi_num);
    } else {
      LOGW("MAX_ROI_NUM ENV is invalid, now set to default value %d", this->max_roi_num_);
    }
  } else {
    LOGW("MAX_ROI_NUM ENV is not set, now set to default value %d", this->max_roi_num_);
  }
  auto ids = std::getenv(ENV_CORE_IDS);
  if (ids != nullptr) {
    auto iter = std::find_if(ids, ids + strlen(ids), [](char val) { return (val < '0' || val > '9') && val != ','; });
    std::vector<int> core_ids;
    if (iter != ids) {
      *iter = '\0';
      char *saveptr;
      char *p = strtok_r(ids, DELIM, &saveptr);
      while (p != nullptr) {
        int id = atoi(p);
        p = strtok_r(NULL, DELIM, &saveptr);
        if (id > MAX_CORE_ID || id < 0) {
          LOGE("id is out of range");
          continue;
        }
        if (std::find(core_ids.begin(), core_ids.end(), id) != core_ids.end()) {
          continue;
        }
        core_ids.push_back(id);
      }
    }
    if (!core_ids.empty()) {
      this->core_ids_ = core_ids;
    } else {
      std::string message =
        "CORE_IDS ENV is invalid, now set to default value {" + std::to_string(this->core_ids_.front()) + "}";
      LOGW(message.c_str());
    }
  } else {
    std::string message =
      "CORE_IDS ENV is not set, now set to default value {" + std::to_string(this->core_ids_.front()) + "}";
    LOGW(message.c_str());
  }
}
}  // namespace nnie
}  // namespace mindspore
