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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_PERF_INFO_COLLECTOR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_PERF_INFO_COLLECTOR_H_

#include <map>
#include <string>

#include "minddata/dataset/util/status.h"

namespace mindspore::dataset {
enum InfoLevel : uint8_t { kDeveloper = 0, kUser = 1 };
enum InfoType : uint8_t { kAll = 0, kMemory = 1, kTime = 2 };
enum TimeType : uint8_t { kStart = 0, kEnd = 1, kStamp = 2 };

Status CollectPipelineInfoStart(const std::string &event, const std::string &stage,
                                const std::map<std::string, std::string> &custom_info = {});

Status CollectPipelineInfoEnd(const std::string &event, const std::string &stage,
                              const std::map<std::string, std::string> &custom_info = {});

Status CollectOpInfoStart(const std::string &event, const std::string &stage,
                          const std::map<std::string, std::string> &custom_info = {});

Status CollectOpInfoEnd(const std::string &event, const std::string &stage,
                        const std::map<std::string, std::string> &custom_info = {});
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_PERF_INFO_COLLECTOR_H_
