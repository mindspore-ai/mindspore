/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_POOL_RUNNER_CONFIG_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_POOL_RUNNER_CONFIG_H_
#include <memory>
#include <string>
#include <map>
#include "include/api/model_parallel_runner.h"
namespace mindspore {
struct RunnerConfig::Data {
  int workers_num = 0;
  std::shared_ptr<Context> context = nullptr;
  std::map<std::string, std::map<std::string, std::string>> config_info;
  std::string config_path = "";
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_CXX_API_MODEL_POOL_RUNNER_CONFIG_H_
