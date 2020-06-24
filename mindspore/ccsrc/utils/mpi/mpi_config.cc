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

#include "utils/mpi/mpi_config.h"

namespace mindspore {
std::shared_ptr<MpiConfig> MpiConfig::instance_ = nullptr;

std::shared_ptr<MpiConfig> MpiConfig::GetInstance() {
  if (instance_ == nullptr) {
    MS_LOG(DEBUG) << "Create new mpi config instance.";
    instance_.reset(new (std::nothrow) MpiConfig());
  }
  return instance_;
}
}  // namespace mindspore
