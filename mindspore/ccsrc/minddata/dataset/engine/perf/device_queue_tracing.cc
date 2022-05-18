/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/perf/device_queue_tracing.h"
#include <fstream>
#include <string>
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/path.h"
#include "mindspore/core/utils/ms_utils.h"

namespace mindspore {
namespace dataset {

Path DeviceQueueTracing::GetFileName(const std::string &dir_path, const std::string &rank_id) {
  return Path(dir_path) / Path("device_queue_profiling_" + rank_id + ".txt");
}
}  // namespace dataset
}  // namespace mindspore
