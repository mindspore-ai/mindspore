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

#ifndef MINDSPORE_DEVICE_QUEUE_TRACING_H
#define MINDSPORE_DEVICE_QUEUE_TRACING_H

#include <string>
#include <vector>
#include "minddata/dataset/engine/perf/profiling.h"
#include "minddata/dataset/util/path.h"

namespace mindspore {
namespace dataset {
constexpr int32_t RECORDS_PER_STEP_DEVICE_QUEUE = 4;
class DeviceQueueTracing : public Tracing {
 public:
  // Constructor
  DeviceQueueTracing() : Tracing(RECORDS_PER_STEP_DEVICE_QUEUE) {}

  // Destructor
  ~DeviceQueueTracing() override = default;

  std::string Name() const override { return kDeviceQueueTracingName; };

  Status Init() override;

  Status GetPipelineTime(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) override;
  Status GetPushTime(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) override;
  Status GetBatchTime(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) override;
  Status GetConnectorSize(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) override;
  Status GetConnectorCapacity(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) override;
  Status GetEmptyQueueFrequency(int32_t start_step, int32_t end_step, float_t *empty_queue_freq) override;

 private:
  Path GetFileName(const std::string &dir_path, const std::string &rank_id) override;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_DEVICE_QUEUE_TRACING_H
