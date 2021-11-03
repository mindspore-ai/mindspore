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
#ifndef ENABLE_ANDROID
#include "utils/log_adapter.h"
#else
#include "mindspore/lite/src/common/log_adapter.h"
#endif
#include "minddata/dataset/util/path.h"
#include "mindspore/core/utils/ms_utils.h"

namespace mindspore {
namespace dataset {

constexpr int32_t PUSH_TIME_OFFSET = 0;
constexpr int32_t BATCH_TIME_OFFSET = 1;
constexpr int32_t PIPELINE_TIME_OFFSET = 2;
constexpr int32_t CONNECTOR_DEPTH_OFFSET = 3;

Status DeviceQueueTracing::Init(const std::string &dir_path, const std::string &device_id) {
  file_path_ = (Path(dir_path) / Path("device_queue_profiling_" + device_id + ".txt")).ToString();
  (void)ts_.emplace_back(0);
  return Status::OK();
}

Status DeviceQueueTracing::GetPipelineTime(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) {
  return GetRecordEntryFieldValue(start_step, end_step, PIPELINE_TIME_OFFSET, "value", result);
}

Status DeviceQueueTracing::GetPushTime(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) {
  return GetRecordEntryFieldValue(start_step, end_step, PUSH_TIME_OFFSET, "value", result);
}

Status DeviceQueueTracing::GetBatchTime(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) {
  return GetRecordEntryFieldValue(start_step, end_step, BATCH_TIME_OFFSET, "value", result);
}

Status DeviceQueueTracing::GetConnectorSize(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) {
  return GetRecordEntryFieldValue(start_step, end_step, CONNECTOR_DEPTH_OFFSET, "value", result);
}

Status DeviceQueueTracing::GetEmptyQueueFrequency(int32_t start_step, int32_t end_step, float_t *empty_queue_freq) {
  std::lock_guard<std::mutex> guard(lock_);
  auto total_steps = records_.size() / records_per_step_;
  MS_LOG(DEBUG) << "start_step: " << start_step << " end_step: " << end_step;
  CHECK_FAIL_RETURN_UNEXPECTED(start_step <= total_steps,
                               "Expected start_step <= total_steps. Got start_step: " + std::to_string(start_step) +
                                 " total_steps: " + std::to_string(total_steps));
  CHECK_FAIL_RETURN_UNEXPECTED(end_step <= total_steps,
                               "Expected end_step <= total_steps. Got end_step: " + std::to_string(end_step) +
                                 " total_steps: " + std::to_string(total_steps));
  CHECK_FAIL_RETURN_UNEXPECTED(start_step <= end_step,
                               "Expected start_step <= end_step. Got start_step: " + std::to_string(start_step) +
                                 " end_step: " + std::to_string(end_step));

  uint32_t total = end_step - start_step + 1;
  uint32_t count = 0U;
  for (auto step_num = start_step; step_num <= end_step; step_num++) {
    auto idx = (step_num - 1) * records_per_step_ + CONNECTOR_DEPTH_OFFSET;
    count += static_cast<uint32_t>(records_[idx].value == 0);
  }
  *empty_queue_freq = static_cast<float_t>(count) / static_cast<float_t>(total);
  return Status::OK();
}

Status DeviceQueueTracing::GetConnectorCapacity(int32_t start_step, int32_t end_step, std::vector<int32_t> *result) {
  return GetRecordEntryFieldValue(start_step, end_step, CONNECTOR_DEPTH_OFFSET, "extra_info", result);
}
}  // namespace dataset
}  // namespace mindspore
