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

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/include/config.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

// Config operations for setting and getting the configuration.
namespace config {

std::shared_ptr<ConfigManager> _config = GlobalContext::config_manager();

// Function to set the seed to be used in any random generator
bool set_seed(int32_t seed) {
  if (seed < 0) {
    MS_LOG(ERROR) << "Seed given is not within the required range: " << seed;
    return false;
  }
  _config->set_seed((uint32_t)seed);
  return true;
}

// Function to get the seed
uint32_t get_seed() { return _config->seed(); }

// Function to set the number of rows to be prefetched
bool set_prefetch_size(int32_t prefetch_size) {
  if (prefetch_size <= 0) {
    MS_LOG(ERROR) << "Prefetch size given is not within the required range: " << prefetch_size;
    return false;
  }
  _config->set_op_connector_size(prefetch_size);
  return true;
}

// Function to get prefetch size in number of rows
int32_t get_prefetch_size() { return _config->op_connector_size(); }

// Function to set the default number of parallel workers
bool set_num_parallel_workers(int32_t num_parallel_workers) {
  if (num_parallel_workers <= 0) {
    MS_LOG(ERROR) << "Number of parallel workers given is not within the required range: " << num_parallel_workers;
    return false;
  }
  _config->set_num_parallel_workers(num_parallel_workers);
  return true;
}

// Function to get the default number of parallel workers
int32_t get_num_parallel_workers() { return _config->num_parallel_workers(); }

// Function to set the default interval (in milliseconds) for monitor sampling
bool set_monitor_sampling_interval(int32_t interval) {
  if (interval <= 0) {
    MS_LOG(ERROR) << "Interval given is not within the required range: " << interval;
    return false;
  }
  _config->set_monitor_sampling_interval((uint32_t)interval);
  return true;
}

// Function to get the default interval of performance monitor sampling
int32_t get_monitor_sampling_interval() { return _config->monitor_sampling_interval(); }

// Function to set the default timeout (in seconds) for DSWaitedCallback
bool set_callback_timeback(int32_t timeout) {
  if (timeout <= 0) {
    MS_LOG(ERROR) << "Timeout given is not within the required range: " << timeout;
    return false;
  }
  _config->set_callback_timeout((uint32_t)timeout);
  return true;
}

// Function to get the default timeout for DSWaitedCallback
int32_t get_callback_timeout() { return _config->callback_timeout(); }

// Function to load configurations from a file
bool load(const std::vector<char> &file) {
  Status rc = _config->LoadFile(CharToString(file));
  if (rc.IsError()) {
    MS_LOG(ERROR) << rc << file;
    return false;
  }
  return true;
}

}  // namespace config
}  // namespace dataset
}  // namespace mindspore
