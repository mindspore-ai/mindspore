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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_CONFIG_MANAGER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_CONFIG_MANAGER_H_

#include <atomic>
#include <ostream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"

// Config settings for the client-side
// example config file:
// {
//    "rowsPerBuffer": 3
// }
//

namespace mindspore {
namespace dataset {
// The ConfigManager is a class for managing default values.  When a user is constructing any objects
// in the framework, often they may choose to omit some settings instead of overriding them.
// This class manages some of the default values, for cases when the user does not manually specify
// those values.
class ConfigManager {
 public:
  ConfigManager();

  // destructor
  ~ConfigManager() = default;

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  void Print(std::ostream &out) const;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param cS - reference to the ConfigManager to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const ConfigManager &cS) {
    cS.Print(out);
    return out;
  }

  // Another debug print helper.  Converts the print info to a string for you.
  // @return The string version of the debug print
  std::string ToString() {
    std::stringstream ss;
    ss << *this;
    return ss.str();
  }

  // Loads a json file with the default settings and populates all the settings
  // @param settingsFile - A json file with a set of default settings
  // @return Status error code
  Status LoadFile(const std::string &settingsFile);

  // getter function
  // @return The rows per buffer setting
  int32_t rows_per_buffer() const { return rows_per_buffer_; }

  // getter function
  // @return The number of workers setting
  int32_t num_parallel_workers() const { return num_parallel_workers_; }

  // getter function
  // @return The queue size of the operator's output connector
  int32_t op_connector_size() const { return op_connector_size_; }

  // getter function
  // @return The sending batches that will send to device
  int64_t sending_batches() const { return sending_batches_; }

  // getter function
  // @return The internal worker-to-master connector queue size
  int32_t worker_connector_size() const { return worker_connector_size_; }

  int32_t num_cpu_threads() const { return num_cpu_threads_; }

  // getter function
  // @return The hostname of cache server
  std::string cache_host() const { return cache_host_; }

  // getter function
  // @return The port of cache server
  int32_t cache_port() const { return cache_port_; }

  /// getter function
  /// \return Number of tcp/ip connection
  int32_t num_connections() const { return num_connections_; }

  /// getter function
  /// \return Prefetch size
  int32_t prefetch_size() const { return prefetch_size_; }

  /// getter function
  /// \return auto_num_workers_
  bool auto_num_workers() const { return auto_num_workers_; }

  // setter function
  // @param rows_per_buffer - The setting to apply to the config
  void set_rows_per_buffer(int32_t rows_per_buffer);

  // setter function
  // @param num_parallel_workers - The setting to apply to the config
  void set_num_parallel_workers(int32_t num_parallel_workers);

  // setter function
  // @param connector_size - The setting to apply to the config
  void set_worker_connector_size(int32_t connector_size);

  // setter function
  // @param connector_size - The setting to apply to the config
  void set_op_connector_size(int32_t connector_size);

  // setter function
  // @param sending_batches - The setting to apply to the config
  void set_sending_batches(int64_t sending_batches);

  // setter function
  // @param cache_host - The hostname of cache server
  void set_cache_host(std::string cache_host);

  // setter function
  // @param cache_port - The port of cache server
  void set_cache_port(int32_t cache_port);

  /// setter function
  /// \param num_connections
  void set_num_connections(int32_t num_connections);

  /// setter function
  /// \param prefetch_size
  void set_prefetch_size(int32_t prefetch_size);

  /// setter function
  /// \param numa_switch
  void set_numa_enable(bool numa_enable);

  /// getter function
  /// Now we want to separate the numa link to _c_dataengine in the CMakeLists,
  /// so we want user to choose whether to open numa switch.
  /// @return Get the current numa switch state.
  bool numa_enable() const { return numa_enable_; }

  // getter function
  // This rank_id is for numa and device_queue, one process work with only one rank_id
  // for standalone scenario, this rank_id may come from env 'CUDA_VISIBLE_DEVICES',
  // but for distribute scenario, this rank_id come from _get_global_rank() in python
  // @return Get the current device id, for one process, it's only with one rank_id.
  int32_t rank_id() const { return rank_id_; }

  // setter function
  // @param rank_id - Set the current device id
  void set_rank_id(int32_t rank_id);

  uint32_t seed() const;

  // setter function
  // @param seed - The default seed to use
  void set_seed(uint32_t seed);

  // setter function
  // @param interval - The setting to apply to the config
  void set_monitor_sampling_interval(uint32_t interval);

  // getter function
  // @return The interval of monitor sampling
  int32_t monitor_sampling_interval() const { return monitor_sampling_interval_; }

  // setter function
  // @param stop_profiler - The setting to apply to the config
  void stop_dataset_profiler(bool stop_profiler);

  // getter function
  // @return The status of stop profiler
  bool stop_profiler_status() const { return stop_profiler_; }

  // setter function
  // @param file_ready - The setting to apply to the config
  void set_profiler_file_status(bool file_ready);

  // getter function
  // @return The status of profiler file, whether generated
  bool get_profiler_file_status() const { return file_ready_; }

  // setter function
  // @param auto_num_workers - whether assign threads to each op automatically
  void set_auto_num_workers(bool auto_num_workers) { auto_num_workers_ = auto_num_workers; }

  // setter function
  // this function will be called when a distributed sampler (RT and Obj) is created and will be used by AutoWorkerPass
  // This is to get around the limitation of PreBuildSampler (which doesn't have a getter for sharding params)
  // @param num_shards
  void set_num_shards_for_auto_num_workers(int32_t num_shards) { auto_num_workers_num_shards_ = num_shards; }

  // getter function, will be called by AutoNumWorker, user discretion above AutoNumWorker is advised
  // @param num_shards_
  int32_t get_num_shards_for_auto_num_workers() const { return auto_num_workers_num_shards_; }

  // setter function
  // @param timeout - The setting to apply to the config
  void set_callback_timeout(uint32_t timeout);

  // getter function
  // @return The timeout DSWaitedCallback would wait for before raising an error
  int32_t callback_timeout() const { return callback_timout_; }

  // getter function
  // E.g. 0 would corresponds to a 1:1:1 ratio of num_worker among leaf batch and map.
  // please refer to AutoWorkerPass for detail on what each option is.
  // @return The experimental config used by AutoNumWorker, each 1 refers to a different setup configuration
  uint8_t get_auto_worker_config() { return auto_worker_config_; }

  // setter function
  // E.g. set the value of 0 would corresponds to a 1:1:1 ratio of num_worker among leaf batch and map.
  // please refer to AutoWorkerPass for detail on what each option is.
  // @return The experimental config used by AutoNumWorker, each 1 refers to a different setup configuration
  void set_auto_worker_config_(uint8_t cfg) { auto_worker_config_ = cfg; }

 private:
  int32_t rows_per_buffer_;
  int32_t num_parallel_workers_;
  int32_t worker_connector_size_;
  int32_t op_connector_size_;
  int64_t sending_batches_;
  // This rank_id is for numa and device_queue, one process work with only one rank_id,
  // for standalone scenario, this rank_id may come from env 'CUDA_VISIBLE_DEVICES',
  // but for distribute scenario, this rank_id come from _get_global_rank() in python
  int32_t rank_id_;
  uint32_t seed_;
  uint32_t monitor_sampling_interval_;
  std::atomic_bool stop_profiler_;
  std::atomic_bool file_ready_;
  uint32_t callback_timout_;
  std::string cache_host_;
  int32_t cache_port_;
  int32_t num_connections_;
  bool numa_enable_;
  int32_t prefetch_size_;
  bool auto_num_workers_;
  const int32_t num_cpu_threads_;
  int32_t auto_num_workers_num_shards_;
  uint8_t auto_worker_config_;
  // Private helper function that takes a nlohmann json format and populates the settings
  // @param j - The json nlohmann json info
  Status FromJson(const nlohmann::json &j);
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_CONFIG_MANAGER_H_
