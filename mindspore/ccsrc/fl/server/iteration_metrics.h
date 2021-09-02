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

#ifndef MINDSPORE_CCSRC_FL_SERVER_ITERATION_METRICS_H_
#define MINDSPORE_CCSRC_FL_SERVER_ITERATION_METRICS_H_

#include <map>
#include <string>
#include <memory>
#include <fstream>
#include "ps/ps_context.h"
#include "ps/core/configuration.h"
#include "ps/core/file_configuration.h"
#include "fl/server/local_meta_store.h"
#include "fl/server/iteration.h"

namespace mindspore {
namespace fl {
namespace server {
constexpr auto kFLName = "flName";
constexpr auto kInstanceStatus = "instanceStatus";
constexpr auto kFLIterationNum = "flIterationNum";
constexpr auto kCurIteration = "currentIteration";
constexpr auto kJoinedClientNum = "joinedClientNum";
constexpr auto kRejectedClientNum = "rejectedClientNum";
constexpr auto kMetricsAuc = "metricsAuc";
constexpr auto kMetricsLoss = "metricsLoss";
constexpr auto kIterExecutionTime = "iterationExecutionTime";

const std::map<InstanceState, std::string> kInstanceStateName = {
  {InstanceState::kRunning, "running"}, {InstanceState::kDisable, "disable"}, {InstanceState::kFinish, "finish"}};

template <typename T>
inline T JsonGetKeyWithException(const nlohmann::json &json, const std::string &key) {
  if (!json.contains(key)) {
    MS_LOG(EXCEPTION) << "The key " << key << "does not exist in json " << json.dump();
  }
  return json[key].get<T>();
}

constexpr auto kMetrics = "metrics";

class IterationMetrics {
 public:
  explicit IterationMetrics(const std::string &config_file)
      : config_file_path_(config_file),
        config_(nullptr),
        fl_name_(""),
        fl_iteration_num_(0),
        cur_iteration_num_(0),
        instance_state_(InstanceState::kFinish),
        loss_(0.0),
        accuracy_(0.0),
        joined_client_num_(0),
        rejected_client_num_(0),
        iteration_time_cost_(0) {}
  ~IterationMetrics() = default;

  bool Initialize();

  // Gather the details of this iteration and output to the persistent storage.
  bool Summarize();

  // Clear data in persistent storage.
  bool Clear();

  // Setters for the metrics data.
  void set_fl_name(const std::string &fl_name);
  void set_fl_iteration_num(size_t fl_iteration_num);
  void set_cur_iteration_num(size_t cur_iteration_num);
  void set_instance_state(InstanceState state);
  void set_loss(float loss);
  void set_accuracy(float acc);
  void set_joined_client_num(size_t joined_client_num);
  void set_rejected_client_num(size_t rejected_client_num);
  void set_iteration_time_cost(uint64_t iteration_time_cost);

 private:
  // This is the main config file set by ps context.
  std::string config_file_path_;
  std::unique_ptr<ps::core::FileConfiguration> config_;

  // The metrics file object.
  std::fstream metrics_file_;

  // The metrics file path.
  std::string metrics_file_path_;

  // Json object of metrics data.
  nlohmann::basic_json<std::map, std::vector, std::string, bool, int64_t, uint64_t, float> js_;

  // The federated learning job name. Set by ps_context.
  std::string fl_name_;

  // Federated learning iteration number. Set by ps_context.
  // If this number of iterations are completed, one instance is finished.
  size_t fl_iteration_num_;

  // Current iteration number.
  size_t cur_iteration_num_;

  // Current instance state.
  InstanceState instance_state_;

  // The training loss after this federated learning iteration, passed by worker.
  float loss_;

  // The evaluation result after this federated learning iteration, passed by worker.
  float accuracy_;

  // The number of clients which join the federated aggregation.
  size_t joined_client_num_;

  // The number of clients which are not involved in federated aggregation.
  size_t rejected_client_num_;

  // The time cost in millisecond for this completed iteration.
  uint64_t iteration_time_cost_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_ITERATION_METRICS_H_
