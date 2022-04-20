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

#include "fl/server/iteration_metrics.h"

#include <fstream>
#include <string>

#include "include/common/debug/common.h"
#include "ps/constants.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace fl {
namespace server {
bool IterationMetrics::Initialize() {
  config_ = std::make_unique<ps::core::FileConfiguration>(config_file_path_);
  MS_EXCEPTION_IF_NULL(config_);
  if (!config_->Initialize()) {
    MS_LOG(EXCEPTION) << "Initializing for Config file path failed!" << config_file_path_
                      << " may be invalid or not exist.";
    return false;
  }
  ps::core::FileConfig metrics_config;
  if (!ps::core::CommUtil::ParseAndCheckConfigJson(config_.get(), kMetrics, &metrics_config)) {
    MS_LOG(WARNING) << "Metrics parament in config is not correct";
    return false;
  }
  metrics_file_path_ = metrics_config.storage_file_path;
  auto realpath = Common::CreatePrefixPath(metrics_file_path_.c_str());
  if (!realpath.has_value()) {
    MS_LOG(EXCEPTION) << "Creating path for " << metrics_file_path_ << " failed.";
    return false;
  }
  metrics_file_.open(realpath.value(), std::ios::app | std::ios::out);
  metrics_file_.close();
  return true;
}

bool IterationMetrics::Summarize() {
  metrics_file_.open(metrics_file_path_, std::ios::out | std::ios::app);
  if (!metrics_file_.is_open()) {
    MS_LOG(ERROR) << "The metrics file is not opened.";
    return false;
  }

  js_[kInstanceName] = instance_name_;
  js_[kStartTime] = start_time_.time_str_mill;
  js_[kEndTime] = end_time_.time_str_mill;
  js_[kFLName] = fl_name_;
  js_[kInstanceStatus] = kInstanceStateName.at(instance_state_);
  js_[kFLIterationNum] = fl_iteration_num_;
  js_[kCurIteration] = cur_iteration_num_;
  js_[kMetricsAuc] = accuracy_;
  js_[kMetricsLoss] = loss_;
  js_[kIterExecutionTime] = iteration_time_cost_;
  js_[kClientVisitedInfo] = round_client_num_map_;
  js_[kIterationResult] = kIterationResultName.at(iteration_result_);

  metrics_file_ << js_ << "\n";
  (void)metrics_file_.flush();
  metrics_file_.close();
  return true;
}

bool IterationMetrics::Clear() {
  if (metrics_file_.is_open()) {
    MS_LOG(INFO) << "Clear the old metrics file " << metrics_file_path_;
    metrics_file_.close();
    metrics_file_.open(metrics_file_path_, std::ios::ate | std::ios::out);
  }
  return true;
}

void IterationMetrics::set_fl_name(const std::string &fl_name) { fl_name_ = fl_name; }

void IterationMetrics::set_fl_iteration_num(size_t fl_iteration_num) { fl_iteration_num_ = fl_iteration_num; }

void IterationMetrics::set_cur_iteration_num(size_t cur_iteration_num) { cur_iteration_num_ = cur_iteration_num; }

void IterationMetrics::set_instance_state(InstanceState state) { instance_state_ = state; }

void IterationMetrics::set_loss(float loss) { loss_ = loss; }

void IterationMetrics::set_accuracy(float acc) { accuracy_ = acc; }

void IterationMetrics::set_iteration_time_cost(uint64_t iteration_time_cost) {
  iteration_time_cost_ = iteration_time_cost;
}

void IterationMetrics::set_round_client_num_map(const std::map<std::string, size_t> round_client_num_map) {
  round_client_num_map_ = round_client_num_map;
}

void IterationMetrics::set_iteration_result(IterationResult iteration_result) { iteration_result_ = iteration_result; }

void IterationMetrics::SetStartTime(const ps::core::Time &start_time) { start_time_ = start_time; }

void IterationMetrics::SetEndTime(const ps::core::Time &end_time) { end_time_ = end_time; }

void IterationMetrics::SetInstanceName(const std::string &instance_name) { instance_name_ = instance_name; }
}  // namespace server
}  // namespace fl
}  // namespace mindspore
