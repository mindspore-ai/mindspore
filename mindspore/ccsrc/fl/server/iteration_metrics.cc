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
#include <string>
#include <fstream>
#include "debug/common.h"
#include "ps/constants.h"

namespace mindspore {
namespace fl {
namespace server {
bool IterationMetrics::Initialize() {
  config_ = std::make_unique<ps::core::FileConfiguration>(config_file_path_);
  MS_EXCEPTION_IF_NULL(config_);
  if (!config_->Initialize()) {
    MS_LOG(WARNING) << "Initializing for metrics failed. Config file path " << config_file_path_
                    << " may be invalid or not exist.";
    return false;
  }

  // Read the metrics file path. If file is not set or not exits, create one.
  if (!config_->Exists(kMetrics)) {
    MS_LOG(WARNING) << "Metrics config is not set. Don't write metrics.";
    return false;
  } else {
    std::string value = config_->Get(kMetrics, "");
    nlohmann::json value_json;
    try {
      value_json = nlohmann::json::parse(value);
    } catch (const std::exception &e) {
      MS_LOG(EXCEPTION) << "The hyper-parameter data is not in json format.";
      return false;
    }

    // Parse the storage type.
    uint32_t storage_type = JsonGetKeyWithException<uint32_t>(value_json, ps::kStoreType);
    if (std::to_string(storage_type) != ps::kFileStorage) {
      MS_LOG(EXCEPTION) << "Storage type " << storage_type << " is not supported.";
      return false;
    }

    // Parse storage file path.
    metrics_file_path_ = JsonGetKeyWithException<std::string>(value_json, ps::kStoreFilePath);
    auto realpath = Common::CreatePrefixPath(metrics_file_path_.c_str());
    if (!realpath.has_value()) {
      MS_LOG(EXCEPTION) << "Creating path for " << metrics_file_path_ << " failed.";
      return false;
    }

    metrics_file_.open(realpath.value(), std::ios::ate | std::ios::out);
    metrics_file_.close();
  }
  return true;
}

bool IterationMetrics::Summarize() {
  metrics_file_.open(metrics_file_path_, std::ios::ate | std::ios::out);
  if (!metrics_file_.is_open()) {
    MS_LOG(ERROR) << "The metrics file is not opened.";
    return false;
  }

  js_[kFLName] = fl_name_;
  js_[kInstanceStatus] = kInstanceStateName.at(instance_state_);
  js_[kFLIterationNum] = fl_iteration_num_;
  js_[kCurIteration] = cur_iteration_num_;
  js_[kJoinedClientNum] = joined_client_num_;
  js_[kRejectedClientNum] = rejected_client_num_;
  js_[kMetricsAuc] = accuracy_;
  js_[kMetricsLoss] = loss_;
  js_[kIterExecutionTime] = iteration_time_cost_;
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

void IterationMetrics::set_joined_client_num(size_t joined_client_num) { joined_client_num_ = joined_client_num; }

void IterationMetrics::set_rejected_client_num(size_t rejected_client_num) {
  rejected_client_num_ = rejected_client_num;
}

void IterationMetrics::set_iteration_time_cost(uint64_t iteration_time_cost) {
  iteration_time_cost_ = iteration_time_cost;
}
}  // namespace server
}  // namespace fl
}  // namespace mindspore
