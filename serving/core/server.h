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
#ifndef MINDSPORE_SERVER_H
#define MINDSPORE_SERVER_H

#include <string>
#include <mutex>
#include <vector>
#include <memory>
#include "util/status.h"
#include "version_control/model.h"
#include "include/inference.h"
#include "serving/ms_service.pb.h"
#include "serving/ms_service.grpc.pb.h"

namespace mindspore {
namespace serving {

using ms_serving::PredictReply;
using ms_serving::PredictRequest;
using inference::Status;
using inference::SUCCESS;
using inference::FAILED;
using inference::INVALID_INPUTS;

class Session {
 public:
  static Session &Instance();
  Status CreatDeviceSession(const std::string &device, uint32_t device_id);
  // Status Predict(const inference::MultiTensor &inputs, inference::MultiTensor &output);
  Status Predict(const PredictRequest &request, PredictReply &reply);
  Status Warmup(const MindSporeModelPtr model);
  Status Clear();

 private:
  Session() = default;
  ~Session() = default;
  int sesseion_id_{0};
  std::shared_ptr<inference::InferSession> session_{nullptr};
  bool model_loaded_ = false;
  uint32_t graph_id_{0};
  std::mutex mutex_;
  std::string device_type_;
};

class Server {
 public:
  Server() = default;
  ~Server() = default;
  Status BuildAndStart();
};
}  // namespace serving
}  // namespace mindspore
#endif  // MINDSPORE_SERVER_H
