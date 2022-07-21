/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "fl/worker/fl_cloud_worker.h"
#include "fl/armour/secure_protocol/key_agreement.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace fl {
namespace worker {
FLCloudWorker &FLCloudWorker::GetInstance() {
  static FLCloudWorker instance;
  return instance;
}

bool FLCloudWorker::Run() {
  if (running_.load()) {
    return true;
  }
  running_ = true;
  MS_LOG(INFO) << "Begin to run federated learning cloud worker.";
  server_domain_ = ps::PSContext::instance()->server_domain();
  fl_id_ = ps::PSContext::instance()->node_id();
  MS_LOG(INFO) << "fl id is:" << fl_id_ << ". Request will be sent to server domain:" << server_domain_;
  http_client_ = std::make_shared<ps::core::HttpClient>(server_domain_);

  http_client_->SetMessageCallback([&](const size_t request_id, const std::string &kernel_path,
                                       const std::shared_ptr<std::vector<unsigned char>> &response_msg) {
    if (handlers_.count(kernel_path) <= 0) {
      MS_LOG(WARNING) << "The kernel path of response message is invalid.";
      return;
    }
    MS_LOG(DEBUG) << "Received the response"
                  << ", kernel_path is " << kernel_path << ", request_id is " << request_id << ", response_msg size is "
                  << response_msg->size();
    const auto &callback = handlers_[kernel_path];
    callback(response_msg);
    NotifyMessageArrival(request_id);
  });

  http_client_->Init();
  return true;
}

bool FLCloudWorker::Finish(const uint32_t &) {
  if (!http_client_->Stop()) {
    MS_LOG(ERROR) << "Stopping http client failed.";
    return false;
  }
  return true;
}

bool FLCloudWorker::SendToServerSync(const std::string kernel_path, const std::string content_type, const void *data,
                                     size_t data_size) {
  MS_ERROR_IF_NULL_W_RET_VAL(data, false);
  if (data_size == 0) {
    MS_LOG(WARNING) << "Sending request for data size must be > 0";
    return false;
  }
  uint64_t request_id = AddMessageTrack(1);
  if (!http_client_->SendMessage(kernel_path, content_type, data, data_size, request_id)) {
    MS_LOG(WARNING) << "Sending request for " << kernel_path << " to server " << server_domain_ << " failed.";
    return false;
  }
  if (!Wait(request_id)) {
    http_client_->BreakLoopEvent();
    return false;
  }
  return true;
}

void FLCloudWorker::RegisterMessageCallback(const std::string kernel_path, const MessageReceive &cb) {
  if (handlers_.count(kernel_path) > 0) {
    MS_LOG(DEBUG) << "Http handlers has already register kernel path:" << kernel_path;
    return;
  }
  handlers_[kernel_path] = cb;
  MS_LOG(INFO) << "Http handlers register kernel path:" << kernel_path;
}

void FLCloudWorker::set_fl_iteration_num(uint64_t iteration_num) { iteration_num_ = iteration_num; }

uint64_t FLCloudWorker::fl_iteration_num() const { return iteration_num_.load(); }

void FLCloudWorker::set_data_size(int data_size) { data_size_ = data_size; }

void FLCloudWorker::set_secret_pk(armour::PrivateKey *secret_pk) { secret_pk_ = secret_pk; }

void FLCloudWorker::set_pw_salt(const std::vector<uint8_t> pw_salt) { pw_salt_ = pw_salt; }

void FLCloudWorker::set_pw_iv(const std::vector<uint8_t> pw_iv) { pw_iv_ = pw_iv; }

void FLCloudWorker::set_public_keys_list(const std::vector<EncryptPublicKeys> public_keys_list) {
  public_keys_list_ = public_keys_list;
}

int FLCloudWorker::data_size() const { return data_size_; }

armour::PrivateKey *FLCloudWorker::secret_pk() const { return secret_pk_; }

std::vector<uint8_t> FLCloudWorker::pw_salt() const { return pw_salt_; }

std::vector<uint8_t> FLCloudWorker::pw_iv() const { return pw_iv_; }

std::vector<EncryptPublicKeys> FLCloudWorker::public_keys_list() const { return public_keys_list_; }

std::string FLCloudWorker::fl_name() const { return ps::kServerModeFL; }

std::string FLCloudWorker::fl_id() const { return fl_id_; }
}  // namespace worker
}  // namespace fl
}  // namespace mindspore
