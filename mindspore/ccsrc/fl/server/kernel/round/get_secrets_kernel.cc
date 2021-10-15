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

#include "fl/server/kernel/round/get_secrets_kernel.h"
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <utility>
#include "fl/armour/cipher/cipher_shares.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void GetSecretsKernel::InitKernel(size_t) {
  if (LocalMetaStore::GetInstance().has_value(kCtxTotalTimeoutDuration)) {
    iteration_time_window_ = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  }

  executor_ = &Executor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor_);
  if (!executor_->initialized()) {
    MS_LOG(EXCEPTION) << "Executor must be initialized in server pipeline.";
    return;
  }

  cipher_share_ = &armour::CipherShares::GetInstance();
}

bool GetSecretsKernel::CountForGetSecrets(const std::shared_ptr<FBBuilder> &fbb,
                                          const schema::GetShareSecrets *get_secrets_req, const int iter_num) {
  MS_ERROR_IF_NULL_W_RET_VAL(get_secrets_req, false);
  if (!DistributedCountService::GetInstance().Count(name_, get_secrets_req->fl_id()->str())) {
    std::string reason = "Counting for get secrets kernel request failed. Please retry later.";
    cipher_share_->BuildGetSecretsRsp(fbb, schema::ResponseCode_OutOfTime, IntToSize(iter_num),
                                      std::to_string(CURRENT_TIME_MILLI.count()), nullptr);
    MS_LOG(ERROR) << reason;
    return false;
  }
  return true;
}

bool GetSecretsKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                              const std::vector<AddressPtr> &outputs) {
  bool response = false;
  size_t iter_num = LocalMetaStore::GetInstance().curr_iter_num();
  MS_LOG(INFO) << "ITERATION NUMBER IS : " << LocalMetaStore::GetInstance().curr_iter_num();
  std::string next_timestamp = std::to_string(CURRENT_TIME_MILLI.count());
  size_t total_duration = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  MS_LOG(INFO) << "ITERATION NUMBER IS : " << iter_num << ", Total GetSecretsKernel allowed Duration Is "
               << total_duration;
  clock_t start_time = clock();

  if (inputs.size() != 1 || outputs.size() != 1) {
    std::string reason = "inputs or outputs size is invalid.";
    MS_LOG(ERROR) << reason;
    return false;
  }

  std::shared_ptr<server::FBBuilder> fbb = std::make_shared<server::FBBuilder>();
  void *req_data = inputs[0]->addr;
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    return false;
  }

  const schema::GetShareSecrets *get_secrets_req = flatbuffers::GetRoot<schema::GetShareSecrets>(req_data);
  size_t iter_client = IntToSize(get_secrets_req->iteration());
  if (iter_num != iter_client) {
    MS_LOG(ERROR) << "GetSecretsKernel iteration invalid. server now iteration is " << iter_num
                  << ". client request iteration is " << iter_client;
    cipher_share_->BuildGetSecretsRsp(fbb, schema::ResponseCode_OutOfTime, iter_num, next_timestamp, nullptr);
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(ERROR) << "Current amount for GetSecretsKernel is enough.";
  }

  response = cipher_share_->GetSecrets(get_secrets_req, fbb, next_timestamp);
  if (!response) {
    MS_LOG(WARNING) << "get secret shares is failed.";
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  if (!CountForGetSecrets(fbb, get_secrets_req, SizeToInt(iter_num))) {
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
  clock_t end_time = clock();
  double duration = static_cast<double>((end_time - start_time) * 1.0 / CLOCKS_PER_SEC);
  MS_LOG(INFO) << "GetSecretsKernel DURATION TIME is : " << duration;
  return true;
}

bool GetSecretsKernel::Reset() {
  MS_LOG(INFO) << "ITERATION NUMBER IS : " << LocalMetaStore::GetInstance().curr_iter_num();
  MS_LOG(INFO) << "GetSecretsKernel reset!";
  cipher_share_->ClearShareSecrets();
  DistributedCountService::GetInstance().ResetCounter(name_);
  StopTimer();
  return true;
}

REG_ROUND_KERNEL(getSecrets, GetSecretsKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
