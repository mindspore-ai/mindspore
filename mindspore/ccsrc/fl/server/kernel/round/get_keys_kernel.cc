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

#include "fl/server/kernel/round/get_keys_kernel.h"
#include <vector>
#include <memory>

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void GetKeysKernel::InitKernel(size_t) {
  if (LocalMetaStore::GetInstance().has_value(kCtxTotalTimeoutDuration)) {
    iteration_time_window_ = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  }

  executor_ = &Executor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor_);
  if (!executor_->initialized()) {
    MS_LOG(EXCEPTION) << "Executor must be initialized in server pipeline.";
    return;
  }

  cipher_key_ = &armour::CipherKeys::GetInstance();
}

bool GetKeysKernel::CountForGetKeys(const std::shared_ptr<FBBuilder> &fbb, const schema::GetExchangeKeys *get_keys_req,
                                    const int iter_num) {
  MS_ERROR_IF_NULL_W_RET_VAL(get_keys_req, false);
  if (!DistributedCountService::GetInstance().Count(name_, get_keys_req->fl_id()->str())) {
    std::string reason = "Counting for getkeys kernel request failed. Please retry later.";
    cipher_key_->BuildGetKeysRsp(
      fbb, schema::ResponseCode_OutOfTime, IntToSize(iter_num),
      std::to_string(LocalMetaStore::GetInstance().value<uint64_t>(kCtxIterationNextRequestTimestamp)), false);
    MS_LOG(ERROR) << reason;
    return false;
  }
  return true;
}

bool GetKeysKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                           const std::vector<AddressPtr> &outputs) {
  MS_LOG(INFO) << "Launching GetKeys kernel.";
  bool response = false;
  size_t iter_num = LocalMetaStore::GetInstance().curr_iter_num();
  size_t total_duration = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  MS_LOG(INFO) << "ITERATION NUMBER IS : " << iter_num << ", Total GetKeysKernel allowed Duration Is "
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

  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(ERROR) << "Current amount for GetKeysKernel is enough.";
  }

  const schema::GetExchangeKeys *get_exchange_keys_req = flatbuffers::GetRoot<schema::GetExchangeKeys>(req_data);
  size_t iter_client = IntToSize(get_exchange_keys_req->iteration());
  if (iter_num != iter_client) {
    MS_LOG(ERROR) << "GetKeysKernel iteration invalid. server now iteration is " << iter_num
                  << ". client request iteration is " << iter_client;
    cipher_key_->BuildGetKeysRsp(fbb, schema::ResponseCode_OutOfTime, iter_num,
                                 std::to_string(CURRENT_TIME_MILLI.count()), false);
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  response = cipher_key_->GetKeys(iter_num, std::to_string(CURRENT_TIME_MILLI.count()), get_exchange_keys_req, fbb);
  if (!response) {
    MS_LOG(WARNING) << "get public keys is failed.";
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  if (!CountForGetKeys(fbb, get_exchange_keys_req, SizeToInt(iter_num))) {
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  GenerateOutput(outputs, fbb->GetCurrentBufferPointer(), fbb->GetSize());
  clock_t end_time = clock();
  double duration = static_cast<double>((end_time - start_time) * 1.0 / CLOCKS_PER_SEC);
  MS_LOG(INFO) << "GetKeysKernel DURATION TIME IS : " << duration;
  return true;
}

bool GetKeysKernel::Reset() {
  MS_LOG(INFO) << "get keys kernel reset! ITERATION NUMBER IS : " << LocalMetaStore::GetInstance().curr_iter_num();
  cipher_key_->ClearKeys();
  DistributedCountService::GetInstance().ResetCounter(name_);
  StopTimer();
  return true;
}

REG_ROUND_KERNEL(getKeys, GetKeysKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
