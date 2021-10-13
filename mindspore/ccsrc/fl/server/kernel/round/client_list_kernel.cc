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

#include "fl/server/kernel/round/client_list_kernel.h"
#include <utility>
#include <string>
#include <vector>
#include <memory>
#include "schema/cipher_generated.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void ClientListKernel::InitKernel(size_t) {
  if (LocalMetaStore::GetInstance().has_value(kCtxTotalTimeoutDuration)) {
    iteration_time_window_ = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  }

  executor_ = &Executor::GetInstance();
  MS_EXCEPTION_IF_NULL(executor_);
  if (!executor_->initialized()) {
    MS_LOG(EXCEPTION) << "Executor must be initialized in server pipeline.";
    return;
  }
  cipher_init_ = &armour::CipherInit::GetInstance();
}

bool ClientListKernel::DealClient(const size_t iter_num, const schema::GetClientList *get_clients_req,
                                  const std::shared_ptr<server::FBBuilder> &fbb) {
  std::vector<string> client_list;
  std::vector<string> empty_client_list;
  std::string fl_id = get_clients_req->fl_id()->str();

  if (!LocalMetaStore::GetInstance().has_value(kCtxUpdateModelThld)) {
    MS_LOG(ERROR) << "update_model_client_threshold is not set.";
    BuildClientListRsp(fbb, schema::ResponseCode_SystemError, "update_model_client_threshold is not set.",
                       empty_client_list, std::to_string(CURRENT_TIME_MILLI.count()), SizeToInt(iter_num));
    return false;
  }
  uint64_t update_model_client_needed = LocalMetaStore::GetInstance().value<uint64_t>(kCtxUpdateModelThld);
  PBMetadata client_list_pb_out = DistributedMetadataStore::GetInstance().GetMetadata(kCtxUpdateModelClientList);
  const UpdateModelClientList &client_list_pb = client_list_pb_out.client_list();
  for (size_t i = 0; i < IntToSize(client_list_pb.fl_id_size()); ++i) {
    client_list.push_back(client_list_pb.fl_id(SizeToInt(i)));
  }
  if (static_cast<uint64_t>(client_list.size()) < update_model_client_needed) {
    MS_LOG(INFO) << "The server is not ready. update_model_client_needed: " << update_model_client_needed;
    MS_LOG(INFO) << "now update_model_client_num: " << client_list_pb.fl_id_size();
    BuildClientListRsp(fbb, schema::ResponseCode_SucNotReady, "The server is not ready.", empty_client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), SizeToInt(iter_num));
    return false;
  }

  if (find(client_list.begin(), client_list.end(), fl_id) == client_list.end()) {  // client not in update model clients
    std::string reason = "fl_id: " + fl_id + " is not in the update_model_clients";
    MS_LOG(INFO) << reason;
    BuildClientListRsp(fbb, schema::ResponseCode_RequestError, reason, empty_client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), SizeToInt(iter_num));
    return false;
  }

  bool retcode_client =
    cipher_init_->cipher_meta_storage_.UpdateClientToServer(fl::server::kCtxGetUpdateModelClientList, fl_id);
  if (!retcode_client) {
    std::string reason = "update get update model clients failed";
    MS_LOG(ERROR) << reason;
    BuildClientListRsp(fbb, schema::ResponseCode_SucNotReady, reason, empty_client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), SizeToInt(iter_num));
    return false;
  }
  if (!DistributedCountService::GetInstance().Count(name_, get_clients_req->fl_id()->str())) {
    std::string reason = "Counting for get user list request failed. Please retry later.";
    BuildClientListRsp(fbb, schema::ResponseCode_OutOfTime, reason, empty_client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), SizeToInt(iter_num));
    MS_LOG(ERROR) << reason;
    return false;
  }
  MS_LOG(INFO) << "send clients_list succeed!";
  MS_LOG(INFO) << "UpdateModel client list: ";
  for (size_t i = 0; i < client_list.size(); ++i) {
    MS_LOG(INFO) << " fl_id : " << client_list[i];
  }
  MS_LOG(INFO) << "update_model_client_needed: " << update_model_client_needed;
  BuildClientListRsp(fbb, schema::ResponseCode_SUCCEED, "send clients_list succeed!", client_list,
                     std::to_string(CURRENT_TIME_MILLI.count()), SizeToInt(iter_num));
  return true;
}

bool ClientListKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                              const std::vector<AddressPtr> &outputs) {
  size_t iter_num = LocalMetaStore::GetInstance().curr_iter_num();
  size_t total_duration = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  MS_LOG(INFO) << "Iteration number is " << iter_num << ", ClientListKernel total duration is " << total_duration;
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
  std::vector<string> client_list;
  const schema::GetClientList *get_clients_req = flatbuffers::GetRoot<schema::GetClientList>(req_data);
  size_t iter_client = IntToSize(get_clients_req->iteration());
  if (iter_num != iter_client) {
    MS_LOG(ERROR) << "client list iteration number is invalid: server now iteration is " << iter_num
                  << ". client request iteration is " << iter_client;
    BuildClientListRsp(fbb, schema::ResponseCode_OutOfTime, "iter num is error.", client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), SizeToInt(iter_num));
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(ERROR) << "Current amount for GetClientList is enough.";
  }

  (void)DealClient(iter_num, get_clients_req, fbb);
  GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
  clock_t end_time = clock();
  double duration = static_cast<double>((end_time - start_time) * 1.0 / CLOCKS_PER_SEC);
  MS_LOG(INFO) << "client_list_kernel success time is : " << duration;
  return true;
}  // namespace fl

bool ClientListKernel::Reset() {
  MS_LOG(INFO) << "ITERATION NUMBER IS : " << LocalMetaStore::GetInstance().curr_iter_num();
  MS_LOG(INFO) << "Get Client list kernel reset!";
  DistributedCountService::GetInstance().ResetCounter(name_);
  DistributedMetadataStore::GetInstance().ResetMetadata(kCtxGetUpdateModelClientList);
  StopTimer();
  return true;
}

void ClientListKernel::BuildClientListRsp(const std::shared_ptr<server::FBBuilder> &client_list_resp_builder,
                                          const schema::ResponseCode retcode, const string &reason,
                                          std::vector<std::string> clients, const string &next_req_time,
                                          const int iteration) {
  auto rsp_reason = client_list_resp_builder->CreateString(reason);
  auto rsp_next_req_time = client_list_resp_builder->CreateString(next_req_time);
  std::vector<flatbuffers::Offset<flatbuffers::String>> clients_vector;
  for (auto client : clients) {
    auto client_fb = client_list_resp_builder->CreateString(client);
    clients_vector.push_back(client_fb);
    MS_LOG(WARNING) << "update client list: ";
    MS_LOG(WARNING) << client;
  }
  auto clients_fb = client_list_resp_builder->CreateVector(clients_vector);
  schema::ReturnClientListBuilder rsp_builder(*(client_list_resp_builder.get()));
  rsp_builder.add_retcode(retcode);
  rsp_builder.add_reason(rsp_reason);
  rsp_builder.add_clients(clients_fb);
  rsp_builder.add_iteration(iteration);
  rsp_builder.add_next_req_time(rsp_next_req_time);
  auto rsp_exchange_keys = rsp_builder.Finish();
  client_list_resp_builder->Finish(rsp_exchange_keys);
  return;
}

REG_ROUND_KERNEL(getClientList, ClientListKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
