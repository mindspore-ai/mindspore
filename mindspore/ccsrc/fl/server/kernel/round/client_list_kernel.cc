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
#include <map>
#include "schema/cipher_generated.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void ClientListKernel::InitKernel(size_t) {
  if (LocalMetaStore::GetInstance().has_value(kCtxTotalTimeoutDuration)) {
    iteration_time_window_ = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  }
  cipher_init_ = &armour::CipherInit::GetInstance();
}

sigVerifyResult ClientListKernel::VerifySignature(const schema::GetClientList *get_clients_req) {
  auto fbs_fl_id = get_clients_req->fl_id();
  MS_EXCEPTION_IF_NULL(fbs_fl_id);
  std::string fl_id = fbs_fl_id->str();
  auto fbs_timestamp = get_clients_req->fl_id();
  MS_EXCEPTION_IF_NULL(fbs_timestamp);
  std::string timestamp = fbs_timestamp->str();
  int iteration = get_clients_req->iteration();
  std::string iter_str = std::to_string(iteration);
  auto fbs_signature = get_clients_req->signature();
  std::vector<unsigned char> signature;
  if (fbs_signature == nullptr) {
    MS_LOG(ERROR) << "signature in get_clients_req is nullptr";
    return sigVerifyResult::FAILED;
  }
  signature.assign(fbs_signature->begin(), fbs_signature->end());
  std::map<std::string, std::string> key_attestations;
  const fl::PBMetadata &key_attestations_pb_out =
    fl::server::DistributedMetadataStore::GetInstance().GetMetadata(kCtxClientKeyAttestation);
  const fl::KeyAttestation &key_attestation_pb = key_attestations_pb_out.key_attestation();
  auto iter = key_attestation_pb.key_attestations().begin();
  for (; iter != key_attestation_pb.key_attestations().end(); ++iter) {
    (void)key_attestations.emplace(std::pair<std::string, std::string>(iter->first, iter->second));
  }
  if (key_attestations.find(fl_id) == key_attestations.end()) {
    MS_LOG(ERROR) << "can not find key attestation for fl_id: " << fl_id;
    return sigVerifyResult::FAILED;
  }

  std::vector<unsigned char> src_data;
  (void)src_data.insert(src_data.end(), timestamp.begin(), timestamp.end());
  (void)src_data.insert(src_data.end(), iter_str.begin(), iter_str.end());
  auto certVerify = mindspore::ps::server::CertVerify::GetInstance();
  unsigned char srcDataHash[SHA256_DIGEST_LENGTH];
  certVerify.sha256Hash(src_data.data(), SizeToInt(src_data.size()), srcDataHash, SHA256_DIGEST_LENGTH);
  if (!certVerify.verifyRSAKey(key_attestations[fl_id], srcDataHash, signature.data(), SHA256_DIGEST_LENGTH)) {
    return sigVerifyResult::FAILED;
  }
  if (!certVerify.verifyTimeStamp(fl_id, timestamp)) {
    return sigVerifyResult::TIMEOUT;
  }
  MS_LOG(INFO) << "verify signature for fl_id: " << fl_id << " success.";
  return sigVerifyResult::PASSED;
}

bool ClientListKernel::DealClient(const size_t iter_num, const schema::GetClientList *get_clients_req,
                                  const std::shared_ptr<server::FBBuilder> &fbb) {
  std::vector<string> client_list;
  std::vector<string> empty_client_list;
  std::string fl_id = get_clients_req->fl_id()->str();

  if (!LocalMetaStore::GetInstance().has_value(kCtxUpdateModelThld)) {
    MS_LOG(ERROR) << "update_model_client_threshold is not set.";
    BuildClientListRsp(fbb, schema::ResponseCode_SystemError, "update_model_client_threshold is not set.",
                       empty_client_list, std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    return false;
  }
  uint64_t update_model_client_needed = LocalMetaStore::GetInstance().value<uint64_t>(kCtxUpdateModelThld);
  PBMetadata client_list_pb_out = DistributedMetadataStore::GetInstance().GetMetadata(kCtxUpdateModelClientList);
  const UpdateModelClientList &client_list_pb = client_list_pb_out.client_list();
  for (size_t i = 0; i < IntToSize(client_list_pb.fl_id_size()); ++i) {
    client_list.push_back(client_list_pb.fl_id(SizeToInt(i)));
  }
  if (client_list.size() < update_model_client_needed) {
    MS_LOG(INFO) << "The server is not ready. update_model_client_needed: " << update_model_client_needed;
    MS_LOG(INFO) << "now update_model_client_num: " << client_list_pb.fl_id_size();
    BuildClientListRsp(fbb, schema::ResponseCode_SucNotReady, "The server is not ready.", empty_client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    return false;
  }

  if (find(client_list.begin(), client_list.end(), fl_id) == client_list.end()) {  // client not in update model clients
    std::string reason = "fl_id: " + fl_id + " is not in the update_model_clients";
    MS_LOG(INFO) << reason;
    BuildClientListRsp(fbb, schema::ResponseCode_RequestError, reason, empty_client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    return false;
  }

  bool retcode_client =
    cipher_init_->cipher_meta_storage_.UpdateClientToServer(fl::server::kCtxGetUpdateModelClientList, fl_id);
  if (!retcode_client) {
    std::string reason = "update get update model clients failed";
    MS_LOG(ERROR) << reason;
    BuildClientListRsp(fbb, schema::ResponseCode_SucNotReady, reason, empty_client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    return false;
  }

  if (!DistributedCountService::GetInstance().Count(name_, get_clients_req->fl_id()->str())) {
    std::string reason = "Counting for get user list request failed. Please retry later.";
    BuildClientListRsp(fbb, schema::ResponseCode_OutOfTime, reason, empty_client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    MS_LOG(ERROR) << reason;
    return false;
  }
  MS_LOG(INFO) << "update_model_client_needed: " << update_model_client_needed;
  BuildClientListRsp(fbb, schema::ResponseCode_SUCCEED, "send clients_list succeed!", client_list,
                     std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
  return true;
}

bool ClientListKernel::Launch(const uint8_t *req_data, size_t len,
                              const std::shared_ptr<ps::core::MessageHandler> &message) {
  size_t iter_num = LocalMetaStore::GetInstance().curr_iter_num();
  MS_LOG(INFO) << "Launching ClientListKernel, Iteration number is " << iter_num;

  std::shared_ptr<server::FBBuilder> fbb = std::make_shared<server::FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    return false;
  }
  std::vector<string> client_list;
  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::GetClientList>()) {
    std::string reason = "The schema of GetClientList is invalid.";
    BuildClientListRsp(fbb, schema::ResponseCode_RequestError, reason, client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    MS_LOG(ERROR) << reason;
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  const schema::GetClientList *get_clients_req = flatbuffers::GetRoot<schema::GetClientList>(req_data);
  if (get_clients_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for GetClientList.";
    BuildClientListRsp(fbb, schema::ResponseCode_RequestError, reason, client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    MS_LOG(ERROR) << reason;
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  // verify signature
  if (ps::PSContext::instance()->pki_verify()) {
    sigVerifyResult verify_result = VerifySignature(get_clients_req);
    if (verify_result == sigVerifyResult::FAILED) {
      std::string reason = "verify signature failed.";
      BuildClientListRsp(fbb, schema::ResponseCode_RequestError, reason, client_list,
                         std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
      MS_LOG(ERROR) << reason;
      GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }
    if (verify_result == sigVerifyResult::TIMEOUT) {
      std::string reason = "verify signature timestamp failed.";
      BuildClientListRsp(fbb, schema::ResponseCode_OutOfTime, reason, client_list,
                         std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
      MS_LOG(ERROR) << reason;
      GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }
    MS_LOG(INFO) << "verify signature passed!";
  }

  size_t iter_client = IntToSize(get_clients_req->iteration());
  if (iter_num != iter_client) {
    MS_LOG(ERROR) << "client list iteration number is invalid: server now iteration is " << iter_num
                  << ". client request iteration is " << iter_client;
    BuildClientListRsp(fbb, schema::ResponseCode_OutOfTime, "iter num is error.", client_list,
                       std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(WARNING) << "Current amount for GetClientList is enough.";
  }

  if (!DealClient(iter_num, get_clients_req, fbb)) {
    MS_LOG(WARNING) << "Get Client List not ready.";
  }
  GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
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

void ClientListKernel::BuildClientListRsp(const std::shared_ptr<server::FBBuilder> &fbb,
                                          const schema::ResponseCode retcode, const string &reason,
                                          std::vector<std::string> clients, const string &next_req_time,
                                          const size_t iteration) {
  auto rsp_reason = fbb->CreateString(reason);
  auto rsp_next_req_time = fbb->CreateString(next_req_time);
  std::vector<flatbuffers::Offset<flatbuffers::String>> clients_vector;
  for (auto client : clients) {
    auto client_fb = fbb->CreateString(client);
    clients_vector.push_back(client_fb);
    MS_LOG(WARNING) << "update client list: ";
    MS_LOG(WARNING) << client;
  }
  auto clients_fb = fbb->CreateVector(clients_vector);
  schema::ReturnClientListBuilder rsp_builder(*(fbb.get()));
  rsp_builder.add_retcode(SizeToInt(retcode));
  rsp_builder.add_reason(rsp_reason);
  rsp_builder.add_clients(clients_fb);
  rsp_builder.add_iteration(SizeToInt(iteration));
  rsp_builder.add_next_req_time(rsp_next_req_time);
  auto rsp_exchange_keys = rsp_builder.Finish();
  fbb->Finish(rsp_exchange_keys);
  return;
}

REG_ROUND_KERNEL(getClientList, ClientListKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
