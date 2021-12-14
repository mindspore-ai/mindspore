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

#include "fl/server/kernel/round/push_list_sign_kernel.h"
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
void PushListSignKernel::InitKernel(size_t) {
  if (LocalMetaStore::GetInstance().has_value(kCtxTotalTimeoutDuration)) {
    iteration_time_window_ = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  }
  cipher_init_ = &armour::CipherInit::GetInstance();
}

bool PushListSignKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                const std::vector<AddressPtr> &outputs) {
  size_t iter_num = LocalMetaStore::GetInstance().curr_iter_num();
  MS_LOG(INFO) << "Launching PushListSignKernel, Iteration number is " << iter_num;
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
  flatbuffers::Verifier verifier(reinterpret_cast<uint8_t *>(req_data), inputs[0]->size);
  if (!verifier.VerifyBuffer<schema::SendClientListSign>()) {
    std::string reason = "The schema of PushClientListSign is invalid.";
    BuildPushListSignKernelRsp(fbb, schema::ResponseCode_RequestError, reason,
                               std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    MS_LOG(ERROR) << reason;
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  const schema::SendClientListSign *client_list_sign_req = flatbuffers::GetRoot<schema::SendClientListSign>(req_data);
  if (client_list_sign_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for PushClientListSign.";
    BuildPushListSignKernelRsp(fbb, schema::ResponseCode_RequestError, reason,
                               std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    MS_LOG(ERROR) << reason;
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  // verify signature
  if (ps::PSContext::instance()->pki_verify()) {
    sigVerifyResult verify_result = VerifySignature(client_list_sign_req);
    if (verify_result == sigVerifyResult::FAILED) {
      std::string reason = "verify signature failed.";
      BuildPushListSignKernelRsp(fbb, schema::ResponseCode_RequestError, reason,
                                 std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
      MS_LOG(ERROR) << reason;
      GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }
    if (verify_result == sigVerifyResult::TIMEOUT) {
      std::string reason = "verify signature timestamp failed.";
      BuildPushListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, reason,
                                 std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
      MS_LOG(ERROR) << reason;
      GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }
    if (verify_result == sigVerifyResult::PASSED) {
      MS_LOG(INFO) << "verify signature passed!";
    }
  }
  return LaunchForPushListSign(client_list_sign_req, iter_num, fbb, outputs);
}

bool PushListSignKernel::LaunchForPushListSign(const schema::SendClientListSign *client_list_sign_req,
                                               const size_t &iter_num, const std::shared_ptr<server::FBBuilder> &fbb,
                                               const std::vector<AddressPtr> &outputs) {
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req, false);
  size_t iter_client = IntToSize(client_list_sign_req->iteration());
  if (iter_num != iter_client) {
    std::string reason = "push list sign iteration number is invalid";
    MS_LOG(WARNING) << reason;
    MS_LOG(WARNING) << "server now iteration is " << iter_num << ". client request iteration is " << iter_client;
    BuildPushListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, reason, std::to_string(CURRENT_TIME_MILLI.count()),
                               iter_num);
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  std::vector<string> update_model_clients;
  const PBMetadata update_model_clients_pb_out =
    DistributedMetadataStore::GetInstance().GetMetadata(kCtxUpdateModelClientList);
  const UpdateModelClientList &update_model_clients_pb = update_model_clients_pb_out.client_list();
  for (size_t i = 0; i < IntToSize(update_model_clients_pb.fl_id_size()); ++i) {
    update_model_clients.push_back(update_model_clients_pb.fl_id(i));
  }
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req->fl_id(), false);
  std::string fl_id = client_list_sign_req->fl_id()->str();
  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(ERROR) << "Current amount for PushListSignKernel is enough.";
    if (find(update_model_clients.begin(), update_model_clients.end(), fl_id) != update_model_clients.end()) {
      // client in get update model client list.
      BuildPushListSignKernelRsp(fbb, schema::ResponseCode_SUCCEED, "Current amount for PushListSignKernel is enough.",
                                 std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    } else {
      BuildPushListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime,
                                 "Current amount for PushListSignKernel is enough.",
                                 std::to_string(CURRENT_TIME_MILLI.count()), iter_num);
    }
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  if (!PushListSign(iter_num, std::to_string(CURRENT_TIME_MILLI.count()), client_list_sign_req, fbb,
                    update_model_clients)) {
    MS_LOG(ERROR) << "push client list sign failed.";
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  std::string count_reason = "";
  if (!DistributedCountService::GetInstance().Count(name_, fl_id, &count_reason)) {
    std::string reason = "Counting for push list sign request failed. Please retry later. " + count_reason;
    BuildPushListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, reason, std::to_string(CURRENT_TIME_MILLI.count()),
                               iter_num);
    MS_LOG(ERROR) << reason;
    return true;
  }
  GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
  return true;
}

sigVerifyResult PushListSignKernel::VerifySignature(const schema::SendClientListSign *client_list_sign_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req, sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req->fl_id(), sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req->timestamp(), sigVerifyResult::FAILED);

  std::string fl_id = client_list_sign_req->fl_id()->str();
  std::string timestamp = client_list_sign_req->timestamp()->str();
  int iteration = client_list_sign_req->iteration();
  std::string iter_str = std::to_string(iteration);
  auto fbs_signature = client_list_sign_req->req_signature();
  std::vector<unsigned char> signature;
  if (fbs_signature == nullptr) {
    MS_LOG(ERROR) << "signature in client_list_sign_req is nullptr";
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
  mindspore::ps::server::CertVerify certVerify;
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

bool PushListSignKernel::PushListSign(const size_t cur_iterator, const std::string &next_req_time,
                                      const schema::SendClientListSign *client_list_sign_req,
                                      const std::shared_ptr<fl::server::FBBuilder> &fbb,
                                      const std::vector<std::string> &update_model_clients) {
  MS_LOG(INFO) << "CipherMgr::PushClientListSign START";
  std::vector<std::string> get_client_list;  // the clients which get update model client list
  cipher_init_->cipher_meta_storage_.GetClientListFromServer(fl::server::kCtxGetUpdateModelClientList,
                                                             &get_client_list);
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req, false);
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req->fl_id(), false);

  std::string fl_id = client_list_sign_req->fl_id()->str();
  if (find(get_client_list.begin(), get_client_list.end(), fl_id) == get_client_list.end()) {
    // client not in get update model client list.
    std::string reason = "client send signature is not in get update model client list. && client is illegal";
    MS_LOG(WARNING) << reason;
    if (find(update_model_clients.begin(), update_model_clients.end(), fl_id) != update_model_clients.end()) {
      // client in update model client list, client can move to next round
      BuildPushListSignKernelRsp(fbb, schema::ResponseCode_SUCCEED, reason, next_req_time, cur_iterator);
    } else {
      BuildPushListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, reason, next_req_time, cur_iterator);
    }
    return false;
  }
  std::vector<std::string> send_signs_clients;
  const fl::PBMetadata &clients_sign_pb_out =
    fl::server::DistributedMetadataStore::GetInstance().GetMetadata(kCtxClientListSigns);
  const fl::ClientListSign &clients_sign_pb = clients_sign_pb_out.client_list_sign();
  auto iter = clients_sign_pb.client_list_sign().begin();
  for (; iter != clients_sign_pb.client_list_sign().end(); ++iter) {
    send_signs_clients.push_back(iter->first);
  }
  if (find(send_signs_clients.begin(), send_signs_clients.end(), fl_id) != send_signs_clients.end()) {
    // the client has sended signature, return false.
    std::string reason = "The server has received the request, please do not request again.";
    MS_LOG(ERROR) << reason;
    BuildPushListSignKernelRsp(fbb, schema::ResponseCode_SUCCEED, reason, next_req_time, cur_iterator);
    return false;
  }
  auto fbs_signature = client_list_sign_req->signature();
  std::vector<char> signature;
  if (fbs_signature != nullptr) {
    signature.assign(fbs_signature->begin(), fbs_signature->end());
  }
  fl::PairClientListSign pair_client_list_sign_pb;
  pair_client_list_sign_pb.set_fl_id(fl_id);
  pair_client_list_sign_pb.set_signature(signature.data(), signature.size());
  fl::PBMetadata pb_data;
  pb_data.mutable_pair_client_list_sign()->MergeFrom(pair_client_list_sign_pb);
  bool retcode = fl::server::DistributedMetadataStore::GetInstance().UpdateMetadata(kCtxClientListSigns, pb_data);
  if (!retcode) {
    std::string reason = "store client list signature failed";
    MS_LOG(ERROR) << reason;
    BuildPushListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, reason, next_req_time, cur_iterator);
    return false;
  }
  std::string reason = "send update model client list signature success. ";
  BuildPushListSignKernelRsp(fbb, schema::ResponseCode_SUCCEED, reason, next_req_time, cur_iterator);
  MS_LOG(INFO) << "CipherMgr::PushClientListSign Success";
  return true;
}

bool PushListSignKernel::Reset() {
  MS_LOG(INFO) << "ITERATION NUMBER IS : " << LocalMetaStore::GetInstance().curr_iter_num();
  MS_LOG(INFO) << "Push list sign kernel reset!";
  DistributedCountService::GetInstance().ResetCounter(name_);
  DistributedMetadataStore::GetInstance().ResetMetadata(kCtxClientListSigns);
  StopTimer();
  return true;
}

void PushListSignKernel::BuildPushListSignKernelRsp(const std::shared_ptr<server::FBBuilder> &fbb,
                                                    const schema::ResponseCode retcode, const string &reason,
                                                    const string &next_req_time, const size_t iteration) {
  auto rsp_reason = fbb->CreateString(reason);
  auto rsp_next_req_time = fbb->CreateString(next_req_time);
  schema::ResponseClientListSignBuilder rsp_builder(*(fbb.get()));
  rsp_builder.add_retcode(static_cast<int>(retcode));
  rsp_builder.add_reason(rsp_reason);
  rsp_builder.add_next_req_time(rsp_next_req_time);
  rsp_builder.add_iteration(SizeToInt(iteration));
  auto rsp_push_list_sign = rsp_builder.Finish();
  fbb->Finish(rsp_push_list_sign);
  return;
}

REG_ROUND_KERNEL(pushListSign, PushListSignKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
