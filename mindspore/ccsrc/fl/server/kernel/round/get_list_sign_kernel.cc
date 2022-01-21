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

#include "fl/server/kernel/round/get_list_sign_kernel.h"
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
void GetListSignKernel::InitKernel(size_t) {
  if (LocalMetaStore::GetInstance().has_value(kCtxTotalTimeoutDuration)) {
    iteration_time_window_ = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  }
  cipher_init_ = &armour::CipherInit::GetInstance();
}

sigVerifyResult GetListSignKernel::VerifySignature(const schema::RequestAllClientListSign *client_list_sign_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req, sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req->fl_id(), sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(client_list_sign_req->timestamp(), sigVerifyResult::FAILED);

  std::string fl_id = client_list_sign_req->fl_id()->str();
  std::string timestamp = client_list_sign_req->timestamp()->str();
  int iteration = client_list_sign_req->iteration();
  std::string iter_str = std::to_string(iteration);
  auto fbs_signature = client_list_sign_req->signature();
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

bool GetListSignKernel::Launch(const uint8_t *req_data, size_t len,
                               const std::shared_ptr<ps::core::MessageHandler> &message) {
  size_t iter_num = LocalMetaStore::GetInstance().curr_iter_num();
  MS_LOG(INFO) << "Launching GetListSign kernel,  Iteration number is " << iter_num;
  std::shared_ptr<server::FBBuilder> fbb = std::make_shared<server::FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    return false;
  }
  std::map<std::string, std::vector<unsigned char>> list_signs;
  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::RequestAllClientListSign>()) {
    std::string reason = "The schema of RequestAllClientListSign is invalid.";
    BuildGetListSignKernelRsp(fbb, schema::ResponseCode_RequestError, reason,
                              std::to_string(CURRENT_TIME_MILLI.count()), iter_num, list_signs);
    MS_LOG(ERROR) << reason;
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  const schema::RequestAllClientListSign *get_list_sign_req =
    flatbuffers::GetRoot<schema::RequestAllClientListSign>(req_data);
  if (get_list_sign_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for RequestAllClientListSign.";
    BuildGetListSignKernelRsp(fbb, schema::ResponseCode_RequestError, reason,
                              std::to_string(CURRENT_TIME_MILLI.count()), iter_num, list_signs);
    MS_LOG(ERROR) << reason;
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  // verify signature
  if (ps::PSContext::instance()->pki_verify()) {
    sigVerifyResult verify_result = VerifySignature(get_list_sign_req);
    if (verify_result == sigVerifyResult::FAILED) {
      std::string reason = "verify signature failed.";
      BuildGetListSignKernelRsp(fbb, schema::ResponseCode_RequestError, reason,
                                std::to_string(CURRENT_TIME_MILLI.count()), iter_num, list_signs);
      MS_LOG(ERROR) << reason;
      GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }

    if (verify_result == sigVerifyResult::TIMEOUT) {
      std::string reason = "verify signature timestamp failed.";
      BuildGetListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, reason, std::to_string(CURRENT_TIME_MILLI.count()),
                                iter_num, list_signs);
      MS_LOG(ERROR) << reason;
      GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }

    if (verify_result == sigVerifyResult::PASSED) {
      MS_LOG(INFO) << "verify signature passed!";
    }
  }

  size_t iter_client = IntToSize(get_list_sign_req->iteration());
  if (iter_num != iter_client) {
    MS_LOG(ERROR) << "get list sign iteration number is invalid: server now iteration is " << iter_num
                  << ". client request iteration is " << iter_client;
    BuildGetListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, "iter num is error.",
                              std::to_string(CURRENT_TIME_MILLI.count()), iter_num, list_signs);
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  std::string fl_id = get_list_sign_req->fl_id()->str();
  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(WARNING) << "Current amount for GetListSignKernel is enough.";
  }
  if (!GetListSign(iter_num, std::to_string(CURRENT_TIME_MILLI.count()), get_list_sign_req, fbb)) {
    MS_LOG(WARNING) << "get list signs not ready.";
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  std::string count_reason = "";
  if (!DistributedCountService::GetInstance().Count(name_, fl_id, &count_reason)) {
    std::string reason = "Counting for get list sign request failed. Please retry later. " + count_reason;
    BuildGetListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, reason, std::to_string(CURRENT_TIME_MILLI.count()),
                              iter_num, list_signs);
    MS_LOG(ERROR) << reason;
    return true;
  }
  GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
  return true;
}

bool GetListSignKernel::GetListSign(const size_t cur_iterator, const std::string &next_req_time,
                                    const schema::RequestAllClientListSign *get_list_sign_req,
                                    const std::shared_ptr<fl::server::FBBuilder> &fbb) {
  MS_LOG(INFO) << "CipherMgr::SendClientListSign START";
  std::map<std::string, std::vector<unsigned char>> client_list_signs_empty;
  std::map<std::string, std::vector<unsigned char>> client_list_signs_all;
  const fl::PBMetadata &clients_sign_pb_out =
    fl::server::DistributedMetadataStore::GetInstance().GetMetadata(kCtxClientListSigns);
  const fl::ClientListSign &clients_sign_pb = clients_sign_pb_out.client_list_sign();
  size_t cur_clients_sign_num = IntToSize(clients_sign_pb.client_list_sign_size());
  if (cur_clients_sign_num < cipher_init_->push_list_sign_threshold) {
    MS_LOG(INFO) << "The server is not ready. push_list_sign_needed: " << cipher_init_->push_list_sign_threshold;
    MS_LOG(INFO) << "now push_sign_client_num: " << clients_sign_pb.client_list_sign_size();
    BuildGetListSignKernelRsp(fbb, schema::ResponseCode_SucNotReady, "The server is not ready.", next_req_time,
                              cur_iterator, client_list_signs_empty);
    return false;
  }

  std::vector<string> update_model_clients;
  const PBMetadata update_model_clients_pb_out =
    DistributedMetadataStore::GetInstance().GetMetadata(kCtxUpdateModelClientList);
  const UpdateModelClientList &update_model_clients_pb = update_model_clients_pb_out.client_list();
  for (size_t i = 0; i < IntToSize(update_model_clients_pb.fl_id_size()); ++i) {
    update_model_clients.push_back(update_model_clients_pb.fl_id(SizeToInt(i)));
  }

  auto iter = clients_sign_pb.client_list_sign().begin();
  for (; iter != clients_sign_pb.client_list_sign().end(); ++iter) {
    std::vector<unsigned char> signature(iter->second.begin(), iter->second.end());
    (void)client_list_signs_all.emplace(std::pair<std::string, std::vector<unsigned char>>(iter->first, signature));
  }

  MS_ERROR_IF_NULL_W_RET_VAL(get_list_sign_req, false);
  MS_ERROR_IF_NULL_W_RET_VAL(get_list_sign_req->fl_id(), false);
  std::string fl_id = get_list_sign_req->fl_id()->str();
  if (client_list_signs_all.find(fl_id) == client_list_signs_all.end()) {
    std::string reason;
    if (find(update_model_clients.begin(), update_model_clients.end(), fl_id) != update_model_clients.end()) {
      reason = "client not send list signature, but in update model client list.";
      BuildGetListSignKernelRsp(fbb, schema::ResponseCode_SUCCEED, reason, next_req_time, cur_iterator,
                                client_list_signs_all);
    } else {
      reason = "client not send list signature, && client is illegal";
      BuildGetListSignKernelRsp(fbb, schema::ResponseCode_OutOfTime, reason, next_req_time, cur_iterator,
                                client_list_signs_empty);
    }
    MS_LOG(WARNING) << reason;
    return false;
  }

  if (client_list_signs_all.find(fl_id) != client_list_signs_all.end()) {
    // the client has sended signature, return false.
    std::string reason = "The server has received the request, please do not request again.";
    MS_LOG(WARNING) << reason;
    BuildGetListSignKernelRsp(fbb, schema::ResponseCode_SUCCEED, reason, next_req_time, cur_iterator,
                              client_list_signs_all);
    return false;
  }

  std::string reason = "send update model client list signature success. ";
  BuildGetListSignKernelRsp(fbb, schema::ResponseCode_SUCCEED, reason, next_req_time, cur_iterator,
                            client_list_signs_all);
  MS_LOG(INFO) << "CipherMgr::Send Client ListSign Success";
  return true;
}

bool GetListSignKernel::Reset() {
  MS_LOG(INFO) << "ITERATION NUMBER IS : " << LocalMetaStore::GetInstance().curr_iter_num();
  MS_LOG(INFO) << "Get List Signature kernel reset!";
  DistributedCountService::GetInstance().ResetCounter(name_);
  StopTimer();
  return true;
}

void GetListSignKernel::BuildGetListSignKernelRsp(const std::shared_ptr<server::FBBuilder> &fbb,
                                                  const schema::ResponseCode retcode, const string &reason,
                                                  const string &next_req_time, const size_t iteration,
                                                  const std::map<std::string, std::vector<unsigned char>> &list_signs) {
  auto rsp_reason = fbb->CreateString(reason);
  auto rsp_next_req_time = fbb->CreateString(next_req_time);
  if (list_signs.size() == 0) {
    schema::ReturnAllClientListSignBuilder rsp_builder(*(fbb.get()));
    rsp_builder.add_retcode(static_cast<int>(retcode));
    rsp_builder.add_reason(rsp_reason);
    rsp_builder.add_next_req_time(rsp_next_req_time);
    rsp_builder.add_iteration(SizeToInt(iteration));
    auto rsp_get_list_sign = rsp_builder.Finish();
    fbb->Finish(rsp_get_list_sign);
    return;
  }
  std::vector<flatbuffers::Offset<schema::ClientListSign>> client_list_signs;
  for (auto iter = list_signs.begin(); iter != list_signs.end(); ++iter) {
    auto fbs_fl_id = fbb->CreateString(iter->first);
    auto fbs_sign = fbb->CreateVector(iter->second.data(), iter->second.size());
    auto cur_sign = schema::CreateClientListSign(*fbb, fbs_fl_id, fbs_sign);
    client_list_signs.push_back(cur_sign);
  }
  auto all_signs = fbb->CreateVector(client_list_signs);
  schema::ReturnAllClientListSignBuilder rsp_builder(*(fbb.get()));
  rsp_builder.add_retcode(static_cast<int>(retcode));
  rsp_builder.add_reason(rsp_reason);
  rsp_builder.add_next_req_time(rsp_next_req_time);
  rsp_builder.add_iteration(SizeToInt(iteration));
  rsp_builder.add_client_list_sign(all_signs);
  auto rsp_get_list_sign = rsp_builder.Finish();
  fbb->Finish(rsp_get_list_sign);
  return;
}

REG_ROUND_KERNEL(getListSign, GetListSignKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
