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

#include "fl/server/kernel/round/share_secrets_kernel.h"
#include <vector>
#include <memory>
#include <map>
#include <utility>

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void ShareSecretsKernel::InitKernel(size_t) {
  if (LocalMetaStore::GetInstance().has_value(kCtxTotalTimeoutDuration)) {
    iteration_time_window_ = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  }
  cipher_share_ = &armour::CipherShares::GetInstance();
}

bool ShareSecretsKernel::CountForShareSecrets(const std::shared_ptr<FBBuilder> &fbb,
                                              const schema::RequestShareSecrets *share_secrets_req,
                                              const size_t iter_num) {
  if (!DistributedCountService::GetInstance().Count(name_, share_secrets_req->fl_id()->str())) {
    std::string reason = "Counting for share secret kernel request failed. Please retry later.";
    cipher_share_->BuildShareSecretsRsp(fbb, schema::ResponseCode_OutOfTime, reason,
                                        std::to_string(CURRENT_TIME_MILLI.count()), SizeToInt(iter_num));
    MS_LOG(ERROR) << reason;
    return false;
  }
  return true;
}

sigVerifyResult ShareSecretsKernel::VerifySignature(const schema::RequestShareSecrets *share_secrets_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(share_secrets_req, sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(share_secrets_req->fl_id(), sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(share_secrets_req->timestamp(), sigVerifyResult::FAILED);

  std::string fl_id = share_secrets_req->fl_id()->str();
  std::string timestamp = share_secrets_req->timestamp()->str();
  int iteration = share_secrets_req->iteration();
  std::string iter_str = std::to_string(iteration);
  auto fbs_signature = share_secrets_req->signature();
  std::vector<unsigned char> signature;
  if (fbs_signature == nullptr) {
    MS_LOG(ERROR) << "signature in share_secrets_req is nullptr";
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

bool ShareSecretsKernel::Launch(const uint8_t *req_data, size_t len,
                                const std::shared_ptr<ps::core::MessageHandler> &message) {
  bool response = false;
  size_t iter_num = LocalMetaStore::GetInstance().curr_iter_num();
  MS_LOG(INFO) << "Launching ShareSecretsKernel, ITERATION NUMBER IS : " << iter_num;

  std::shared_ptr<server::FBBuilder> fbb = std::make_shared<server::FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    return false;
  }
  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(ERROR) << "Current amount for ShareSecretsKernel is enough.";
    cipher_share_->BuildShareSecretsRsp(fbb, schema::ResponseCode_OutOfTime,
                                        "Current amount for ShareSecretsKernel is enough.",
                                        std::to_string(CURRENT_TIME_MILLI.count()), SizeToInt(iter_num));
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::RequestShareSecrets>()) {
    std::string reason = "The schema of RequestShareSecrets is invalid.";
    cipher_share_->BuildShareSecretsRsp(fbb, schema::ResponseCode_RequestError, reason,
                                        std::to_string(CURRENT_TIME_MILLI.count()), SizeToInt(iter_num));
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  const schema::RequestShareSecrets *share_secrets_req = flatbuffers::GetRoot<schema::RequestShareSecrets>(req_data);
  if (share_secrets_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for RequestShareSecrets.";
    cipher_share_->BuildShareSecretsRsp(fbb, schema::ResponseCode_RequestError, reason,
                                        std::to_string(CURRENT_TIME_MILLI.count()), SizeToInt(iter_num));
    MS_LOG(ERROR) << reason;
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  // verify signature
  if (ps::PSContext::instance()->pki_verify()) {
    sigVerifyResult verify_result = VerifySignature(share_secrets_req);
    if (verify_result == sigVerifyResult::FAILED) {
      std::string reason = "verify signature failed.";
      cipher_share_->BuildShareSecretsRsp(fbb, schema::ResponseCode_RequestError, reason,
                                          std::to_string(CURRENT_TIME_MILLI.count()), SizeToInt(iter_num));
      MS_LOG(ERROR) << reason;
      SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }

    if (verify_result == sigVerifyResult::TIMEOUT) {
      std::string reason = "verify signature timestamp failed.";
      cipher_share_->BuildShareSecretsRsp(fbb, schema::ResponseCode_OutOfTime, reason,
                                          std::to_string(CURRENT_TIME_MILLI.count()), SizeToInt(iter_num));
      MS_LOG(ERROR) << reason;
      SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }
    MS_LOG(INFO) << "verify signature passed!";
  }

  size_t iter_client = IntToSize(share_secrets_req->iteration());
  if (iter_num != iter_client) {
    MS_LOG(ERROR) << "ShareSecretsKernel iteration invalid. server now iteration is " << iter_num
                  << ". client request iteration is " << iter_client;
    cipher_share_->BuildShareSecretsRsp(fbb, schema::ResponseCode_OutOfTime, "ShareSecretsKernel iteration invalid",
                                        std::to_string(CURRENT_TIME_MILLI.count()), SizeToInt(iter_num));
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  response = cipher_share_->ShareSecrets(SizeToInt(iter_num), share_secrets_req, fbb,
                                         std::to_string(CURRENT_TIME_MILLI.count()));
  if (!response) {
    MS_LOG(ERROR) << "update secret shares is failed.";
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  if (!CountForShareSecrets(fbb, share_secrets_req, iter_num)) {
    SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  SendResponseMsg(message, fbb->GetBufferPointer(), fbb->GetSize());
  return true;
}

bool ShareSecretsKernel::Reset() {
  MS_LOG(INFO) << "share_secrets_kernel reset! ITERATION NUMBER IS : " << LocalMetaStore::GetInstance().curr_iter_num();
  DistributedCountService::GetInstance().ResetCounter(name_);
  StopTimer();
  return true;
}

REG_ROUND_KERNEL(shareSecrets, ShareSecretsKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
