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
  cipher_share_ = &armour::CipherShares::GetInstance();
}

bool GetSecretsKernel::CountForGetSecrets(const std::shared_ptr<FBBuilder> &fbb,
                                          const schema::GetShareSecrets *get_secrets_req, const size_t iter_num) {
  MS_ERROR_IF_NULL_W_RET_VAL(get_secrets_req, false);
  if (!DistributedCountService::GetInstance().Count(name_, get_secrets_req->fl_id()->str())) {
    std::string reason = "Counting for get secrets kernel request failed. Please retry later.";
    cipher_share_->BuildGetSecretsRsp(fbb, schema::ResponseCode_OutOfTime, iter_num,
                                      std::to_string(CURRENT_TIME_MILLI.count()), nullptr);
    MS_LOG(ERROR) << reason;
    return false;
  }
  return true;
}

sigVerifyResult GetSecretsKernel::VerifySignature(const schema::GetShareSecrets *get_secrets_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(get_secrets_req, sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(get_secrets_req->fl_id(), sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(get_secrets_req->timestamp(), sigVerifyResult::FAILED);

  std::string fl_id = get_secrets_req->fl_id()->str();
  std::string timestamp = get_secrets_req->timestamp()->str();
  int iteration = get_secrets_req->iteration();
  std::string iter_str = std::to_string(iteration);
  auto fbs_signature = get_secrets_req->signature();
  std::vector<unsigned char> signature;
  if (fbs_signature == nullptr) {
    MS_LOG(ERROR) << "signature in get_secrets_req is nullptr";
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

bool GetSecretsKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                              const std::vector<AddressPtr> &outputs) {
  size_t iter_num = LocalMetaStore::GetInstance().curr_iter_num();
  std::string next_timestamp = std::to_string(CURRENT_TIME_MILLI.count());
  MS_LOG(INFO) << "Launching get secrets kernel, ITERATION NUMBER IS : " << iter_num;

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
  if (!verifier.VerifyBuffer<schema::GetShareSecrets>()) {
    std::string reason = "The schema of GetShareSecrets is invalid.";
    cipher_share_->BuildGetSecretsRsp(fbb, schema::ResponseCode_RequestError, iter_num, next_timestamp, nullptr);
    MS_LOG(ERROR) << reason;
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  const schema::GetShareSecrets *get_secrets_req = flatbuffers::GetRoot<schema::GetShareSecrets>(req_data);
  if (get_secrets_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for GetExchangeKeys.";
    cipher_share_->BuildGetSecretsRsp(fbb, schema::ResponseCode_RequestError, iter_num, next_timestamp, nullptr);
    MS_LOG(ERROR) << reason;
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  // verify signature
  if (ps::PSContext::instance()->pki_verify()) {
    sigVerifyResult verify_result = VerifySignature(get_secrets_req);
    if (verify_result == sigVerifyResult::FAILED) {
      std::string reason = "verify signature failed.";
      cipher_share_->BuildGetSecretsRsp(fbb, schema::ResponseCode_RequestError, iter_num, next_timestamp, nullptr);
      MS_LOG(ERROR) << reason;
      GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }

    if (verify_result == sigVerifyResult::TIMEOUT) {
      std::string reason = "verify signature timestamp failed.";
      cipher_share_->BuildGetSecretsRsp(fbb, schema::ResponseCode_OutOfTime, iter_num, next_timestamp, nullptr);
      MS_LOG(ERROR) << reason;
      GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }

    if (verify_result == sigVerifyResult::PASSED) {
      MS_LOG(INFO) << "verify signature passed!";
    }
  }
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

  bool response = cipher_share_->GetSecrets(get_secrets_req, fbb, next_timestamp);
  if (!response) {
    MS_LOG(WARNING) << "get secret shares not ready.";
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  if (!CountForGetSecrets(fbb, get_secrets_req, iter_num)) {
    GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  GenerateOutput(outputs, fbb->GetBufferPointer(), fbb->GetSize());
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
