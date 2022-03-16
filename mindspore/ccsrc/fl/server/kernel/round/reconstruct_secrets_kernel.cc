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

#include "fl/server/kernel/round/reconstruct_secrets_kernel.h"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <utility>

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
void ReconstructSecretsKernel::InitKernel(size_t required_cnt) {
  if (LocalMetaStore::GetInstance().has_value(kCtxTotalTimeoutDuration)) {
    iteration_time_window_ = LocalMetaStore::GetInstance().value<size_t>(kCtxTotalTimeoutDuration);
  }
  auto last_cnt_handler = [&](std::shared_ptr<ps::core::MessageHandler>) {
    if (ps::PSContext::instance()->resetter_round() == ps::ResetterRound::kReconstructSeccrets) {
      MS_LOG(INFO) << "start FinishIteration";
      FinishIteration(true);
      MS_LOG(INFO) << "end FinishIteration";
    }
    return;
  };
  auto first_cnt_handler = [&](std::shared_ptr<ps::core::MessageHandler>) { return; };
  name_unmask_ = "UnMaskKernel";
  MS_LOG(INFO) << "ReconstructSecretsKernel Init, ITERATION NUMBER IS : "
               << LocalMetaStore::GetInstance().curr_iter_num();
  DistributedCountService::GetInstance().RegisterCounter(name_unmask_, required_cnt,
                                                         {first_cnt_handler, last_cnt_handler});
}

sigVerifyResult ReconstructSecretsKernel::VerifySignature(const schema::SendReconstructSecret *reconstruct_secret_req) {
  MS_ERROR_IF_NULL_W_RET_VAL(reconstruct_secret_req, sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(reconstruct_secret_req->fl_id(), sigVerifyResult::FAILED);
  MS_ERROR_IF_NULL_W_RET_VAL(reconstruct_secret_req->timestamp(), sigVerifyResult::FAILED);

  std::string fl_id = reconstruct_secret_req->fl_id()->str();
  std::string timestamp = reconstruct_secret_req->timestamp()->str();
  int iteration = reconstruct_secret_req->iteration();
  std::string iter_str = std::to_string(iteration);
  auto fbs_signature = reconstruct_secret_req->signature();
  std::vector<unsigned char> signature;
  if (fbs_signature == nullptr) {
    MS_LOG(ERROR) << "signature in reconstruct_secret_req is nullptr";
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

bool ReconstructSecretsKernel::Launch(const uint8_t *req_data, size_t len,
                                      const std::shared_ptr<ps::core::MessageHandler> &message) {
  bool response = false;
  size_t iter_num = LocalMetaStore::GetInstance().curr_iter_num();
  MS_LOG(INFO) << "Launching ReconstructSecrets Kernel, Iteration number is " << iter_num;

  std::shared_ptr<server::FBBuilder> fbb = std::make_shared<server::FBBuilder>();
  if (fbb == nullptr || req_data == nullptr) {
    std::string reason = "FBBuilder builder or req_data is nullptr.";
    MS_LOG(ERROR) << reason;
    return false;
  }

  // get client list from memory server.
  std::vector<string> update_model_clients;
  const PBMetadata update_model_clients_pb_out =
    DistributedMetadataStore::GetInstance().GetMetadata(kCtxUpdateModelClientList);
  const UpdateModelClientList &update_model_clients_pb = update_model_clients_pb_out.client_list();

  for (size_t i = 0; i < IntToSize(update_model_clients_pb.fl_id_size()); ++i) {
    update_model_clients.push_back(update_model_clients_pb.fl_id(SizeToInt(i)));
  }
  flatbuffers::Verifier verifier(req_data, len);
  if (!verifier.VerifyBuffer<schema::SendReconstructSecret>()) {
    std::string reason = "The schema of SendReconstructSecret is invalid.";
    cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_RequestError, reason, SizeToInt(iter_num),
                                                   std::to_string(CURRENT_TIME_MILLI.count()));
    MS_LOG(ERROR) << reason;
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  const schema::SendReconstructSecret *reconstruct_secret_req =
    flatbuffers::GetRoot<schema::SendReconstructSecret>(req_data);
  if (reconstruct_secret_req == nullptr) {
    std::string reason = "Building flatbuffers schema failed for SendReconstructSecret.";
    cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_RequestError, reason, SizeToInt(iter_num),
                                                   std::to_string(CURRENT_TIME_MILLI.count()));
    MS_LOG(ERROR) << reason;
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }
  // verify signature
  if (ps::PSContext::instance()->pki_verify()) {
    sigVerifyResult verify_result = VerifySignature(reconstruct_secret_req);
    if (verify_result == sigVerifyResult::FAILED) {
      std::string reason = "verify signature failed.";
      cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_RequestError, reason,
                                                     SizeToInt(iter_num), std::to_string(CURRENT_TIME_MILLI.count()));
      MS_LOG(ERROR) << reason;
      GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }

    if (verify_result == sigVerifyResult::TIMEOUT) {
      std::string reason = "verify signature timestamp failed.";
      cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_OutOfTime, reason, SizeToInt(iter_num),
                                                     std::to_string(CURRENT_TIME_MILLI.count()));
      MS_LOG(ERROR) << reason;
      GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
      return true;
    }
    MS_LOG(INFO) << "verify signature passed!";
  }

  std::string fl_id = reconstruct_secret_req->fl_id()->str();
  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(ERROR) << "Current amount for ReconstructSecretsKernel is enough.";
    if (find(update_model_clients.begin(), update_model_clients.end(), fl_id) != update_model_clients.end()) {
      // client in get update model client list.
      cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_SUCCEED,
                                                     "Current amount for ReconstructSecretsKernel is enough.",
                                                     SizeToInt(iter_num), std::to_string(CURRENT_TIME_MILLI.count()));
    } else {
      cipher_reconstruct_.BuildReconstructSecretsRsp(fbb, schema::ResponseCode_OutOfTime,
                                                     "Current amount for ReconstructSecretsKernel is enough.",
                                                     SizeToInt(iter_num), std::to_string(CURRENT_TIME_MILLI.count()));
    }
    GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());
    return true;
  }

  response = cipher_reconstruct_.ReconstructSecrets(SizeToInt(iter_num), std::to_string(CURRENT_TIME_MILLI.count()),
                                                    reconstruct_secret_req, fbb, update_model_clients);
  if (response) {
    (void)DistributedCountService::GetInstance().Count(name_, reconstruct_secret_req->fl_id()->str());
  }
  if (DistributedCountService::GetInstance().CountReachThreshold(name_)) {
    MS_LOG(INFO) << "Current amount for ReconstructSecretsKernel is enough.";
  }
  GenerateOutput(message, fbb->GetBufferPointer(), fbb->GetSize());

  MS_LOG(INFO) << "reconstruct_secrets_kernel success.";
  if (!response) {
    MS_LOG(INFO) << "reconstruct_secrets_kernel response not ready.";
  }
  return true;
}

void ReconstructSecretsKernel::OnLastCountEvent(const std::shared_ptr<ps::core::MessageHandler> &) {
  MS_LOG(INFO) << "ITERATION NUMBER IS : " << LocalMetaStore::GetInstance().curr_iter_num();
  if (ps::PSContext::instance()->encrypt_type() == ps::kPWEncryptType) {
    int sleep_time = 5;
    while (!Executor::GetInstance().IsAllWeightAggregationDone()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    }
    MS_LOG(INFO) << "start unmask";
    while (!Executor::GetInstance().Unmask()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
    }
    MS_LOG(INFO) << "end unmask";
    Executor::GetInstance().set_unmasked(true);
    std::string worker_id = std::to_string(DistributedCountService::GetInstance().local_rank());
    (void)DistributedCountService::GetInstance().Count(name_unmask_, worker_id);
  }
}

bool ReconstructSecretsKernel::Reset() {
  MS_LOG(INFO) << "ITERATION NUMBER IS : " << LocalMetaStore::GetInstance().curr_iter_num();
  MS_LOG(INFO) << "reconstruct secrets kernel reset!";
  DistributedCountService::GetInstance().ResetCounter(name_);
  DistributedCountService::GetInstance().ResetCounter(name_unmask_);
  StopTimer();
  Executor::GetInstance().set_unmasked(false);
  cipher_reconstruct_.ClearReconstructSecrets();
  return true;
}

REG_ROUND_KERNEL(reconstructSecrets, ReconstructSecretsKernel)
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
