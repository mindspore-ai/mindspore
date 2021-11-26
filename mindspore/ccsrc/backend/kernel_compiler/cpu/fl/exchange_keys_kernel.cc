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

#include "backend/kernel_compiler/cpu/fl/exchange_keys_kernel.h"

namespace mindspore {
namespace kernel {
constexpr int iv_vec_len = 16;
constexpr int salt_len = 32;

bool ExchangeKeysKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                const std::vector<AddressPtr> &) {
  MS_LOG(INFO) << "Launching client ExchangeKeysKernel";
  if (!BuildExchangeKeysReq(fbb_)) {
    MS_LOG(EXCEPTION) << "Building request for ExchangeKeys failed.";
    return false;
  }

  std::shared_ptr<std::vector<unsigned char>> exchange_keys_rsp_msg = nullptr;
  if (!fl::worker::FLWorker::GetInstance().SendToServer(target_server_rank_, fbb_->GetBufferPointer(), fbb_->GetSize(),
                                                        ps::core::TcpUserCommand::kExchangeKeys,
                                                        &exchange_keys_rsp_msg)) {
    MS_LOG(EXCEPTION) << "Sending request for ExchangeKeys to server " << target_server_rank_ << " failed.";
    return false;
  }
  if (exchange_keys_rsp_msg == nullptr) {
    MS_LOG(EXCEPTION) << "Received message pointer is nullptr.";
    return false;
  }
  flatbuffers::Verifier verifier(exchange_keys_rsp_msg->data(), exchange_keys_rsp_msg->size());
  if (!verifier.VerifyBuffer<schema::ResponseExchangeKeys>()) {
    MS_LOG(EXCEPTION) << "The schema of ResponseExchangeKeys is invalid.";
    return false;
  }

  const schema::ResponseExchangeKeys *exchange_keys_rsp =
    flatbuffers::GetRoot<schema::ResponseExchangeKeys>(exchange_keys_rsp_msg->data());
  MS_EXCEPTION_IF_NULL(exchange_keys_rsp);
  auto response_code = exchange_keys_rsp->retcode();
  if ((response_code != schema::ResponseCode_SUCCEED) && (response_code != schema::ResponseCode_OutOfTime)) {
    MS_LOG(EXCEPTION) << "Launching exchange keys job for worker failed. Reason: " << exchange_keys_rsp->reason();
  }

  MS_LOG(INFO) << "Exchange keys successfully.";
  return true;
}

void ExchangeKeysKernel::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  fl_id_ = fl::worker::FLWorker::GetInstance().fl_id();
  server_num_ = fl::worker::FLWorker::GetInstance().server_num();
  rank_id_ = fl::worker::FLWorker::GetInstance().rank_id();
  if (rank_id_ == UINT32_MAX) {
    MS_LOG(EXCEPTION) << "Federated worker is not initialized yet.";
    return;
  }

  if (server_num_ <= 0) {
    MS_LOG(EXCEPTION) << "Server number should be larger than 0, but got: " << server_num_;
    return;
  }
  target_server_rank_ = rank_id_ % server_num_;

  MS_LOG(INFO) << "Initializing ExchangeKeys kernel"
               << ", fl_id: " << fl_id_ << ". Request will be sent to server " << target_server_rank_;

  fbb_ = std::make_shared<fl::FBBuilder>();
  MS_EXCEPTION_IF_NULL(fbb_);
  input_size_list_.push_back(sizeof(int));
  output_size_list_.push_back(sizeof(float));
  MS_LOG(INFO) << "Initialize ExchangeKeys kernel successfully.";
}

void ExchangeKeysKernel::InitKernel(const CNodePtr &kernel_node) { return; }

bool ExchangeKeysKernel::BuildExchangeKeysReq(const std::shared_ptr<fl::FBBuilder> &fbb) {
  MS_EXCEPTION_IF_NULL(fbb);
  // generate initialization vector value used for generate pairwise noise
  std::vector<uint8_t> pw_iv_(iv_vec_len);
  std::vector<uint8_t> pw_salt_(salt_len);
  auto ret = RAND_bytes(pw_iv_.data(), iv_vec_len);
  if (ret != 1) {
    MS_LOG(ERROR) << "RAND_bytes error, failed to init pw_iv_.";
    return false;
  }
  // generate salt value used for generate pairwise noise
  ret = RAND_bytes(pw_salt_.data(), salt_len);
  if (ret != 1) {
    MS_LOG(ERROR) << "RAND_bytes error, failed to init pw_salt_.";
    return false;
  }

  // save pw_salt and pw_iv at local
  fl::worker::FLWorker::GetInstance().set_pw_salt(pw_salt_);
  fl::worker::FLWorker::GetInstance().set_pw_iv(pw_iv_);

  // get public key bytes
  std::vector<uint8_t> pubkey_bytes = GetPubicKeyBytes();
  if (pubkey_bytes.size() == 0) {
    MS_LOG(EXCEPTION) << "Get public key failed.";
    return false;
  }

  // build data which will be send to server
  int iter = fl::worker::FLWorker::GetInstance().fl_iteration_num();
  auto fbs_fl_id = fbb->CreateString(fl_id_);
  auto fbs_public_key = fbb->CreateVector(pubkey_bytes.data(), pubkey_bytes.size());
  auto fbs_pw_iv = fbb->CreateVector(pw_iv_.data(), iv_vec_len);
  auto fbs_pw_salt = fbb->CreateVector(pw_salt_.data(), salt_len);
  schema::RequestExchangeKeysBuilder req_exchange_key_builder(*(fbb.get()));
  req_exchange_key_builder.add_fl_id(fbs_fl_id);
  req_exchange_key_builder.add_s_pk(fbs_public_key);
  req_exchange_key_builder.add_iteration(iter);
  req_exchange_key_builder.add_pw_iv(fbs_pw_iv);
  req_exchange_key_builder.add_pw_salt(fbs_pw_salt);
  auto req_fl_job = req_exchange_key_builder.Finish();
  fbb->Finish(req_fl_job);
  MS_LOG(INFO) << "BuildExchangeKeysReq successfully.";
  return true;
}

std::vector<uint8_t> ExchangeKeysKernel::GetPubicKeyBytes() {
  // generate private key of secret
  armour::PrivateKey *sPriKeyPtr = armour::KeyAgreement::GeneratePrivKey();
  fl::worker::FLWorker::GetInstance().set_secret_pk(sPriKeyPtr);

  // get public bytes length
  size_t pubLen;
  uint8_t *secret_pubkey_ptr = NULL;
  auto ret = sPriKeyPtr->GetPublicBytes(&pubLen, secret_pubkey_ptr);
  if (ret != 0 || pubLen == 0) {
    MS_LOG(ERROR) << "GetPublicBytes error, failed to get public_key bytes length.";
    return {};
  }
  // pubLen has been updated, now get public_key bytes
  secret_pubkey_ptr = reinterpret_cast<uint8_t *>(malloc(pubLen));
  if (secret_pubkey_ptr == nullptr) {
    MS_LOG(ERROR) << "secret_pubkey_ptr is nullptr, malloc failed.";
    return {};
  }
  ret = sPriKeyPtr->GetPublicBytes(&pubLen, secret_pubkey_ptr);
  if (ret != 0) {
    free(secret_pubkey_ptr);
    MS_LOG(ERROR) << "GetPublicBytes error, failed to get public_key bytes.";
    return {};
  }

  // transform key buffer to uint8_t vector
  std::vector<uint8_t> pubkey_bytes(pubLen);
  for (int i = 0; i < SizeToInt(pubLen); i++) {
    pubkey_bytes[i] = secret_pubkey_ptr[i];
  }
  free(secret_pubkey_ptr);
  return pubkey_bytes;
}

MS_REG_CPU_KERNEL(ExchangeKeys, KernelAttr().AddOutputAttr(kNumberTypeFloat32), ExchangeKeysKernel);
}  // namespace kernel
}  // namespace mindspore
