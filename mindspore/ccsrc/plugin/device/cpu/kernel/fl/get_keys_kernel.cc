/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/fl/get_keys_kernel.h"

namespace mindspore {
namespace kernel {
bool GetKeysKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                              const std::vector<AddressPtr> &) {
  MS_LOG(INFO) << "Launching client GetKeysKernelMod";
  BuildGetKeysReq(fbb_);

  std::shared_ptr<std::vector<unsigned char>> get_keys_rsp_msg = nullptr;
  if (!fl::worker::FLWorker::GetInstance().SendToServer(target_server_rank_, fbb_->GetBufferPointer(), fbb_->GetSize(),
                                                        ps::core::TcpUserCommand::kGetKeys, &get_keys_rsp_msg)) {
    MS_LOG(EXCEPTION) << "Sending request for GetKeys to server " << target_server_rank_ << " failed.";
    return false;
  }
  if (get_keys_rsp_msg == nullptr) {
    MS_LOG(EXCEPTION) << "Received message pointer is nullptr.";
    return false;
  }
  flatbuffers::Verifier verifier(get_keys_rsp_msg->data(), get_keys_rsp_msg->size());
  if (!verifier.VerifyBuffer<schema::ReturnExchangeKeys>()) {
    MS_LOG(EXCEPTION) << "The schema of ResponseGetKeys is invalid.";
    return false;
  }

  const schema::ReturnExchangeKeys *get_keys_rsp =
    flatbuffers::GetRoot<schema::ReturnExchangeKeys>(get_keys_rsp_msg->data());
  MS_EXCEPTION_IF_NULL(get_keys_rsp);
  auto response_code = get_keys_rsp->retcode();
  if ((response_code != schema::ResponseCode_SUCCEED) && (response_code != schema::ResponseCode_OutOfTime)) {
    MS_LOG(EXCEPTION) << "Launching get keys job for worker failed. response_code: " << response_code;
  }

  bool save_keys_succeed = SavePublicKeyList(get_keys_rsp->remote_publickeys());
  if (!save_keys_succeed) {
    MS_LOG(EXCEPTION) << "Save received remote keys failed.";
    return false;
  }

  MS_LOG(INFO) << "Get keys successfully.";
  return true;
}

void GetKeysKernelMod::Init(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  if (cnode_ptr_.lock() == nullptr) {
    cnode_ptr_ = kernel_node;
  }

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

  MS_LOG(INFO) << "Initializing GetKeys kernel"
               << ", fl_id: " << fl_id_ << ". Request will be sent to server " << target_server_rank_;

  fbb_ = std::make_shared<fl::FBBuilder>();
  MS_EXCEPTION_IF_NULL(fbb_);
  input_size_list_.push_back(sizeof(int));
  output_size_list_.push_back(sizeof(float));
  MS_LOG(INFO) << "Initialize GetKeys kernel successfully.";
}

void GetKeysKernelMod::InitKernel(const CNodePtr &kernel_node) { return; }

void GetKeysKernelMod::BuildGetKeysReq(const std::shared_ptr<fl::FBBuilder> &fbb) {
  MS_EXCEPTION_IF_NULL(fbb);
  int iter = fl::worker::FLWorker::GetInstance().fl_iteration_num();
  auto fbs_fl_id = fbb->CreateString(fl_id_);
  schema::GetExchangeKeysBuilder get_keys_builder(*(fbb.get()));
  get_keys_builder.add_fl_id(fbs_fl_id);
  get_keys_builder.add_iteration(iter);
  auto req_fl_job = get_keys_builder.Finish();
  fbb->Finish(req_fl_job);
  MS_LOG(INFO) << "BuildGetKeysReq successfully.";
}

bool GetKeysKernelMod::SavePublicKeyList(
  const flatbuffers::Vector<flatbuffers::Offset<mindspore::schema::ClientPublicKeys>> *remote_public_key) {
  if (remote_public_key == nullptr) {
    MS_LOG(EXCEPTION) << "Input remote_pubic_key is nullptr.";
  }

  int client_num = remote_public_key->size();
  if (client_num <= 0) {
    MS_LOG(EXCEPTION) << "Received client keys length is <= 0, please check it!";
    return false;
  }

  // save client keys list
  std::vector<EncryptPublicKeys> saved_remote_public_keys;
  for (auto iter = remote_public_key->begin(); iter != remote_public_key->end(); ++iter) {
    std::string fl_id = iter->fl_id()->str();
    auto fbs_spk = iter->s_pk();
    auto fbs_pw_iv = iter->pw_iv();
    auto fbs_pw_salt = iter->pw_salt();
    if (fbs_spk == nullptr || fbs_pw_iv == nullptr || fbs_pw_salt == nullptr) {
      MS_LOG(WARNING) << "public key, pw_iv or pw_salt in remote_publickeys is nullptr.";
    } else {
      std::vector<uint8_t> spk_vector;
      std::vector<uint8_t> pw_iv_vector;
      std::vector<uint8_t> pw_salt_vector;
      spk_vector.assign(fbs_spk->begin(), fbs_spk->end());
      pw_iv_vector.assign(fbs_pw_iv->begin(), fbs_pw_iv->end());
      pw_salt_vector.assign(fbs_pw_salt->begin(), fbs_pw_salt->end());
      EncryptPublicKeys public_keys_i;
      public_keys_i.flID = fl_id;
      public_keys_i.publicKey = spk_vector;
      public_keys_i.pwIV = pw_iv_vector;
      public_keys_i.pwSalt = pw_salt_vector;
      saved_remote_public_keys.push_back(public_keys_i);
      MS_LOG(INFO) << "Add public keys of client:" << fl_id << " successfully.";
    }
  }
  fl::worker::FLWorker::GetInstance().set_public_keys_list(saved_remote_public_keys);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, GetKeys, GetKeysKernelMod);
}  // namespace kernel
}  // namespace mindspore
