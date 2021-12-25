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

#include "fl/armour/cipher/cipher_shares.h"
#include "fl/server/common.h"
#include "fl/armour/cipher/cipher_meta_storage.h"

namespace mindspore {
namespace armour {
bool CipherShares::ShareSecrets(const int cur_iterator, const schema::RequestShareSecrets *share_secrets_req,
                                const std::shared_ptr<fl::server::FBBuilder> &share_secrets_resp_builder,
                                const string next_req_time) {
  MS_LOG(INFO) << "CipherShares::ShareSecrets START";
  if (share_secrets_req == nullptr) {
    std::string reason = "Request is nullptr";
    MS_LOG(ERROR) << reason;
    BuildShareSecretsRsp(share_secrets_resp_builder, schema::ResponseCode_RequestError, reason, next_req_time,
                         cur_iterator);
    return false;
  }
  if (cipher_init_ == nullptr) {
    std::string reason = "cipher_init_ is nullptr";
    MS_LOG(ERROR) << reason;
    BuildShareSecretsRsp(share_secrets_resp_builder, schema::ResponseCode_SystemError, reason, next_req_time,
                         cur_iterator);
    return false;
  }
  // step 1: get client list and share secrets from memory server.
  clock_t start_time = clock();

  int iteration = share_secrets_req->iteration();
  std::vector<std::string> get_keys_clients;
  cipher_init_->cipher_meta_storage_.GetClientListFromServer(fl::server::kCtxGetKeysClientList, &get_keys_clients);
  std::vector<std::string> clients_share_list;
  cipher_init_->cipher_meta_storage_.GetClientListFromServer(fl::server::kCtxShareSecretsClientList,
                                                             &clients_share_list);

  std::map<std::string, std::vector<clientshare_str>> encrypted_shares_all;
  cipher_init_->cipher_meta_storage_.GetClientSharesFromServer(fl::server::kCtxClientsEncryptedShares,
                                                               &encrypted_shares_all);

  MS_LOG(INFO) << "Client of get keys size : " << get_keys_clients.size()
               << "client of update shares size : " << clients_share_list.size()
               << "updated shares size: " << encrypted_shares_all.size();

  // step 2: update new item to memory server. serialise: update pb struct to memory server.

  std::string fl_id_src = share_secrets_req->fl_id()->str();
  if (find(get_keys_clients.begin(), get_keys_clients.end(), fl_id_src) == get_keys_clients.end()) {
    // the client not in get keys clients
    BuildShareSecretsRsp(share_secrets_resp_builder, schema::ResponseCode_RequestError,
                         ("client share secret is not in getkeys list. && client is illegal"), next_req_time,
                         iteration);
    return false;
  }

  if (encrypted_shares_all.find(fl_id_src) != encrypted_shares_all.end()) {  // the client is already exists
    BuildShareSecretsRsp(share_secrets_resp_builder, schema::ResponseCode_SUCCEED,
                         ("client sharesecret already exists."), next_req_time, iteration);
    return false;
  }

  // update new item to memory server.
  const flatbuffers::Vector<flatbuffers::Offset<mindspore::schema::ClientShare>> *encrypted_shares =
    (share_secrets_req->encrypted_shares());
  bool retcode_client =
    cipher_init_->cipher_meta_storage_.UpdateClientToServer(fl::server::kCtxShareSecretsClientList, fl_id_src);
  bool retcode_share = cipher_init_->cipher_meta_storage_.UpdateClientShareToServer(
    fl::server::kCtxClientsEncryptedShares, fl_id_src, encrypted_shares);
  if (!(retcode_share && retcode_client)) {
    BuildShareSecretsRsp(share_secrets_resp_builder, schema::ResponseCode_OutOfTime,
                         "update client of shares and shares failed", next_req_time, iteration);
    MS_LOG(ERROR) << "CipherShares::ShareSecrets update client of shares and shares failed ";
    return false;
  }
  BuildShareSecretsRsp(share_secrets_resp_builder, schema::ResponseCode_SUCCEED, "OK", next_req_time, iteration);
  MS_LOG(INFO) << "CipherShares::ShareSecrets Success";
  clock_t end_time = clock();
  double duration = static_cast<double>((end_time - start_time) * 1.0 / CLOCKS_PER_SEC);
  MS_LOG(INFO) << "ShareSecrets get + deal + update data time is : " << duration;
  return true;
}

bool CipherShares::GetSecrets(const schema::GetShareSecrets *get_secrets_req,
                              const std::shared_ptr<fl::server::FBBuilder> &fbb, const std::string &next_req_time) {
  MS_LOG(INFO) << "CipherShares::GetSecrets START";
  clock_t start_time = clock();
  // step 0: check whether the parameters are legal.
  if (get_secrets_req == nullptr) {
    BuildGetSecretsRsp(fbb, schema::ResponseCode_RequestError, 0, next_req_time, nullptr);
    MS_LOG(ERROR) << "GetSecrets: get_secrets_req is nullptr.";
    return false;
  }
  int iteration = get_secrets_req->iteration();
  if (cipher_init_ == nullptr) {
    MS_LOG(ERROR) << "cipher_init_ is nullptr";
    BuildGetSecretsRsp(fbb, schema::ResponseCode_SystemError, IntToSize(iteration), next_req_time, nullptr);
    return false;
  }
  // step 1: get client list and client shares list from memory server.
  std::map<std::string, std::vector<clientshare_str>> encrypted_shares_all;
  cipher_init_->cipher_meta_storage_.GetClientSharesFromServer(fl::server::kCtxClientsEncryptedShares,
                                                               &encrypted_shares_all);
  size_t encrypted_shares_num = encrypted_shares_all.size();
  if (cipher_init_->share_secrets_threshold > encrypted_shares_num) {  // the client num is not enough, return false.
    BuildGetSecretsRsp(fbb, schema::ResponseCode_SucNotReady, IntToSize(iteration), next_req_time, nullptr);
    MS_LOG(INFO) << "GetSecrets: the encrypted shares num is not enough: share_secrets_threshold: "
                 << cipher_init_->share_secrets_threshold << "encrypted_shares_num: " << encrypted_shares_num;
    return false;
  }

  std::string fl_id = get_secrets_req->fl_id()->str();
  // the client is not in share secrets client list.
  if (encrypted_shares_all.find(fl_id) == encrypted_shares_all.end()) {
    BuildGetSecretsRsp(fbb, schema::ResponseCode_RequestError, IntToSize(iteration), next_req_time, nullptr);
    MS_LOG(ERROR) << "GetSecrets: client is not in share secrets client list.";
    return false;
  }

  bool retcode_client =
    cipher_init_->cipher_meta_storage_.UpdateClientToServer(fl::server::kCtxGetSecretsClientList, fl_id);
  if (!retcode_client) {
    MS_LOG(ERROR) << "update get secrets clients failed";
    BuildGetSecretsRsp(fbb, schema::ResponseCode_SucNotReady, IntToSize(iteration), next_req_time, nullptr);
    return false;
  }

  // get the result client shares.
  std::vector<clientshare_str> encrypted_shares_add;
  for (auto encrypted_shares_iterator = encrypted_shares_all.begin();
       encrypted_shares_iterator != encrypted_shares_all.end(); ++encrypted_shares_iterator) {
    std::string fl_id_src_now = encrypted_shares_iterator->first;
    std::vector<clientshare_str> &clientshare_str_now = encrypted_shares_iterator->second;
    clientshare_str client_share_str_new;
    bool find_flag = false;
    for (size_t index_clientshare = 0; index_clientshare < clientshare_str_now.size(); ++index_clientshare) {
      std::string fl_id_des = clientshare_str_now[index_clientshare].fl_id;
      if (fl_id_des == fl_id) {
        client_share_str_new.fl_id = fl_id_src_now;
        client_share_str_new.index = clientshare_str_now[index_clientshare].index;
        client_share_str_new.share = clientshare_str_now[index_clientshare].share;
        find_flag = true;
        break;
      }
    }
    if (find_flag) {
      encrypted_shares_add.push_back(client_share_str_new);
    }
  }

  // serialise clientshares
  size_t size_shares = encrypted_shares_add.size();
  std::vector<flatbuffers::Offset<mindspore::schema::ClientShare>> encrypted_shares;
  std::vector<clientshare_str>::iterator ptr_start = encrypted_shares_add.begin();
  std::vector<clientshare_str>::iterator ptr_end = ptr_start + size_shares;
  for (std::vector<clientshare_str>::iterator ptr = ptr_start; ptr < ptr_end; ++ptr) {
    auto one_fl_id = fbb->CreateString(ptr->fl_id);
    auto two_share = fbb->CreateVector(ptr->share.data(), ptr->share.size());
    auto third_index = ptr->index;
    auto one_clientshare = schema::CreateClientShare(*fbb, one_fl_id, two_share, third_index);
    encrypted_shares.push_back(one_clientshare);
  }

  BuildGetSecretsRsp(fbb, schema::ResponseCode_SUCCEED, IntToSize(iteration), next_req_time, &encrypted_shares);
  MS_LOG(INFO) << "CipherShares::GetSecrets Success";
  clock_t end_time = clock();
  double duration = static_cast<double>((end_time - start_time) * 1.0 / CLOCKS_PER_SEC);
  MS_LOG(INFO) << "Getsecrets Duration Time is : " << duration;
  return true;
}

void CipherShares::BuildGetSecretsRsp(
  const std::shared_ptr<fl::server::FBBuilder> &fbb, const schema::ResponseCode retcode, size_t iteration,
  const std::string &next_req_time,
  const std::vector<flatbuffers::Offset<mindspore::schema::ClientShare>> *encrypted_shares) {
  int rsp_retcode = retcode;
  int rsp_iteration = SizeToInt(iteration);
  auto rsp_next_req_time = fbb->CreateString(next_req_time);
  if (encrypted_shares == nullptr) {
    auto get_secrets_rsp = schema::CreateReturnShareSecrets(*fbb, rsp_retcode, rsp_iteration, 0, rsp_next_req_time);
    fbb->Finish(get_secrets_rsp);
  } else {
    auto encrypted_shares_rsp = fbb->CreateVector(*encrypted_shares);
    auto get_secrets_rsp =
      CreateReturnShareSecrets(*fbb, rsp_retcode, rsp_iteration, encrypted_shares_rsp, rsp_next_req_time);
    fbb->Finish(get_secrets_rsp);
  }
  return;
}

void CipherShares::BuildShareSecretsRsp(const std::shared_ptr<fl::server::FBBuilder> &share_secrets_resp_builder,
                                        const schema::ResponseCode retcode, const string &reason,
                                        const string &next_req_time, const int iteration) {
  auto rsp_reason = share_secrets_resp_builder->CreateString(reason);
  auto rsp_next_req_time = share_secrets_resp_builder->CreateString(next_req_time);
  auto share_secrets_rsp =
    schema::CreateResponseShareSecrets(*share_secrets_resp_builder, retcode, rsp_reason, rsp_next_req_time, iteration);
  share_secrets_resp_builder->Finish(share_secrets_rsp);
  return;
}

void CipherShares::ClearShareSecrets() {
  fl::server::DistributedMetadataStore::GetInstance().ResetMetadata(fl::server::kCtxShareSecretsClientList);
  fl::server::DistributedMetadataStore::GetInstance().ResetMetadata(fl::server::kCtxClientsEncryptedShares);
  fl::server::DistributedMetadataStore::GetInstance().ResetMetadata(fl::server::kCtxGetSecretsClientList);
}
}  // namespace armour
}  // namespace mindspore
