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

#include "armour/cipher/cipher_shares.h"
#include "fl/server/common.h"
#include "armour/cipher/cipher_meta_storage.h"

namespace mindspore {
namespace armour {
bool CipherShares::ShareSecrets(const int cur_iterator, const schema::RequestShareSecrets *share_secrets_req,
                                std::shared_ptr<ps::server::FBBuilder> share_secrets_resp_builder,
                                const string next_req_time) {
  MS_LOG(INFO) << "CipherShares::ShareSecrets START";
  if (share_secrets_req == nullptr) {
    MS_LOG(ERROR) << "Request is nullptr or Response builder is nullptr.";
    std::string reason = "Request is nullptr or Response builder is nullptr.";
    BuildShareSecretsRsp(share_secrets_resp_builder, schema::ResponseCode_RequestError, reason, next_req_time,
                         cur_iterator);
    return false;
  }

  // step 1: get client list and share secrets from memory server.
  clock_t start_time = clock();
  std::vector<std::string> clients_share_list;
  cipher_init_->cipher_meta_storage_.GetClientListFromServer(ps::server::kCtxShareSecretsClientList,
                                                             &clients_share_list);
  std::vector<std::string> clients_exchange_list;
  cipher_init_->cipher_meta_storage_.GetClientListFromServer(ps::server::kCtxExChangeKeysClientList,
                                                             &clients_exchange_list);
  std::map<std::string, std::vector<clientshare_str>> encrypted_shares_all;
  cipher_init_->cipher_meta_storage_.GetClientSharesFromServer(ps::server::kCtxClientsEncryptedShares,
                                                               &encrypted_shares_all);

  MS_LOG(INFO) << "Client of keys size : " << clients_exchange_list.size()
               << "client of shares size : " << clients_share_list.size() << "shares size"
               << encrypted_shares_all.size();
  if (encrypted_shares_all.size() != clients_share_list.size()) {
    BuildShareSecretsRsp(share_secrets_resp_builder, schema::ResponseCode_OutOfTime,
                         "client of shares and shares size are not equal", next_req_time, cur_iterator);
    MS_LOG(ERROR) << "client of shares and shares size are not equal. client of shares size : "
                  << clients_share_list.size() << "shares size" << encrypted_shares_all.size();
  }

  // step 2: update new item to memory server. serialise: update pb struct to memory server.
  int iteration = share_secrets_req->iteration();
  std::string fl_id_src = share_secrets_req->fl_id()->str();
  if (find(clients_exchange_list.begin(), clients_exchange_list.end(), fl_id_src) ==
      clients_exchange_list.end()) {  // the client not in clients_exchange_list, return false.
    BuildShareSecretsRsp(share_secrets_resp_builder, schema::ResponseCode_RequestError,
                         ("client share secret is not in clients_exchange list. && client is illegal"), next_req_time,
                         iteration);
    return false;
  }
  if (find(clients_share_list.begin(), clients_share_list.end(), fl_id_src) !=
      clients_share_list.end()) {  // the client is already exists, return false.
    BuildShareSecretsRsp(share_secrets_resp_builder, schema::ResponseCode_SUCCEED,
                         ("client sharesecret already exists."), next_req_time, iteration);
    return false;
  }

  // update new item to memory server.
  const flatbuffers::Vector<flatbuffers::Offset<mindspore::schema::ClientShare>> *encrypted_shares =
    (share_secrets_req->encrypted_shares());
  bool retcode_share = cipher_init_->cipher_meta_storage_.UpdateClientShareToServer(
    ps::server::kCtxClientsEncryptedShares, fl_id_src, encrypted_shares);
  bool retcode_client =
    cipher_init_->cipher_meta_storage_.UpdateClientToServer(ps::server::kCtxShareSecretsClientList, fl_id_src);
  bool retcode = retcode_share && retcode_client;
  if (retcode) {
    BuildShareSecretsRsp(share_secrets_resp_builder, schema::ResponseCode_SUCCEED, "OK", next_req_time, iteration);
    MS_LOG(INFO) << "CipherShares::ShareSecrets Success";
  } else {
    BuildShareSecretsRsp(share_secrets_resp_builder, schema::ResponseCode_OutOfTime,
                         "update client of shares and shares failed", next_req_time, iteration);
    MS_LOG(ERROR) << "CipherShares::ShareSecrets update client of shares and shares failed ";
  }

  clock_t end_time = clock();
  double duration = static_cast<double>((end_time - start_time) * 1.0 / CLOCKS_PER_SEC);
  MS_LOG(INFO) << "ShareSecrets get + deal + update data time is : " << duration;
  return retcode;
}

bool CipherShares::GetSecrets(const schema::GetShareSecrets *get_secrets_req,
                              std::shared_ptr<ps::server::FBBuilder> get_secrets_resp_builder,
                              const std::string &next_req_time) {
  MS_LOG(INFO) << "CipherShares::GetSecrets START";
  clock_t start_time = clock();
  // step 0: check whether the parameters are legal.
  if (get_secrets_req == nullptr) {
    BuildGetSecretsRsp(get_secrets_resp_builder, schema::ResponseCode_SystemError, 0, next_req_time, 0);
    MS_LOG(ERROR) << "GetSecrets: get_secrets_req is nullptr.";
    return false;
  }

  // step 1: get client list and client shares list from memory server.
  std::vector<std::string> clients_share_list;
  cipher_init_->cipher_meta_storage_.GetClientListFromServer(ps::server::kCtxShareSecretsClientList,
                                                             &clients_share_list);
  std::map<std::string, std::vector<clientshare_str>> encrypted_shares_all;
  cipher_init_->cipher_meta_storage_.GetClientSharesFromServer(ps::server::kCtxClientsEncryptedShares,
                                                               &encrypted_shares_all);
  int iteration = get_secrets_req->iteration();
  size_t share_clients_num = clients_share_list.size();
  size_t cients_has_shares = encrypted_shares_all.size();
  if (share_clients_num != cients_has_shares) {
    BuildGetSecretsRsp(get_secrets_resp_builder, schema::ResponseCode_OutOfTime, iteration, next_req_time, 0);
    MS_LOG(ERROR) << "cients_has_shares: " << cients_has_shares << "share_clients_num: " << share_clients_num;
  }
  if (cipher_init_->share_clients_num_need_ > share_clients_num) {  // the client num is not enough, return false.
    BuildGetSecretsRsp(get_secrets_resp_builder, schema::ResponseCode_SucNotReady, iteration, next_req_time, 0);
    MS_LOG(INFO) << "GetSecrets: the client num is not enough: share_clients_num_need_: "
                 << cipher_init_->share_clients_num_need_ << "share_clients_num: " << share_clients_num;
    return false;
  }
  std::string fl_id = get_secrets_req->fl_id()->str();
  if (find(clients_share_list.begin(), clients_share_list.end(), fl_id) ==
      clients_share_list.end()) {  // the client is not in client list, return false.
    BuildGetSecretsRsp(get_secrets_resp_builder, schema::ResponseCode_RequestError, iteration, next_req_time, 0);
    MS_LOG(ERROR) << "GetSecrets: client is not in client list.";
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
    auto one_fl_id = get_secrets_resp_builder->CreateString(ptr->fl_id);
    auto two_share = get_secrets_resp_builder->CreateVector(ptr->share.data(), ptr->share.size());
    auto third_index = ptr->index;
    auto one_clientshare = schema::CreateClientShare(*get_secrets_resp_builder, one_fl_id, two_share, third_index);
    encrypted_shares.push_back(one_clientshare);
  }

  BuildGetSecretsRsp(get_secrets_resp_builder, schema::ResponseCode_SUCCEED, iteration, next_req_time,
                     &encrypted_shares);

  MS_LOG(INFO) << "CipherShares::GetSecrets Success";
  clock_t end_time = clock();
  double duration = static_cast<double>((end_time - start_time) * 1.0 / CLOCKS_PER_SEC);
  MS_LOG(INFO) << "Getsecrets Duration Time is : " << duration;
  return true;
}

void CipherShares::BuildGetSecretsRsp(
  std::shared_ptr<ps::server::FBBuilder> get_secrets_resp_builder, schema::ResponseCode retcode, int iteration,
  std::string next_req_time, std::vector<flatbuffers::Offset<mindspore::schema::ClientShare>> *encrypted_shares) {
  int rsp_retcode = retcode;
  int rsp_iteration = iteration;
  auto rsp_next_req_time = get_secrets_resp_builder->CreateString(next_req_time);
  if (encrypted_shares == 0) {
    auto get_secrets_rsp =
      schema::CreateReturnShareSecrets(*get_secrets_resp_builder, rsp_retcode, rsp_iteration, 0, rsp_next_req_time);
    get_secrets_resp_builder->Finish(get_secrets_rsp);
  } else {
    auto encrypted_shares_rsp = get_secrets_resp_builder->CreateVector(*encrypted_shares);
    auto get_secrets_rsp = CreateReturnShareSecrets(*get_secrets_resp_builder, rsp_retcode, rsp_iteration,
                                                    encrypted_shares_rsp, rsp_next_req_time);
    get_secrets_resp_builder->Finish(get_secrets_rsp);
  }

  return;
}

void CipherShares::BuildShareSecretsRsp(std::shared_ptr<ps::server::FBBuilder> share_secrets_resp_builder,
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
  ps::server::DistributedMetadataStore::GetInstance().ResetMetadata(ps::server::kCtxShareSecretsClientList);
  ps::server::DistributedMetadataStore::GetInstance().ResetMetadata(ps::server::kCtxClientsEncryptedShares);
}

}  // namespace armour
}  // namespace mindspore
