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

#include "fl/armour/cipher/cipher_keys.h"
#include "fl/armour/cipher/cipher_meta_storage.h"

namespace mindspore {
namespace armour {
bool CipherKeys::GetKeys(const int cur_iterator, const std::string &next_req_time,
                         const schema::GetExchangeKeys *get_exchange_keys_req,
                         const std::shared_ptr<fl::server::FBBuilder> &get_exchange_keys_resp_builder) {
  MS_LOG(INFO) << "CipherMgr::GetKeys START";
  if (get_exchange_keys_req == nullptr || get_exchange_keys_resp_builder == nullptr) {
    MS_LOG(ERROR) << "Request is nullptr or Response builder is nullptr.";
    BuildGetKeys(get_exchange_keys_resp_builder, schema::ResponseCode_SystemError, cur_iterator, next_req_time, false);
    return false;
  }

  // get clientlist from memory server.
  std::vector<std::string> clients;

  cipher_init_->cipher_meta_storage_.GetClientListFromServer(fl::server::kCtxExChangeKeysClientList, &clients);

  size_t cur_clients_num = clients.size();
  std::string fl_id = get_exchange_keys_req->fl_id()->str();

  if (find(clients.begin(), clients.end(), fl_id) == clients.end()) {
    MS_LOG(INFO) << "The fl_id is not in clients.";
    BuildGetKeys(get_exchange_keys_resp_builder, schema::ResponseCode_RequestError, cur_iterator, next_req_time, false);
    return false;
  }
  if (cur_clients_num < cipher_init_->client_num_need_) {
    MS_LOG(INFO) << "The server is not ready yet: cur_clients_num < client_num_need";
    MS_LOG(INFO) << "cur_clients_num : " << cur_clients_num << ", client_num_need : " << cipher_init_->client_num_need_;
    BuildGetKeys(get_exchange_keys_resp_builder, schema::ResponseCode_SucNotReady, cur_iterator, next_req_time, false);
    return false;
  }

  MS_LOG(INFO) << "GetKeys client list: ";
  for (size_t i = 0; i < clients.size(); i++) {
    MS_LOG(INFO) << "fl_id: " << clients[i];
  }

  bool flag =
    BuildGetKeys(get_exchange_keys_resp_builder, schema::ResponseCode_SUCCEED, cur_iterator, next_req_time, true);
  return flag;
}  // namespace armour

bool CipherKeys::ExchangeKeys(const int cur_iterator, const std::string &next_req_time,
                              const schema::RequestExchangeKeys *exchange_keys_req,
                              const std::shared_ptr<fl::server::FBBuilder> &exchange_keys_resp_builder) {
  MS_LOG(INFO) << "CipherMgr::ExchangeKeys START";
  // step 0: judge if the input param is legal.
  if (exchange_keys_req == nullptr || exchange_keys_resp_builder == nullptr) {
    MS_LOG(ERROR) << "Request is nullptr or Response builder is nullptr.";
    std::string reason = "Request is nullptr or Response builder is nullptr.";
    BuildExchangeKeysRsp(exchange_keys_resp_builder, schema::ResponseCode_RequestError, reason, next_req_time,
                         cur_iterator);
    return false;
  }

  // step 1: get clientlist and client keys from memory server.
  std::map<std::string, std::vector<std::vector<unsigned char>>> record_public_keys;
  std::vector<std::string> client_list;
  cipher_init_->cipher_meta_storage_.GetClientListFromServer(fl::server::kCtxExChangeKeysClientList, &client_list);
  cipher_init_->cipher_meta_storage_.GetClientKeysFromServer(fl::server::kCtxClientsKeys, &record_public_keys);

  // step2: process new item data. and update new item data to memory server.
  size_t cur_clients_num = client_list.size();
  size_t cur_clients_has_keys_num = record_public_keys.size();
  if (cur_clients_num != cur_clients_has_keys_num) {
    std::string reason = "client num and keys num are not equal.";
    MS_LOG(ERROR) << reason;
    MS_LOG(ERROR) << "cur_clients_num is " << cur_clients_num << ". cur_clients_has_keys_num is "
                  << cur_clients_has_keys_num;
    BuildExchangeKeysRsp(exchange_keys_resp_builder, schema::ResponseCode_OutOfTime, reason, next_req_time,
                         cur_iterator);

    return false;
  }

  MS_LOG(INFO) << "client_num_need_ " << cipher_init_->client_num_need_ << ". cur_clients_num " << cur_clients_num;
  std::string fl_id = exchange_keys_req->fl_id()->str();
  if (cur_clients_num >= cipher_init_->client_num_need_) {  // the client num is enough, return false.
    MS_LOG(ERROR) << "The server has received enough requests and refuse this request.";
    BuildExchangeKeysRsp(exchange_keys_resp_builder, schema::ResponseCode_OutOfTime,
                         "The server has received enough requests and refuse this request.", next_req_time,
                         cur_iterator);
    return false;
  }
  if (record_public_keys.find(fl_id) != record_public_keys.end()) {  // the client already exists, return false.
    MS_LOG(INFO) << "The server has received the request, please do not request again.";
    BuildExchangeKeysRsp(exchange_keys_resp_builder, schema::ResponseCode_SUCCEED,
                         "The server has received the request, please do not request again.", next_req_time,
                         cur_iterator);
    return false;
  }

  // Gets the members of the deserialized data ： exchange_keys_req
  auto fbs_cpk = exchange_keys_req->c_pk();
  size_t cpk_len = fbs_cpk->size();
  auto fbs_spk = exchange_keys_req->s_pk();
  size_t spk_len = fbs_spk->size();

  // transform fbs (fbs_cpk & fbs_spk) to a vector: public_key
  std::vector<std::vector<unsigned char>> cur_public_key;
  std::vector<unsigned char> cpk(cpk_len);
  std::vector<unsigned char> spk(spk_len);
  bool ret_create_code_cpk = CreateArray<unsigned char>(&cpk, *fbs_cpk);
  bool ret_create_code_spk = CreateArray<unsigned char>(&spk, *fbs_spk);
  if (!(ret_create_code_cpk && ret_create_code_spk)) {
    MS_LOG(ERROR) << "create cur_public_key failed";
    BuildExchangeKeysRsp(exchange_keys_resp_builder, schema::ResponseCode_OutOfTime, "update key or client failed",
                         next_req_time, cur_iterator);
    return false;
  }
  cur_public_key.push_back(cpk);
  cur_public_key.push_back(spk);

  bool retcode_key =
    cipher_init_->cipher_meta_storage_.UpdateClientKeyToServer(fl::server::kCtxClientsKeys, fl_id, cur_public_key);
  bool retcode_client =
    cipher_init_->cipher_meta_storage_.UpdateClientToServer(fl::server::kCtxExChangeKeysClientList, fl_id);
  if (retcode_key && retcode_client) {
    MS_LOG(INFO) << "The client " << fl_id << " CipherMgr::ExchangeKeys Success";
    BuildExchangeKeysRsp(exchange_keys_resp_builder, schema::ResponseCode_SUCCEED,
                         "Success, but the server is not ready yet.", next_req_time, cur_iterator);
    return true;
  } else {
    MS_LOG(ERROR) << "update key or client failed";
    BuildExchangeKeysRsp(exchange_keys_resp_builder, schema::ResponseCode_OutOfTime, "update key or client failed",
                         next_req_time, cur_iterator);
    return false;
  }
}

void CipherKeys::BuildExchangeKeysRsp(const std::shared_ptr<fl::server::FBBuilder> &exchange_keys_resp_builder,
                                      const schema::ResponseCode retcode, const std::string &reason,
                                      const std::string &next_req_time, const int iteration) {
  auto rsp_reason = exchange_keys_resp_builder->CreateString(reason);
  auto rsp_next_req_time = exchange_keys_resp_builder->CreateString(next_req_time);
  schema::ResponseExchangeKeysBuilder rsp_builder(*(exchange_keys_resp_builder.get()));
  rsp_builder.add_retcode(retcode);
  rsp_builder.add_reason(rsp_reason);
  rsp_builder.add_next_req_time(rsp_next_req_time);
  rsp_builder.add_iteration(iteration);
  auto rsp_exchange_keys = rsp_builder.Finish();
  exchange_keys_resp_builder->Finish(rsp_exchange_keys);
  return;
}

bool CipherKeys::BuildGetKeys(const std::shared_ptr<fl::server::FBBuilder> &fbb, const schema::ResponseCode retcode,
                              const int iteration, const std::string &next_req_time, bool is_good) {
  bool flag = true;
  if (is_good) {
    // convert client keys to standard keys list.
    std::vector<flatbuffers::Offset<schema::ClientPublicKeys>> public_keys_list;
    MS_LOG(INFO) << "Get Keys: ";
    std::map<std::string, std::vector<std::vector<unsigned char>>> record_public_keys;
    cipher_init_->cipher_meta_storage_.GetClientKeysFromServer(fl::server::kCtxClientsKeys, &record_public_keys);
    if (record_public_keys.size() < cipher_init_->client_num_need_) {
      MS_LOG(INFO) << "NOT READY. keys num: " << record_public_keys.size()
                   << "clients num: " << cipher_init_->client_num_need_;
      flag = false;
      auto fbs_next_req_time = fbb->CreateString(next_req_time);
      schema::ReturnExchangeKeysBuilder rsp_buider(*(fbb.get()));
      rsp_buider.add_retcode(retcode);
      rsp_buider.add_iteration(iteration);
      rsp_buider.add_next_req_time(fbs_next_req_time);
      auto rsp_get_keys = rsp_buider.Finish();

      fbb->Finish(rsp_get_keys);
    } else {
      for (auto iter = record_public_keys.begin(); iter != record_public_keys.end(); ++iter) {
        // read (fl_id, c_pk, s_pk) from the map: record_public_keys_
        std::string fl_id = iter->first;
        MS_LOG(INFO) << "fl_id : " << fl_id;

        //   To serialize the members to a new Table：ClientPublicKeys
        auto fbs_fl_id = fbb->CreateString(fl_id);
        auto fbs_c_pk = fbb->CreateVector(iter->second[0].data(), iter->second[0].size());
        auto fbs_s_pk = fbb->CreateVector(iter->second[1].data(), iter->second[1].size());
        auto cur_public_key = schema::CreateClientPublicKeys(*fbb, fbs_fl_id, fbs_c_pk, fbs_s_pk);
        public_keys_list.push_back(cur_public_key);
      }
      auto remote_publickeys = fbb->CreateVector(public_keys_list);
      auto fbs_next_req_time = fbb->CreateString(next_req_time);
      schema::ReturnExchangeKeysBuilder rsp_buider(*(fbb.get()));
      rsp_buider.add_retcode(retcode);
      rsp_buider.add_iteration(iteration);
      rsp_buider.add_remote_publickeys(remote_publickeys);
      rsp_buider.add_next_req_time(fbs_next_req_time);
      auto rsp_get_keys = rsp_buider.Finish();
      fbb->Finish(rsp_get_keys);
      MS_LOG(INFO) << "CipherMgr::GetKeys Success";
    }
  } else {
    auto fbs_next_req_time = fbb->CreateString(next_req_time);
    schema::ReturnExchangeKeysBuilder rsp_buider(*(fbb.get()));
    rsp_buider.add_retcode(retcode);
    rsp_buider.add_iteration(iteration);
    rsp_buider.add_next_req_time(fbs_next_req_time);
    auto rsp_get_keys = rsp_buider.Finish();

    fbb->Finish(rsp_get_keys);
  }
  return flag;
}

void CipherKeys::ClearKeys() {
  fl::server::DistributedMetadataStore::GetInstance().ResetMetadata(fl::server::kCtxExChangeKeysClientList);
  fl::server::DistributedMetadataStore::GetInstance().ResetMetadata(fl::server::kCtxClientsKeys);
}

}  // namespace armour
}  // namespace mindspore
