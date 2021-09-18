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
bool CipherKeys::GetKeys(const size_t cur_iterator, const std::string &next_req_time,
                         const schema::GetExchangeKeys *get_exchange_keys_req,
                         const std::shared_ptr<fl::server::FBBuilder> &fbb) {
  MS_LOG(INFO) << "CipherMgr::GetKeys START";
  if (get_exchange_keys_req == nullptr) {
    MS_LOG(ERROR) << "Request is nullptr";
    BuildGetKeysRsp(fbb, schema::ResponseCode_RequestError, cur_iterator, next_req_time, false);
    return false;
  }
  if (cipher_init_ == nullptr) {
    BuildGetKeysRsp(fbb, schema::ResponseCode_SystemError, cur_iterator, next_req_time, false);
    return false;
  }
  // get clientlist from memory server.
  std::map<std::string, std::vector<std::vector<uint8_t>>> client_public_keys;
  cipher_init_->cipher_meta_storage_.GetClientKeysFromServer(fl::server::kCtxClientsKeys, &client_public_keys);

  size_t cur_exchange_clients_num = client_public_keys.size();
  std::string fl_id = get_exchange_keys_req->fl_id()->str();

  if (cur_exchange_clients_num < cipher_init_->exchange_key_threshold) {
    MS_LOG(INFO) << "The server is not ready yet: cur_exchangekey_clients_num < exchange_key_threshold";
    MS_LOG(INFO) << "cur_exchangekey_clients_num : " << cur_exchange_clients_num
                 << ", exchange_key_threshold : " << cipher_init_->exchange_key_threshold;
    BuildGetKeysRsp(fbb, schema::ResponseCode_SucNotReady, cur_iterator, next_req_time, false);
    return false;
  }

  if (client_public_keys.find(fl_id) == client_public_keys.end()) {
    MS_LOG(INFO) << "Get keys: the fl_id: " << fl_id << "is not in exchange keys clients.";
    BuildGetKeysRsp(fbb, schema::ResponseCode_RequestError, cur_iterator, next_req_time, false);
    return false;
  }

  bool ret = cipher_init_->cipher_meta_storage_.UpdateClientToServer(fl::server::kCtxGetKeysClientList, fl_id);
  if (!ret) {
    MS_LOG(ERROR) << "update get keys clients failed";
    BuildGetKeysRsp(fbb, schema::ResponseCode_OutOfTime, cur_iterator, next_req_time, false);
    return false;
  }

  MS_LOG(INFO) << "GetKeys client list: ";
  BuildGetKeysRsp(fbb, schema::ResponseCode_SUCCEED, cur_iterator, next_req_time, true);
  return true;
}

bool CipherKeys::ExchangeKeys(const size_t cur_iterator, const std::string &next_req_time,
                              const schema::RequestExchangeKeys *exchange_keys_req,
                              const std::shared_ptr<fl::server::FBBuilder> &fbb) {
  MS_LOG(INFO) << "CipherMgr::ExchangeKeys START";
  // step 0: judge if the input param is legal.
  if (exchange_keys_req == nullptr) {
    std::string reason = "Request is nullptr";
    MS_LOG(ERROR) << reason;
    BuildExchangeKeysRsp(fbb, schema::ResponseCode_RequestError, reason, next_req_time, cur_iterator);
    return false;
  }
  if (cipher_init_ == nullptr) {
    std::string reason = "cipher_init_ is nullptr";
    BuildExchangeKeysRsp(fbb, schema::ResponseCode_SystemError, reason, next_req_time, cur_iterator);
    return false;
  }
  std::string fl_id = exchange_keys_req->fl_id()->str();
  mindspore::fl::PBMetadata device_metas =
    fl::server::DistributedMetadataStore::GetInstance().GetMetadata(fl::server::kCtxDeviceMetas);
  mindspore::fl::FLIdToDeviceMeta fl_id_to_meta = device_metas.device_metas();
  MS_LOG(INFO) << "exchange key for fl id " << fl_id;
  if (fl_id_to_meta.fl_id_to_meta().count(fl_id) == 0) {
    std::string reason = "devices_meta for " + fl_id + " is not set. Please retry later.";
    BuildExchangeKeysRsp(fbb, schema::ResponseCode_OutOfTime, reason, next_req_time, cur_iterator);
    MS_LOG(ERROR) << reason;
    return false;
  }

  // step 1: get clientlist and client keys from memory server.
  std::map<std::string, std::vector<std::vector<uint8_t>>> client_public_keys;
  std::vector<std::string> client_list;
  cipher_init_->cipher_meta_storage_.GetClientListFromServer(fl::server::kCtxExChangeKeysClientList, &client_list);
  cipher_init_->cipher_meta_storage_.GetClientKeysFromServer(fl::server::kCtxClientsKeys, &client_public_keys);

  // step2: process new item data. and update new item data to memory server.
  size_t cur_clients_num = client_list.size();
  size_t cur_clients_has_keys_num = client_public_keys.size();
  if (cur_clients_num != cur_clients_has_keys_num) {
    std::string reason = "client num and keys num are not equal.";
    MS_LOG(WARNING) << reason;
    MS_LOG(WARNING) << "cur_clients_num is " << cur_clients_num << ". cur_clients_has_keys_num is "
                    << cur_clients_has_keys_num;
  }
  MS_LOG(WARNING) << "exchange_key_threshold " << cipher_init_->exchange_key_threshold << ". cur_clients_num "
                  << cur_clients_num << ". cur_clients_keys_num " << cur_clients_has_keys_num;

  if (client_public_keys.find(fl_id) != client_public_keys.end()) {  // the client already exists, return false.
    MS_LOG(ERROR) << "The server has received the request, please do not request again.";
    BuildExchangeKeysRsp(fbb, schema::ResponseCode_SUCCEED,
                         "The server has received the request, please do not request again.", next_req_time,
                         cur_iterator);
    return false;
  }

  bool retcode_key =
    cipher_init_->cipher_meta_storage_.UpdateClientKeyToServer(fl::server::kCtxClientsKeys, exchange_keys_req);
  bool retcode_client =
    cipher_init_->cipher_meta_storage_.UpdateClientToServer(fl::server::kCtxExChangeKeysClientList, fl_id);
  if (retcode_key && retcode_client) {
    MS_LOG(INFO) << "The client " << fl_id << " CipherMgr::ExchangeKeys Success";
    BuildExchangeKeysRsp(fbb, schema::ResponseCode_SUCCEED, "Success, but the server is not ready yet.", next_req_time,
                         cur_iterator);
    return true;
  } else {
    MS_LOG(ERROR) << "update key or client failed";
    BuildExchangeKeysRsp(fbb, schema::ResponseCode_OutOfTime, "update key or client failed", next_req_time,
                         cur_iterator);
    return false;
  }
}

void CipherKeys::BuildExchangeKeysRsp(const std::shared_ptr<fl::server::FBBuilder> &fbb,
                                      const schema::ResponseCode retcode, const std::string &reason,
                                      const std::string &next_req_time, const size_t iteration) {
  auto rsp_reason = fbb->CreateString(reason);
  auto rsp_next_req_time = fbb->CreateString(next_req_time);

  schema::ResponseExchangeKeysBuilder rsp_builder(*(fbb.get()));
  rsp_builder.add_retcode(retcode);
  rsp_builder.add_reason(rsp_reason);
  rsp_builder.add_next_req_time(rsp_next_req_time);
  rsp_builder.add_iteration(SizeToInt(iteration));
  auto rsp_exchange_keys = rsp_builder.Finish();
  fbb->Finish(rsp_exchange_keys);
  return;
}

void CipherKeys::BuildGetKeysRsp(const std::shared_ptr<fl::server::FBBuilder> &fbb, const schema::ResponseCode retcode,
                                 const size_t iteration, const std::string &next_req_time, bool is_good) {
  if (!is_good) {
    auto fbs_next_req_time = fbb->CreateString(next_req_time);
    schema::ReturnExchangeKeysBuilder rsp_builder(*(fbb.get()));
    rsp_builder.add_retcode(static_cast<int>(retcode));
    rsp_builder.add_iteration(SizeToInt(iteration));
    rsp_builder.add_next_req_time(fbs_next_req_time);
    auto rsp_get_keys = rsp_builder.Finish();
    fbb->Finish(rsp_get_keys);
    return;
  }
  const fl::PBMetadata &clients_keys_pb_out =
    fl::server::DistributedMetadataStore::GetInstance().GetMetadata(fl::server::kCtxClientsKeys);
  const fl::ClientKeys &clients_keys_pb = clients_keys_pb_out.client_keys();
  std::vector<flatbuffers::Offset<schema::ClientPublicKeys>> public_keys_list;
  for (auto iter = clients_keys_pb.client_keys().begin(); iter != clients_keys_pb.client_keys().end(); ++iter) {
    std::string fl_id = iter->first;
    fl::KeysPb keys_pb = iter->second;
    auto fbs_fl_id = fbb->CreateString(fl_id);
    std::vector<uint8_t> cpk(keys_pb.key(0).begin(), keys_pb.key(0).end());
    std::vector<uint8_t> spk(keys_pb.key(1).begin(), keys_pb.key(1).end());
    auto fbs_c_pk = fbb->CreateVector(cpk.data(), cpk.size());
    auto fbs_s_pk = fbb->CreateVector(spk.data(), spk.size());
    std::vector<uint8_t> pw_iv(keys_pb.pw_iv().begin(), keys_pb.pw_iv().end());
    auto fbs_pw_iv = fbb->CreateVector(pw_iv.data(), pw_iv.size());
    std::vector<uint8_t> pw_salt(keys_pb.pw_salt().begin(), keys_pb.pw_salt().end());
    auto fbs_pw_salt = fbb->CreateVector(pw_salt.data(), pw_salt.size());
    auto cur_public_key = schema::CreateClientPublicKeys(*fbb, fbs_fl_id, fbs_c_pk, fbs_s_pk, fbs_pw_iv, fbs_pw_salt);
    public_keys_list.push_back(cur_public_key);
  }
  auto remote_publickeys = fbb->CreateVector(public_keys_list);
  auto fbs_next_req_time = fbb->CreateString(next_req_time);
  schema::ReturnExchangeKeysBuilder rsp_builder(*(fbb.get()));
  rsp_builder.add_retcode(static_cast<int>(retcode));
  rsp_builder.add_iteration(SizeToInt(iteration));
  rsp_builder.add_remote_publickeys(remote_publickeys);
  rsp_builder.add_next_req_time(fbs_next_req_time);
  auto rsp_get_keys = rsp_builder.Finish();
  fbb->Finish(rsp_get_keys);
  MS_LOG(INFO) << "CipherMgr::GetKeys Success";
  return;
}

void CipherKeys::ClearKeys() {
  fl::server::DistributedMetadataStore::GetInstance().ResetMetadata(fl::server::kCtxExChangeKeysClientList);
  fl::server::DistributedMetadataStore::GetInstance().ResetMetadata(fl::server::kCtxClientsKeys);
  fl::server::DistributedMetadataStore::GetInstance().ResetMetadata(fl::server::kCtxGetKeysClientList);
}
}  // namespace armour
}  // namespace mindspore
