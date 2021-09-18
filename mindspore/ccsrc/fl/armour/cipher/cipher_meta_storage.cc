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

#include "fl/armour/cipher/cipher_meta_storage.h"

namespace mindspore {
namespace armour {
void CipherMetaStorage::GetClientSharesFromServer(
  const char *list_name, std::map<std::string, std::vector<clientshare_str>> *clients_shares_list) {
  if (clients_shares_list == nullptr) {
    MS_LOG(ERROR) << "input clients_shares_list is nullptr";
    return;
  }
  const fl::PBMetadata &clients_shares_pb_out =
    fl::server::DistributedMetadataStore::GetInstance().GetMetadata(list_name);
  const fl::ClientShares &clients_shares_pb = clients_shares_pb_out.client_shares();
  auto iter = clients_shares_pb.client_secret_shares().begin();
  for (; iter != clients_shares_pb.client_secret_shares().end(); ++iter) {
    std::string fl_id = iter->first;
    const fl::SharesPb &shares_pb = iter->second;
    std::vector<clientshare_str> encrpted_shares_new;
    size_t client_share_num = IntToSize(shares_pb.clientsharestrs_size());
    for (size_t index_shares = 0; index_shares < client_share_num; ++index_shares) {
      const fl::ClientShareStr &client_share_str_pb = shares_pb.clientsharestrs(index_shares);
      clientshare_str new_clientshare;
      new_clientshare.fl_id = client_share_str_pb.fl_id();
      new_clientshare.index = client_share_str_pb.index();
      new_clientshare.share.assign(client_share_str_pb.share().begin(), client_share_str_pb.share().end());
      encrpted_shares_new.push_back(new_clientshare);
    }
    clients_shares_list->insert(std::pair<std::string, std::vector<clientshare_str>>(fl_id, encrpted_shares_new));
  }
}

void CipherMetaStorage::GetClientListFromServer(const char *list_name, std::vector<std::string> *clients_list) {
  if (clients_list == nullptr) {
    MS_LOG(ERROR) << "input clients_list is nullptr";
    return;
  }
  const fl::PBMetadata &client_list_pb_out = fl::server::DistributedMetadataStore::GetInstance().GetMetadata(list_name);
  const fl::UpdateModelClientList &client_list_pb = client_list_pb_out.client_list();
  size_t client_list_num = IntToSize(client_list_pb.fl_id_size());
  for (size_t i = 0; i < client_list_num; ++i) {
    std::string fl_id = client_list_pb.fl_id(SizeToInt(i));
    clients_list->push_back(fl_id);
  }
}

void CipherMetaStorage::GetClientKeysFromServer(
  const char *list_name, std::map<std::string, std::vector<std::vector<uint8_t>>> *clients_keys_list) {
  if (clients_keys_list == nullptr) {
    MS_LOG(ERROR) << "input clients_keys_list is nullptr";
    return;
  }
  const fl::PBMetadata &clients_keys_pb_out =
    fl::server::DistributedMetadataStore::GetInstance().GetMetadata(list_name);
  const fl::ClientKeys &clients_keys_pb = clients_keys_pb_out.client_keys();

  for (auto iter = clients_keys_pb.client_keys().begin(); iter != clients_keys_pb.client_keys().end(); ++iter) {
    std::string fl_id = iter->first;
    fl::KeysPb keys_pb = iter->second;
    std::vector<uint8_t> cpk(keys_pb.key(0).begin(), keys_pb.key(0).end());
    std::vector<uint8_t> spk(keys_pb.key(1).begin(), keys_pb.key(1).end());
    std::vector<std::vector<uint8_t>> cur_keys;
    cur_keys.push_back(cpk);
    cur_keys.push_back(spk);
    (void)clients_keys_list->emplace(std::pair<std::string, std::vector<std::vector<uint8_t>>>(fl_id, cur_keys));
  }
}

void CipherMetaStorage::GetClientIVsFromServer(
  const char *list_name, std::map<std::string, std::vector<std::vector<uint8_t>>> *clients_ivs_list) {
  if (clients_ivs_list == nullptr) {
    MS_LOG(ERROR) << "input clients_ivs_list is nullptr";
    return;
  }
  const fl::PBMetadata &clients_keys_pb_out =
    fl::server::DistributedMetadataStore::GetInstance().GetMetadata(list_name);
  const fl::ClientKeys &clients_keys_pb = clients_keys_pb_out.client_keys();

  for (auto iter = clients_keys_pb.client_keys().begin(); iter != clients_keys_pb.client_keys().end(); ++iter) {
    std::string fl_id = iter->first;
    fl::KeysPb keys_pb = iter->second;
    std::vector<uint8_t> ind_iv(keys_pb.ind_iv().begin(), keys_pb.ind_iv().end());
    std::vector<uint8_t> pw_iv(keys_pb.pw_iv().begin(), keys_pb.pw_iv().end());
    std::vector<uint8_t> pw_salt(keys_pb.pw_salt().begin(), keys_pb.pw_salt().end());

    std::vector<std::vector<uint8_t>> cur_ivs;
    cur_ivs.push_back(ind_iv);
    cur_ivs.push_back(pw_iv);
    cur_ivs.push_back(pw_salt);
    (void)clients_ivs_list->emplace(std::pair<std::string, std::vector<std::vector<uint8_t>>>(fl_id, cur_ivs));
  }
}

bool CipherMetaStorage::GetClientNoisesFromServer(const char *list_name, std::vector<float> *cur_public_noise) {
  if (cur_public_noise == nullptr) {
    MS_LOG(ERROR) << "input cur_public_noise is nullptr";
    return false;
  }
  const fl::PBMetadata &clients_noises_pb_out =
    fl::server::DistributedMetadataStore::GetInstance().GetMetadata(list_name);
  const fl::ClientNoises &clients_noises_pb = clients_noises_pb_out.client_noises();
  int count = 0;
  const int count_thld = 1000;
  while (clients_noises_pb.has_one_client_noises() == false) {
    const int register_time = 500;
    std::this_thread::sleep_for(std::chrono::milliseconds(register_time));
    count++;
    if (count >= count_thld) break;
  }
  if (clients_noises_pb.has_one_client_noises() == false) {
    MS_LOG(WARNING) << "GetClientNoisesFromServer Count: " << count;
    return false;
  }
  cur_public_noise->assign(clients_noises_pb.one_client_noises().noise().begin(),
                           clients_noises_pb.one_client_noises().noise().end());
  return true;
}

bool CipherMetaStorage::GetPrimeFromServer(const char *prime_name, uint8_t *prime) {
  if (prime == nullptr) {
    MS_LOG(ERROR) << "input prime is nullptr";
    return false;
  }
  const fl::PBMetadata &prime_pb_out = fl::server::DistributedMetadataStore::GetInstance().GetMetadata(prime_name);
  fl::Prime prime_pb(prime_pb_out.prime());
  std::string str = *(prime_pb.mutable_prime());
  if (str.size() != PRIME_MAX_LEN) {
    MS_LOG(ERROR) << "get prime size is :" << str.size();
    return false;
  } else {
    if (memcpy_s(prime, PRIME_MAX_LEN, str.data(), str.size()) != 0) {
      MS_LOG(ERROR) << "Memcpy_s error";
      return false;
    }
    return true;
  }
}

bool CipherMetaStorage::UpdateClientToServer(const char *list_name, const std::string &fl_id) {
  fl::FLId fl_id_pb;
  fl_id_pb.set_fl_id(fl_id);
  fl::PBMetadata client_pb;
  client_pb.mutable_fl_id()->MergeFrom(fl_id_pb);
  bool retcode = fl::server::DistributedMetadataStore::GetInstance().UpdateMetadata(list_name, client_pb);
  return retcode;
}

void CipherMetaStorage::RegisterPrime(const char *list_name, const std::string &prime) {
  MS_LOG(INFO) << "register prime: " << prime;
  fl::Prime prime_id_pb;
  prime_id_pb.set_prime(prime);
  fl::PBMetadata prime_pb;
  prime_pb.mutable_prime()->MergeFrom(prime_id_pb);
  fl::server::DistributedMetadataStore::GetInstance().RegisterMetadata(list_name, prime_pb);
  uint32_t time = 1;
  (void)sleep(time);
}

bool CipherMetaStorage::UpdateClientKeyToServer(const char *list_name, const std::string &fl_id,
                                                const std::vector<std::vector<uint8_t>> &cur_public_key) {
  const size_t correct_size = 2;
  if (cur_public_key.size() < correct_size) {
    MS_LOG(ERROR) << "cur_public_key's size must is 2. actual size is " << cur_public_key.size();
    return false;
  }
  // update new item to memory server.
  fl::KeysPb keys;
  keys.add_key()->assign(cur_public_key[0].begin(), cur_public_key[0].end());
  keys.add_key()->assign(cur_public_key[1].begin(), cur_public_key[1].end());
  fl::PairClientKeys pair_client_keys_pb;
  pair_client_keys_pb.set_fl_id(fl_id);
  pair_client_keys_pb.mutable_client_keys()->MergeFrom(keys);
  fl::PBMetadata client_and_keys_pb;
  client_and_keys_pb.mutable_pair_client_keys()->MergeFrom(pair_client_keys_pb);
  bool retcode = fl::server::DistributedMetadataStore::GetInstance().UpdateMetadata(list_name, client_and_keys_pb);
  return retcode;
}

bool CipherMetaStorage::UpdateClientKeyToServer(const char *list_name,
                                                const schema::RequestExchangeKeys *exchange_keys_req) {
  std::string fl_id = exchange_keys_req->fl_id()->str();
  auto fbs_cpk = exchange_keys_req->c_pk();
  auto fbs_spk = exchange_keys_req->s_pk();
  if (fbs_cpk == nullptr || fbs_spk == nullptr) {
    MS_LOG(ERROR) << "public key from exchange_keys_req is null";
    return false;
  }

  size_t spk_len = fbs_spk->size();
  size_t cpk_len = fbs_cpk->size();

  // transform fbs (fbs_cpk & fbs_spk) to a vector: public_key
  std::vector<std::vector<uint8_t>> cur_public_key;
  std::vector<uint8_t> cpk(cpk_len);
  std::vector<uint8_t> spk(spk_len);
  bool ret_create_code_cpk = CreateArray<uint8_t>(&cpk, *fbs_cpk);
  bool ret_create_code_spk = CreateArray<uint8_t>(&spk, *fbs_spk);
  if (!(ret_create_code_cpk && ret_create_code_spk)) {
    MS_LOG(ERROR) << "create array for public keys failed";
    return false;
  }
  cur_public_key.push_back(cpk);
  cur_public_key.push_back(spk);

  auto fbs_ind_iv = exchange_keys_req->ind_iv();
  std::vector<char> ind_iv;
  if (fbs_ind_iv == nullptr) {
    MS_LOG(WARNING) << "ind_iv in exchange_keys_req is nullptr";
  } else {
    ind_iv.assign(fbs_ind_iv->begin(), fbs_ind_iv->end());
  }

  auto fbs_pw_iv = exchange_keys_req->pw_iv();
  std::vector<char> pw_iv;
  if (fbs_pw_iv == nullptr) {
    MS_LOG(WARNING) << "pw_iv in exchange_keys_req is nullptr";
  } else {
    pw_iv.assign(fbs_pw_iv->begin(), fbs_pw_iv->end());
  }

  auto fbs_pw_salt = exchange_keys_req->pw_salt();
  std::vector<char> pw_salt;
  if (fbs_pw_salt == nullptr) {
    MS_LOG(WARNING) << "pw_salt in exchange_keys_req is nullptr";
  } else {
    pw_salt.assign(fbs_pw_salt->begin(), fbs_pw_salt->end());
  }

  // update new item to memory server.
  fl::KeysPb keys;
  keys.add_key()->assign(cur_public_key[0].begin(), cur_public_key[0].end());
  keys.add_key()->assign(cur_public_key[1].begin(), cur_public_key[1].end());
  keys.set_ind_iv(ind_iv.data(), ind_iv.size());
  keys.set_pw_iv(pw_iv.data(), pw_iv.size());
  keys.set_pw_salt(pw_salt.data(), pw_salt.size());
  fl::PairClientKeys pair_client_keys_pb;
  pair_client_keys_pb.set_fl_id(fl_id);
  pair_client_keys_pb.mutable_client_keys()->MergeFrom(keys);
  fl::PBMetadata client_and_keys_pb;
  client_and_keys_pb.mutable_pair_client_keys()->MergeFrom(pair_client_keys_pb);
  bool retcode = fl::server::DistributedMetadataStore::GetInstance().UpdateMetadata(list_name, client_and_keys_pb);
  return retcode;
}

bool CipherMetaStorage::UpdateClientNoiseToServer(const char *list_name, const std::vector<float> &cur_public_noise) {
  // update new item to memory server.
  fl::OneClientNoises noises_pb;
  *noises_pb.mutable_noise() = {cur_public_noise.begin(), cur_public_noise.end()};
  fl::PBMetadata client_noises_pb;
  client_noises_pb.mutable_one_client_noises()->MergeFrom(noises_pb);
  bool ret = fl::server::DistributedMetadataStore::GetInstance().UpdateMetadata(list_name, client_noises_pb);
  return ret;
}

bool CipherMetaStorage::UpdateClientShareToServer(
  const char *list_name, const std::string &fl_id,
  const flatbuffers::Vector<flatbuffers::Offset<mindspore::schema::ClientShare>> *shares) {
  if (shares == nullptr) {
    return false;
  }
  size_t size_shares = shares->size();
  fl::SharesPb shares_pb;
  for (size_t index = 0; index < size_shares; ++index) {
    // new item
    fl::ClientShareStr *client_share_str_new_p = shares_pb.add_clientsharestrs();
    std::string fl_id_new = (*shares)[SizeToInt(index)]->fl_id()->str();
    int index_new = (*shares)[SizeToInt(index)]->index();
    auto share = (*shares)[SizeToInt(index)]->share();
    if (share == nullptr) return false;
    client_share_str_new_p->set_share(reinterpret_cast<const char *>(share->data()), share->size());
    client_share_str_new_p->set_fl_id(fl_id_new);
    client_share_str_new_p->set_index(index_new);
  }
  fl::PairClientShares pair_client_shares_pb;
  pair_client_shares_pb.set_fl_id(fl_id);
  pair_client_shares_pb.mutable_client_shares()->MergeFrom(shares_pb);
  fl::PBMetadata client_and_shares_pb;
  client_and_shares_pb.mutable_pair_client_shares()->MergeFrom(pair_client_shares_pb);
  bool retcode = fl::server::DistributedMetadataStore::GetInstance().UpdateMetadata(list_name, client_and_shares_pb);
  return retcode;
}

void CipherMetaStorage::RegisterClass() {
  fl::PBMetadata exchange_keys_client_list;
  fl::server::DistributedMetadataStore::GetInstance().RegisterMetadata(fl::server::kCtxExChangeKeysClientList,
                                                                       exchange_keys_client_list);
  fl::PBMetadata get_keys_client_list;
  fl::server::DistributedMetadataStore::GetInstance().RegisterMetadata(fl::server::kCtxGetKeysClientList,
                                                                       get_keys_client_list);
  fl::PBMetadata clients_keys;
  fl::server::DistributedMetadataStore::GetInstance().RegisterMetadata(fl::server::kCtxClientsKeys, clients_keys);
  fl::PBMetadata reconstruct_client_list;
  fl::server::DistributedMetadataStore::GetInstance().RegisterMetadata(fl::server::kCtxReconstructClientList,
                                                                       reconstruct_client_list);
  fl::PBMetadata clients_reconstruct_shares;
  fl::server::DistributedMetadataStore::GetInstance().RegisterMetadata(fl::server::kCtxClientsReconstructShares,
                                                                       clients_reconstruct_shares);
  fl::PBMetadata share_secretes_client_list;
  fl::server::DistributedMetadataStore::GetInstance().RegisterMetadata(fl::server::kCtxShareSecretsClientList,
                                                                       share_secretes_client_list);
  fl::PBMetadata get_secretes_client_list;
  fl::server::DistributedMetadataStore::GetInstance().RegisterMetadata(fl::server::kCtxGetSecretsClientList,
                                                                       get_secretes_client_list);
  fl::PBMetadata clients_encrypt_shares;
  fl::server::DistributedMetadataStore::GetInstance().RegisterMetadata(fl::server::kCtxClientsEncryptedShares,
                                                                       clients_encrypt_shares);
  fl::PBMetadata get_update_clients_list;
  fl::server::DistributedMetadataStore::GetInstance().RegisterMetadata(fl::server::kCtxGetUpdateModelClientList,
                                                                       get_update_clients_list);
  fl::PBMetadata client_noises;
  fl::server::DistributedMetadataStore::GetInstance().RegisterMetadata(fl::server::kCtxClientNoises, client_noises);
}
}  // namespace armour
}  // namespace mindspore
