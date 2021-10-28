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

#include "fl/armour/cipher/cipher_reconstruct.h"
#include "fl/server/common.h"
#include "fl/armour/secure_protocol/masking.h"
#include "fl/armour/secure_protocol/key_agreement.h"
#include "fl/armour/cipher/cipher_meta_storage.h"

namespace mindspore {
namespace armour {
bool CipherReconStruct::CombineMask(std::vector<Share *> *shares_tmp,
                                    std::map<std::string, std::vector<float>> *client_noise,
                                    const std::vector<std::string> &clients_share_list,
                                    const std::map<std::string, std::vector<std::vector<uint8_t>>> &record_public_keys,
                                    const std::map<std::string, std::vector<clientshare_str>> &reconstruct_secret_list,
                                    const std::vector<string> &client_list,
                                    const std::map<std::string, std::vector<std::vector<uint8_t>>> &client_ivs) {
  bool retcode = true;
#ifdef _WIN32
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  retcode = false;
#else
  if (shares_tmp == nullptr || client_noise == nullptr) {
    MS_LOG(ERROR) << "shares_tmp or client_noise is nullptr.";
    return false;
  }
  for (auto iter = reconstruct_secret_list.begin(); iter != reconstruct_secret_list.end(); ++iter) {
    // define flag_share: judge we need b or s
    bool flag_share = true;
    const std::string fl_id = iter->first;
    if (find(client_list.begin(), client_list.end(), fl_id) != client_list.end()) {
      // the client is online
      flag_share = false;
    }
    MS_LOG(INFO) << "fl_id_src : " << fl_id;
    BIGNUM *prime = BN_new();
    if (prime == nullptr) {
      return false;
    }
    auto publicparam_ = CipherInit::GetInstance().GetPublicParams();
    (void)BN_bin2bn(publicparam_->prime, PRIME_MAX_LEN, prime);
    if (iter->second.size() >= cipher_init_->secrets_minnums_) {  // combine private key seed.
      MS_LOG(INFO) << "start assign secrets shares to public shares ";
      for (int i = 0; i < static_cast<int>(cipher_init_->secrets_minnums_); ++i) {
        shares_tmp->at(i)->index = (iter->second)[i].index;
        shares_tmp->at(i)->len = (iter->second)[i].share.size();
        if (memcpy_s(shares_tmp->at(i)->data, IntToSize(SHARE_MAX_SIZE), (iter->second)[i].share.data(),
                     shares_tmp->at(i)->len) != 0) {
          MS_LOG(ERROR) << "shares_tmp copy failed";
          retcode = false;
        }
      }
      MS_LOG(INFO) << "end assign secrets shares to public shares ";

      size_t length;
      uint8_t secret[SECRET_MAX_LEN] = {0};
      SecretSharing combine(prime);
      if (combine.Combine(cipher_init_->secrets_minnums_, *shares_tmp, secret, &length) < 0) retcode = false;
      length = SECRET_MAX_LEN;
      MS_LOG(INFO) << "combine secrets shares Success.";

      if (flag_share) {
        // reconstruct pairwise noise
        MS_LOG(INFO) << "start reconstruct pairwise noise.";
        std::vector<float> noise(cipher_init_->featuremap_, 0.0);
        if (GetSuvNoise(clients_share_list, record_public_keys, client_ivs, fl_id, &noise, secret, length) == false) {
          MS_LOG(ERROR) << "GetSuvNoise failed";
          BN_clear_free(prime);
          if (memset_s(secret, SECRET_MAX_LEN, 0, length) != 0) {
            MS_LOG(EXCEPTION) << "Memset failed.";
          }
          return false;
        }
        (void)client_noise->emplace(std::pair<std::string, std::vector<float>>(fl_id, noise));
      } else {
        // reconstruct individual noise
        MS_LOG(INFO) << "start reconstruct individual noise.";
        std::vector<float> noise;
        auto it = client_ivs.find(fl_id);
        if (it == client_ivs.end()) {
          MS_LOG(ERROR) << "cannot get ivs for client: " << fl_id;
          return false;
        }
        if (it->second.size() != IV_NUM) {
          MS_LOG(ERROR) << "get " << it->second.size() << " ivs, the iv num required is: " << IV_NUM;
          return false;
        }
        std::vector<uint8_t> ind_iv = it->second[0];
        if (Masking::GetMasking(&noise, SizeToInt(cipher_init_->featuremap_), (const uint8_t *)secret, SECRET_MAX_LEN,
                                ind_iv.data(), SizeToInt(ind_iv.size())) < 0) {
          MS_LOG(ERROR) << "Get Masking failed";
          if (memset_s(secret, SECRET_MAX_LEN, 0, length) != 0) {
            MS_LOG(EXCEPTION) << "Memset failed.";
          }
          BN_clear_free(prime);
          return false;
        }
        for (size_t index_noise = 0; index_noise < cipher_init_->featuremap_; index_noise++) {
          noise[index_noise] *= -1;
        }
        (void)client_noise->emplace(std::pair<std::string, std::vector<float>>(fl_id, noise));
      }
      BN_clear_free(prime);
      if (memset_s(secret, SECRET_MAX_LEN, 0, length) != 0) {
        MS_LOG(EXCEPTION) << "Memset failed.";
      }
    } else {
      MS_LOG(ERROR) << "reconstruct secret failed: the number of secret shares for fl_id: " << fl_id
                    << " is not enough";
      MS_LOG(ERROR) << "get " << iter->second.size()
                    << "shares, however the secrets_minnums_ required is: " << cipher_init_->secrets_minnums_;
      return false;
    }
  }
#endif
  return retcode;
}

bool CipherReconStruct::ReconstructSecretsGenNoise(const std::vector<string> &client_list) {
  // get reconstruct_secrets from memory server
  MS_LOG(INFO) << "CipherReconStruct::ReconstructSecretsGenNoise START";
  bool retcode = true;
  std::map<std::string, std::vector<clientshare_str>> reconstruct_secrets;
  cipher_init_->cipher_meta_storage_.GetClientSharesFromServer(fl::server::kCtxClientsReconstructShares,
                                                               &reconstruct_secrets);
  std::map<std::string, std::vector<std::vector<uint8_t>>> record_public_keys;
  cipher_init_->cipher_meta_storage_.GetClientKeysFromServer(fl::server::kCtxClientsKeys, &record_public_keys);

  std::map<std::string, std::vector<std::vector<uint8_t>>> client_ivs;
  cipher_init_->cipher_meta_storage_.GetClientIVsFromServer(fl::server::kCtxClientsKeys, &client_ivs);

  std::vector<std::string> clients_share_list;
  cipher_init_->cipher_meta_storage_.GetClientListFromServer(fl::server::kCtxShareSecretsClientList,
                                                             &clients_share_list);
  if (record_public_keys.size() < cipher_init_->exchange_key_threshold ||
      clients_share_list.size() < cipher_init_->share_secrets_threshold ||
      record_public_keys.size() != client_ivs.size()) {
    MS_LOG(ERROR) << "send share client size: " << clients_share_list.size()
                  << ", send public-key client size: " << record_public_keys.size()
                  << ", send ivs client size: " << client_ivs.size();
    MS_LOG(ERROR) << "get data from server memory failed";
    return false;
  }

  std::map<std::string, std::vector<clientshare_str>> reconstruct_secret_list;
  if (!ConvertSharesToShares(reconstruct_secrets, &reconstruct_secret_list)) {
    MS_LOG(ERROR) << "ConvertSharesToShares failed.";
    return false;
  }

  MS_LOG(ERROR) << "recombined shares";
  for (auto iter = reconstruct_secret_list.begin(); iter != reconstruct_secret_list.end(); ++iter) {
    MS_LOG(ERROR) << "fl_id: " << iter->first;
    MS_LOG(ERROR) << "share size: " << iter->second.size();
  }
  std::vector<Share *> shares_tmp;
  if (!MallocShares(&shares_tmp, (SizeToInt)(cipher_init_->secrets_minnums_))) {
    MS_LOG(ERROR) << "Reconstruct malloc shares_tmp invalid.";
    DeleteShares(&shares_tmp);
    return false;
  }

  MS_LOG(INFO) << "Reconstruct secrets shares: ";
  std::map<std::string, std::vector<float>> client_noise;
  retcode = CombineMask(&shares_tmp, &client_noise, clients_share_list, record_public_keys, reconstruct_secret_list,
                        client_list, client_ivs);
  DeleteShares(&shares_tmp);
  if (retcode) {
    std::vector<float> noise;
    if (!GetNoiseMasksSum(&noise, client_noise)) {
      MS_LOG(ERROR) << " GetNoiseMasksSum failed";
      return false;
    }
    client_noise.clear();
    MS_LOG(INFO) << " ReconstructSecretsGenNoise updata noise to server";

    if (!cipher_init_->cipher_meta_storage_.UpdateClientNoiseToServer(fl::server::kCtxClientNoises, noise)) {
      MS_LOG(ERROR) << " ReconstructSecretsGenNoise failed. because UpdateClientNoiseToServer failed";
      return false;
    }
    MS_LOG(INFO) << " ReconstructSecretsGenNoise Success";
  } else {
    MS_LOG(ERROR) << " ReconstructSecretsGenNoise failed. because gen noise inside failed";
  }
  return retcode;
}

bool CipherReconStruct::CheckInputs(const schema::SendReconstructSecret *reconstruct_secret_req,
                                    const std::shared_ptr<fl::server::FBBuilder> &fbb, const int cur_iterator,
                                    const std::string &next_req_time) {
  if (reconstruct_secret_req == nullptr) {
    std::string reason = "Request is nullptr";
    MS_LOG(ERROR) << reason;
    BuildReconstructSecretsRsp(fbb, schema::ResponseCode_RequestError, reason, cur_iterator, next_req_time);
    return false;
  }
  if (cipher_init_ == nullptr) {
    std::string reason = "cipher_init_ is nullptr";
    MS_LOG(ERROR) << reason;
    BuildReconstructSecretsRsp(fbb, schema::ResponseCode_SystemError, reason, cur_iterator, next_req_time);
    return false;
  }
  return true;
}

// reconstruct secrets
bool CipherReconStruct::ReconstructSecrets(const int cur_iterator, const std::string &next_req_time,
                                           const schema::SendReconstructSecret *reconstruct_secret_req,
                                           const std::shared_ptr<fl::server::FBBuilder> &fbb,
                                           const std::vector<std::string> &client_list) {
  MS_LOG(INFO) << "CipherReconStruct::ReconstructSecrets START";
  clock_t start_time = clock();
  bool inputs_check = CheckInputs(reconstruct_secret_req, fbb, cur_iterator, next_req_time);
  if (!inputs_check) return false;

  int iterator = reconstruct_secret_req->iteration();
  std::string fl_id = reconstruct_secret_req->fl_id()->str();
  if (iterator != cur_iterator) {
    BuildReconstructSecretsRsp(fbb, schema::ResponseCode_OutOfTime,
                               "The iteration round of the client does not match the current iteration.", cur_iterator,
                               next_req_time);
    MS_LOG(ERROR) << "Client " << fl_id << " The iteration round of the client does not match the current iteration.";
    return false;
  }

  if (client_list.size() < cipher_init_->reconstruct_secrets_threshold) {
    MS_LOG(ERROR) << "illegal parameters. update model client_list size: " << client_list.size();
    BuildReconstructSecretsRsp(
      fbb, schema::ResponseCode_RequestError,
      "illegal parameters: update model client_list size must larger than reconstruct_clients_num_need", cur_iterator,
      next_req_time);
    return false;
  }

  std::vector<std::string> get_clients_list;
  cipher_init_->cipher_meta_storage_.GetClientListFromServer(fl::server::kCtxGetUpdateModelClientList,
                                                             &get_clients_list);
  // client not in get client list.
  if (find(get_clients_list.begin(), get_clients_list.end(), fl_id) == get_clients_list.end()) {
    std::string reason;
    MS_LOG(INFO) << "The client " << fl_id << " is not in get update model client list.";
    // client in update model client list.
    if (find(client_list.begin(), client_list.end(), fl_id) != client_list.end()) {
      reason = "The client " + fl_id + " is not in get clients list, but in update model client list.";
      MS_LOG(INFO) << reason;
      BuildReconstructSecretsRsp(fbb, schema::ResponseCode_SUCCEED, reason, cur_iterator, next_req_time);
      return false;
    }
    reason = "The client " + fl_id + " is not in get clients list, and not in update model client list.";
    BuildReconstructSecretsRsp(fbb, schema::ResponseCode_RequestError, "The client is not in update model client list.",
                               cur_iterator, next_req_time);

    return false;
  }

  std::map<std::string, std::vector<clientshare_str>> reconstruct_shares;
  cipher_init_->cipher_meta_storage_.GetClientSharesFromServer(fl::server::kCtxClientsReconstructShares,
                                                               &reconstruct_shares);
  size_t count_client_num = reconstruct_shares.size();
  if (reconstruct_shares.find(fl_id) != reconstruct_shares.end()) {
    BuildReconstructSecretsRsp(fbb, schema::ResponseCode_SUCCEED, "Client has sended messages.", cur_iterator,
                               next_req_time);
    MS_LOG(INFO) << "Error, client " << fl_id << " has sended messages.";
    return false;
  }

  auto reconstruct_secret_shares = reconstruct_secret_req->reconstruct_secret_shares();
  bool retcode_client =
    cipher_init_->cipher_meta_storage_.UpdateClientToServer(fl::server::kCtxReconstructClientList, fl_id);
  bool retcode_share = cipher_init_->cipher_meta_storage_.UpdateClientShareToServer(
    fl::server::kCtxClientsReconstructShares, fl_id, reconstruct_secret_shares);
  if (!(retcode_client && retcode_share)) {
    BuildReconstructSecretsRsp(fbb, schema::ResponseCode_OutOfTime, "reconstruct update shares or client failed.",
                               cur_iterator, next_req_time);
    MS_LOG(ERROR) << "reconstruct update shares or client failed.";
    return false;
  }

  count_client_num = count_client_num + 1;
  if (count_client_num < cipher_init_->reconstruct_secrets_threshold) {
    BuildReconstructSecretsRsp(fbb, schema::ResponseCode_SUCCEED,
                               "Success, but the server is not ready to reconstruct secret yet.", cur_iterator,
                               next_req_time);
    MS_LOG(INFO) << "Get reconstruct shares from " << fl_id << " Success, but count " << count_client_num
                 << " is not enough.";
    return true;
  }
  const fl::PBMetadata &clients_noises_pb_out =
    fl::server::DistributedMetadataStore::GetInstance().GetMetadata(fl::server::kCtxClientNoises);
  const fl::ClientNoises &clients_noises_pb = clients_noises_pb_out.client_noises();
  if (clients_noises_pb.has_one_client_noises() == false) {
    MS_LOG(INFO) << "Success, the secret will be reconstructed.";
    if (ReconstructSecretsGenNoise(client_list)) {
      BuildReconstructSecretsRsp(fbb, schema::ResponseCode_SUCCEED, "Success,the secret is reconstructing.",
                                 cur_iterator, next_req_time);
      MS_LOG(INFO) << "CipherReconStruct::ReconstructSecrets" << fl_id << " Success, reconstruct ok.";
    } else {
      BuildReconstructSecretsRsp(fbb, schema::ResponseCode_OutOfTime, "the secret restructs failed.", cur_iterator,
                                 next_req_time);
      MS_LOG(ERROR) << "CipherReconStruct::ReconstructSecrets" << fl_id << " failed.";
    }
  } else {
    BuildReconstructSecretsRsp(fbb, schema::ResponseCode_SUCCEED, "Clients' number is full.", cur_iterator,
                               next_req_time);
    MS_LOG(INFO) << "CipherReconStruct::ReconstructSecrets" << fl_id << " Success : no need reconstruct.";
  }
  clock_t end_time = clock();
  double duration = static_cast<double>((end_time - start_time) * 1.0 / CLOCKS_PER_SEC);
  MS_LOG(INFO) << "Reconstruct get + gennoise data time is : " << duration;
  return true;
}

bool CipherReconStruct::GetNoiseMasksSum(std::vector<float> *result,
                                         const std::map<std::string, std::vector<float>> &client_noise) {
  std::vector<float> sum(cipher_init_->featuremap_, 0.0);
  for (auto iter = client_noise.begin(); iter != client_noise.end(); iter++) {
    if (iter->second.size() != cipher_init_->featuremap_) {
      return false;
    }
    for (size_t i = 0; i < cipher_init_->featuremap_; i++) {
      sum[i] += iter->second[i];
    }
  }
  for (size_t i = 0; i < cipher_init_->featuremap_; i++) {
    result->push_back(sum[i]);
  }
  return true;
}

void CipherReconStruct::ClearReconstructSecrets() {
  MS_LOG(INFO) << "CipherReconStruct::ClearReconstructSecrets START";
  fl::server::DistributedMetadataStore::GetInstance().ResetMetadata(fl::server::kCtxReconstructClientList);
  fl::server::DistributedMetadataStore::GetInstance().ResetMetadata(fl::server::kCtxClientsReconstructShares);
  fl::server::DistributedMetadataStore::GetInstance().ResetMetadata(fl::server::kCtxClientNoises);
  MS_LOG(INFO) << "CipherReconStruct::ClearReconstructSecrets Success";
}

void CipherReconStruct::BuildReconstructSecretsRsp(const std::shared_ptr<fl::server::FBBuilder> &fbb,
                                                   const schema::ResponseCode retcode, const std::string &reason,
                                                   const int iteration, const std::string &next_req_time) {
  auto fbs_reason = fbb->CreateString(reason);
  auto fbs_next_req_time = fbb->CreateString(next_req_time);
  schema::ReconstructSecretBuilder rsp_reconstruct_secret_builder(*(fbb.get()));
  rsp_reconstruct_secret_builder.add_retcode(static_cast<int>(retcode));
  rsp_reconstruct_secret_builder.add_reason(fbs_reason);
  rsp_reconstruct_secret_builder.add_iteration(iteration);
  rsp_reconstruct_secret_builder.add_next_req_time(fbs_next_req_time);
  auto rsp_reconstruct_secret = rsp_reconstruct_secret_builder.Finish();
  fbb->Finish(rsp_reconstruct_secret);
  return;
}

bool CipherReconStruct::GetSuvNoise(const std::vector<std::string> &clients_share_list,
                                    const std::map<std::string, std::vector<std::vector<uint8_t>>> &record_public_keys,
                                    const std::map<std::string, std::vector<std::vector<uint8_t>>> &client_ivs,
                                    const string &fl_id, std::vector<float> *noise, const uint8_t *secret,
                                    size_t length) {
  for (auto p_key = clients_share_list.begin(); p_key != clients_share_list.end(); ++p_key) {
    if (*p_key != fl_id) {
      PrivateKey *privKey = KeyAgreement::FromPrivateBytes(secret, length);
      if (privKey == NULL) {
        MS_LOG(ERROR) << "create privKey failed\n";
        return false;
      }
      std::vector<uint8_t> public_key = record_public_keys.at(*p_key)[1];
      std::string iv_fl_id;
      if (fl_id < *p_key) {
        iv_fl_id = fl_id;
      } else {
        iv_fl_id = *p_key;
      }
      auto iter = client_ivs.find(iv_fl_id);
      if (iter == client_ivs.end()) {
        MS_LOG(ERROR) << "cannot get ivs for client: " << iv_fl_id;
        delete privKey;
        return false;
      }
      if (iter->second.size() != IV_NUM) {
        MS_LOG(ERROR) << "get " << iter->second.size() << " ivs, the iv num required is: " << IV_NUM;
        return false;
      }
      std::vector<uint8_t> pw_iv = iter->second[PW_IV_INDEX];
      std::vector<uint8_t> pw_salt = iter->second[PW_SALT_INDEX];
      PublicKey *pubKey = KeyAgreement::FromPublicBytes(public_key.data(), public_key.size());
      if (pubKey == NULL) {
        MS_LOG(ERROR) << "create pubKey failed\n";
        return false;
      }
      MS_LOG(INFO) << "private_key fl_id : " << fl_id << " public_key fl_id : " << *p_key;
      uint8_t secret1[SECRET_MAX_LEN] = {0};
      int ret = KeyAgreement::ComputeSharedKey(privKey, pubKey, SECRET_MAX_LEN, pw_salt.data(),
                                               SizeToInt(pw_salt.size()), secret1);
      delete privKey;
      delete pubKey;
      if (ret < 0) {
        MS_LOG(ERROR) << "ComputeSharedKey failed\n";
        return false;
      }

      std::vector<float> noise_tmp;
      if (Masking::GetMasking(&noise_tmp, SizeToInt(cipher_init_->featuremap_), (const uint8_t *)secret1,
                              SECRET_MAX_LEN, pw_iv.data(), pw_iv.size()) < 0) {
        MS_LOG(ERROR) << "Get Masking failed\n";
        return false;
      }
      bool symbol_noise = GetSymbol(fl_id, *p_key);
      size_t index = 0;
      size_t size_noise = noise_tmp.size();
      if (symbol_noise == false) {
        for (; index < size_noise; ++index) {
          noise_tmp[index] = noise_tmp[index] * -1;
          noise->at(index) += noise_tmp[index];
        }
      } else {
        for (; index < size_noise; ++index) {
          noise->at(index) += noise_tmp[index];
        }
      }
    }
  }
  return true;
}

bool CipherReconStruct::GetSymbol(const std::string &str1, const std::string &str2) const {
  if (str1 > str2) {
    return true;
  } else {
    return false;
  }
}

// recombined shares by their source fl_id (ownners)
bool CipherReconStruct::ConvertSharesToShares(const std::map<std::string, std::vector<clientshare_str>> &src,
                                              std::map<std::string, std::vector<clientshare_str>> *des) {
  if (des == nullptr) return false;
  for (auto iter = src.begin(); iter != src.end(); ++iter) {
    std::string des_id = iter->first;
    auto &cur_clientshare_str = iter->second;
    for (size_t index_clientshare = 0; index_clientshare < cur_clientshare_str.size(); ++index_clientshare) {
      std::string src_id = cur_clientshare_str[index_clientshare].fl_id;
      clientshare_str value;
      value.fl_id = des_id;
      value.share = cur_clientshare_str[index_clientshare].share;
      value.index = cur_clientshare_str[index_clientshare].index;
      if (des->find(src_id) == des->end()) {  // src_id is not in recombined shares list
        std::vector<clientshare_str> value_list;
        value_list.push_back(value);
        (void)des->emplace(std::pair<std::string, std::vector<clientshare_str>>(src_id, value_list));
      } else {
        des->at(src_id).push_back(value);
      }
    }
  }
  return true;
}

bool CipherReconStruct::MallocShares(std::vector<Share *> *shares_tmp, size_t shares_size) {
  if (shares_tmp == nullptr) return false;
  for (size_t i = 0; i < shares_size; ++i) {
    Share *share_i = new (std::nothrow) Share();
    if (share_i == nullptr) {
      MS_LOG(ERROR) << "new Share failed.";
      return false;
    }
    share_i->data = new uint8_t[SHARE_MAX_SIZE];
    if (share_i->data == nullptr) {
      MS_LOG(ERROR) << "malloc memory failed.";
      delete share_i;
      return false;
    }
    share_i->index = 0;
    share_i->len = SHARE_MAX_SIZE;
    shares_tmp->push_back(share_i);
  }
  return true;
}

void CipherReconStruct::DeleteShares(std::vector<Share *> *shares_tmp) {
  if (shares_tmp == nullptr) return;
  if (shares_tmp->size() != 0) {
    for (size_t i = 0; i < shares_tmp->size(); ++i) {
      if (shares_tmp->at(i) != nullptr && shares_tmp->at(i)->data != nullptr) {
        delete[](shares_tmp->at(i)->data);
        shares_tmp->at(i)->data = nullptr;
      }
      delete shares_tmp->at(i);
      shares_tmp->at(i) = nullptr;
    }
  }
  return;
}
}  // namespace armour
}  // namespace mindspore
