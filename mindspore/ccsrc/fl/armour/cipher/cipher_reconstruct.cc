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
#include "fl/armour/secure_protocol/random.h"
#include "fl/armour/secure_protocol/key_agreement.h"
#include "fl/armour/cipher/cipher_meta_storage.h"

namespace mindspore {
namespace armour {
bool CipherReconStruct::CombineMask(
  std::vector<Share *> *shares_tmp, std::map<std::string, std::vector<float>> *client_keys,
  const std::vector<std::string> &clients_share_list,
  const std::map<std::string, std::vector<std::vector<unsigned char>>> &record_public_keys,
  const std::map<std::string, std::vector<clientshare_str>> &reconstruct_secret_list,
  const std::vector<string> &client_list) {
  bool retcode = true;
#ifdef _WIN32
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  retcode = false;
#else
  for (auto iter = reconstruct_secret_list.begin(); iter != reconstruct_secret_list.end(); ++iter) {
    // define flag_share: judge we need b or s
    bool flag_share = true;
    const std::string fl_id = iter->first;
    std::vector<std::string>::const_iterator ptr = client_list.begin();
    for (; ptr < client_list.end(); ++ptr) {
      if (*ptr == fl_id) {
        flag_share = false;
        break;
      }
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
        if (memcpy_s(shares_tmp->at(i)->data, shares_tmp->at(i)->len, (iter->second)[i].share.data(),
                     shares_tmp->at(i)->len) != 0) {
          MS_LOG(ERROR) << "shares_tmp copy failed";
          retcode = false;
        }
        MS_LOG(INFO) << "fl_id_des : " << (iter->second)[i].fl_id;
        std::string print_share_data(reinterpret_cast<const char *>(shares_tmp->at(i)->data), shares_tmp->at(i)->len);
      }
      MS_LOG(INFO) << "end assign secrets shares to public shares ";

      size_t length;
      uint8_t secret[SECRET_MAX_LEN] = {0};
      SecretSharing combine(prime);
      if (combine.Combine(cipher_init_->secrets_minnums_, *shares_tmp, secret, &length) < 0) retcode = false;
      length = SECRET_MAX_LEN;
      MS_LOG(INFO) << "combine secrets shares Success.";

      if (flag_share) {
        MS_LOG(INFO) << "start get complete s_uv.";
        std::vector<float> noise(cipher_init_->featuremap_, 0.0);
        if (GetSuvNoise(clients_share_list, record_public_keys, fl_id, &noise, secret, length) == false)
          retcode = false;
        client_keys->insert(std::pair<std::string, std::vector<float>>(fl_id, noise));
        MS_LOG(INFO) << " fl_id : " << fl_id;
        MS_LOG(INFO) << "end get complete s_uv.";
      } else {
        std::vector<float> noise;
        if (Random::RandomAESCTR(&noise, cipher_init_->featuremap_, (const unsigned char *)secret, SECRET_MAX_LEN) < 0)
          retcode = false;
        for (size_t index_noise = 0; index_noise < cipher_init_->featuremap_; index_noise++) {
          noise[index_noise] *= -1;
        }
        client_keys->insert(std::pair<std::string, std::vector<float>>(fl_id, noise));
        MS_LOG(INFO) << " fl_id : " << fl_id;
      }
    }
  }
#endif
  return retcode;
}

bool CipherReconStruct::ReconstructSecretsGenNoise(const std::vector<string> &client_list) {
  // get reconstruct_secret_list_ori from memory server
  MS_LOG(INFO) << "CipherReconStruct::ReconstructSecretsGenNoise START";
  bool retcode = true;
  std::map<std::string, std::vector<clientshare_str>> reconstruct_secret_list_ori;
  cipher_init_->cipher_meta_storage_.GetClientSharesFromServer(fl::server::kCtxClientsReconstructShares,
                                                               &reconstruct_secret_list_ori);
  std::map<std::string, std::vector<std::vector<unsigned char>>> record_public_keys;
  cipher_init_->cipher_meta_storage_.GetClientKeysFromServer(fl::server::kCtxClientsKeys, &record_public_keys);
  std::vector<std::string> clients_reconstruct_list;
  cipher_init_->cipher_meta_storage_.GetClientListFromServer(fl::server::kCtxReconstructClientList,
                                                             &clients_reconstruct_list);
  std::vector<std::string> clients_share_list;
  cipher_init_->cipher_meta_storage_.GetClientListFromServer(fl::server::kCtxShareSecretsClientList,
                                                             &clients_share_list);
  if (reconstruct_secret_list_ori.size() != clients_reconstruct_list.size() ||
      record_public_keys.size() < cipher_init_->client_num_need_ ||
      clients_share_list.size() < cipher_init_->share_clients_num_need_) {
    MS_LOG(ERROR) << "get data from server memory failed";
    return false;
  }

  std::map<std::string, std::vector<clientshare_str>> reconstruct_secret_list;
  ConvertSharesToShares(reconstruct_secret_list_ori, &reconstruct_secret_list);
  std::vector<Share *> shares_tmp;
  if (MallocShares(&shares_tmp, cipher_init_->secrets_minnums_) == false) {
    MS_LOG(ERROR) << "Reconstruct malloc shares_tmp invalid.";
    return false;
  }
  MS_LOG(INFO) << "Reconstruct client list: ";
  std::vector<std::string>::const_iterator ptr_tmp = client_list.begin();
  for (; ptr_tmp < client_list.end(); ++ptr_tmp) {
    MS_LOG(INFO) << *ptr_tmp;
  }
  MS_LOG(INFO) << "Reconstruct secrets shares: ";
  std::map<std::string, std::vector<float>> client_keys;

  retcode = CombineMask(&shares_tmp, &client_keys, clients_share_list, record_public_keys, reconstruct_secret_list,
                        client_list);

  DeleteShares(&shares_tmp);
  if (retcode) {
    std::vector<float> noise;
    if (GetNoiseMasksSum(&noise, client_keys) == false) {
      MS_LOG(ERROR) << " GetNoiseMasksSum failed";
      return false;
    }
    client_keys.clear();
    MS_LOG(INFO) << " ReconstructSecretsGenNoise updata noise to server";

    if (cipher_init_->cipher_meta_storage_.UpdateClientNoiseToServer(fl::server::kCtxClientNoises, noise) == false)
      return false;
    MS_LOG(INFO) << " ReconstructSecretsGenNoise Success";
  } else {
    MS_LOG(INFO) << " ReconstructSecretsGenNoise failed. because gen noise inside failed";
  }

  return retcode;
}

// reconstruct secrets
bool CipherReconStruct::ReconstructSecrets(
  const int cur_iterator, const std::string &next_req_time, const schema::SendReconstructSecret *reconstruct_secret_req,
  const std::shared_ptr<fl::server::FBBuilder> &reconstruct_secret_resp_builder,
  const std::vector<std::string> &client_list) {
  MS_LOG(INFO) << "CipherReconStruct::ReconstructSecrets START";
  clock_t start_time = clock();
  if (reconstruct_secret_req == nullptr || reconstruct_secret_resp_builder == nullptr) {
    MS_LOG(ERROR) << "Request is nullptr or Response builder is nullptr. ";
    BuildReconstructSecretsRsp(reconstruct_secret_resp_builder, schema::ResponseCode_RequestError,
                               "Request is nullptr or Response builder is nullptr.", cur_iterator, next_req_time);
    return false;
  }
  if (client_list.size() < cipher_init_->reconstruct_clients_num_need_) {
    MS_LOG(ERROR) << "illegal parameters. update model client_list size: " << client_list.size();
    BuildReconstructSecretsRsp(
      reconstruct_secret_resp_builder, schema::ResponseCode_RequestError,
      "illegal parameters: update model client_list size must larger than reconstruct_clients_num_need", cur_iterator,
      next_req_time);
    return false;
  }
  std::vector<std::string> clients_reconstruct_list;
  cipher_init_->cipher_meta_storage_.GetClientListFromServer(fl::server::kCtxReconstructClientList,
                                                             &clients_reconstruct_list);
  std::map<std::string, std::vector<clientshare_str>> clients_shares_all;
  cipher_init_->cipher_meta_storage_.GetClientSharesFromServer(fl::server::kCtxClientsReconstructShares,
                                                               &clients_shares_all);

  size_t count_client_num = clients_shares_all.size();
  if (count_client_num != clients_reconstruct_list.size()) {
    BuildReconstructSecretsRsp(reconstruct_secret_resp_builder, schema::ResponseCode_OutOfTime,
                               "shares client size and client size are not equal.", cur_iterator, next_req_time);
    MS_LOG(ERROR) << "shares client size and client size are not equal.";
    return false;
  }
  int iterator = reconstruct_secret_req->iteration();
  std::string fl_id = reconstruct_secret_req->fl_id()->str();
  if (iterator != cur_iterator) {
    BuildReconstructSecretsRsp(reconstruct_secret_resp_builder, schema::ResponseCode_OutOfTime,
                               "The iteration round of the client does not match the current iteration.", cur_iterator,
                               next_req_time);
    MS_LOG(ERROR) << "Client " << fl_id << " The iteration round of the client does not match the current iteration.";
    return false;
  }

  if (find(client_list.begin(), client_list.end(), fl_id) == client_list.end()) {  // client not in client list.
    BuildReconstructSecretsRsp(reconstruct_secret_resp_builder, schema::ResponseCode_OutOfTime,
                               "The client is not in update model client list.", cur_iterator, next_req_time);
    MS_LOG(ERROR) << "The client " << fl_id << " is not in update model client list.";
    return false;
  }
  if (find(clients_reconstruct_list.begin(), clients_reconstruct_list.end(), fl_id) != clients_reconstruct_list.end()) {
    BuildReconstructSecretsRsp(reconstruct_secret_resp_builder, schema::ResponseCode_SUCCEED,
                               "Client has sended messages.", cur_iterator, next_req_time);
    MS_LOG(INFO) << "Error, client " << fl_id << " has sended messages.";
    return false;
  }
  auto reconstruct_secret_shares = reconstruct_secret_req->reconstruct_secret_shares();
  bool retcode_client =
    cipher_init_->cipher_meta_storage_.UpdateClientToServer(fl::server::kCtxReconstructClientList, fl_id);
  bool retcode_share = cipher_init_->cipher_meta_storage_.UpdateClientShareToServer(
    fl::server::kCtxClientsReconstructShares, fl_id, reconstruct_secret_shares);
  if (!(retcode_client && retcode_share)) {
    BuildReconstructSecretsRsp(reconstruct_secret_resp_builder, schema::ResponseCode_OutOfTime,
                               "reconstruct update shares or client failed.", cur_iterator, next_req_time);
    MS_LOG(ERROR) << "reconstruct update shares or client failed.";
    return false;
  }
  count_client_num = count_client_num + 1;
  if (count_client_num < cipher_init_->reconstruct_clients_num_need_) {
    BuildReconstructSecretsRsp(reconstruct_secret_resp_builder, schema::ResponseCode_SUCCEED,
                               "Success,but the server is not ready to reconstruct secret yet.", cur_iterator,
                               next_req_time);
    MS_LOG(INFO) << "ReconstructSecrets" << fl_id << " Success, but count " << count_client_num << "is not enough.";
    return true;
  } else {
    bool retcode_result = true;
    const fl::PBMetadata &clients_noises_pb_out =
      fl::server::DistributedMetadataStore::GetInstance().GetMetadata(fl::server::kCtxClientNoises);
    const fl::ClientNoises &clients_noises_pb = clients_noises_pb_out.client_noises();
    if (clients_noises_pb.has_one_client_noises() == false) {
      MS_LOG(INFO) << "Success,the secret will be reconstructed.";
      retcode_result = ReconstructSecretsGenNoise(client_list);
      if (retcode_result) {
        BuildReconstructSecretsRsp(reconstruct_secret_resp_builder, schema::ResponseCode_SUCCEED,
                                   "Success,the secret is reconstructing.", cur_iterator, next_req_time);
        MS_LOG(INFO) << "CipherReconStruct::ReconstructSecrets" << fl_id << " Success, reconstruct ok.";
      } else {
        BuildReconstructSecretsRsp(reconstruct_secret_resp_builder, schema::ResponseCode_OutOfTime,
                                   "the secret restructs failed.", cur_iterator, next_req_time);
        MS_LOG(ERROR) << "the secret restructs failed.";
      }
    } else {
      BuildReconstructSecretsRsp(reconstruct_secret_resp_builder, schema::ResponseCode_SUCCEED,
                                 "Clients' number is full.", cur_iterator, next_req_time);
      MS_LOG(INFO) << "CipherReconStruct::ReconstructSecrets" << fl_id << " Success : no need reconstruct.";
    }
    clock_t end_time = clock();
    double duration = static_cast<double>((end_time - start_time) * 1.0 / CLOCKS_PER_SEC);
    MS_LOG(INFO) << "Reconstruct get + gennoise data time is : " << duration;
    return retcode_result;
  }
}

bool CipherReconStruct::GetNoiseMasksSum(std::vector<float> *result,
                                         const std::map<std::string, std::vector<float>> &client_keys) {
  std::vector<float> sum(cipher_init_->featuremap_, 0.0);
  for (auto iter = client_keys.begin(); iter != client_keys.end(); iter++) {
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
  rsp_reconstruct_secret_builder.add_retcode(retcode);
  rsp_reconstruct_secret_builder.add_reason(fbs_reason);
  rsp_reconstruct_secret_builder.add_iteration(iteration);
  rsp_reconstruct_secret_builder.add_next_req_time(fbs_next_req_time);
  auto rsp_reconstruct_secret = rsp_reconstruct_secret_builder.Finish();
  fbb->Finish(rsp_reconstruct_secret);
  return;
}

bool CipherReconStruct::GetSuvNoise(
  const std::vector<std::string> &clients_share_list,
  const std::map<std::string, std::vector<std::vector<unsigned char>>> &record_public_keys, const string &fl_id,
  std::vector<float> *noise, uint8_t *secret, int length) {
  for (auto p_key = clients_share_list.begin(); p_key != clients_share_list.end(); ++p_key) {
    if (*p_key != fl_id) {
      PrivateKey *privKey1 = KeyAgreement::FromPrivateBytes((unsigned char *)secret, length);
      if (privKey1 == NULL) {
        MS_LOG(ERROR) << "create privKey1 failed\n";
        return false;
      }
      std::vector<unsigned char> public_key = record_public_keys.at(*p_key)[1];
      PublicKey *pubKey1 = KeyAgreement::FromPublicBytes(public_key.data(), public_key.size());
      if (pubKey1 == NULL) {
        MS_LOG(ERROR) << "create pubKey1 failed\n";
        return false;
      }
      MS_LOG(INFO) << "fl_id : " << fl_id << "other id : " << *p_key;
      unsigned char secret1[SECRET_MAX_LEN] = {0};
      unsigned char salt[SECRET_MAX_LEN] = {0};
      if (KeyAgreement::ComputeSharedKey(privKey1, pubKey1, SECRET_MAX_LEN, salt, SECRET_MAX_LEN, secret1) < 0) {
        MS_LOG(ERROR) << "ComputeSharedKey failed\n";
        return false;
      }

      std::vector<float> noise_tmp;
      if (Random::RandomAESCTR(&noise_tmp, cipher_init_->featuremap_, (const unsigned char *)secret1, SECRET_MAX_LEN) <
          0) {
        MS_LOG(ERROR) << "RandomAESCTR failed\n";
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
      for (int i = 0; i < 5; i++) {
        MS_LOG(INFO) << "index " << i << " : " << noise_tmp[i];
      }
    }
  }
  return true;
}

bool CipherReconStruct::GetSymbol(const std::string &str1, const std::string &str2) {
  if (str1 > str2) {
    return true;
  } else {
    return false;
  }
}

void CipherReconStruct::ConvertSharesToShares(const std::map<std::string, std::vector<clientshare_str>> &src,
                                              std::map<std::string, std::vector<clientshare_str>> *des) {
  for (auto iter_ori = src.begin(); iter_ori != src.end(); ++iter_ori) {
    std::string fl_des = iter_ori->first;
    auto &cur_clientshare_str = iter_ori->second;
    for (size_t index_clientshare = 0; index_clientshare < cur_clientshare_str.size(); ++index_clientshare) {
      std::string fl_src = cur_clientshare_str[index_clientshare].fl_id;
      clientshare_str value;
      value.fl_id = fl_des;
      value.share = cur_clientshare_str[index_clientshare].share;
      value.index = cur_clientshare_str[index_clientshare].index;
      if (des->find(fl_src) == des->end()) {  // fl_id_des is not in reconstruct_secret_list_
        std::vector<clientshare_str> value_list;
        value_list.push_back(value);
        des->insert(std::pair<std::string, std::vector<clientshare_str>>(fl_src, value_list));
      } else {  // fl_id_des is in reconstruct_secret_list_
        des->at(fl_src).push_back(value);
      }
    }
  }
}

bool CipherReconStruct::MallocShares(std::vector<Share *> *shares_tmp, int shares_size) {
  for (int i = 0; i < shares_size; ++i) {
    Share *share_i = new Share;
    if (share_i == nullptr) {
      MS_LOG(ERROR) << "shares_tmp " << i << " memory to cipher is invalid.";
      DeleteShares(shares_tmp);
      return false;
    }
    share_i->data = new unsigned char[SHARE_MAX_SIZE];
    if (share_i->data == nullptr) {
      MS_LOG(ERROR) << "shares_tmp's data " << i << " memory to cipher is invalid.";
      DeleteShares(shares_tmp);
      return false;
    }
    share_i->index = 0;
    share_i->len = SHARE_MAX_SIZE;
    shares_tmp->push_back(share_i);
  }
  return true;
}

void CipherReconStruct::DeleteShares(std::vector<Share *> *shares_tmp) {
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
