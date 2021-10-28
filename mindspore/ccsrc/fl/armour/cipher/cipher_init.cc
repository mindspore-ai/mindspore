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

#include "fl/armour/cipher/cipher_init.h"

#include "fl/armour/cipher/cipher_meta_storage.h"
#include "fl/server/common.h"
#include "fl/server/model_store.h"

namespace mindspore {
namespace armour {
bool CipherInit::Init(const CipherPublicPara &param, size_t time_out_mutex, size_t cipher_exchange_keys_cnt,
                      size_t cipher_get_keys_cnt, size_t cipher_share_secrets_cnt, size_t cipher_get_secrets_cnt,
                      size_t cipher_get_clientlist_cnt, size_t cipher_reconstruct_secrets_up_cnt) {
  MS_LOG(INFO) << "CipherInit::Init START";
  if (publicparam_.p == nullptr || param.p == nullptr || param.prime == nullptr || publicparam_.prime == nullptr) {
    MS_LOG(ERROR) << "CipherInit::input data invalid.";
    return false;
  }
  if (memcpy_s(publicparam_.p, SECRET_MAX_LEN, param.p, sizeof(param.p)) != 0) {
    MS_LOG(ERROR) << "CipherInit::memory copy failed.";
    return false;
  }

  publicparam_.g = param.g;
  publicparam_.t = param.t;
  secrets_minnums_ = IntToSize(param.t);
  featuremap_ = fl::server::ModelStore::GetInstance().model_size() / sizeof(float);

  exchange_key_threshold = cipher_exchange_keys_cnt;
  get_key_threshold = cipher_get_keys_cnt;
  share_secrets_threshold = cipher_share_secrets_cnt;
  get_secrets_threshold = cipher_get_secrets_cnt;
  client_list_threshold = cipher_get_clientlist_cnt;
  reconstruct_secrets_threshold = cipher_reconstruct_secrets_up_cnt;

  time_out_mutex_ = time_out_mutex;
  publicparam_.dp_eps = param.dp_eps;
  publicparam_.dp_delta = param.dp_delta;
  publicparam_.dp_norm_clip = param.dp_norm_clip;
  publicparam_.encrypt_type = param.encrypt_type;

  if (param.encrypt_type == mindspore::ps::kDPEncryptType) {
    MS_LOG(INFO) << "DP parameters init, dp_eps: " << param.dp_eps;
    MS_LOG(INFO) << "DP parameters init, dp_delta: " << param.dp_delta;
    MS_LOG(INFO) << "DP parameters init, dp_norm_clip: " << param.dp_norm_clip;
  }

  if (param.encrypt_type == mindspore::ps::kPWEncryptType) {
    cipher_meta_storage_.RegisterClass();
    const std::string new_prime(reinterpret_cast<const char *>(param.prime), PRIME_MAX_LEN);
    cipher_meta_storage_.RegisterPrime(fl::server::kCtxCipherPrimer, new_prime);
    if (!cipher_meta_storage_.GetPrimeFromServer(fl::server::kCtxCipherPrimer, publicparam_.prime)) {
      MS_LOG(ERROR) << "Cipher Param Update is invalid.";
      return false;
    }
    MS_LOG(INFO) << " CipherInit exchange_key_threshold : " << exchange_key_threshold;
    MS_LOG(INFO) << " CipherInit get_key_threshold : " << get_key_threshold;
    MS_LOG(INFO) << " CipherInit share_secrets_threshold : " << share_secrets_threshold;
    MS_LOG(INFO) << " CipherInit get_secrets_threshold : " << get_secrets_threshold;
    MS_LOG(INFO) << " CipherInit client_list_threshold : " << client_list_threshold;
    MS_LOG(INFO) << " CipherInit reconstruct_secrets_threshold : " << reconstruct_secrets_threshold;
    MS_LOG(INFO) << " CipherInit featuremap_ : " << featuremap_;
    if (!Check_Parames()) {
      MS_LOG(ERROR) << "Cipher parameters are illegal.";
      return false;
    }
    MS_LOG(INFO) << " CipherInit::Init Success";
  }
  return true;
}

bool CipherInit::Check_Parames() {
  if (featuremap_ < 1) {
    MS_LOG(ERROR) << "Featuremap size should be positive, but got " << featuremap_;
    return false;
  }

  if (share_secrets_threshold < reconstruct_secrets_threshold) {
    MS_LOG(ERROR) << "reconstruct_secrets_threshold should not be larger "
                     "than share_secrets_threshold, but got they are:"
                  << reconstruct_secrets_threshold << ", " << share_secrets_threshold;
    return false;
  }

  return true;
}
}  // namespace armour
}  // namespace mindspore
