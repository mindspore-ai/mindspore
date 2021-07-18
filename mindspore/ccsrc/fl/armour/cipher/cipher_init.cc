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

bool CipherInit::Init(const CipherPublicPara &param, size_t time_out_mutex, size_t cipher_initial_client_cnt,
                      size_t cipher_exchange_secrets_cnt, size_t cipher_share_secrets_cnt,
                      size_t cipher_get_clientlist_cnt, size_t cipher_reconstruct_secrets_down_cnt,
                      size_t cipher_reconstruct_secrets_up_cnt) {
  MS_LOG(INFO) << "CipherInit::Init START";
  int return_num = 0;
  return_num = memcpy_s(publicparam_.p, SECRET_MAX_LEN, param.p, SECRET_MAX_LEN);
  if (return_num != 0) {
    return false;
  }

  publicparam_.g = param.g;
  publicparam_.t = param.t;
  secrets_minnums_ = param.t;
  client_num_need_ = cipher_initial_client_cnt;
  featuremap_ = fl::server::ModelStore::GetInstance().model_size() / sizeof(float);
  share_clients_num_need_ = cipher_share_secrets_cnt;
  reconstruct_clients_num_need_ = cipher_reconstruct_secrets_down_cnt + 1;
  get_model_num_need_ = cipher_get_clientlist_cnt;
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
    MS_LOG(INFO) << " CipherInit client_num_need_ : " << client_num_need_;
    MS_LOG(INFO) << " CipherInit share_clients_num_need_ : " << share_clients_num_need_;
    MS_LOG(INFO) << " CipherInit reconstruct_clients_num_need_ : " << reconstruct_clients_num_need_;
    MS_LOG(INFO) << " CipherInit get_model_num_need_ : " << get_model_num_need_;
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

  if (share_clients_num_need_ < reconstruct_clients_num_need_) {
    MS_LOG(ERROR)
      << "reconstruct_clients_num_need (which is reconstruct_secrets_threshold + 1) should not be larger "
         "than share_clients_num_need (which is start_fl_job_threshold*share_secrets_ratio), but got they are:"
      << reconstruct_clients_num_need_ << ", " << share_clients_num_need_;
    return false;
  }

  return true;
}

}  // namespace armour
}  // namespace mindspore
