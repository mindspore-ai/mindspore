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

#ifndef MINDSPORE_CCSRC_ARMOUR_CIPHER_INIT_H
#define MINDSPORE_CCSRC_ARMOUR_CIPHER_INIT_H

#include <vector>
#include <string>
#include "armour/secure_protocol/secret_sharing.h"
#include "proto/ps.pb.h"
#include "utils/log_adapter.h"
#include "schema/fl_job_generated.h"
#include "schema/cipher_generated.h"
#include "armour/cipher/cipher_meta_storage.h"

namespace mindspore {
namespace armour {

template <typename T1>
bool CreateArray(std::vector<T1> *newData, const flatbuffers::Vector<T1> &fbs_arr) {
  size_t size = newData->size();
  size_t size_fbs_arr = fbs_arr.size();
  if (size != size_fbs_arr) return false;
  for (size_t i = 0; i < size; ++i) {
    newData->at(i) = fbs_arr.Get(i);
  }
  return true;
}

// Initialization of secure aggregation.
class CipherInit {
 public:
  static CipherInit &GetInstance() {
    static CipherInit instance;
    return instance;
  }

  // Initialize the parameters of the secure aggregation.
  bool Init(const CipherPublicPara &param, size_t time_out_mutex, size_t cipher_initial_client_cnt,
            size_t cipher_exchange_secrets_cnt, size_t cipher_share_secrets_cnt, size_t cipher_get_clientlist_cnt,
            size_t cipher_reconstruct_secrets_down_cnt, size_t cipher_reconstruct_secrets_up_cnt);

  // Check whether the parameters are valid.
  bool Check_Parames();

  // Get public params. which is given to start fl job thread.
  CipherPublicPara *GetPublicParams() { return &publicparam_; }

  size_t share_clients_num_need_;        // the minimum number of clients to share secrets.
  size_t reconstruct_clients_num_need_;  // the minimum number of clients to reconstruct secret mask.
  size_t client_num_need_;               // the minimum number of clients to update model.
  size_t get_model_num_need_;            // the minimum number of clients to get model.

  size_t secrets_minnums_;  // the minimum number of secret fragment s to reconstruct secret mask.
  size_t featuremap_;       // the size of data to deal.
  size_t time_out_mutex_;   // timeout mutex.

  CipherPublicPara publicparam_;  // the param containing encrypted public parameters.
  CipherMetaStorage cipher_meta_storage_;
};
}  // namespace armour
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_ARMOUR_CIPHER_COMMON_H
