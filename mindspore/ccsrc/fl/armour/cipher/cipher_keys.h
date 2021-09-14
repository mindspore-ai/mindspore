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

#ifndef MINDSPORE_CCSRC_ARMOUR_CIPHER_KEYS_H
#define MINDSPORE_CCSRC_ARMOUR_CIPHER_KEYS_H

#include <vector>
#include <string>
#include <memory>
#include <map>
#include "fl/armour/secure_protocol/secret_sharing.h"
#include "proto/ps.pb.h"
#include "utils/log_adapter.h"
#include "fl/armour/cipher/cipher_init.h"
#include "fl/armour/cipher/cipher_meta_storage.h"
#include "fl/server/common.h"

namespace mindspore {
namespace armour {

// The process of exchange keys and get keys in the secure aggregation
class CipherKeys {
 public:
  // initialize: get cipher_init_
  CipherKeys() { cipher_init_ = &CipherInit::GetInstance(); }
  ~CipherKeys() = default;

  static CipherKeys &GetInstance() {
    static CipherKeys instance;
    return instance;
  }

  // handle the client's request of get keys.
  bool GetKeys(const size_t cur_iterator, const std::string &next_req_time,
               const schema::GetExchangeKeys *get_exchange_keys_req, const std::shared_ptr<fl::server::FBBuilder> &fbb);

  // handle the client's request of exchange keys.
  bool ExchangeKeys(const size_t cur_iterator, const std::string &next_req_time,
                    const schema::RequestExchangeKeys *exchange_keys_req,
                    const std::shared_ptr<fl::server::FBBuilder> &fbb);

  // build response code of get keys.
  void BuildGetKeysRsp(const std::shared_ptr<fl::server::FBBuilder> &fbb, const schema::ResponseCode retcode,
                       const size_t iteration, const std::string &next_req_time, bool is_good);
  // build response code of exchange keys.
  void BuildExchangeKeysRsp(const std::shared_ptr<fl::server::FBBuilder> &fbb, const schema::ResponseCode retcode,
                            const std::string &reason, const std::string &next_req_time, const size_t iteration);
  // clear the shared memory.
  void ClearKeys();

 private:
  CipherInit *cipher_init_;  // the parameter of the secure aggregation
};
}  // namespace armour
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_ARMOUR_CIPHER_KEYS_H
