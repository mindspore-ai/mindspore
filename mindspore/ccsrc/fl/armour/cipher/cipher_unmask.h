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

#ifndef MINDSPORE_CCSRC_ARMOUR_CIPHER_UNMASK_H
#define MINDSPORE_CCSRC_ARMOUR_CIPHER_UNMASK_H

#include <vector>
#include <string>
#include <map>
#include "fl/armour/secure_protocol/secret_sharing.h"
#include "proto/ps.pb.h"
#include "utils/log_adapter.h"
#include "fl/armour/cipher/cipher_init.h"
#include "fl/armour/cipher/cipher_meta_storage.h"

namespace mindspore {
namespace armour {

class CipherUnmask {
 public:
  // initialize: get cipher_init_
  CipherUnmask() { cipher_init_ = &CipherInit::GetInstance(); }
  ~CipherUnmask() = default;
  // unmask the data by secret mask.
  bool UnMask(const std::map<std::string, AddressPtr> &data);

 private:
  CipherInit *cipher_init_;  // the parameter of the secure aggregation
};
}  // namespace armour
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_ARMOUR_CIPHER_UNMASK_H
