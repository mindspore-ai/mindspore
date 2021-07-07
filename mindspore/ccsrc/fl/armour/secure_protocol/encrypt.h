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

#ifndef MINDSPORE_ARMOUR_ENCRYPT_H
#define MINDSPORE_ARMOUR_ENCRYPT_H

#ifndef _WIN32
#include <openssl/evp.h>
#endif
#include "utils/log_adapter.h"

#define INIT_VEC_SIZE 16

namespace mindspore {
namespace armour {
class Encrypt {};
enum AES_MODE {
  AES_CBC = 0,
  AES_CTR = 1,
};
class SymmetricEncrypt : Encrypt {};

class AESEncrypt : SymmetricEncrypt {
 public:
  AESEncrypt(const unsigned char *key, int key_len, unsigned char *ivec, int ivec_len, AES_MODE mode);
  ~AESEncrypt();
  int EncryptData(const unsigned char *data, const int len, unsigned char *encrypt_data, int *encrypt_len);
  int DecryptData(const unsigned char *encrypt_data, const int encrypt_len, unsigned char *data, int *len);

 private:
  const unsigned char *privKey;
  int privKeyLen;
  unsigned char *iVec;
  int iVecLen;
  AES_MODE aesMode;
  int evp_aes_encrypt(const unsigned char *data, const int len, const unsigned char *key, unsigned char *ivec,
                      unsigned char *encrypt_data, int *encrypt_len);
  int evp_aes_decrypt(const unsigned char *encrypt_data, const int len, const unsigned char *key, unsigned char *ivec,
                      unsigned char *decrypt_data, int *decrypt_len);
};

}  // namespace armour
}  // namespace mindspore
#endif  // MINDSPORE_ARMOUR_ENCRYPT_H
