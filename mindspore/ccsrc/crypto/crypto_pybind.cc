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

#include "crypto/crypto_pybind.h"
namespace mindspore {
namespace crypto {
py::bytes PyEncrypt(char *plain_data, const int64_t plain_len, char *key, const int32_t key_len, std::string enc_mode) {
  int64_t encrypt_len;
  char *encrypt_data;
  encrypt_data = reinterpret_cast<char *>(Encrypt(&encrypt_len, reinterpret_cast<Byte *>(plain_data), plain_len,
                                                  reinterpret_cast<Byte *>(key), key_len, enc_mode));
  return py::bytes(encrypt_data, encrypt_len);
}

py::bytes PyDecrypt(std::string encrypt_data_path, char *key, const int32_t key_len, std::string dec_mode) {
  int64_t decrypt_len;
  char *decrypt_data;
  decrypt_data = reinterpret_cast<char *>(
    Decrypt(&decrypt_len, encrypt_data_path, reinterpret_cast<Byte *>(key), key_len, dec_mode));
  return py::bytes(decrypt_data, decrypt_len);
}
bool PyIsCipherFile(std::string file_path) { return IsCipherFile(file_path); }
}  // namespace crypto
}  // namespace mindspore
