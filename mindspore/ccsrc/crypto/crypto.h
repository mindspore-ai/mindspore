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

#ifndef MINDSPORE_CCSRC_CRYPTO_CRYPTO_H
#define MINDSPORE_CCSRC_CRYPTO_CRYPTO_H

#if not defined(_WIN32)
#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#endif

#include <stdio.h>
#include <fstream>
#include <string>
#include <regex>
#include "utils/log_adapter.h"

typedef unsigned char Byte;

namespace mindspore {
namespace crypto {
const int MAX_BLOCK_SIZE = 512 * 1024 * 1024;  // Maximum ciphertext segment, units is Byte
const unsigned int MAGIC_NUM = 0x7F3A5ED8;     // Magic number

Byte *Encrypt(int64_t *encrypt_len, Byte *plain_data, const int64_t plain_len, Byte *key, const int32_t key_len,
              const std::string &enc_mode);
Byte *Decrypt(int64_t *decrypt_len, const std::string &encrypt_data_path, Byte *key, const int32_t key_len,
              const std::string &dec_mode);
bool IsCipherFile(const std::string file_path);
}  // namespace crypto
}  // namespace mindspore
#endif
