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

#include "fl/armour/secure_protocol/masking.h"

namespace mindspore {
namespace armour {
#ifdef _WIN32
int Masking::GetMasking(std::vector<float> *noise, int noise_len, const uint8_t *seed, int seed_len,
                        const uint8_t *ivec, int ivec_size) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return -1;
}

#else
int Masking::GetMasking(std::vector<float> *noise, int noise_len, const uint8_t *secret, int secret_len,
                        const uint8_t *ivec, int ivec_size) {
  if ((secret_len != KEY_LENGTH_16 && secret_len != KEY_LENGTH_32) || secret == NULL) {
    MS_LOG(ERROR) << "secret is invalid!";
    return -1;
  }
  if (noise == NULL || noise_len <= 0) {
    MS_LOG(ERROR) << "noise is invalid!";
    return -1;
  }
  if (ivec == NULL || ivec_size != AES_IV_SIZE) {
    MS_LOG(ERROR) << "ivec is invalid!";
    return -1;
  }
  int size = noise_len * sizeof(int);
  std::vector<uint8_t> data(size, 0);
  std::vector<uint8_t> encrypt_data(size, 0);
  int encrypt_len = 0;
  AESEncrypt encrypt(secret, secret_len, ivec, AES_IV_SIZE, AES_CTR);
  if (encrypt.EncryptData(data.data(), size, encrypt_data.data(), &encrypt_len) != 0) {
    MS_LOG(ERROR) << "call AES-CTR failed!";
    return -1;
  }

  for (int i = 0; i < noise_len; i++) {
    auto value = *(reinterpret_cast<int32_t *>(encrypt_data.data()) + i);
    noise->emplace_back(static_cast<float>(value) / INT32_MAX);
  }
  return 0;
}
#endif
}  // namespace armour
}  // namespace mindspore
