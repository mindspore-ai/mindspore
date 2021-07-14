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

#include "fl/armour/secure_protocol/random.h"

namespace mindspore {
namespace armour {
Random::Random(size_t init_seed) { generator.seed(init_seed); }

Random::~Random() {}

#ifdef _WIN32
int Random::GetRandomBytes(unsigned char *secret, int num_bytes) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return -1;
}

int Random::RandomAESCTR(std::vector<float> *noise, int noise_len, const unsigned char *seed, int seed_len) {
  MS_LOG(ERROR) << "Unsupported feature in Windows platform.";
  return -1;
}

#else
int Random::GetRandomBytes(unsigned char *secret, int num_bytes) {
  int retval = RAND_priv_bytes(secret, num_bytes);
  return retval;
}

int Random::RandomAESCTR(std::vector<float> *noise, int noise_len, const unsigned char *seed, int seed_len) {
  if (seed_len != 16 && seed_len != 32) {
    MS_LOG(ERROR) << "seed length must be 16 or 32!";
    return -1;
  }
  int size = noise_len * sizeof(int);
  std::vector<unsigned char> data(size, 0);
  std::vector<unsigned char> encrypt_data(size, 0);
  std::vector<unsigned char> ivec(INIT_VEC_SIZE, 0);
  int encrypt_len = 0;
  AESEncrypt encrypt(seed, seed_len, ivec.data(), INIT_VEC_SIZE, AES_CTR);
  if (encrypt.EncryptData(data.data(), size, encrypt_data.data(), &encrypt_len) != 0) {
    MS_LOG(ERROR) << "call encryptData fail!";
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
