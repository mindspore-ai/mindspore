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

#include "armour/secure_protocol/random.h"

namespace mindspore {
namespace armour {
Random::Random(size_t init_seed) { generator.seed(init_seed); }

Random::~Random() {}

void Random::RandUniform(float *array, int size) {
  std::uniform_real_distribution<double> rand(0, 1);
  for (int i = 0; i < size; i++) {
    *(reinterpret_cast<float *>(array) + i) = rand(generator);
  }
}

void Random::RandNorminal(float *array, int size) {
  std::normal_distribution<double> randn(0, 1);
  for (int i = 0; i < size; i++) {
    *(reinterpret_cast<float *>(array) + i) = randn(generator);
  }
}

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
  int retval = RAND_priv_bytes(secret, RANDOM_LEN);
  return retval;
}

int Random::RandomAESCTR(std::vector<float> *noise, int noise_len, const unsigned char *seed, int seed_len) {
  if (seed_len != 16 && seed_len != 32) {
    std::cout << "seed length must be 16 or 32!" << std::endl;
    return -1;
  }
  int size = noise_len * sizeof(int);
  unsigned char data[size];
  unsigned char encrypt_data[size];
  for (int i = 0; i < size; i++) {
    data[i] = 0;
    encrypt_data[i] = 0;
  }
  unsigned char ivec[INIT_VEC_SIZE];
  for (size_t i = 0; i < INIT_VEC_SIZE; i++) {
    ivec[i] = 0;
  }
  int encrypt_len;
  AESEncrypt encrypt(seed, seed_len, ivec, INIT_VEC_SIZE, AES_CTR);
  if (encrypt.EncryptData(data, size, encrypt_data, &encrypt_len) != 0) {
    std::cout << "call encryptData fail!" << std::endl;
    return -1;
  }

  for (int i = 0; i < noise_len; i++) {
    auto value = *(reinterpret_cast<int32_t *>(encrypt_data) + i);
    noise->emplace_back(static_cast<float>(value) / INT32_MAX);
  }
  return 0;
}
#endif

}  // namespace armour
}  // namespace mindspore
