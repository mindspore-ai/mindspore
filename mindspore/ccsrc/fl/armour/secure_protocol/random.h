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

#ifndef MINDSPORE_ARMOUR_RANDOM_H
#define MINDSPORE_ARMOUR_RANDOM_H

#include <random>
#include <vector>
#ifndef _WIN32
#include <openssl/rand.h>
#endif
#include "fl/armour/secure_protocol/encrypt.h"

namespace mindspore {
namespace armour {

#define RANDOM_LEN 8

class Random {
 public:
  explicit Random(size_t init_seed);
  ~Random();
  // use openssl RAND_priv_bytes
  static int GetRandomBytes(unsigned char *secret, int num_bytes);

  static int RandomAESCTR(std::vector<float> *noise, int noise_len, const unsigned char *seed, int seed_len);

 private:
  std::default_random_engine generator;
};
}  // namespace armour
}  // namespace mindspore
#endif  // MINDSPORE_ARMOUR_RANDOM_H
