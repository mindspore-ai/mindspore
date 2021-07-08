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

#ifndef MINDSPORE_SECRET_SHARING_H
#define MINDSPORE_SECRET_SHARING_H
#ifndef _WIN32
#include <gmp.h>
#include "openssl/rand.h"
#endif
#include <string>
#include <vector>
#include "utils/log_adapter.h"

namespace mindspore {
namespace armour {
#define SECRET_MAX_LEN 32
#define PRIME_MAX_LEN 33

struct Share {
  unsigned int index;
  unsigned char *data;
  size_t len;
  ~Share();
};

#ifndef _WIN32
void secure_zero(void *s, size_t);
int GetRandInteger(mpz_t x, mpz_t prim);
int GetRandomPrime(mpz_t prim);
void PrintBigInteger(mpz_t x);
void PrintBigInteger(mpz_t x, int hex);

class SecretSharing {
 public:
  explicit SecretSharing(mpz_t prim);
  ~SecretSharing();
  // split the input secret into multiple shares
  int Split(int n, const int k, const char *secret, size_t length, const std::vector<Share *> &shares);
  // reconstruct the secret from multiple shares
  int Combine(int k, const std::vector<Share *> &shares, char *secret, size_t *length);

 private:
  mpz_t prim_;
  size_t degree_;
  // calculate shares from a polynomial
  int CalculateShares(const mpz_t coeff[], int k, int n, const std::vector<Share *> &shares);
  // inversion in finite field
  void field_invert(mpz_t z, const mpz_t x);
  // addition in finite field
  void field_add(mpz_t z, const mpz_t x, const mpz_t y);
  // multiplication in finite field
  void field_mult(mpz_t z, const mpz_t x, const mpz_t y);
  // evaluate polynomial at x
  void GetPolyVal(int k, mpz_t y, const mpz_t x, const mpz_t coeff[]);
  // convert secret sharing from Share type to mpz_t type
  void GetShare(mpz_t x, mpz_t share, Share *s_share);
};
#endif

}  // namespace armour
}  // namespace mindspore
#endif  // MINDSPORE_SECRET_SHARING_H
