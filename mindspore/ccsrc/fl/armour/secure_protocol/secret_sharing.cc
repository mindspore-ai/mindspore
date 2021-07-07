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

#include "fl/armour/secure_protocol/secret_sharing.h"

namespace mindspore {
namespace armour {
void secure_zero(unsigned char *s, size_t n) {
  volatile unsigned char *p = s;
  if (p)
    while (n--) *p++ = '\0';
}

#ifndef _WIN32
int GetRandInteger(mpz_t x, mpz_t prim) {
  size_t bytes_len = (mpz_sizeinbase(prim, 2) + 8 - 1) / 8;
  unsigned char buf[bytes_len];
  while (true) {
    if (!RAND_bytes(buf, bytes_len)) {
      MS_LOG(WARNING) << "Get Rand Integer failed!";
      continue;
    }
    mpz_import(x, bytes_len, 1, 1, 0, 0, buf);
    secure_zero(buf, sizeof(buf));
    if (mpz_cmp_ui(x, 0) > 0 && mpz_cmp(x, prim) < 0) {
      return 0;
    }
  }
}

int GetRandomPrime(mpz_t prim) {
  mpz_t rand;
  mpz_init(rand);
  const int max_prime_len = SECRET_MAX_LEN + 1;
  unsigned char buf[max_prime_len];
  if (!RAND_bytes(buf, max_prime_len)) {
    MS_LOG(ERROR) << "Get Rand Integer failed!";
    return -1;
  }
  mpz_import(rand, max_prime_len, 1, 1, 0, 0, buf);
  mpz_nextprime(prim, rand);
  mpz_clear(rand);
  secure_zero(buf, sizeof(buf));
  return 0;
}

void PrintBigInteger(mpz_t x) {
  char *tmp = mpz_get_str(NULL, 16, x);
  std::string Str = tmp;
  MS_LOG(INFO) << Str;
  void (*freefunc)(void *, size_t);
  mp_get_memory_functions(NULL, NULL, &freefunc);
  freefunc(tmp, strlen(tmp) + 1);
}

void PrintBigInteger(mpz_t x, int hex) {
  char *tmp = mpz_get_str(NULL, hex, x);
  std::string Str = tmp;

  MS_LOG(INFO) << Str;
  void (*freefunc)(void *, size_t);
  mp_get_memory_functions(NULL, NULL, &freefunc);
  freefunc(tmp, strlen(tmp) + 1);
}

Share::~Share() {
  if (this->data != nullptr) free(this->data);
}

SecretSharing::SecretSharing(mpz_t prim) {
  mpz_init(this->prim_);
  mpz_set(this->prim_, prim);
}

SecretSharing::~SecretSharing() { mpz_clear(this->prim_); }

void SecretSharing::GetPolyVal(int k, mpz_t y, const mpz_t x, const mpz_t coeff[]) {
  int i;
  mpz_set_ui(y, 0);
  for (i = k - 1; i >= 0; i--) {
    field_mult(y, y, x);
    field_add(y, y, coeff[i]);
  }
}

void SecretSharing::field_invert(mpz_t z, const mpz_t x) { mpz_invert(z, x, this->prim_); }

void SecretSharing::field_add(mpz_t z, const mpz_t x, const mpz_t y) {
  mpz_add(z, x, y);
  mpz_mod(z, z, this->prim_);
}

void SecretSharing::field_mult(mpz_t z, const mpz_t x, const mpz_t y) {
  mpz_mul(z, x, y);
  mpz_mod(z, z, this->prim_);
}

int SecretSharing::CalculateShares(const mpz_t coeff[], int k, int n, const std::vector<Share *> &shares) {
  mpz_t x, y;
  mpz_init(x);
  mpz_init(y);
  for (int i = 0; i < n; i++) {
    mpz_set_ui(x, i + 1);
    GetPolyVal(k, y, x, coeff);
    shares[i]->index = i + 1;
    size_t share_len = (mpz_sizeinbase(y, 2) + 8 - 1) / 8;
    shares[i]->data = (unsigned char *)malloc(share_len + 1);
    mpz_export(shares[i]->data, &(shares[i]->len), 1, 1, 0, 0, y);
    if (shares[i]->len != share_len) {
      MS_LOG(ERROR) << "share_len is not equal";
      return -1;
    }
    MS_LOG(INFO) << "share_" << i + 1 << ": ";
    PrintBigInteger(y);
  }
  mpz_clear(x);
  mpz_clear(y);
  return 0;
}

int SecretSharing::Split(int n, const int k, const char *secret, const size_t length,
                         const std::vector<Share *> &shares) {
  if (k <= 1 || k > n) {
    MS_LOG(ERROR) << "invalid parameters";
    return -1;
  }
  if (static_cast<int>(shares.size()) != n) {
    MS_LOG(ERROR) << "the size of shares must be equal to n";
    return -1;
  }
  this->degree_ = length * 8;
  const int kCoeffLen = k;
  mpz_t coeff[kCoeffLen];
  int ret = 0;
  int i = 0;
  mpz_init(coeff[i]);
  mpz_import(coeff[i], length, 1, 1, 0, 0, secret);
  i++;
  for (; i < k && ret == 0; i++) {
    mpz_init(coeff[i]);
    ret = GetRandInteger(coeff[i], this->prim_);
    if (ret != 0) {
      break;
    }
    MS_LOG(INFO) << "coeff_" << i << ":";
    PrintBigInteger(coeff[i]);
  }
  if (ret == 0) {
    ret = CalculateShares(coeff, k, n, shares);
  }
  for (i = 0; i < k; i++) mpz_clear(coeff[i]);
  return ret;
}

void SecretSharing::GetShare(mpz_t x, mpz_t share, Share *s_share) {
  mpz_set_ui(x, s_share->index);
  mpz_import(share, s_share->len, 1, 1, 0, 0, s_share->data);
}

int SecretSharing::Combine(int k, const std::vector<Share *> &shares, char *secret, size_t *length) {
  int ret = 0;
  mpz_t y[k], x[k], denses[k], nums[k];
  int i, j, m;

  for (i = 0; i < k; i++) {
    mpz_init(x[i]);
    mpz_init(y[i]);
    mpz_init(denses[i]);
    mpz_init(nums[i]);
    GetShare(x[i], y[i], shares[i]);
    MS_LOG(INFO) << "combine -- share_" << mpz_get_str(NULL, 10, x[i]) << ": ";
    PrintBigInteger(y[i]);
    MS_LOG(INFO) << "index is : " << shares[i]->index;
    MS_LOG(INFO) << "len is %zu " << shares[i]->len;
  }

  mpz_t sum;
  mpz_init(sum);
  mpz_set_ui(sum, 0);
  for (j = 0; j < k; j++) {
    mpz_set_ui(denses[j], 1);
    mpz_set_ui(nums[j], 1);
    mpz_t tmp;
    mpz_init(tmp);
    for (m = 0; m < k; m++) {
      if (m != j) {
        field_mult(nums[j], nums[j], x[m]);
        mpz_mul_si(tmp, x[j], -1);
        field_add(tmp, x[m], tmp);
        field_mult(denses[j], denses[j], tmp);
      }
    }
    field_invert(tmp, denses[j]);

    field_mult(tmp, tmp, nums[j]);
    field_mult(tmp, tmp, y[j]);
    field_add(sum, sum, tmp);
    mpz_clear(tmp);
  }

  mpz_export(secret, length, 1, 1, 0, 0, sum);
  PrintBigInteger(sum);
  mpz_clear(sum);
  for (i = 0; i < k; i++) {
    mpz_clear(x[i]);
    mpz_clear(y[i]);
    mpz_clear(nums[i]);
    mpz_clear(denses[i]);
  }
  return ret;
}
#endif
}  // namespace armour
}  // namespace mindspore
