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
void secure_zero(uint8_t *s, size_t n) {
  volatile uint8_t *p = s;
  if (p != nullptr)
    while (n--) *p++ = '\0';
}

#ifndef _WIN32
int GetPrime(BIGNUM *prim) {
  constexpr int byteBits = 8;
  const int max_prime_len = SECRET_MAX_LEN + 1;
  const int maxCount = 500;
  int count = 1;
  int ret = 0;
  while (count < maxCount) {
    ret = BN_generate_prime_ex(prim, max_prime_len * byteBits, 1, NULL, NULL, NULL);
    if (ret == 1) {
      break;
    }
    count++;
  }
  if (ret != 1 || BN_num_bytes(prim) != max_prime_len) {
    MS_LOG(ERROR) << "Get prim failed, get count: " << count;
    MS_LOG(ERROR) << "BN_num_bytes: " << BN_num_bytes(prim) << ", max_prime_len: " << max_prime_len;
    return -1;
  }
  MS_LOG(INFO) << "Get prim success, get count: " << count;
  return 0;
}

Share::~Share() {
  if (this->data != nullptr) free(this->data);
}

SecretSharing::SecretSharing(BIGNUM *prim) {
  if (prim != nullptr) {
    this->bn_prim_ = BN_dup(prim);
  } else {
    this->bn_prim_ = nullptr;
  }
}

SecretSharing::~SecretSharing() {
  if (this->bn_prim_ != nullptr) {
    BN_clear_free(this->bn_prim_);
  }
}

bool SecretSharing::field_mult(BIGNUM *z, const BIGNUM *x, const BIGNUM *y, BN_CTX *ctx) {
  if (BN_mul(z, x, y, ctx) != 1) {
    return false;
  }
  if (BN_mod(z, z, this->bn_prim_, ctx) != 1) {
    return false;
  }
  return true;
}

bool SecretSharing::field_add(BIGNUM *z, const BIGNUM *x, const BIGNUM *y, BN_CTX *ctx) {
  if (BN_add(z, x, y) != 1) {
    return false;
  }
  if (BN_mod(z, z, this->bn_prim_, ctx) != 1) {
    return false;
  }
  return true;
}

bool SecretSharing::field_sub(BIGNUM *z, const BIGNUM *x, const BIGNUM *y, BN_CTX *ctx) {
  if (BN_sub(z, x, y) != 1) {
    return false;
  }
  if (BN_mod(z, z, this->bn_prim_, ctx) != 1) {
    return false;
  }
  return true;
}

bool SecretSharing::GetShare(BIGNUM *x, BIGNUM *share, Share *s_share) {
  if (x == nullptr || share == nullptr || s_share == nullptr) {
    return false;
  }
  if (BN_set_word(x, s_share->index) != 1) {
    return false;
  }
  (void)BN_bin2bn(s_share->data, SizeToInt(s_share->len), share);
  return true;
}

void SecretSharing::FreeBNVector(std::vector<BIGNUM *> bns) {
  for (size_t i = 0; i < bns.size(); i++) {
    if (bns[i] != nullptr) {
      BN_clear_free(bns[i]);
    }
  }
}

int SecretSharing::CheckShares(Share *share_i, BIGNUM *x_i, BIGNUM *y_i, BIGNUM *denses_i, BIGNUM *nums_i) {
  if (x_i == nullptr || y_i == nullptr || denses_i == nullptr || nums_i == nullptr) {
    MS_LOG(ERROR) << "new bn object failed";
    return -1;
  } else {
    if (!GetShare(x_i, y_i, share_i)) {
      MS_LOG(ERROR) << "get share failed";
      return -1;
    }
  }
  return 0;
}

int SecretSharing::CheckSum(BIGNUM *sum) const {
  int ret = 0;
  if (sum == nullptr) {
    MS_LOG(ERROR) << "new bn object failed";
    ret = -1;
  } else {
    if (BN_zero(sum) != 1) {
      ret = -1;
    }
  }
  return ret;
}

int SecretSharing::LagrangeCal(BIGNUM *nums_j, BIGNUM *x_m, BIGNUM *x_j, BIGNUM *denses_j, BIGNUM *tmp, BN_CTX *ctx) {
  if (!field_mult(nums_j, nums_j, x_m, ctx)) {
    return -1;
  }
  if (!field_sub(tmp, x_m, x_j, ctx)) {
    return -1;
  }
  if (!field_mult(denses_j, denses_j, tmp, ctx)) {
    return -1;
  }
  return 0;
}

int SecretSharing::InputCheck(size_t k, const std::vector<Share *> &shares, uint8_t *secret, size_t *length) const {
  if (secret == nullptr || length == nullptr || k < 1 || shares.size() < k || this->bn_prim_ == nullptr) {
    return -1;
  }
  return 0;
}

void SecretSharing::ReleaseNum(BIGNUM *bigNum) const {
  if (bigNum != nullptr) {
    BN_clear_free(bigNum);
  }
}

int SecretSharing::Combine(size_t k, const std::vector<Share *> &shares, uint8_t *secret, size_t *length) {
  int check_result = InputCheck(k, shares, secret, length);
  if (check_result == -1) return -1;
  BN_CTX *ctx = BN_CTX_new();
  if (ctx == nullptr) {
    MS_LOG(ERROR) << "new bn ctx failed";
    return -1;
  }
  int ret = 0;
  std::vector<BIGNUM *> y(k);
  std::vector<BIGNUM *> x(k);
  std::vector<BIGNUM *> denses(k);
  std::vector<BIGNUM *> nums(k);
  BIGNUM *sum = nullptr;

  for (size_t i = 0; i < k; i++) {
    x[i] = BN_new();
    y[i] = BN_new();
    denses[i] = BN_new();
    nums[i] = BN_new();
    ret = CheckShares(shares[i], x[i], y[i], denses[i], nums[i]);
    if (ret == -1) break;
  }

  if (ret != -1) {
    sum = BN_new();
    ret = CheckSum(sum);
  }

  if (ret != -1) {
    for (size_t j = 0; j < k; j++) {
      if (BN_one(denses[j]) != 1 || BN_one(nums[j]) != 1) {
        ret = -1;
        break;
      }
      BIGNUM *tmp = BN_new();
      if (tmp == nullptr) {
        MS_LOG(ERROR) << "new bn object failed";
        ret = -1;
        break;
      }

      for (size_t m = 0; m < k; m++) {
        if (m != j) {
          ret = LagrangeCal(nums[j], x[m], x[j], denses[j], tmp, ctx);
          if (ret == -1) break;
        }
      }

      if (ret == -1) {
        BN_clear_free(tmp);
        break;
      }
      (void)BN_mod_inverse(tmp, denses[j], this->bn_prim_, ctx);
      if (!field_mult(tmp, tmp, nums[j], ctx)) {
        ret = -1;
        BN_clear_free(tmp);
        break;
      }
      if (!field_mult(tmp, tmp, y[j], ctx)) {
        ret = -1;
        BN_clear_free(tmp);
        break;
      }
      if (!field_add(sum, sum, tmp, ctx)) {
        ret = -1;
        BN_clear_free(tmp);
        break;
      }
      BN_clear_free(tmp);
    }
    *length = BN_bn2bin(sum, secret);
  }
  BN_CTX_free(ctx);
  ReleaseNum(sum);
  FreeBNVector(x);
  FreeBNVector(y);
  FreeBNVector(denses);
  FreeBNVector(nums);
  return ret;
}
#endif
}  // namespace armour
}  // namespace mindspore
