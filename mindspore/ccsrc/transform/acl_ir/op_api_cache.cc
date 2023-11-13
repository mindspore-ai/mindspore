/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "transform/acl_ir/op_api_cache.h"

namespace mindspore::transform {
thread_local char g_hash_buf[g_hash_buf_size];
thread_local int g_hash_offset = 0;

typedef void (*AddTensorAddrToCachedList)(void *addr);

void GatherInfo(const mindspore::kernel::KernelTensor *tensor) {
  if (tensor == nullptr) {
    return;
  }

  // Normal tensor
  static const auto add_tensor_addr_to_cached_list = transform::GetOpApiFunc("AddTensorAddrToCachedList");
  if (add_tensor_addr_to_cached_list == nullptr) {
    MS_LOG(EXCEPTION) << "AddTensorAddrToCachedList not in " << transform::GetOpApiLibName() << ", please check!";
  }
  AddTensorAddrToCachedList add_tensor_addr_to_cached_list_func =
    reinterpret_cast<AddTensorAddrToCachedList>(add_tensor_addr_to_cached_list);
  MS_EXCEPTION_IF_NULL(add_tensor_addr_to_cached_list_func);

  auto shape = tensor->GetShapeVector();
  const auto shape_size = shape.size();
  // view shape
  MEMCPY_TO_BUF(shape.data(), static_cast<int64_t>(shape_size * sizeof(int64_t)));
  // data type
  auto dtype = tensor->dtype_id();
  MEMCPY_TO_BUF(&dtype, sizeof(int));
  // separator
  MEMCPY_TO_BUF(",", 1);
  // strides(current hasn't uncontinus tensor)
  MEMCPY_TO_BUF(",", 1);
  // offset(current hasn't uncontinus tensor)
  MEMCPY_TO_BUF(",", 1);
  // storage shape(current hasn't special format)
  MEMCPY_TO_BUF(shape.data(), static_cast<int64_t>(shape_size * sizeof(int64_t)));

  add_tensor_addr_to_cached_list_func(tensor->device_ptr());
}

void GatherInfo(const string &s) { MEMCPY_TO_BUF(s.c_str(), static_cast<int64_t>(s.size())); }

void GatherInfo() {}

constexpr int g_rShift33Bits = 33;
constexpr uint64_t MIX_STEP1 = 18397679294719823053LLU;
constexpr uint64_t MIX_STEP2 = 14181476777654086739LLU;

inline uint64_t rotating_left(uint64_t x, uint8_t n) { return (x << n) | (x >> (64 - n)); }

inline uint64_t mixture(uint64_t x) {
  // constants step1(18397679294719823053) and step2(14181476777654086739) are used to allow
  // hash values to be more evenly distributed after multiplication.
  x ^= x >> g_rShift33Bits;
  x *= MIX_STEP1;
  x ^= x >> g_rShift33Bits;
  x *= MIX_STEP2;
  x ^= x >> g_rShift33Bits;

  return x;
}

void gen_hash_tmp(const uint64_t *blocks, const int block_num, const uint32_t seed, uint64_t *rhas, uint64_t *rhax) {
  MS_EXCEPTION_IF_NULL(blocks);

  // use 9782798678568883157 and 5545529020109919103 for blocking and obfuscation of input data
  const uint64_t c1 = 9782798678568883157LLU;
  const uint64_t c2 = 5545529020109919103LLU;

  uint64_t has = seed;
  uint64_t hax = seed;
  for (int i = 0; i < block_num; i++) {
    int even_num = 2;
    uint64_t tmp1 = blocks[i * even_num];
    uint64_t tmp2 = blocks[i * even_num + 1];

    int8_t bits_31 = 31;
    tmp1 *= c1;
    tmp1 = rotating_left(tmp1, bits_31);
    tmp1 *= c2;
    has ^= tmp1;

    int8_t bits_27 = 27;
    has = rotating_left(has, bits_27);
    has += hax;
    // increase randomness by mul by 5 and adding a constant
    has = has * 5 + 1390208809;

    int8_t bits_33 = 33;
    tmp2 *= c2;
    tmp2 = rotating_left(tmp2, bits_33);
    tmp2 *= c1;
    hax ^= tmp2;

    hax = rotating_left(hax, bits_31);
    hax += has;
    // increase randomness by mul by 5 and adding a constant
    hax = hax * 5 + 944331445;
  }

  *rhas = has;
  *rhax = hax;
}

uint64_t gen_hash(const void *key, const int len, const uint32_t seed = 0xdeadb0d7) {
  const uint8_t *data = (const uint8_t *)key;
  // the length of each block is 16 bytes
  const int block_num = len / 16;
  // has and hax are literal appromix to hash, and hax is the return value of this function.
  uint64_t has = seed;
  uint64_t hax = seed;

  // use 9782798678568883157 and 5545529020109919103 for blocking and obfuscation of input data
  const uint64_t c1 = 9782798678568883157LLU;
  const uint64_t c2 = 5545529020109919103LLU;

  const uint64_t *blocks = (const uint64_t *)(data);

  // update hax
  gen_hash_tmp(blocks, block_num, seed, &has, &hax);

  // the length of each block is 16 bytes
  const uint8_t *tail = (const uint8_t *)(data + block_num * 16);
  uint64_t t1 = 0;
  uint64_t t2 = 0;
  // because the size of a block is 16, different offsets are calculated for tail blocks
  // for different sizes
  switch (static_cast<uint64_t>(len) & 15) {
    case 15:
      t2 ^= ((uint64_t)tail[14]) << 48;
      [[fallthrough]];
      {}
    case 14:
      t2 ^= ((uint64_t)tail[13]) << 40;
      [[fallthrough]];
      {}
    case 13:
      t2 ^= ((uint64_t)tail[12]) << 32;
      [[fallthrough]];
      {}
    case 12:
      t2 ^= ((uint64_t)tail[11]) << 24;
      [[fallthrough]];
      {}
    case 11:
      t2 ^= ((uint64_t)tail[10]) << 16;
      [[fallthrough]];
      {}
    case 10:
      t2 ^= ((uint64_t)tail[9]) << 8;
      [[fallthrough]];
      {}
    case 9:
      t2 ^= ((uint64_t)tail[8]) << 0;
      t2 *= c2;
      t2 = rotating_left(t2, 33);
      t2 *= c1;
      hax ^= t2;
      [[fallthrough]];
      {}
    case 8:
      t1 ^= ((uint64_t)tail[7]) << 56;
      [[fallthrough]];
      {}
    case 7:
      t1 ^= ((uint64_t)tail[6]) << 48;
      [[fallthrough]];
      {}
    case 6:
      t1 ^= ((uint64_t)tail[5]) << 40;
      [[fallthrough]];
      {}
    case 5:
      t1 ^= ((uint64_t)tail[4]) << 32;
      [[fallthrough]];
      {}
    case 4:
      t1 ^= ((uint64_t)tail[3]) << 24;
      [[fallthrough]];
      {}
    case 3:
      t1 ^= ((uint64_t)tail[2]) << 16;
      [[fallthrough]];
      {}
    case 2:
      t1 ^= ((uint64_t)tail[1]) << 8;
      [[fallthrough]];
      {}
    case 1:
      t1 ^= ((uint64_t)tail[0]) << 0;
      t1 *= c1;
      t1 = rotating_left(t1, 31);
      t1 *= c2;
      has ^= t1;
      [[fallthrough]];
      {}
    default:
      [[fallthrough]];
      {}
  }

  has ^= static_cast<uint64_t>(len);
  hax ^= static_cast<uint64_t>(len);

  has += hax;
  hax += has;

  has = mixture(has);
  hax = mixture(hax);

  has += hax;
  hax += has;
  return hax;
}

uint64_t calc_hash_id() {
  if (g_hash_offset == g_hash_buf_max_size) {
    return 0;
  }
  uint64_t hash_id = gen_hash(g_hash_buf, g_hash_offset);
  return hash_id;
}
}  // namespace mindspore::transform
