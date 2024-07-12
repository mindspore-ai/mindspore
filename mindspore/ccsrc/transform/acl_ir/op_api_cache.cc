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
namespace {
void Gather(mindspore::kernel::KernelTensor *tensor) {
  if (tensor == nullptr || tensor->type_id() == kMetaTypeNone) {
    MemcpyToBuf("None", kSizeFive);
    return;
  }

  const auto &shape = tensor->GetShapeVector();
  const auto shape_size = shape.size();
  // view shape
  if (!shape.empty()) {
    MemcpyToBuf(shape.data(), static_cast<int64_t>(shape_size * sizeof(int64_t)));
  }

  // data type
  auto dtype = tensor->dtype_id();
  MemcpyToBuf(&dtype, sizeof(int));

  const auto &storage_info = tensor->tensor_storage_info();
  if (storage_info != nullptr) {
    // strides
    MemcpyToBuf(storage_info->strides.data(), static_cast<int64_t>(storage_info->strides.size() * sizeof(int64_t)));

    // offset
    MemcpyToBuf(&storage_info->storage_offset, sizeof(int64_t));

    // origin shape
    MemcpyToBuf(storage_info->ori_shape.data(), static_cast<int64_t>(storage_info->ori_shape.size()) * sizeof(int64_t));
  }
}

void Gather(const device::DeviceAddressPtr &device_address) {
  if (device_address == nullptr) {
    MemcpyToBuf("None", 5);
    return;
  }

  const auto &shape = device_address->GetShapeVector();
  const auto shape_size = shape.size();
  // view shape
  if (!shape.empty()) {
    MemcpyToBuf(shape.data(), static_cast<int64_t>(shape_size * sizeof(int64_t)));
  }

  // data type
  auto dtype = device_address->type_id();
  MemcpyToBuf(&dtype, sizeof(int));

  const auto &storage_info = device_address->address_common()->tensor_storage_info_;
  if (storage_info != nullptr) {
    // strides
    MemcpyToBuf(storage_info->strides.data(), static_cast<int64_t>(storage_info->strides.size() * sizeof(int64_t)));

    // offset
    MemcpyToBuf(&storage_info->storage_offset, sizeof(int64_t));

    // origin shape
    MemcpyToBuf(storage_info->ori_shape.data(), static_cast<int64_t>(storage_info->ori_shape.size()) * sizeof(int64_t));
  }
}

void Gather(const mindspore::tensor::BaseTensorPtr &tensor) {
  if (tensor == nullptr) {
    return;
  }

  // "t" for tensor
  MemcpyToBuf("t", 1);

  const auto &shape = tensor->shape();
  const auto shape_size = shape.size();
  // view shape
  if (!shape.empty()) {
    MemcpyToBuf(shape.data(), static_cast<int64_t>(shape_size * sizeof(int64_t)));
  }
  // data type
  auto dtype = tensor->data_type();
  MemcpyToBuf(&dtype, sizeof(int));

  auto storage_info = tensor->storage_info();
  if (storage_info != nullptr) {
    // strides
    MemcpyToBuf(storage_info->strides.data(), static_cast<int64_t>(storage_info->strides.size() * sizeof(int64_t)));

    // offset
    MemcpyToBuf(&storage_info->storage_offset, sizeof(int64_t));

    // origin shape
    MemcpyToBuf(storage_info->ori_shape.data(), static_cast<int64_t>(storage_info->ori_shape.size()) * sizeof(int64_t));
  }

  // storage shape(current hasn't special format)
}
}  // namespace
thread_local char g_hash_buf[g_hash_buf_size];
thread_local int g_hash_offset = 0;

typedef void (*AddTensorAddrToCachedList)(void *addr);

void GatherInfo(mindspore::kernel::KernelTensor *tensor) {
  Gather(tensor);
  RefreshAddr(tensor);
}

void GatherInfo(const device::DeviceAddressPtr &device_address) {
  Gather(device_address);
  RefreshAddr(device_address);
}

void GatherInfo(const std::pair<mindspore::kernel::KernelTensor *, bool> &tensor_and_trans) {
  auto tensor = tensor_and_trans.first;
  auto trans = tensor_and_trans.second;
  GatherInfo(tensor);
  // trans
  MemcpyToBuf(&trans, 1);
}

void GatherInfo(const std::vector<mindspore::kernel::KernelTensor *> &tensor_list) {
  for (auto tensor : tensor_list) {
    GatherInfo(tensor);
  }
}

void GatherInfo(const mindspore::tensor::BaseTensorPtr &tensor) {
  Gather(tensor);
  RefreshAddr(tensor);
}

void GatherInfo(const std::optional<tensor::BaseTensorPtr> &tensor) {
  // "ot" for optional tensor
  MemcpyToBuf("ot", 2);
  if (tensor.has_value()) {
    GatherInfo(tensor.value());
  }
}

void GatherInfo(const std::vector<tensor::BaseTensorPtr> &tensors) {
  for (const auto &tensor : tensors) {
    GatherInfo(tensor);
  }
}

void GatherInfo(const mindspore::tensor::TensorPtr &tensor) { GatherInfo(tensor->cast<tensor::BaseTensorPtr>()); }

void GatherInfo(const std::optional<tensor::TensorPtr> &tensor) {
  // "ot" for optional tensor
  MemcpyToBuf("ot", 2);
  if (tensor.has_value()) {
    GatherInfo(tensor.value());
  }
}

void GatherInfo(const std::vector<tensor::TensorPtr> &tensors) {
  for (const auto &tensor : tensors) {
    GatherInfo(tensor);
  }
}

void GatherInfo(const ScalarPtr &scalar) {
  if (scalar == nullptr) {
    MemcpyToBuf("None", 5);
    return;
  }
  // "s" for scalar
  MemcpyToBuf("s", 1);
  if (scalar->isa<BoolImm>()) {
    auto value = GetValue<bool>(scalar);
    MemcpyToBuf(&value, sizeof(bool));
  } else if (scalar->isa<Int64Imm>()) {
    auto value = GetValue<int64_t>(scalar);
    MemcpyToBuf(&value, sizeof(int64_t));
  } else if (scalar->isa<FP32Imm>()) {
    auto value = GetValue<float>(scalar);
    MemcpyToBuf(&value, sizeof(float));
  } else if (scalar->isa<Int32Imm>()) {
    auto value = GetValue<int32_t>(scalar);
    MemcpyToBuf(&value, sizeof(int32_t));
  } else if (scalar->isa<Int8Imm>()) {
    auto value = GetValue<int8_t>(scalar);
    MemcpyToBuf(&value, sizeof(int8_t));
  } else if (scalar->isa<Int16Imm>()) {
    auto value = GetValue<int16_t>(scalar);
    MemcpyToBuf(&value, sizeof(int16_t));
  } else if (scalar->isa<UInt8Imm>()) {
    auto value = GetValue<uint8_t>(scalar);
    MemcpyToBuf(&value, sizeof(uint8_t));
  } else if (scalar->isa<FP64Imm>()) {
    auto value = GetValue<double>(scalar);
    MemcpyToBuf(&value, sizeof(double));
  } else if (scalar->isa<BF16Imm>()) {
    auto value = GetValue<bfloat16>(scalar);
    MemcpyToBuf(&value, sizeof(int16_t));
  } else {
    MS_LOG(EXCEPTION) << "Currently not support value: " << scalar->ToString();
  }
}

void GatherInfo(const std::optional<ScalarPtr> &scalar) {
  if (scalar.has_value()) {
    GatherInfo(scalar.value());
  } else {
    MemcpyToBuf("None", 5);
  }
}

void GatherInfo(const TypePtr &type) {
  const auto type_id = type->type_id();
  MemcpyToBuf(&type_id, sizeof(int));
}

void GatherInfo(const std::optional<TypePtr> &type) {
  if (type.has_value()) {
    GatherInfo(type.value());
  }
}

void GatherInfo(const string &s) { MemcpyToBuf(s.c_str(), static_cast<int64_t>(s.size())); }

void GatherInfo(const std::optional<string> &s) {
  if (s.has_value()) {
    GatherInfo(s.value());
  }
}

void GatherInfo() {}

void RefreshAddr(mindspore::kernel::KernelTensor *tensor) {
  if (tensor == nullptr || tensor->type_id() == kMetaTypeNone) {
    return;
  }

  static const auto add_tensor_addr_to_cached_list = transform::GetOpApiFunc("AddTensorAddrToCachedList");
  if (add_tensor_addr_to_cached_list == nullptr) {
    MS_LOG(EXCEPTION) << "AddTensorAddrToCachedList not in " << transform::GetOpApiLibName() << ", please check!";
  }
  AddTensorAddrToCachedList add_tensor_addr_to_cached_list_func =
    reinterpret_cast<AddTensorAddrToCachedList>(add_tensor_addr_to_cached_list);
  MS_EXCEPTION_IF_NULL(add_tensor_addr_to_cached_list_func);

  add_tensor_addr_to_cached_list_func(tensor->device_ptr());
}

void RefreshAddr(const device::DeviceAddressPtr &device_address) {
  if (device_address == nullptr) {
    return;
  }

  static const auto add_tensor_addr_to_cached_list = transform::GetOpApiFunc("AddTensorAddrToCachedList");
  if (add_tensor_addr_to_cached_list == nullptr) {
    MS_LOG(EXCEPTION) << "AddTensorAddrToCachedList not in " << transform::GetOpApiLibName() << ", please check!";
  }
  AddTensorAddrToCachedList add_tensor_addr_to_cached_list_func =
    reinterpret_cast<AddTensorAddrToCachedList>(add_tensor_addr_to_cached_list);
  MS_EXCEPTION_IF_NULL(add_tensor_addr_to_cached_list_func);

  add_tensor_addr_to_cached_list_func(device_address->GetMutablePtr());
}

void RefreshAddr(const mindspore::tensor::TensorPtr &tensor) {
  if (tensor == nullptr) {
    return;
  }

  static const auto add_tensor_addr_to_cached_list = transform::GetOpApiFunc("AddTensorAddrToCachedList");
  if (add_tensor_addr_to_cached_list == nullptr) {
    MS_LOG(EXCEPTION) << "AddTensorAddrToCachedList not in " << transform::GetOpApiLibName() << ", please check!";
  }
  AddTensorAddrToCachedList add_tensor_addr_to_cached_list_func =
    reinterpret_cast<AddTensorAddrToCachedList>(add_tensor_addr_to_cached_list);
  MS_EXCEPTION_IF_NULL(add_tensor_addr_to_cached_list_func);

  add_tensor_addr_to_cached_list_func(tensor->device_address()->GetMutablePtr());
}

void RefreshAddr(const std::pair<mindspore::kernel::KernelTensor *, bool> &tensor_and_trans) {
  RefreshAddr(tensor_and_trans.first);
}

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

uint64_t gen_hash(const void *key, const int len, const uint32_t seed) {
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
    default: {
    }
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

void GatherHash(mindspore::kernel::KernelTensor *tensor) { Gather(tensor); }

void GatherHash(const device::DeviceAddressPtr &device_address) { Gather(device_address); }

void GatherHash(const std::pair<mindspore::kernel::KernelTensor *, bool> &tensor_and_trans) {
  auto tensor = tensor_and_trans.first;
  auto trans = tensor_and_trans.second;
  GatherHash(tensor);
  // trans
  MemcpyToBuf(&trans, 1);
}

void GatherHash(const std::vector<mindspore::kernel::KernelTensor *> &tensor_list) {
  for (auto tensor : tensor_list) {
    GatherHash(tensor);
  }
}

void GatherHash(const mindspore::tensor::TensorPtr &tensor) { Gather(tensor); }

void GatherHash(const std::optional<tensor::TensorPtr> &tensor) {
  // "ot" for optional tensor
  MemcpyToBuf("ot", 2);
  if (tensor.has_value()) {
    GatherHash(tensor.value());
  }
}

void GatherHash(const std::vector<tensor::TensorPtr> &tensors) {
  for (const auto &tensor : tensors) {
    GatherHash(tensor);
  }
}

void GatherHash(const ScalarPtr &scalar) {
  if (scalar == nullptr) {
    MemcpyToBuf("None", 5);
    return;
  }
  // "s" for scalar
  MemcpyToBuf("s", 1);
  if (scalar->isa<BoolImm>()) {
    auto value = GetValue<bool>(scalar);
    MemcpyToBuf(&value, sizeof(bool));
  } else if (scalar->isa<Int64Imm>()) {
    auto value = GetValue<int64_t>(scalar);
    MemcpyToBuf(&value, sizeof(int64_t));
  } else if (scalar->isa<FP32Imm>()) {
    auto value = GetValue<float>(scalar);
    MemcpyToBuf(&value, sizeof(float));
  } else if (scalar->isa<Int32Imm>()) {
    auto value = GetValue<int32_t>(scalar);
    MemcpyToBuf(&value, sizeof(int32_t));
  } else if (scalar->isa<Int8Imm>()) {
    auto value = GetValue<int8_t>(scalar);
    MemcpyToBuf(&value, sizeof(int8_t));
  } else if (scalar->isa<Int16Imm>()) {
    auto value = GetValue<int16_t>(scalar);
    MemcpyToBuf(&value, sizeof(int16_t));
  } else if (scalar->isa<UInt8Imm>()) {
    auto value = GetValue<uint8_t>(scalar);
    MemcpyToBuf(&value, sizeof(uint8_t));
  } else if (scalar->isa<FP64Imm>()) {
    auto value = GetValue<double>(scalar);
    MemcpyToBuf(&value, sizeof(double));
  } else if (scalar->isa<BF16Imm>()) {
    auto value = GetValue<bfloat16>(scalar);
    MemcpyToBuf(&value, sizeof(int16_t));
  } else {
    MS_LOG(EXCEPTION) << "Currently not support value: " << scalar->ToString();
  }
}

void GatherHash(const std::optional<ScalarPtr> &scalar) {
  if (scalar.has_value()) {
    GatherHash(scalar.value());
  } else {
    MemcpyToBuf("None", 5);
  }
}

void GatherHash(const TypePtr &type) {
  const auto type_id = type->type_id();
  MemcpyToBuf(&type_id, sizeof(int));
}

void GatherHash(const std::optional<TypePtr> &type) {
  if (type.has_value()) {
    GatherHash(type.value());
  }
}

void GatherHash(const string &s) { MemcpyToBuf(s.c_str(), static_cast<int64_t>(s.size())); }

void GatherHash(const std::optional<string> &s) {
  if (s.has_value()) {
    GatherHash(s.value());
  }
}

void GatherHash() {}
}  // namespace mindspore::transform
