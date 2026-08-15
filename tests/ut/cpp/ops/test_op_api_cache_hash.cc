/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

/*
 * Unit tests for aclnn cache GatherInfo semantics used by op_api_cache.h.
 *
 * op_api_cache.h pulls Ascend ACL headers and cannot be included in CORE UT, so this
 * file mirrors only the POD GatherInfo overloads under test. Keep them in sync with:
 *   mindspore/ops/kernel/ascend/acl_ir/op_api_cache.h
 *
 * Overload order matters: vector must be declared before optional so the nested
 * GatherInfo(value.value()) call can resolve to the vector overload (two-phase lookup).
 */
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/common_test.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr int kHashBufSize = 16384;

char g_hash_buf[kHashBufSize];
int g_hash_offset = 0;

void MemcpyToBuf(const void *data, size_t size) {
  if (size == 0) {
    return;
  }
  if (g_hash_offset + static_cast<int>(size) > kHashBufSize) {
    ADD_FAILURE() << "Hash buf overflow in UT harness";
    return;
  }
  (void)memcpy(g_hash_buf + g_hash_offset, data, size);
  g_hash_offset += static_cast<int>(size);
}

template <typename T>
void GatherInfo(const T &value) {
  MemcpyToBuf(&value, sizeof(T));
}

// Must match production: vector before optional (two-phase name lookup).
template <typename T>
void GatherInfo(const std::vector<T> &values) {
  const size_t size = values.size();
  MemcpyToBuf(&size, sizeof(size_t));
  if (size != 0) {
    MemcpyToBuf(values.data(), size * sizeof(T));
  }
}

template <typename T>
void GatherInfo(const std::optional<T> &value) {
  const bool has_value = value.has_value();
  MemcpyToBuf(&has_value, sizeof(bool));
  if (has_value) {
    GatherInfo(value.value());
  }
}

template <typename T>
void GatherInfo(const std::pair<std::vector<T>, bool> &p) {
  GatherInfo(p.first);
  MemcpyToBuf(&p.second, sizeof(bool));
}

template <typename T>
void GatherInfo(const std::pair<std::optional<std::vector<T>>, bool> &p) {
  GatherInfo(p.first);
  MemcpyToBuf(&p.second, sizeof(bool));
}

void ResetHashBuf() { g_hash_offset = 0; }

std::string CaptureHashPayload() { return std::string(g_hash_buf, static_cast<size_t>(g_hash_offset)); }

template <typename T>
std::string HashValue(const T &value) {
  ResetHashBuf();
  GatherInfo(value);
  return CaptureHashPayload();
}
}  // namespace

class TestOpApiCacheHash : public UT::Common {
 public:
  TestOpApiCacheHash() = default;
  ~TestOpApiCacheHash() override = default;

  void SetUp() override { ResetHashBuf(); }
};

/// Feature: aclnn cache hash for vector content.
/// Description: Same vector content with different heap allocations.
/// Expectation: hash payloads are equal.
TEST_F(TestOpApiCacheHash, SameVectorContentDifferentAllocation) {
  std::vector<int64_t> lhs = {1, 2, 3, 4};
  std::vector<int64_t> rhs = {1, 2, 3, 4};
  ASSERT_NE(lhs.data(), rhs.data());
  EXPECT_EQ(HashValue(lhs), HashValue(rhs));
}

/// Feature: aclnn cache hash for vector content.
/// Description: Same length but one element differs.
/// Expectation: hash payloads differ.
TEST_F(TestOpApiCacheHash, SameLengthDifferentElement) {
  std::vector<int64_t> lhs = {1, 2, 3, 4};
  std::vector<int64_t> rhs = {1, 2, 3, 5};
  EXPECT_NE(HashValue(lhs), HashValue(rhs));
}

/// Feature: aclnn cache hash for optional presence.
/// Description: nullopt versus engaged empty vector.
/// Expectation: hash payloads differ.
TEST_F(TestOpApiCacheHash, NulloptVsEngagedEmptyVector) {
  std::optional<std::vector<int64_t>> nullopt_value = std::nullopt;
  std::optional<std::vector<int64_t>> empty_value = std::vector<int64_t>{};
  EXPECT_NE(HashValue(nullopt_value), HashValue(empty_value));
}

/// Feature: aclnn cache hash for vector length prefix.
/// Description: Different lengths that share a common prefix.
/// Expectation: hash payloads differ.
TEST_F(TestOpApiCacheHash, DifferentLengthCommonPrefix) {
  std::vector<int64_t> short_vec = {1, 2, 3};
  std::vector<int64_t> long_vec = {1, 2, 3, 4};
  EXPECT_NE(HashValue(short_vec), HashValue(long_vec));
}

/// Feature: aclnn cache hash for pair bool flag.
/// Description: Same vector content with different pair bool.
/// Expectation: hash payloads differ.
TEST_F(TestOpApiCacheHash, PairBoolDifferent) {
  using IntArrayPair = std::pair<std::vector<int64_t>, bool>;
  IntArrayPair with_true = {std::vector<int64_t>{1, 2}, true};
  IntArrayPair with_false = {std::vector<int64_t>{1, 2}, false};
  EXPECT_NE(HashValue(with_true), HashValue(with_false));
}

/// Feature: aclnn cache hash for adjacent vectors.
/// Description: Different partitions whose raw bytes would concatenate equally without size prefixes.
/// Expectation: hash payloads differ because each vector writes a length prefix.
TEST_F(TestOpApiCacheHash, AdjacentVectorsDifferentPartitions) {
  ResetHashBuf();
  GatherInfo(std::vector<int64_t>{1, 2, 3});
  GatherInfo(std::vector<int64_t>{4});
  const auto split_a = CaptureHashPayload();

  ResetHashBuf();
  GatherInfo(std::vector<int64_t>{1, 2});
  GatherInfo(std::vector<int64_t>{3, 4});
  const auto split_b = CaptureHashPayload();

  EXPECT_NE(split_a, split_b);
}

/// Feature: aclnn cache hash for pair<optional<vector>, bool>.
/// Description: Same size optional-vector pairs with different contents (FIAS actual_seq_lengths path).
/// Expectation: hash payloads differ.
TEST_F(TestOpApiCacheHash, OptionalVectorPairContentSensitive) {
  using OptionalIntArrayPair = std::pair<std::optional<std::vector<int64_t>>, bool>;
  OptionalIntArrayPair lhs = {std::vector<int64_t>{10, 20, 30}, true};
  OptionalIntArrayPair rhs = {std::vector<int64_t>{10, 20, 31}, true};
  EXPECT_NE(HashValue(lhs), HashValue(rhs));
}

/// Feature: aclnn cache hash nested optional-vector recursion.
/// Description: Two independently allocated optional-vector pairs with identical element content.
/// Expectation: hash payloads are equal (proves elements are hashed, not heap pointers).
TEST_F(TestOpApiCacheHash, OptionalVectorPairSameContentDifferentAllocation) {
  using OptionalIntArrayPair = std::pair<std::optional<std::vector<int64_t>>, bool>;
  OptionalIntArrayPair lhs = {std::vector<int64_t>{10, 20, 30}, true};
  OptionalIntArrayPair rhs = {std::vector<int64_t>{10, 20, 30}, true};
  ASSERT_TRUE(lhs.first.has_value() && rhs.first.has_value());
  ASSERT_NE(lhs.first->data(), rhs.first->data());
  EXPECT_EQ(HashValue(lhs), HashValue(rhs));
}

/// Feature: aclnn cache hash nested optional-vector recursion.
/// Description: Same optional-vector object; keep pointer/size/capacity, mutate one element in place.
/// Expectation: hash payloads differ (catches hashing object representation instead of elements).
TEST_F(TestOpApiCacheHash, OptionalVectorPairInPlaceElementMutation) {
  using OptionalIntArrayPair = std::pair<std::optional<std::vector<int64_t>>, bool>;
  OptionalIntArrayPair value = {std::vector<int64_t>{10, 20, 30}, true};
  ASSERT_TRUE(value.first.has_value());
  const auto *data_ptr = value.first->data();
  const auto size_before = value.first->size();
  const auto capacity_before = value.first->capacity();

  const auto hash_before = HashValue(value);
  (*value.first)[1] = 21;
  ASSERT_EQ(value.first->data(), data_ptr);
  ASSERT_EQ(value.first->size(), size_before);
  ASSERT_EQ(value.first->capacity(), capacity_before);
  const auto hash_after = HashValue(value);

  EXPECT_NE(hash_before, hash_after);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
