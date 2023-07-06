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

#include <vector>
#include <numeric>

#include "common/common_test.h"
#include "plugin/device/cpu/hal/device/cpu_hash_table.h"

namespace mindspore {
namespace device {
namespace cpu {
class TestCPUHashTable : public UT::Common {
 public:
  TestCPUHashTable() = default;
  virtual ~TestCPUHashTable() = default;

  void SetUp() override {}
  void TearDown() override {}
};

using Key = int;
using Value = float;
/// Feature: test cpu hash table all api.
/// Description: test cpu hash table data structure and interface.
/// Expectation: all interface work normally.
TEST_F(TestCPUHashTable, test_cpu_hash_table) {
  size_t value_dim = 4;
  size_t key_num = 10;
  size_t erase_key_num = 5;
  CPUHashTable<Key, Value> hash_table(value_dim, "ones");

  // Keys and values to insert.
  std::vector<Key> keys_to_insert(key_num);
  std::iota(keys_to_insert.begin(), keys_to_insert.end(), 0);

  std::vector<Value> value_to_insert(key_num * value_dim);
  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < value_dim; j++) {
      value_to_insert[i * value_dim + j] = static_cast<Value>(i);
    }
  }

  // Statuses to insert.
  std::vector<HashTableElementStatus> statuses_to_insert(key_num, HashTableElementStatus::kModified);

  // Keys and values check map.
  std::unordered_map<Key, std::vector<Value>> keys_values;
  for (size_t i = 0; i < key_num; ++i) {
    keys_values.emplace(keys_to_insert[i], std::vector<Value>(value_to_insert.begin() + i * value_dim,
                                                              value_to_insert.begin() + (i + 1) * value_dim));
  }
  EXPECT_EQ(keys_values.size(), key_num);

  // Keys, values and statuses check tensor.
  std::vector<Key> keys_to_check(key_num);
  std::vector<Value> values_to_check(key_num * value_dim);
  std::vector<HashTableElementStatus> statuses_to_check(key_num);

  // Test all api of cpu hash table.
  EXPECT_TRUE(hash_table.Reserve(key_num, nullptr));
  EXPECT_TRUE(hash_table.Insert(keys_to_insert.data(), key_num, value_to_insert.data(), nullptr));
  EXPECT_TRUE(hash_table.Find(keys_to_insert.data(), key_num, false, values_to_check.data(), nullptr));
  EXPECT_EQ(values_to_check, value_to_insert);

  EXPECT_TRUE(hash_table.is_dirty());

  EXPECT_EQ(hash_table.size(), key_num);

  EXPECT_EQ(hash_table.capacity(), key_num);

  EXPECT_TRUE(hash_table.GetKeysAndValues(keys_to_check.data(), values_to_check.data(), nullptr));

  for (size_t i = 0; i < key_num; ++i) {
    EXPECT_TRUE(keys_values.find(keys_to_check[i]) != keys_values.end());
    EXPECT_EQ(keys_values[keys_to_check[i]], std::vector<Value>(values_to_check.begin() + i * value_dim,
                                                                values_to_check.begin() + (i + 1) * value_dim));
  }

  HashTableExportData full_export_data, incre_export_data;
  EXPECT_NO_THROW(full_export_data = hash_table.Export(false));
  EXPECT_NO_THROW(incre_export_data = hash_table.Export(true));
  EXPECT_FALSE(hash_table.is_dirty());

  EXPECT_EQ(*full_export_data[0], *incre_export_data[0]);
  EXPECT_EQ(*full_export_data[1], *incre_export_data[1]);
  EXPECT_EQ(*full_export_data[2], *incre_export_data[2]);

  EXPECT_EQ(full_export_data[0]->size(), key_num * sizeof(Key));
  EXPECT_EQ(full_export_data[1]->size(), key_num * value_dim * sizeof(Value));
  EXPECT_EQ(full_export_data[2]->size(), key_num * sizeof(HashTableElementStatus));

  auto ret = memcpy_s(keys_to_check.data(), keys_to_check.size() * sizeof(Key), full_export_data[0]->data(),
                      key_num * sizeof(Key));
  EXPECT_EQ(ret, EOK);
  ret = memcpy_s(values_to_check.data(), values_to_check.size() * sizeof(Value), full_export_data[1]->data(),
                 key_num * value_dim * sizeof(Value));
  EXPECT_EQ(ret, EOK);
  ret = memcpy_s(statuses_to_check.data(), statuses_to_check.size() * sizeof(HashTableElementStatus),
                 full_export_data[2]->data(), key_num * sizeof(HashTableElementStatus));
  EXPECT_EQ(ret, EOK);

  for (size_t i = 0; i < key_num; ++i) {
    EXPECT_TRUE(keys_values.find(keys_to_check[i]) != keys_values.end());
    EXPECT_EQ(keys_values[keys_to_check[i]], std::vector<Value>(values_to_check.begin() + i * value_dim,
                                                                values_to_check.begin() + (i + 1) * value_dim));
    EXPECT_EQ(statuses_to_check[i], HashTableElementStatus::kModified);
  }

  EXPECT_TRUE(
    hash_table.Insert(keys_to_insert.data(), key_num, value_to_insert.data(), statuses_to_insert.data(), nullptr));

  HashTableExportData incre_export_data_after_insert;
  EXPECT_NO_THROW(incre_export_data_after_insert = hash_table.Export(true));
  EXPECT_EQ(*incre_export_data[0], *incre_export_data_after_insert[0]);
  EXPECT_EQ(*incre_export_data[1], *incre_export_data_after_insert[1]);
  EXPECT_EQ(*incre_export_data[2], *incre_export_data_after_insert[2]);

  EXPECT_TRUE(hash_table.Erase(keys_to_insert.data(), erase_key_num, nullptr));
  EXPECT_EQ(hash_table.size(), key_num - erase_key_num);

  EXPECT_TRUE(hash_table.Clear());

  EXPECT_EQ(hash_table.size(), 0);
}

/// Feature: test cpu hash table find api mainly.
/// Description: test cpu hash table data find API.
/// Expectation: find function works normally.
TEST_F(TestCPUHashTable, test_cpu_hash_table_find) {
  size_t value_dim = 4;
  size_t key_num = 10;
  size_t erase_key_num = 5;
  CPUHashTable<Key, Value> hash_table(value_dim, 0.0);

  // Keys and values to insert.
  std::vector<Key> keys_to_insert(key_num);
  std::iota(keys_to_insert.begin(), keys_to_insert.end(), 0);

  std::vector<Value> value_to_insert(key_num * value_dim);
  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < value_dim; j++) {
      value_to_insert[i * value_dim + j] = static_cast<Value>(i);
    }
  }

  // Keys and values check map.
  std::unordered_map<Key, std::vector<Value>> keys_values;
  for (size_t i = 0; i < key_num; ++i) {
    keys_values.emplace(keys_to_insert[i], std::vector<Value>(value_to_insert.begin() + i * value_dim,
                                                              value_to_insert.begin() + (i + 1) * value_dim));
  }
  EXPECT_EQ(keys_values.size(), key_num);

  // Keys, values and statuses check tensor.
  std::vector<Key> keys_to_check(key_num);
  std::vector<Value> values_to_check(key_num * value_dim);

  /// Test find() parameters of cpu hash table.
  // >>>1. if keys_to_insert exists, it will return true
  // Pad the hash_table
  EXPECT_TRUE(hash_table.Insert(keys_to_insert.data(), key_num, value_to_insert.data(), nullptr));
  EXPECT_TRUE(hash_table.Find(keys_to_insert.data(), key_num, false, values_to_check.data(), nullptr));
  EXPECT_EQ(values_to_check, value_to_insert);

  EXPECT_TRUE(hash_table.is_dirty());

  EXPECT_EQ(hash_table.size(), key_num);

  EXPECT_EQ(hash_table.capacity(), key_num);

  EXPECT_TRUE(hash_table.GetKeysAndValues(keys_to_check.data(), values_to_check.data(), nullptr));

  for (size_t i = 0; i < key_num; ++i) {
    EXPECT_TRUE(keys_values.find(keys_to_check[i]) != keys_values.end());
    EXPECT_EQ(keys_values[keys_to_check[i]], std::vector<Value>(values_to_check.begin() + i * value_dim,
                                                                values_to_check.begin() + (i + 1) * value_dim));
  }
  // Clear the hash_table
  EXPECT_TRUE(hash_table.Erase(keys_to_insert.data(), erase_key_num, nullptr));
  keys_values.clear();
  value_to_insert.clear();
  keys_to_check.clear();
  values_to_check.clear();

  // >>>2. if keys_to_insert doesn't exist, <insert_default_value> is false, <find> will return false directly
  EXPECT_FALSE(hash_table.Find(keys_to_insert.data(), key_num, false, values_to_check.data(), nullptr));

  // >>>3. if keys_to_insert doesn't exists, <insert_default_value> is true, it will insert key-value pair with
  // <Value>default_value
  Value default_value_ = static_cast<Value>(99);
  CPUHashTable<Key, Value> hash_table_d(value_dim, default_value_);

  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < value_dim; j++) {
      value_to_insert[i * value_dim + j] = static_cast<Value>(default_value_);
    }
  }
  // Keys and values check map.
  for (size_t i = 0; i < key_num; ++i) {
    keys_values.emplace(keys_to_insert[i], std::vector<Value>(value_to_insert.begin() + i * value_dim,
                                                              value_to_insert.begin() + (i + 1) * value_dim));
  }
  EXPECT_EQ(keys_values.size(), key_num);

  // There is no keys_to_insert in this empty hash_table, so it will be padded automatically
  EXPECT_TRUE(hash_table_d.Find(keys_to_insert.data(), key_num, true, values_to_check.data(), nullptr));
  EXPECT_EQ(values_to_check, value_to_insert);

  // Get keys and values from hash_table
  EXPECT_TRUE(hash_table_d.GetKeysAndValues(keys_to_check.data(), values_to_check.data(), nullptr));

  // Compare keys_to_check and values_to_check with check_map.
  for (size_t i = 0; i < key_num; ++i) {
    EXPECT_TRUE(keys_values.find(keys_to_check[i]) != keys_values.end());
    EXPECT_EQ(keys_values[keys_to_check[i]], std::vector<Value>(values_to_check.begin() + i * value_dim,
                                                                values_to_check.begin() + (i + 1) * value_dim));
  }
  keys_values.clear();
  value_to_insert.clear();
  keys_to_check.clear();
  values_to_check.clear();

  // >>>4. if miss_keys doesn't exists, <insert_default_value> is true, it will insert key-value pair, value is decided
  // by <String>initializer_
  std::string initializer_ = "ones";
  std::map<std::string, int> initializer_values = {{"ones", 1}, {"zeros", 0}};
  CPUHashTable<Key, Value> hash_table_i(value_dim, initializer_);

  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < value_dim; j++) {
      value_to_insert[i * value_dim + j] = static_cast<Value>(initializer_values[initializer_]);
    }
  }
  // Keys and values check map.
  for (size_t i = 0; i < key_num; ++i) {
    keys_values.emplace(keys_to_insert[i], std::vector<Value>(value_to_insert.begin() + i * value_dim,
                                                              value_to_insert.begin() + (i + 1) * value_dim));
  }
  EXPECT_EQ(keys_values.size(), key_num);
  // There is no keys_to_insert in this empty hash_table, so it will be padded automatically
  EXPECT_TRUE(hash_table_i.Find(keys_to_insert.data(), key_num, true, values_to_check.data(), nullptr));
  EXPECT_EQ(values_to_check, value_to_insert);

  // Get keys and values from hash_table
  EXPECT_TRUE(hash_table_i.GetKeysAndValues(keys_to_check.data(), values_to_check.data(), nullptr));

  // Compare keys_to_check&values_to_check with check_map.
  for (size_t i = 0; i < key_num; ++i) {
    EXPECT_TRUE(keys_values.find(keys_to_check[i]) != keys_values.end());
    EXPECT_EQ(keys_values[keys_to_check[i]], std::vector<Value>(values_to_check.begin() + i * value_dim,
                                                                values_to_check.begin() + (i + 1) * value_dim));
  }
  keys_values.clear();
  value_to_insert.clear();
  keys_to_check.clear();
  values_to_check.clear();
}

/// Feature: test cpu hash table all api.
/// Description: test cpu hash table data structure and interface.
/// Expectation: all interface work normally.
TEST_F(TestCPUHashTable, test_cpu_hash_table_export_slice) {
  size_t value_dim = 1024;
  size_t key_num = 512;

  size_t slice_size_in_mb = 1;
  size_t slice_num = (slice_size_in_mb << 20) / (value_dim * sizeof(Value));
  CPUHashTable<Key, Value> hash_table(value_dim, 0.0);

  // Keys and values to insert.
  std::vector<Key> keys_to_insert(key_num);
  std::iota(keys_to_insert.begin(), keys_to_insert.end(), 0);

  std::vector<Value> value_to_insert(key_num * value_dim);
  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < value_dim; j++) {
      value_to_insert[i * value_dim + j] = static_cast<Value>(i);
    }
  }

  // Keys and values check map.
  std::map<Key, std::vector<Value>> keys_values;
  for (size_t i = 0; i < key_num; ++i) {
    keys_values.emplace(keys_to_insert[i], std::vector<Value>(value_to_insert.begin() + i * value_dim,
                                                              value_to_insert.begin() + (i + 1) * value_dim));
  }
  EXPECT_EQ(keys_values.size(), key_num);

  // Keys, values and statuses check tensor.
  std::vector<Key> keys_to_check(key_num);
  std::vector<Value> values_to_check(key_num * value_dim);
  std::vector<HashTableElementStatus> statuses_to_check(key_num);

  // Test all api of cpu hash table.
  EXPECT_TRUE(hash_table.Insert(keys_to_insert.data(), key_num, value_to_insert.data(), nullptr));

  std::vector<HashTableExportData> full_export_data, incre_export_data;
  bool last_slice = false;
  while (!last_slice) {
    HashTableExportData ret;
    EXPECT_NO_THROW(ret = hash_table.ExportSlice(false, &last_slice, slice_size_in_mb));
    EXPECT_EQ(ret[0]->size(), sizeof(Key) * slice_num);
    EXPECT_EQ(ret[1]->size(), sizeof(Value) * value_dim * slice_num);
    EXPECT_EQ(ret[2]->size(), sizeof(HashTableElementStatus) * slice_num);
    full_export_data.push_back(ret);
  }

  last_slice = false;
  while (!last_slice) {
    HashTableExportData ret;
    EXPECT_NO_THROW(ret = hash_table.ExportSlice(true, &last_slice, slice_size_in_mb));
    EXPECT_EQ(ret[0]->size(), sizeof(Key) * slice_num);
    EXPECT_EQ(ret[1]->size(), sizeof(Value) * value_dim * slice_num);
    EXPECT_EQ(ret[2]->size(), sizeof(HashTableElementStatus) * slice_num);
    incre_export_data.push_back(ret);
  }

  EXPECT_TRUE(hash_table.is_dirty());

  EXPECT_EQ(full_export_data.size(), incre_export_data.size());
  EXPECT_EQ(full_export_data.size(), key_num / slice_num);

  for (size_t i = 0; i < full_export_data.size(); ++i) {
    EXPECT_EQ(*full_export_data[i][0], *incre_export_data[i][0]);
    EXPECT_EQ(*full_export_data[i][1], *incre_export_data[i][1]);
    EXPECT_EQ(*full_export_data[i][2], *incre_export_data[i][2]);
  }

  std::map<Key, std::vector<Value>> keys_values_to_check;
  for (auto &item : full_export_data) {
    auto &key_data = item[0];
    auto &value_data = item[1];
    auto &status_data = item[2];
    size_t slice_key_num = key_data->size() / sizeof(Key);
    Key *key_ptr = reinterpret_cast<Key *>(key_data->data());
    Value *value_ptr = reinterpret_cast<Value *>(value_data->data());
    HashTableElementStatus *status_ptr = reinterpret_cast<HashTableElementStatus *>(status_data->data());

    for (size_t i = 0; i < slice_key_num; i++) {
      keys_values_to_check.emplace(key_ptr[i],
                                   std::vector<Value>(value_ptr + i * value_dim, value_ptr + (i + 1) * value_dim));

      EXPECT_EQ(status_ptr[i], HashTableElementStatus::kModified);
    }
  }

  EXPECT_EQ(keys_values_to_check.size(), key_num);
  EXPECT_EQ(keys_values_to_check, keys_values);

  EXPECT_TRUE(hash_table.Clear());
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore