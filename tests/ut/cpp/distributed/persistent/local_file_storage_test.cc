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

#include "common/common_test.h"

#include <memory>
#include <map>
#include <vector>
#include <string>

#include "distributed/persistent/storage/local_file.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace distributed {
namespace storage {
class TestLocalFileStorage : public UT::Common {
 public:
  TestLocalFileStorage() = default;
  virtual ~TestLocalFileStorage() = default;

  void SetUp() override {}
  void TearDown() override {}
};

/// Feature: Test local file persistent storage.
/// Description: Write memory content to the file, and read it from the file again.
/// Expectation: All interface work normally or throw expectant exception.
TEST_F(TestLocalFileStorage, test_local_file_storage) {
  std::string storage_file_path = "./local_file_storage";
  if (!FileIOUtils::IsFileOrDirExist(storage_file_path)) {
    FileIOUtils::CreateDir(storage_file_path);
  }

  auto ret = FileUtils::GetRealPath(storage_file_path.c_str());
  if (!ret.has_value()) {
    MS_LOG(EXCEPTION) << "Cannot get real path of local file storage.";
  }

  std::map<std::string, std::string> config_map;
  std::string real_storage_file_path = ret.value();
  config_map.emplace(kFileStoragePath, real_storage_file_path);
  size_t embedding_dim = 8;
  config_map.emplace(kElementSize, std::to_string(embedding_dim));
  // The max block length 160 bytes
  size_t max_block_len = 160;
  config_map.emplace(kMaxBlockLength, std::to_string(max_block_len));

  std::unique_ptr<StorageBase<int, float>> local_file = std::make_unique<LocalFile<int, float>>(config_map);
  EXPECT_NE(local_file, nullptr);
  EXPECT_NO_THROW(local_file->Initialize());

  size_t key_num = 10;
  std::vector<int> keys(key_num);
  std::iota(keys.begin(), keys.end(), 0);
  std::vector<float> values_to_write(key_num * embedding_dim);
  std::vector<float> values_to_read(key_num * embedding_dim);

  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < embedding_dim; j++) {
      values_to_write[i * embedding_dim + j] = static_cast<float>(i);
    }
  }

  // Test write and read for keys which doesn't exist, write values first and read the values.
  EXPECT_NO_THROW(local_file->Write({keys.data(), keys.size() * sizeof(int)},
                                    {values_to_write.data(), values_to_write.size() * sizeof(float)}));

  EXPECT_NO_THROW(local_file->Read({keys.data(), keys.size() * sizeof(int)},
                                   {values_to_read.data(), values_to_write.size() * sizeof(float)}));

  EXPECT_EQ(values_to_read, values_to_write);

  // Change values.
  for (size_t i = 0; i < key_num; i++) {
    for (size_t j = 0; j < embedding_dim; j++) {
      values_to_write[i * embedding_dim + j] = static_cast<float>(i) * 10.0;
    }
  }

  // Test write and read for keys which exist, write new values for these keys and then read the values.
  EXPECT_NO_THROW(local_file->Write({keys.data(), keys.size() * sizeof(int)},
                                    {values_to_write.data(), values_to_write.size() * sizeof(float)}));
  EXPECT_NO_THROW(local_file->Read({keys.data(), keys.size() * sizeof(int)},
                                   {values_to_read.data(), values_to_write.size() * sizeof(float)}));
  EXPECT_EQ(values_to_read, values_to_write);

  std::unique_ptr<std::vector<int>> all_keys = nullptr;
  EXPECT_NO_THROW((all_keys = local_file->GetAllKeys()));
  EXPECT_NE(all_keys, nullptr);
  std::sort(all_keys->begin(), all_keys->end());
  EXPECT_EQ(*all_keys, keys);

  int key_not_exist = -1;
  float value_not_exist = 0.0;
  // Test writing values length is not equal keys length.
  EXPECT_THROW(local_file->Write({&key_not_exist, sizeof(int)}, {&value_not_exist, 1}), std::runtime_error);

  // Test reading a key which doesn't exist in file.
  EXPECT_NO_THROW(
    local_file->Read({&key_not_exist, sizeof(int)}, {values_to_read.data(), embedding_dim * sizeof(float)}));

  // Test readding values length is less than keys length.
  EXPECT_THROW(local_file->Read({&key_not_exist, sizeof(int)}, {&value_not_exist, 1}), std::runtime_error);

  EXPECT_NO_THROW(local_file->Finalize());
}
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore
