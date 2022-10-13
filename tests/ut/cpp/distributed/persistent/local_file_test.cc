/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
namespace persistent {
class TestLocalFile : public UT::Common {
 public:
  TestLocalFile() = default;
  virtual ~TestLocalFile() = default;
  void SetUp() override {}
  void TearDown() override {}
};

/// Feature: test parameter persistent storage and resotre.
/// Description: Modify part of the Embedding table content, persist it to the file, and read it from the file again.
/// Expectation: The content after persistent recovery is consistent with expectations.
TEST_F(TestLocalFile, test_read_write_by_ids_normal_size) {
  std::string storage_file_path = "./storage";
  if (!distributed::storage::FileIOUtils::IsFileOrDirExist(storage_file_path)) {
    distributed::storage::FileIOUtils::CreateDir(storage_file_path);
  }

  std::map<std::string, std::string> config_map;
  config_map[distributed::storage::kFileStoragePath] = storage_file_path;
  std::shared_ptr<storage::StorageBase> storage_ =
    std::make_shared<storage::LocalFile>(config_map, sizeof(int32_t), 10 * sizeof(int));
  EXPECT_NO_THROW(storage_->Initialize());

  size_t ids_num = 10000;
  size_t table_size = ids_num * 10;
  size_t miss_num = 0;

  std::vector<int32_t> ids;
  for (int i = 1; i <= ids_num; i++) {
    ids.emplace_back(i);
  }

  std::vector<int> write_data;
  for (int i = 0; i < table_size; i++) {
    write_data.emplace_back(i);
  }

  std::vector<int> read_data(table_size);
  std::vector<size_t> missing(ids_num);

  EXPECT_NO_THROW(storage_->Write(write_data.data(), ids_num, ids.data()));
  EXPECT_NO_THROW(storage_->Read(ids_num, ids.data(), read_data.data(), &miss_num, missing.data()));

  EXPECT_EQ(miss_num, 0);
  for (int i = 0; i < table_size; i++) {
    EXPECT_EQ(i, read_data[i]);
  }
}
}  // namespace persistent
}  // namespace distributed
}  // namespace mindspore