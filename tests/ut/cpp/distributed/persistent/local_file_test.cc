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
    std::make_shared<storage::LocalFile>(config_map, sizeof(int), 10 * sizeof(int));

  EXPECT_NO_THROW(storage_->Initialize());

  std::vector<int> ids;
  for (int i = 1; i <= 10000; i++) {
    ids.emplace_back(i);
  }
  std::vector<int> write_data;
  for (int i = 0; i < 100000; i++) {
    write_data.emplace_back(i);
  }
  std::vector<int> read_data(100000);
  std::vector<int> missing(10000);

  EXPECT_NO_THROW(storage_->Write(write_data.data(), ids));
  EXPECT_NO_THROW(storage_->Read(ids, read_data.data(), &missing));

  for (int i = 0; i < 100000; i++) {
    EXPECT_EQ(i, read_data[i]);
  }
}
}  // namespace persistent
}  // namespace distributed
}  // namespace mindspore