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

#include <unistd.h>
#include <cstdio>
#include "common/common_test.h"
#include "distributed/persistent/storage/file_io_utils.h"
#include "distributed/recovery/file_configuration.h"

namespace mindspore {
namespace distributed {
namespace recovery {
class TestFileConfiguration : public UT::Common {
 public:
  TestFileConfiguration() = default;
  virtual ~TestFileConfiguration() = default;

  void SetUp() override {}
  void TearDown() override {}
};

/// Feature: test save and load function of file configuration.
/// Description: Create a new local file configuration, put some key-value pairs.
/// Expectation: All the key-value pairs could be saved to the local file and the local file could be loaded
/// successfully.
TEST_F(TestFileConfiguration, SaveAndLoadLocalFile) {
  std::string local_file = "metadata.json";
  char *dir = getcwd(nullptr, 0);
  EXPECT_NE(nullptr, dir);

  std::string path = dir;
  free(dir);
  dir = nullptr;

  std::string full_file_path = path + "/" + local_file;
  if (storage::FileIOUtils::IsFileOrDirExist(full_file_path)) {
    remove(full_file_path.c_str());
  }
  EXPECT_TRUE(!storage::FileIOUtils::IsFileOrDirExist(full_file_path));

  std::unique_ptr<FileConfiguration> config = std::make_unique<FileConfiguration>(full_file_path);
  EXPECT_NE(nullptr, config);
  EXPECT_TRUE(config->Initialize());

  for (int i = 0; i < 10; ++i) {
    config->Put("key_" + std::to_string(i), "value_" + std::to_string(i));
  }
  config->Flush();
  config.reset();

  std::unique_ptr<FileConfiguration> recovery_config = std::make_unique<FileConfiguration>(full_file_path);
  EXPECT_NE(nullptr, recovery_config);
  EXPECT_TRUE(recovery_config->Initialize());

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ("value_" + std::to_string(i), recovery_config->Get("key_" + std::to_string(i), ""));
  }
  EXPECT_FALSE(recovery_config->Exists("key_11"));
  recovery_config.reset();

  remove(full_file_path.c_str());
}
}  // namespace recovery
}  // namespace distributed
}  // namespace mindspore
