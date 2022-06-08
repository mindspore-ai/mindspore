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

#include "distributed/persistent/data.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace distributed {
namespace persistent {
class TestPersistStorage : public UT::Common {
 public:
  TestPersistStorage() = default;
  virtual ~TestPersistStorage() = default;

  void SetUp() override {}
  void TearDown() override {}
};

/// Feature: test parameter persistent storage and resotre.
/// Description: Modify part of the Embedding table content, persist it to the file, and read it from the file again.
/// Expectation: The content after persistent recovery is consistent with expectations.
TEST_F(TestPersistStorage, test_embedding_storage) {
  int vocab = 8000;
  int emb_dim = 80;
  int total_dim = vocab * emb_dim;

  std::shared_ptr<std::vector<int>> embedding_shape = std::make_shared<std::vector<int>>();
  embedding_shape->push_back(vocab);
  embedding_shape->push_back(emb_dim);

  std::vector<int> data = std::vector<int>(total_dim, 1);
  auto data_ptr = std::make_shared<std::vector<int>>(data);
  PersistentData<int> embedding_table(data_ptr, embedding_shape);

  std::vector<int> shape = *(embedding_table.shape());
  for (size_t i = 0; i < shape.size(); i++) {
    EXPECT_EQ(shape[i], embedding_shape->at(i));
  }

  std::string storage_file_path = "./storage";
  if (!distributed::storage::FileIOUtils::IsFileOrDirExist(storage_file_path)) {
    distributed::storage::FileIOUtils::CreateDir(storage_file_path);
  }

  auto ret = FileUtils::GetRealPath(storage_file_path.c_str());
  if (!ret.has_value()) {
    MS_LOG(EXCEPTION) << "Cannot get real path of persistent storage file for parameter.";
  }

  std::string real_storage_file_path = ret.value();

  std::map<std::string, std::string> config_map;
  config_map[distributed::storage::kFileStoragePath] = real_storage_file_path;
  embedding_table.Initialize(config_map);
  auto dirty_info = distributed::storage::DirtyInfo();

  EXPECT_NO_THROW(embedding_table.Persist(dirty_info));
  EXPECT_NO_THROW(embedding_table.Restore());

  auto embdding_table_data = embedding_table.MutableData();

  for (size_t i = 0; i < emb_dim * 3; i++) {
    EXPECT_EQ(data[i], embdding_table_data->at(i));
  }

  dirty_info.push_back(1);
  for (size_t i = 0; i < emb_dim; i++) {
    (embedding_table.data())[emb_dim + i] = i;
    data[emb_dim + i] = i;
  }

  EXPECT_NO_THROW(embedding_table.Persist(dirty_info));
  EXPECT_NO_THROW(embedding_table.Restore());

  embdding_table_data = embedding_table.MutableData();

  for (size_t i = 0; i < emb_dim * 3; i++) {
    EXPECT_EQ(data[i], embdding_table_data->at(i));
  }
}
}  // namespace persistent
}  // namespace distributed
}  // namespace mindspore
