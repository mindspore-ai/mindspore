/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "utils/ms_utils.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "minddata/mindrecord/include/shard_reader.h"
#include "minddata/mindrecord/include/shard_sample.h"
#include "ut_common.h"

namespace mindspore {
namespace mindrecord {
class TestShardReader : public UT::Common {
 public:
  TestShardReader() {}
  void SetUp() override { ShardWriterImageNet(); }

  void TearDown() override {
    for (int i = 1; i <= 4; i++) {
      string filename = std::string("./imagenet.shard0") + std::to_string(i);
      string db_name = std::string("./imagenet.shard0") + std::to_string(i) + ".db";
      remove(common::SafeCStr(filename));
      remove(common::SafeCStr(db_name));
    }
  }
};

TEST_F(TestShardReader, TestShardReaderGeneral) {
  MS_LOG(INFO) << FormatInfo("Test read imageNet");
  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name"};

  ShardReader dataset;
  dataset.Open({file_name}, true, 4, column_list);
  dataset.Launch();

  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto &j : x) {
      for (auto &item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << item.key() << ", value: " << item.value().dump();
      }
    }
  }
  dataset.Close();
}

TEST_F(TestShardReader, TestShardReaderLazyLoad) {
  MS_LOG(INFO) << FormatInfo("Test read imageNet");
  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name"};

  ShardReader dataset;
  dataset.Open({file_name}, true, 4, column_list, {}, 0, LoadMode::kLazy);
  dataset.Launch();

  uint32_t count = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto &j : x) {
      for (auto &item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << item.key() << ", value: " << item.value().dump();
      }
    }
    count++;
  }
  ASSERT_TRUE(count == 10);
  dataset.Close();
}

TEST_F(TestShardReader, TestShardReaderSample) {
  MS_LOG(INFO) << FormatInfo("Test read imageNet");
  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name"};

  std::vector<std::shared_ptr<ShardOperator>> ops;
  ops.push_back(std::make_shared<ShardSample>(17));
  ShardReader dataset;
  dataset.Open({file_name}, true, 4, column_list, ops);
  dataset.Launch();

  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto &j : x) {
      for (auto &item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << item.key() << ", value: " << item.value().dump();
      }
    }
  }
  dataset.Close();
}

TEST_F(TestShardReader, TestShardReaderLazyLoadDistributed) {
  MS_LOG(INFO) << FormatInfo("Test read imageNet");
  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name"};

  std::vector<std::shared_ptr<ShardOperator>> ops;
  ops.push_back(std::make_shared<ShardSample>(1, 8));
  ShardReader dataset;
  dataset.Open({file_name}, true, 4, column_list, ops, 0, LoadMode::kLazy);
  dataset.Launch();

  uint32_t count = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto &j : x) {
      for (auto &item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << item.key() << ", value: " << item.value().dump();
      }
    }
    count++;
  }
  ASSERT_TRUE(count == 2);
  dataset.Close();
}

TEST_F(TestShardReader, TestShardReaderEasy) {
  MS_LOG(INFO) << FormatInfo("Test read imageNet");
  std::string file_name = "./imagenet.shard01";
  ShardReader dataset;
  dataset.Open({file_name}, true);
  dataset.Launch();

  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto &j : x) {
      for (auto &item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << item.key() << ", value: " << item.value().dump();
      }
    }
  }
  dataset.Close();
}

TEST_F(TestShardReader, TestShardReaderColumnNotInIndex) {
  MS_LOG(INFO) << FormatInfo("Test read imageNet");
  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"label"};
  ShardReader dataset;
  auto status = dataset.Open({file_name}, true,  4, column_list);
  EXPECT_TRUE(status.IsOk());
  dataset.Launch();

  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto &j : x) {
      for (auto &item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << item.key() << ", value: " << item.value().dump();
      }
    }
  }
  dataset.Close();
}

TEST_F(TestShardReader, TestShardReaderColumnNotInSchema) {
  MS_LOG(INFO) << FormatInfo("Test read imageNet");
  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_namex"};
  ShardReader dataset;
  auto status= dataset.Open({file_name}, true, 4, column_list);
  EXPECT_FALSE(status.IsOk());
}

TEST_F(TestShardReader, TestShardVersion) {
  MS_LOG(INFO) << FormatInfo("Test shard version");
  std::string file_name = "./imagenet.shard01";
  ShardReader dataset;
  auto status = dataset.Open({file_name}, true,  4);
  EXPECT_TRUE(status.IsOk());
  dataset.Launch();

  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto &j : x) {
      MS_LOG(INFO) << "result size: " << std::get<0>(j).size();
      for (auto &item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << common::SafeCStr(item.key()) << ", value: " << common::SafeCStr(item.value().dump());
      }
    }
  }
  dataset.Close();
}

TEST_F(TestShardReader, TestShardReaderDir) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));
  std::string file_name = "./";
  auto column_list = std::vector<std::string>{"file_name"};

  ShardReader dataset;
  auto status = dataset.Open({file_name}, true,  4, column_list);
  EXPECT_FALSE(status.IsOk());
}

TEST_F(TestShardReader, TestShardReaderConsumer) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));
  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name"};

  ShardReader dataset;
  dataset.Open({file_name}, true,  -481565535, column_list);
  dataset.Launch();

  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto &j : x) {
      for (auto &item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << common::SafeCStr(item.key()) << ", value: " << common::SafeCStr(item.value().dump());
      }
    }
  }
  dataset.Close();
}
}  // namespace mindrecord
}  // namespace mindspore
