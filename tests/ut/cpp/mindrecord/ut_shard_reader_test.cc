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

#include "common/utils.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "mindrecord/include/shard_reader.h"
#include "mindrecord/include/shard_sample.h"
#include "ut_common.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

namespace mindspore {
namespace mindrecord {
class TestShardReader : public UT::Common {
 public:
  TestShardReader() {}
};

TEST_F(TestShardReader, TestShardReaderGeneral) {
  MS_LOG(INFO) << FormatInfo("Test read imageNet");
  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name"};

  ShardReader dataset;
  dataset.Open(file_name, 4, column_list);
  dataset.Launch();

  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto& j : x) {
      for (auto& item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << item.key() << ", value: " << item.value().dump();
      }
    }
  }
  dataset.Finish();
}

TEST_F(TestShardReader, TestShardReaderSample) {
  MS_LOG(INFO) << FormatInfo("Test read imageNet");
  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name"};

  std::vector<std::shared_ptr<ShardOperator>> ops;
  ops.push_back(std::make_shared<ShardSample>(17));
  ShardReader dataset;
  dataset.Open(file_name, 4, column_list, ops);
  dataset.Launch();

  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto& j : x) {
      for (auto& item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << item.key() << ", value: " << item.value().dump();
      }
    }
  }
  dataset.Finish();
  dataset.Close();
}

TEST_F(TestShardReader, TestShardReaderBlock) {
  MS_LOG(INFO) << FormatInfo("Test read imageNet with block way");
  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"label"};

  std::vector<std::shared_ptr<ShardOperator>> ops;
  ops.push_back(std::make_shared<ShardSample>(3));
  ShardReader dataset;
  const bool kBlockReader = true;
  dataset.Open(file_name, 4, column_list, ops, kBlockReader);
  dataset.Launch();

  while (true) {
    auto x = dataset.GetBlockNext();
    if (x.empty()) break;
    for (auto& j : x) {
      for (auto& item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << item.key() << ", value: " << item.value().dump();
      }
    }
  }
  dataset.Finish();
  dataset.Close();
}

TEST_F(TestShardReader, TestShardReaderEasy) {
  MS_LOG(INFO) << FormatInfo("Test read imageNet");
  std::string file_name = "./imagenet.shard01";
  ShardReader dataset;
  dataset.Open(file_name);
  dataset.Launch();

  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto& j : x) {
      for (auto& item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << item.key() << ", value: " << item.value().dump();
      }
    }
  }
  dataset.Finish();
}

TEST_F(TestShardReader, TestShardReaderColumnNotInIndex) {
  MS_LOG(INFO) << FormatInfo("Test read imageNet");
  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"label"};
  ShardReader dataset;
  MSRStatus ret = dataset.Open(file_name, 4, column_list);
  ASSERT_EQ(ret, SUCCESS);
  dataset.Launch();

  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto& j : x) {
      for (auto& item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << item.key() << ", value: " << item.value().dump();
      }
    }
  }
  dataset.Finish();
}

TEST_F(TestShardReader, TestShardReaderColumnNotInSchema) {
  MS_LOG(INFO) << FormatInfo("Test read imageNet");
  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_namex"};
  ShardReader dataset;
  MSRStatus ret = dataset.Open(file_name, 4, column_list);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestShardReader, TestShardVersion) {
  MS_LOG(INFO) << FormatInfo("Test shard version");
  std::string file_name = "./imagenet.shard01";
  ShardReader dataset;
  MSRStatus ret = dataset.Open(file_name, 4);
  ASSERT_EQ(ret, SUCCESS);
  dataset.Launch();

  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto& j : x) {
      MS_LOG(INFO) << "result size: " << std::get<0>(j).size();
      for (auto& item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << common::SafeCStr(item.key()) << ", value: " << common::SafeCStr(item.value().dump());
      }
    }
  }
  dataset.Finish();
}

TEST_F(TestShardReader, TestShardReaderDir) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));
  std::string file_name = "./";
  auto column_list = std::vector<std::string>{"file_name"};

  ShardReader dataset;
  MSRStatus ret = dataset.Open(file_name, 4, column_list);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestShardReader, TestShardReaderConsumer) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));
  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name"};

  ShardReader dataset;
  dataset.Open(file_name, -481565535, column_list);
  dataset.Launch();

  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    for (auto& j : x) {
      for (auto& item : std::get<1>(j).items()) {
        MS_LOG(INFO) << "key: " << common::SafeCStr(item.key()) << ", value: " << common::SafeCStr(item.value().dump());
      }
    }
  }
  dataset.Finish();
}
}  // namespace mindrecord
}  // namespace mindspore
