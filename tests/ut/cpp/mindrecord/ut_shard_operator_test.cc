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
#include "minddata/mindrecord/include/shard_category.h"
#include "minddata/mindrecord/include/shard_pk_sample.h"
#include "minddata/mindrecord/include/shard_reader.h"
#include "minddata/mindrecord/include/shard_sample.h"
#include "minddata/mindrecord/include/shard_shuffle.h"
#include "ut_common.h"

namespace mindspore {
namespace mindrecord {
class TestShardOperator : public UT::Common {
 public:
  TestShardOperator() {}

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

TEST_F(TestShardOperator, TestShardSampleBasic) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));

  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name"};

  const int kSampleCount = 8;
  std::vector<std::shared_ptr<ShardOperator>> ops;
  ops.push_back(std::make_shared<ShardSample>(kSampleCount));
  ShardReader dataset;
  dataset.Open({file_name}, true, 4, column_list, ops);
  dataset.Launch();

  int i = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    MS_LOG(INFO) << "index: " << i << ", filename: " << common::SafeCStr((std::get<1>(x[0]))["file_name"]);
    i++;
  }
  dataset.Close();
  ASSERT_TRUE(i <= kSampleCount);
}

TEST_F(TestShardOperator, TestShardSampleWrongNumber) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));

  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name"};

  const int kNum = 5;
  const int kDen = 0;
  std::vector<std::shared_ptr<ShardOperator>> ops;
  ops.push_back(std::make_shared<ShardSample>(kNum, kDen));

  ShardReader dataset;
  dataset.Open({file_name}, true, 4, column_list, ops);
  dataset.Launch();

  int i = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    MS_LOG(INFO) << "index: " << i << ", filename: " << common::SafeCStr((std::get<1>(x[0]))["file_name"]);
    i++;
  }
  dataset.Close();
  ASSERT_TRUE(i <= 5);
}

TEST_F(TestShardOperator, TestShardSampleRatio) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));

  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name"};

  const int kNum = 1;
  const int kDen = 4;
  std::vector<std::shared_ptr<ShardOperator>> ops;
  ops.push_back(std::make_shared<ShardSample>(kNum, kDen));

  ShardReader dataset;
  dataset.Open({file_name}, true, 4, column_list, ops);
  dataset.Launch();

  int i = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    MS_LOG(INFO) << "index: " << i << ", filename: " << common::SafeCStr((std::get<1>(x[0]))["file_name"]);
    i++;
  }
  dataset.Close();
  ASSERT_TRUE(i <= 10);
}

TEST_F(TestShardOperator, TestShardSamplePartition) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));
  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name"};

  const int kNum = 1;
  const int kDen = 4;
  const int kPar = 2;
  std::vector<std::shared_ptr<ShardOperator>> ops;
  ops.push_back(std::make_shared<ShardSample>(kNum, kDen, kPar));

  ShardReader dataset;
  dataset.Open({file_name}, true, 4, column_list, ops);
  dataset.Launch();

  int i = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    MS_LOG(INFO) << "index: " << i << ", filename: " << common::SafeCStr((std::get<1>(x[0]))["file_name"]);
    i++;
  }
  dataset.Close();
  ASSERT_TRUE(i <= 10);
}

TEST_F(TestShardOperator, TestShardPkSamplerBasic) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test pk sampler"));

  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name", "label"};

  std::vector<std::shared_ptr<ShardOperator>> ops;
  ops.push_back(std::make_shared<ShardPkSample>("label", 2, 0));

  ShardReader dataset;
  dataset.Open({file_name},true, 4, column_list, ops);
  dataset.Launch();

  int i = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    std::cout << "index: " << i << ", filename: " << common::SafeCStr((std::get<1>(x[0]))["file_name"])
              << ", label: " << common::SafeCStr((std::get<1>(x[0]))["label"].dump()) << std::endl;
    i++;
  }
  dataset.Close();
  ASSERT_TRUE(i == 20);
}  // namespace mindrecord

TEST_F(TestShardOperator, TestShardPkSamplerNumClass) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test pk sampler"));

  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name", "label"};

  std::vector<std::shared_ptr<ShardOperator>> ops;
  ops.push_back(std::make_shared<ShardPkSample>("label", 2, 3, 0, 0));

  ShardReader dataset;
  dataset.Open({file_name},true, 4, column_list, ops);
  dataset.Launch();

  int i = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;

    std::cout << "index: " << i << ", filename: " << common::SafeCStr((std::get<1>(x[0]))["file_name"])
              << ", label: " << common::SafeCStr((std::get<1>(x[0]))["label"].dump()) << std::endl;
    i++;
  }
  dataset.Close();
  ASSERT_TRUE(i == 6);
}

TEST_F(TestShardOperator, TestShardCategory) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));

  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name", "label"};

  std::vector<std::shared_ptr<ShardOperator>> ops;
  std::vector<std::pair<std::string, std::string>> categories;
  categories.emplace_back("label", "257");
  categories.emplace_back("label", "302");
  categories.emplace_back("label", "132");
  ops.push_back(std::make_shared<ShardCategory>(categories));

  ShardReader dataset;
  dataset.Open({file_name}, true, 4, column_list, ops);
  dataset.Launch();

  int i = 0;
  int category_no = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;

    MS_LOG(INFO) << "index: " << i << ", filename: " << common::SafeCStr((std::get<1>(x[0]))["file_name"])
                 << ", label: " << common::SafeCStr((std::get<1>(x[0]))["label"].dump());
    i++;

    ASSERT_TRUE((std::get<1>(x[0]))["label"] == categories[category_no].second);

    category_no++;
    category_no %= static_cast<int>(categories.size());
  }
  dataset.Close();
}

TEST_F(TestShardOperator, TestShardShuffle) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));

  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name", "label"};

  std::vector<std::shared_ptr<ShardOperator>> ops;
  ops.push_back(std::make_shared<ShardShuffle>(1));

  ShardReader dataset;
  dataset.Open({file_name}, true, 16, column_list, ops);
  dataset.Launch();

  int i = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    MS_LOG(INFO) << "index: " << i << ", filename: " << common::SafeCStr((std::get<1>(x[0]))["file_name"])
                 << ", label: " << common::SafeCStr((std::get<1>(x[0]))["label"].dump());
    i++;
  }
  dataset.Close();
}

TEST_F(TestShardOperator, TestShardSampleShuffle) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));

  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name", "label"};

  std::vector<std::shared_ptr<ShardOperator>> ops;
  ops.push_back(std::make_shared<ShardSample>(35));
  ops.push_back(std::make_shared<ShardShuffle>(1));

  ShardReader dataset;
  dataset.Open({file_name}, true, 4, column_list, ops);
  dataset.Launch();

  int i = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    MS_LOG(INFO) << "index: " << i << ", filename: " << common::SafeCStr((std::get<1>(x[0]))["file_name"])
                 << ", label: " << common::SafeCStr((std::get<1>(x[0]))["label"].dump());
    i++;
  }
  dataset.Close();
  ASSERT_LE(i, 35);
}

TEST_F(TestShardOperator, TestShardShuffleSample) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));

  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name", "label"};

  std::vector<std::shared_ptr<ShardOperator>> ops;
  ops.push_back(std::make_shared<ShardShuffle>(1));
  const int kSampleSize = 1000;
  ops.push_back(std::make_shared<ShardSample>(kSampleSize));

  ShardReader dataset;
  dataset.Open({file_name}, true,  4, column_list, ops);
  dataset.Launch();

  int i = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    MS_LOG(INFO) << "index: " << i << ", filename: " << common::SafeCStr((std::get<1>(x[0]))["file_name"])
                 << ", label: " << common::SafeCStr((std::get<1>(x[0]))["label"].dump());
    i++;
  }
  dataset.Close();
  ASSERT_TRUE(i <= kSampleSize);
}

TEST_F(TestShardOperator, TestShardSampleShuffleSample) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));

  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name", "label"};

  std::vector<std::shared_ptr<ShardOperator>> ops;
  ops.push_back(std::make_shared<ShardSample>(100));
  ops.push_back(std::make_shared<ShardShuffle>(10));
  ops.push_back(std::make_shared<ShardSample>(35));

  ShardReader dataset;
  dataset.Open({file_name}, true, 4, column_list, ops);
  dataset.Launch();

  int i = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    MS_LOG(INFO) << "index: " << i << ", filename: " << common::SafeCStr((std::get<1>(x[0]))["file_name"])
                 << ", label: " << common::SafeCStr((std::get<1>(x[0]))["label"].dump());
    i++;
  }
  dataset.Close();
  ASSERT_LE(i, 35);
}

TEST_F(TestShardOperator, TestShardShuffleCompare) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));
  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name", "label"};

  std::vector<std::shared_ptr<ShardOperator>> ops;
  ops.push_back(std::make_shared<ShardShuffle>(1));

  ShardReader dataset;
  dataset.Open({file_name}, true,  4, column_list, ops);
  dataset.Launch();

  ShardReader compare_dataset;
  compare_dataset.Open({file_name},true, 4, column_list);
  compare_dataset.Launch();

  int i = 0;
  bool different = false;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    MS_LOG(INFO) << "index: " << i << ", filename: " << common::SafeCStr((std::get<1>(x[0]))["file_name"])
                 << ", label: " << common::SafeCStr((std::get<1>(x[0]))["label"].dump());
    i++;

    auto y = compare_dataset.GetNext();
    if ((std::get<1>(x[0]))["file_name"] != (std::get<1>(y[0]))["file_name"]) different = true;
  }
  dataset.Close();
  compare_dataset.Close();
  ASSERT_TRUE(different);
}

TEST_F(TestShardOperator, TestShardCategoryShuffle1) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));

  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name", "label"};

  std::vector<std::shared_ptr<ShardOperator>> ops;
  std::vector<std::pair<std::string, std::string>> categories;
  categories.emplace_back("label", "257");
  categories.emplace_back("label", "302");
  categories.emplace_back("label", "490");
  ops.push_back(std::make_shared<ShardCategory>(categories));
  ops.push_back(std::make_shared<ShardShuffle>(21));

  ShardReader dataset;
  dataset.Open({file_name}, true, 4, column_list, ops);
  dataset.Launch();

  int i = 0;
  int category_no = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    MS_LOG(INFO) << "index: " << i << ", filename: " << common::SafeCStr((std::get<1>(x[0]))["file_name"])
                 << ", label: " << common::SafeCStr((std::get<1>(x[0]))["label"].dump());
    i++;

    ASSERT_TRUE((std::get<1>(x[0]))["label"] == categories[category_no].second);
    category_no++;
    category_no %= static_cast<int>(categories.size());
  }
  dataset.Close();
}

TEST_F(TestShardOperator, TestShardCategoryShuffle2) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));

  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name", "label"};

  std::vector<std::shared_ptr<ShardOperator>> ops;
  std::vector<std::pair<std::string, std::string>> categories;
  categories.emplace_back("label", "257");
  categories.emplace_back("label", "302");
  categories.emplace_back("label", "132");
  ops.push_back(std::make_shared<ShardShuffle>(32));
  ops.push_back(std::make_shared<ShardCategory>(categories));

  ShardReader dataset;
  dataset.Open({file_name}, true, 4, column_list, ops);
  dataset.Launch();

  int i = 0;
  int category_no = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    MS_LOG(INFO) << "index: " << i << ", filename: " << common::SafeCStr((std::get<1>(x[0]))["file_name"])
                 << ", label: " << common::SafeCStr((std::get<1>(x[0]))["label"].dump());
    i++;
    ASSERT_TRUE((std::get<1>(x[0]))["label"] == categories[category_no].second);
    category_no++;
    category_no %= static_cast<int>(categories.size());
  }
  dataset.Close();
}

TEST_F(TestShardOperator, TestShardCategorySample) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));

  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name", "label"};

  std::vector<std::shared_ptr<ShardOperator>> ops;
  std::vector<std::pair<std::string, std::string>> categories;
  categories.emplace_back("label", "257");
  categories.emplace_back("label", "302");
  categories.emplace_back("label", "132");
  const int kSampleSize = 17;
  ops.push_back(std::make_shared<ShardSample>(kSampleSize));
  ops.push_back(std::make_shared<ShardCategory>(categories));

  ShardReader dataset;
  dataset.Open({file_name},true, 4, column_list, ops);
  dataset.Launch();

  int i = 0;
  int category_no = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    MS_LOG(INFO) << "index: " << i << ", filename: " << common::SafeCStr((std::get<1>(x[0]))["file_name"])
                 << ", label: " << common::SafeCStr((std::get<1>(x[0]))["label"].dump());
    i++;

    ASSERT_TRUE((std::get<1>(x[0]))["label"] == categories[category_no].second);
    category_no++;
    category_no %= static_cast<int>(categories.size());
  }
  dataset.Close();
  ASSERT_EQ(category_no, 0);
  ASSERT_TRUE(i <= kSampleSize);
}

TEST_F(TestShardOperator, TestShardCategorySampleShuffle) {
  MS_LOG(INFO) << common::SafeCStr(FormatInfo("Test read imageNet"));

  std::string file_name = "./imagenet.shard01";
  auto column_list = std::vector<std::string>{"file_name", "label"};

  std::vector<std::shared_ptr<ShardOperator>> ops;
  std::vector<std::pair<std::string, std::string>> categories;
  categories.emplace_back("label", "257");
  categories.emplace_back("label", "302");
  categories.emplace_back("label", "132");
  const int kSampleSize = 17;
  ops.push_back(std::make_shared<ShardSample>(kSampleSize));
  ops.push_back(std::make_shared<ShardCategory>(categories));
  ops.push_back(std::make_shared<ShardShuffle>(100));

  ShardReader dataset;
  dataset.Open({file_name}, true, 4, column_list, ops);
  dataset.Launch();

  int i = 0;
  int category_no = 0;
  while (true) {
    auto x = dataset.GetNext();
    if (x.empty()) break;
    MS_LOG(INFO) << "index: " << i << ", filename: " << common::SafeCStr((std::get<1>(x[0]))["file_name"])
                 << ", label: " << common::SafeCStr((std::get<1>(x[0]))["label"].dump());
    i++;

    ASSERT_TRUE((std::get<1>(x[0]))["label"] == categories[category_no].second);
    category_no++;
    category_no %= static_cast<int>(categories.size());
  }
  dataset.Close();
  ASSERT_EQ(category_no, 0);
  ASSERT_TRUE(i <= kSampleSize);
}
}  // namespace mindrecord
}  // namespace mindspore
