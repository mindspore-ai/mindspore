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

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "common/utils.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "mindrecord/include/shard_segment.h"
#include "ut_common.h"

using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

namespace mindspore {
namespace mindrecord {
class TestShardSegment : public UT::Common {
 public:
  TestShardSegment() {}
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

TEST_F(TestShardSegment, TestShardSegment) {
  MS_LOG(INFO) << FormatInfo("Test Shard Segment");
  std::string file_name = "./imagenet.shard01";

  ShardSegment dataset;
  dataset.Open(file_name, 4);

  auto x = dataset.GetCategoryFields();
  for (const auto &fields : x.second) {
    MS_LOG(INFO) << "Get category field: " << fields;
  }

  ASSERT_TRUE(dataset.SetCategoryField("label") == SUCCESS);
  ASSERT_TRUE(dataset.SetCategoryField("laabel_0") == FAILED);

  MS_LOG(INFO) << "Read category info: " << dataset.ReadCategoryInfo().second;

  auto ret = dataset.ReadAtPageByName("822", 0, 10);
  auto images = ret.second;
  MS_LOG(INFO) << "category field: 822, images count: " << images.size() << ", image[0] size: " << images[0].size();

  auto ret1 = dataset.ReadAtPageByName("823", 0, 10);
  auto images2 = ret1.second;
  MS_LOG(INFO) << "category field: 823, images count: " << images2.size();

  auto ret2 = dataset.ReadAtPageById(1, 0, 10);
  auto images3 = ret2.second;
  MS_LOG(INFO) << "category id: 1, images count: " << images3.size() << ", image[0] size: " << images3[0].size();

  auto ret3 = dataset.ReadAllAtPageByName("822", 0, 10);
  auto images4 = ret3.second;
  MS_LOG(INFO) << "category field: 822, images count: " << images4.size();

  auto ret4 = dataset.ReadAllAtPageById(1, 0, 10);
  auto images5 = ret4.second;
  MS_LOG(INFO) << "category id: 1, images count: " << images5.size();
}

TEST_F(TestShardSegment, TestReadAtPageByNameOfCategoryName) {
  MS_LOG(INFO) << FormatInfo("Test ReadAtPageByName of error category_name and category_field");
  std::string file_name = "./imagenet.shard01";

  ShardSegment dataset;
  dataset.Open(file_name, 4);

  auto x = dataset.GetCategoryFields();
  for (const auto &fields : x.second) {
    MS_LOG(INFO) << "Get category field: " << fields;
  }

  string category_name = "82Cus";
  string category_field = "laabel_0";

  ASSERT_TRUE(dataset.SetCategoryField("label") == SUCCESS);
  ASSERT_TRUE(dataset.SetCategoryField(category_field) == FAILED);

  MS_LOG(INFO) << "Read category info: " << dataset.ReadCategoryInfo().second;

  auto ret = dataset.ReadAtPageByName(category_name, 0, 10);
  EXPECT_TRUE(ret.first == FAILED);
}

TEST_F(TestShardSegment, TestReadAtPageByIdOfCategoryId) {
  MS_LOG(INFO) << FormatInfo("Test ReadAtPageById of error categoryId");
  std::string file_name = "./imagenet.shard01";

  ShardSegment dataset;
  dataset.Open(file_name, 4);

  auto x = dataset.GetCategoryFields();
  for (const auto &fields : x.second) {
    MS_LOG(INFO) << "Get category field: " << fields;
  }

  int64_t categoryId = 2251799813685247;
  MS_LOG(INFO) << "Input category id: " << categoryId;

  ASSERT_TRUE(dataset.SetCategoryField("label") == SUCCESS);
  MS_LOG(INFO) << "Read category info: " << dataset.ReadCategoryInfo().second;

  auto ret2 = dataset.ReadAtPageById(categoryId, 0, 10);
  EXPECT_TRUE(ret2.first == FAILED);
}

TEST_F(TestShardSegment, TestReadAtPageByIdOfPageNo) {
  MS_LOG(INFO) << FormatInfo("Test ReadAtPageById of error page_no");
  std::string file_name = "./imagenet.shard01";

  ShardSegment dataset;
  dataset.Open(file_name, 4);

  auto x = dataset.GetCategoryFields();
  for (const auto &fields : x.second) {
    MS_LOG(INFO) << "Get category field: " << fields;
  }

  int64_t page_no = 2251799813685247;
  MS_LOG(INFO) << "Input page no: " << page_no;

  ASSERT_TRUE(dataset.SetCategoryField("label") == SUCCESS);
  MS_LOG(INFO) << "Read category info: " << dataset.ReadCategoryInfo().second;

  auto ret2 = dataset.ReadAtPageById(1, page_no, 10);
  EXPECT_TRUE(ret2.first == FAILED);
}

TEST_F(TestShardSegment, TestReadAtPageByIdOfPageRows) {
  MS_LOG(INFO) << FormatInfo("Test ReadAtPageById of error pageRows");
  std::string file_name = "./imagenet.shard01";

  ShardSegment dataset;
  dataset.Open(file_name, 4);

  auto x = dataset.GetCategoryFields();
  for (const auto &fields : x.second) {
    MS_LOG(INFO) << "Get category field: " << fields;
  }

  int64_t pageRows = 0;
  MS_LOG(INFO) << "Input page rows: " << pageRows;

  ASSERT_TRUE(dataset.SetCategoryField("label") == SUCCESS);
  MS_LOG(INFO) << "Read category info: " << dataset.ReadCategoryInfo().second;

  auto ret2 = dataset.ReadAtPageById(1, 0, pageRows);
  EXPECT_TRUE(ret2.first == FAILED);
}

}  // namespace mindrecord
}  // namespace mindspore
