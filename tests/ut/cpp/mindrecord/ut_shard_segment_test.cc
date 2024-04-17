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

#include "utils/ms_utils.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "minddata/mindrecord/include/shard_segment.h"
#include "ut_common.h"

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
  dataset.Open({file_name}, true, 4);

  auto fields_ptr = std::make_shared<vector<std::string>>();
  auto status = dataset.GetCategoryFields(&fields_ptr);
  for (const auto &fields : *fields_ptr) {
    MS_LOG(INFO) << "Get category field: " << fields;
  }

  status = dataset.SetCategoryField("label");
  EXPECT_TRUE(status.IsOk());
  status = dataset.SetCategoryField("laabel_0");
  EXPECT_FALSE(status.IsOk());


  std::shared_ptr<std::string> category_ptr;
  status = dataset.ReadCategoryInfo(&category_ptr);
  EXPECT_TRUE(status.IsOk());
  MS_LOG(INFO) << "Read category info: " << *category_ptr;

  auto pages_ptr = std::make_shared<std::vector<std::vector<uint8_t>>>();
  status = dataset.ReadAtPageByName("822", 0, 10, &pages_ptr);
  EXPECT_TRUE(status.IsOk());
  MS_LOG(INFO) << "category field: 822, images count: " << pages_ptr->size() << ", image[0] size: " << ((*pages_ptr)[0]).size();

  auto pages_ptr_1 = std::make_shared<std::vector<std::vector<uint8_t>>>();
  status = dataset.ReadAtPageByName("823", 0, 10, &pages_ptr_1);
  MS_LOG(INFO) << "category field: 823, images count: " << pages_ptr_1->size();

  auto pages_ptr_2 = std::make_shared<std::vector<std::vector<uint8_t>>>();
  status = dataset.ReadAtPageById(1, 0, 10, &pages_ptr_2);
  EXPECT_TRUE(status.IsOk());
  MS_LOG(INFO) << "category id: 1, images count: " << pages_ptr_2->size() << ", image[0] size: " << ((*pages_ptr_2)[0]).size();

  auto pages_ptr_3 = std::make_shared<PAGES_WITH_BLOBS>();
  status = dataset.ReadAllAtPageByName("822", 0, 10, &pages_ptr_3);
  MS_LOG(INFO) << "category field: 822, images count: " << pages_ptr_3->size();

  auto pages_ptr_4 = std::make_shared<PAGES_WITH_BLOBS>();
  status = dataset.ReadAllAtPageById(1, 0, 10, &pages_ptr_4);
  MS_LOG(INFO) << "category id: 1, images count: " << pages_ptr_4->size();
}

TEST_F(TestShardSegment, TestReadAtPageByNameOfCategoryName) {
  MS_LOG(INFO) << FormatInfo("Test ReadAtPageByName of error category_name and category_field");
  std::string file_name = "./imagenet.shard01";

  ShardSegment dataset;
  dataset.Open({file_name}, true, 4);

  auto fields_ptr = std::make_shared<vector<std::string>>();
  auto status = dataset.GetCategoryFields(&fields_ptr);
  for (const auto &fields : *fields_ptr) {
    MS_LOG(INFO) << "Get category field: " << fields;
  }

  string category_name = "82Cus";
  string category_field = "laabel_0";

  status = dataset.SetCategoryField("label");
  EXPECT_TRUE(status.IsOk());
  status = dataset.SetCategoryField(category_field);
  EXPECT_FALSE(status.IsOk());

  std::shared_ptr<std::string> category_ptr;
  status = dataset.ReadCategoryInfo(&category_ptr);
  EXPECT_TRUE(status.IsOk());
  MS_LOG(INFO) << "Read category info: " << *category_ptr;

  auto pages_ptr = std::make_shared<std::vector<std::vector<uint8_t>>>();
  status = dataset.ReadAtPageByName(category_name, 0, 10, &pages_ptr);
  EXPECT_FALSE(status.IsOk());
}

TEST_F(TestShardSegment, TestReadAtPageByIdOfCategoryId) {
  MS_LOG(INFO) << FormatInfo("Test ReadAtPageById of error categoryId");
  std::string file_name = "./imagenet.shard01";

  ShardSegment dataset;
  dataset.Open({file_name}, true,  4);

  auto fields_ptr = std::make_shared<vector<std::string>>();
  auto status = dataset.GetCategoryFields(&fields_ptr);
  for (const auto &fields : *fields_ptr) {
    MS_LOG(INFO) << "Get category field: " << fields;
  }

  int64_t categoryId = 2251799813685247;
  MS_LOG(INFO) << "Input category id: " << categoryId;

  status = dataset.SetCategoryField("label");
  EXPECT_TRUE(status.IsOk());
  std::shared_ptr<std::string> category_ptr;
  status = dataset.ReadCategoryInfo(&category_ptr);
  EXPECT_TRUE(status.IsOk());
  MS_LOG(INFO) << "Read category info: " << *category_ptr;

  auto pages_ptr = std::make_shared<std::vector<std::vector<uint8_t>>>();
  status = dataset.ReadAtPageById(categoryId, 0, 10, &pages_ptr);
  EXPECT_FALSE(status.IsOk());
}

TEST_F(TestShardSegment, TestReadAtPageByIdOfPageNo) {
  MS_LOG(INFO) << FormatInfo("Test ReadAtPageById of error page_no");
  std::string file_name = "./imagenet.shard01";

  ShardSegment dataset;
  dataset.Open({file_name}, true, 4);

  auto fields_ptr = std::make_shared<vector<std::string>>();
  auto status = dataset.GetCategoryFields(&fields_ptr);
  for (const auto &fields : *fields_ptr) {
    MS_LOG(INFO) << "Get category field: " << fields;
  }

  int64_t page_no = 2251799813685247;
  MS_LOG(INFO) << "Input page no: " << page_no;

  status = dataset.SetCategoryField("label");
  EXPECT_TRUE(status.IsOk());


  std::shared_ptr<std::string> category_ptr;
  status = dataset.ReadCategoryInfo(&category_ptr);
  EXPECT_TRUE(status.IsOk());
  MS_LOG(INFO) << "Read category info: " << *category_ptr;

  auto pages_ptr = std::make_shared<std::vector<std::vector<uint8_t>>>();
  status = dataset.ReadAtPageById(1, page_no, 10, &pages_ptr);
  EXPECT_FALSE(status.IsOk());
}

TEST_F(TestShardSegment, TestReadAtPageByIdOfPageRows) {
  MS_LOG(INFO) << FormatInfo("Test ReadAtPageById of error pageRows");
  std::string file_name = "./imagenet.shard01";

  ShardSegment dataset;
  dataset.Open({file_name}, true, 4);

  auto fields_ptr = std::make_shared<vector<std::string>>();
  auto status = dataset.GetCategoryFields(&fields_ptr);
  for (const auto &fields : *fields_ptr) {
    MS_LOG(INFO) << "Get category field: " << fields;
  }

  int64_t pageRows = 0;
  MS_LOG(INFO) << "Input page rows: " << pageRows;

  status = dataset.SetCategoryField("label");
  EXPECT_TRUE(status.IsOk());

  std::shared_ptr<std::string> category_ptr;
  status = dataset.ReadCategoryInfo(&category_ptr);
  EXPECT_TRUE(status.IsOk());
  MS_LOG(INFO) << "Read category info: " << *category_ptr;

  auto pages_ptr = std::make_shared<std::vector<std::vector<uint8_t>>>();
  status = dataset.ReadAtPageById(1, 0, pageRows, &pages_ptr);
  EXPECT_FALSE(status.IsOk());
}

}  // namespace mindrecord
}  // namespace mindspore
