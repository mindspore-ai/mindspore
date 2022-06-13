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

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "minddata/mindrecord/include/shard_page.h"
#include "ut_common.h"

using json = nlohmann::json;
using std::ifstream;
using std::pair;
using std::string;
using std::vector;

namespace mindspore {
namespace mindrecord {
class TestShardPage : public UT::Common {
 public:
  TestShardPage() {}
};

TEST_F(TestShardPage, TestBasic) {
  MS_LOG(INFO) << FormatInfo("Test ShardPage Basic");
  const int kGoldenPageId = 12;
  const int kGoldenShardId = 20;

  const std::string kGoldenType = kPageTypeRaw;
  const int kGoldenTypeId = 2;
  const uint64_t kGoldenStart = 10;
  const uint64_t kGoldenEnd = 20;

  std::vector<std::pair<int, uint64_t>> golden_row_group = {{1, 2}, {2, 4}, {4, 6}};
  const uint64_t kGoldenSize = 100;
  const uint64_t kOffset = 6;

  Page page =
    Page(kGoldenPageId, kGoldenShardId, kGoldenType, kGoldenTypeId, kGoldenStart, kGoldenEnd, golden_row_group, kGoldenSize);
  EXPECT_EQ(kGoldenPageId, page.GetPageID());
  EXPECT_EQ(kGoldenShardId, page.GetShardID());
  EXPECT_EQ(kGoldenTypeId, page.GetPageTypeID());
  ASSERT_TRUE(kGoldenType == page.GetPageType());
  EXPECT_EQ(kGoldenSize, page.GetPageSize());
  EXPECT_EQ(kGoldenStart, page.GetStartRowID());
  EXPECT_EQ(kGoldenEnd, page.GetEndRowID());
  ASSERT_TRUE(std::make_pair(4, kOffset) == page.GetLastRowGroupID());
  ASSERT_TRUE(golden_row_group == page.GetRowGroupIds());
}

TEST_F(TestShardPage, TestSetter) {
  MS_LOG(INFO) << FormatInfo("Test ShardPage Setter Functions");
  const int kGoldenPageId = 12;
  const int kGoldenShardId = 20;

  const std::string kGoldenType = kPageTypeBlob;
  const int kGoldenTypeId = 2;
  const uint64_t kGoldenStart = 10;
  const uint64_t kGoldenEnd = 20;

  std::vector<std::pair<int, uint64_t>> golden_row_group = {{1, 2}, {2, 4}, {4, 6}};
  const uint64_t kGoldenSize = 100;
  const uint64_t kOffset1 = 6;
  const uint64_t kOffset2 = 3000;
  const uint64_t kOffset3 = 200;

  Page page =
    Page(kGoldenPageId, kGoldenShardId, kGoldenType, kGoldenTypeId, kGoldenStart, kGoldenEnd, golden_row_group, kGoldenSize);
  EXPECT_EQ(kGoldenPageId, page.GetPageID());
  EXPECT_EQ(kGoldenShardId, page.GetShardID());
  EXPECT_EQ(kGoldenTypeId, page.GetPageTypeID());
  ASSERT_TRUE(kGoldenType == page.GetPageType());
  EXPECT_EQ(kGoldenSize, page.GetPageSize());
  EXPECT_EQ(kGoldenStart, page.GetStartRowID());
  EXPECT_EQ(kGoldenEnd, page.GetEndRowID());
  ASSERT_TRUE(std::make_pair(4, kOffset1) == page.GetLastRowGroupID());
  ASSERT_TRUE(golden_row_group == page.GetRowGroupIds());

  const int kNewEnd = 33;
  const int kNewSize = 300;
  std::vector<std::pair<int, uint64_t>> new_row_group = {{0, 100}, {100, 200}, {200, 3000}};
  page.SetEndRowID(kNewEnd);
  page.SetPageSize(kNewSize);
  page.SetRowGroupIds(new_row_group);
  EXPECT_EQ(kGoldenPageId, page.GetPageID());
  EXPECT_EQ(kGoldenShardId, page.GetShardID());
  EXPECT_EQ(kGoldenTypeId, page.GetPageTypeID());
  ASSERT_TRUE(kGoldenType == page.GetPageType());
  EXPECT_EQ(kNewSize, page.GetPageSize());
  EXPECT_EQ(kGoldenStart, page.GetStartRowID());
  EXPECT_EQ(kNewEnd, page.GetEndRowID());
  ASSERT_TRUE(std::make_pair(200, kOffset2) == page.GetLastRowGroupID());
  ASSERT_TRUE(new_row_group == page.GetRowGroupIds());
  page.DeleteLastGroupId();

  EXPECT_EQ(kGoldenPageId, page.GetPageID());
  EXPECT_EQ(kGoldenShardId, page.GetShardID());
  EXPECT_EQ(kGoldenTypeId, page.GetPageTypeID());
  ASSERT_TRUE(kGoldenType == page.GetPageType());
  EXPECT_EQ(3000, page.GetPageSize());
  EXPECT_EQ(kGoldenStart, page.GetStartRowID());
  EXPECT_EQ(kNewEnd, page.GetEndRowID());
  ASSERT_TRUE(std::make_pair(100, kOffset3) == page.GetLastRowGroupID());
  new_row_group.pop_back();
  ASSERT_TRUE(new_row_group == page.GetRowGroupIds());
}

TEST_F(TestShardPage, TestJson) {
  MS_LOG(INFO) << FormatInfo("Test ShardPage json");
  const int kGoldenPageId = 12;
  const int kGoldenShardId = 20;

  const std::string kGoldenType = kPageTypeRaw;
  const int kGoldenTypeId = 2;
  const uint64_t kGoldenStart = 10;
  const uint64_t kGoldenEnd = 20;

  std::vector<std::pair<int, uint64_t>> golden_row_group = {{1, 2}, {2, 4}, {4, 6}};
  const uint64_t kGoldenSize = 100;

  Page page =
    Page(kGoldenPageId, kGoldenShardId, kGoldenType, kGoldenTypeId, kGoldenStart, kGoldenEnd, golden_row_group, kGoldenSize);

  json json_page = page.GetPage();
  EXPECT_EQ(kGoldenPageId, json_page["page_id"]);
  EXPECT_EQ(kGoldenShardId, json_page["shard_id"]);
  EXPECT_EQ(kGoldenTypeId, json_page["page_type_id"]);
  ASSERT_TRUE(kGoldenType == json_page["page_type"]);
  EXPECT_EQ(kGoldenSize, json_page["page_size"]);
  EXPECT_EQ(kGoldenStart, json_page["start_row_id"]);
  EXPECT_EQ(kGoldenEnd, json_page["end_row_id"]);
  json row_group = json_page["row_group_ids"];
  int i = 0;
  ASSERT_TRUE(golden_row_group.size() == row_group.size());
  for (json &row : row_group) {
    ASSERT_TRUE(golden_row_group[i] == std::make_pair(row["id"], row["offset"]));
    ++i;
  }
}
}  // namespace mindrecord
}  // namespace mindspore
