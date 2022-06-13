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
#include <memory>
#include <string>
#include <vector>

#include "configuration.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "minddata/mindrecord/include/shard_error.h"
#include "minddata/mindrecord/include/shard_index.h"
#include "minddata/mindrecord/include/shard_header.h"
#include "minddata/mindrecord/include/shard_statistics.h"
#include "securec.h"
#include "ut_common.h"

namespace mindspore {
namespace mindrecord {
class TestShard : public UT::Common {
 public:
  TestShard() {}
};

TEST_F(TestShard, TestShardSchemaPart) {
  ShardWriterImageNet();

  MS_LOG(INFO) << FormatInfo("Test schema");

  // read schema json from schema dir
  nlohmann::json j = nlohmann::json::parse(kCvatSchema);
  std::string desc = kCvatSchemaDesc;

  std::shared_ptr<Schema> schema = Schema::Build(desc, j);
  ASSERT_TRUE(schema != nullptr);
  MS_LOG(INFO) << "schema description: " << schema->GetDesc() << ", schema:  " <<
    common::SafeCStr(schema->GetSchema().dump());
  for (int i = 1; i <= 4; i++) {
    string filename = std::string("./imagenet.shard0") + std::to_string(i);
    string db_name = std::string("./imagenet.shard0") + std::to_string(i) + ".db";
    remove(common::SafeCStr(filename));
    remove(common::SafeCStr(db_name));
  }
}

TEST_F(TestShard, TestStatisticPart) {
  MS_LOG(INFO) << FormatInfo("Test statistics");

  // define statistics
  MS_LOG(INFO) << "statistics: " << kStatistics[2];
  std::string desc = "statistic desc";
  nlohmann::json statistic_json = json::parse(kStatistics[2]);
  std::shared_ptr<Statistics> statistics = Statistics::Build(desc, statistic_json);
  ASSERT_TRUE(statistics != nullptr);
  MS_LOG(INFO) << "test get_desc(), result: " << statistics->GetDesc();
  MS_LOG(INFO) << "test get_statistics, result: " << statistics->GetStatistics().dump();

  std::string desc2 = "axis";
  nlohmann::json statistic_json2 = R"({})";
  std::shared_ptr<Statistics> statistics2 = Statistics::Build(desc2, statistic_json2);
  ASSERT_TRUE(statistics2 == nullptr);
}

TEST_F(TestShard, TestShardHeaderPart) {
  MS_LOG(INFO) << FormatInfo("Test ShardHeader");
  json schema_json1 = R"({"name": {"type": "string"}, "type": {"type": "string"}})"_json;

  json statistic_json1 = json::parse(
    "{\"statistics\": "
    "{\"level\": ["
    "{\"key\": \"2018-12\", \"count\": 811}, "
    "{\"key\": \"2019-11\", \"count\": 763}"
    "]}}");

  std::string schema_desc1 = "test schema1";
  std::string statistics_desc1 = "test statistics1";

  std::shared_ptr<Schema> schema1 = Schema::Build(schema_desc1, schema_json1);
  ASSERT_TRUE(schema1 != nullptr);
  std::vector<Schema> validate_schema;
  validate_schema.push_back(*schema1);

  std::shared_ptr<Statistics> statistics1 = Statistics::Build(statistics_desc1, statistic_json1["statistics"]);
  ASSERT_TRUE(statistics1 != nullptr);
  std::vector<Statistics> validate_statistics;
  validate_statistics.push_back(*statistics1);

  // init shardHeader
  mindrecord::ShardHeader header_data;

  int res = header_data.AddSchema(schema1);
  ASSERT_EQ(res, 0);
  header_data.AddStatistic(statistics1);
  std::vector<Schema> re_schemas;
  for (auto &schema_ptr : header_data.GetSchemas()) {
    re_schemas.push_back(*schema_ptr);
  }
  ASSERT_EQ(re_schemas, validate_schema);

  std::vector<Statistics> re_statistics;
  for (auto &statistic : header_data.GetStatistics()) {
    re_statistics.push_back(*statistic);
  }
  ASSERT_EQ(re_statistics, validate_statistics);
  std::shared_ptr<Statistics> statistics_ptr;

  auto status = header_data.GetStatisticByID(-1, &statistics_ptr);
  EXPECT_FALSE(status.IsOk());
  status = header_data.GetStatisticByID(10, &statistics_ptr);
  EXPECT_FALSE(status.IsOk());

  // test add index fields
  std::vector<std::pair<uint64_t, std::string>> fields;
  std::pair<uint64_t, std::string> pair1(0, "name");
  fields.push_back(pair1);
  status = header_data.AddIndexFields(fields);
  EXPECT_TRUE(status.IsOk());
  std::vector<std::pair<uint64_t, std::string>> resFields = header_data.GetFields();
  ASSERT_EQ(resFields, fields);
}

}  // namespace mindrecord
}  // namespace mindspore
