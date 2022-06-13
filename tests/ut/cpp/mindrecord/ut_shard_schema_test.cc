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

#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "minddata/mindrecord/include/shard_page.h"
#include "minddata/mindrecord/include/shard_schema.h"
#include "minddata/mindrecord/include/shard_statistics.h"
#include "securec.h"
#include "ut_common.h"

using json = nlohmann::json;
using std::ifstream;
using std::pair;
using std::string;
using std::vector;

namespace mindspore {
namespace mindrecord {
class TestShardSchema : public UT::Common {
 public:
  TestShardSchema() {}
};

class TestStatistics : public UT::Common {
 public:
  TestStatistics() {}
};

TEST_F(TestShardSchema, BuildSchema) {
  MS_LOG(INFO) << FormatInfo("Test ShardSchema: build schema");

  std::string desc = "this is a test";
  json schema_content = R"({"name": {"type": "string"},
                           "label": {"type": "int32", "shape": [-1]}})"_json;

  std::shared_ptr<Schema> schema = Schema::Build(desc, schema_content);
  ASSERT_NE(schema, nullptr);
  // checkout field name

  schema_content["name%"] = R"({"type": "string"})"_json;
  schema = Schema::Build(desc, schema_content);
  ASSERT_EQ(schema, nullptr);
  schema_content.erase("name%");

  schema_content["na-me"] = R"({"type": "string"})"_json;
  schema = Schema::Build(desc, schema_content);
  ASSERT_EQ(schema, nullptr);
  schema_content.erase("na-me");

  schema_content["name_type.2"] = R"({"type": "string"})"_json;
  schema = Schema::Build(desc, schema_content);
  ASSERT_EQ(schema, nullptr);
  schema_content.erase("name_type.2");

  schema_content["3_name"] = R"({"type": "string"})"_json;
  schema = Schema::Build(desc, schema_content);
  ASSERT_NE(schema, nullptr);
  schema_content.erase("3_name");

  schema_content["test"] = R"({"type": "test"})"_json;
  schema = Schema::Build(desc, schema_content);
  ASSERT_EQ(schema, nullptr);
  schema_content.erase("test");

  schema_content["test"] = R"({"type": "string", "test": "this is for test"})"_json;
  schema = Schema::Build(desc, schema_content);
  ASSERT_EQ(schema, nullptr);
  schema_content.erase("test");
}

TEST_F(TestShardSchema, TestFunction) {
  std::string desc = "this is a test";
  json schema_content = R"({"name": {"type": "string"},
                           "label": {"type": "int32", "shape": [-1]}})"_json;

  std::shared_ptr<Schema> schema = Schema::Build(desc, schema_content);
  ASSERT_NE(schema, nullptr);

  ASSERT_EQ(schema->GetDesc(), desc);

  json schema_json = schema->GetSchema();
  ASSERT_EQ(schema_json["desc"], desc);
  ASSERT_EQ(schema_json["schema"], schema_content);

  ASSERT_EQ(schema->GetSchemaID(), -1);
  schema->SetSchemaID(2);
  ASSERT_EQ(schema->GetSchemaID(), 2);
}

TEST_F(TestStatistics, StatisticPart) {
  MS_LOG(INFO) << FormatInfo("Test statistics");

  std::string statistic =
    "{\"level\": ["
    "{\"key\": \"2018-12\", \"count\": 811}, {\"key\": \"2019-01\", \"count\": 805}, "
    "{\"key\": \"2019-02\", \"count\": 763}, {\"key\": \"2019-03\", \"count\": 793}, "
    "{\"key\": \"2019-04\", \"count\": 773}, {\"key\": \"2019-05\", \"count\": 432}"
    "]}";

  // define statistics
  nlohmann::json statistic_json = json::parse(statistic);
  MS_LOG(INFO) << "statistics: " << statistic;
  std::string desc = "axis";

  std::shared_ptr<Statistics> statistics = Statistics::Build(desc, statistic_json);

  ASSERT_NE(statistics, nullptr);

  MS_LOG(INFO) << "test GetDesc(), result: " << statistics->GetDesc();
  MS_LOG(INFO) << "test GetStatistics, result: " << statistics->GetStatistics().dump();

  statistic_json["test"] = "test";
  statistics = Statistics::Build(desc, statistic_json);
  ASSERT_EQ(statistics, nullptr);

  statistic_json.erase("level");
  statistics = Statistics::Build(desc, statistic_json);
  ASSERT_EQ(statistics, nullptr);
}

}  // namespace mindrecord
}  // namespace mindspore
