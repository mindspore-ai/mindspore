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
#include "mindrecord/include/shard_error.h"
#include "mindrecord/include/shard_index_generator.h"
#include "mindrecord/include/shard_index.h"
#include "mindrecord/include/shard_statistics.h"
#include "securec.h"
#include "ut_common.h"

using json = nlohmann::json;
using std::ifstream;
using std::pair;
using std::string;
using std::vector;

using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

namespace mindspore {
namespace mindrecord {
class TestShardIndexGenerator : public UT::Common {
 public:
  TestShardIndexGenerator() {}
};

TEST_F(TestShardIndexGenerator, GetField) {
  MS_LOG(INFO) << FormatInfo("Test ShardIndex: get field");

  int max_num = 1;
  string input_path1 = install_root + "/test/testCBGData/data/annotation.data";
  std::vector<json> json_buffer1;  // store the image_raw_meta.data
  Common::LoadData(input_path1, json_buffer1, max_num);

  MS_LOG(INFO) << "Fetch fields: ";
  for (auto &j : json_buffer1) {
    auto v_name = ShardIndexGenerator::GetField("anno_tool", j);
    auto v_attr_name = ShardIndexGenerator::GetField("entity_instances.attributes.attr_name", j);
    auto v_entity_name = ShardIndexGenerator::GetField("entity_instances.entity_name", j);
    vector<string> names = {"\"CVAT\""};
    for (unsigned int i = 0; i != names.size(); i++) {
      ASSERT_EQ(names[i], v_name[i]);
    }
    vector<string> attr_names = {"\"脸部评分\"", "\"特征点\"", "\"points_example\"", "\"polyline_example\"",
                                 "\"polyline_example\""};
    for (unsigned int i = 0; i != attr_names.size(); i++) {
      ASSERT_EQ(attr_names[i], v_attr_name[i]);
    }
    vector<string> entity_names = {"\"276点人脸\"", "\"points_example\"", "\"polyline_example\"",
                                   "\"polyline_example\""};
    for (unsigned int i = 0; i != entity_names.size(); i++) {
      ASSERT_EQ(entity_names[i], v_entity_name[i]);
    }
  }
}
TEST_F(TestShardIndexGenerator, TakeFieldType) {
  MS_LOG(INFO) << FormatInfo("Test ShardSchema: take field Type");

  json schema1 = R"({
    "type": "object",
    "properties": {
      "number":      { "type": "number" },
      "street_name": { "type": "string" },
      "street_type": { "type": "array",
                       "items": { "type": "array",
                                 "items":{ "type": "number"}
                                }
                     }
                  }})"_json;
  json schema2 = R"({"name": {"type": "string"},
                           "label": {"type": "array", "items": {"type": "number"}}})"_json;
  auto type1 = ShardIndexGenerator::TakeFieldType("number", schema1);
  ASSERT_EQ("number", type1);
  auto type2 = ShardIndexGenerator::TakeFieldType("street_name", schema1);
  ASSERT_EQ("string", type2);
  auto type3 = ShardIndexGenerator::TakeFieldType("street_type", schema1);
  ASSERT_EQ("array", type3);

  auto type4 = ShardIndexGenerator::TakeFieldType("name", schema2);
  ASSERT_EQ("string", type4);
  auto type5 = ShardIndexGenerator::TakeFieldType("label", schema2);
  ASSERT_EQ("array", type5);
}
}  // namespace mindrecord
}  // namespace mindspore
