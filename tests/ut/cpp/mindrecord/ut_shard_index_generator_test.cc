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
#include "minddata/mindrecord/include/shard_error.h"
#include "minddata/mindrecord/include/shard_index_generator.h"
#include "minddata/mindrecord/include/shard_index.h"
#include "minddata/mindrecord/include/shard_statistics.h"
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
