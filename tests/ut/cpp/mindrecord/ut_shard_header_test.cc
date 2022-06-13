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
#include "minddata/mindrecord/include/shard_reader.h"
#include "minddata/mindrecord/include/shard_writer.h"
#include "minddata/mindrecord/include/shard_index.h"
#include "minddata/mindrecord/include/shard_header.h"
#include "minddata/mindrecord/include/shard_schema.h"
#include "minddata/mindrecord/include/shard_statistics.h"
#include "securec.h"
#include "ut_common.h"

namespace mindspore {
namespace mindrecord {
class TestShardHeader : public UT::Common {
 public:
  TestShardHeader() {}
};

TEST_F(TestShardHeader, AddIndexFields) {
  MS_LOG(INFO) << FormatInfo("Test ShardHeader: add index fields");

  std::string desc1 = "this is a test1";
  json schema_content1 = R"({"name": {"type": "string"},
                            "box": {"type": "string"},
                            "label": {"type": "int32", "shape": [-1]}})"_json;

  std::string desc2 = "this is a test2";
  json schema_content2 = R"({"names": {"type": "string"},
                            "labels": {"type": "array", "items": {"type": "number"}}})"_json;
  std::shared_ptr<Schema> schema1 = Schema::Build(desc1, schema_content1);
  ASSERT_NE(schema1, nullptr);
  std::shared_ptr<Schema> schema2 = Schema::Build(desc2, schema_content2);
  ASSERT_EQ(schema2, nullptr);

  mindrecord::ShardHeader header_data;
  int schema_id1 = header_data.AddSchema(schema1);
  int schema_id2 = header_data.AddSchema(schema2);
  ASSERT_EQ(schema_id2, -1);
  ASSERT_EQ(header_data.GetSchemas().size(), 1);

  // check out fields
  std::vector<std::pair<uint64_t, std::string>> fields;

  std::pair<uint64_t, std::string> index_field1(schema_id1, "name");
  std::pair<uint64_t, std::string> index_field2(schema_id1, "box");
  fields.push_back(index_field1);
  fields.push_back(index_field2);
  Status status = header_data.AddIndexFields(fields);
  EXPECT_TRUE(status.IsOk());

  ASSERT_EQ(header_data.GetFields().size(), 2);

  fields.clear();
  std::pair<uint64_t, std::string> index_field3(schema_id1, "name");
  fields.push_back(index_field3);
  status = header_data.AddIndexFields(fields);
  EXPECT_FALSE(status.IsOk());
  ASSERT_EQ(header_data.GetFields().size(), 2);

  fields.clear();
  std::pair<uint64_t, std::string> index_field4(schema_id1, "names");
  fields.push_back(index_field4);
  status = header_data.AddIndexFields(fields);
  EXPECT_FALSE(status.IsOk());
  ASSERT_EQ(header_data.GetFields().size(), 2);

  fields.clear();
  std::pair<uint64_t, std::string> index_field5(schema_id1 + 1, "name");
  fields.push_back(index_field5);
  status = header_data.AddIndexFields(fields);
  EXPECT_FALSE(status.IsOk());
  ASSERT_EQ(header_data.GetFields().size(), 2);

  fields.clear();
  std::pair<uint64_t, std::string> index_field6(schema_id1, "label");
  fields.push_back(index_field6);
  status = header_data.AddIndexFields(fields);
  EXPECT_FALSE(status.IsOk());
  ASSERT_EQ(header_data.GetFields().size(), 2);

  std::string desc_new = "this is a test1";
  json schemaContent_new = R"({"name": {"type": "string"},
                            "box": {"type": "string"},
                            "label": {"type": "int32", "shape": [-1]}})"_json;

  std::shared_ptr<Schema> schema_new = Schema::Build(desc_new, schemaContent_new);
  ASSERT_NE(schema_new, nullptr);

  mindrecord::ShardHeader header_data_new;
  header_data_new.AddSchema(schema_new);
  ASSERT_EQ(header_data_new.GetSchemas().size(), 1);

  // test add fields
  std::vector<std::string> single_fields;

  single_fields.push_back("name");
  single_fields.push_back("name");
  single_fields.push_back("box");
  status = header_data_new.AddIndexFields(single_fields);
  EXPECT_FALSE(status.IsOk());
  ASSERT_EQ(header_data_new.GetFields().size(), 1);

  single_fields.push_back("name");
  single_fields.push_back("box");
  status = header_data_new.AddIndexFields(single_fields);
  EXPECT_FALSE(status.IsOk());
  ASSERT_EQ(header_data_new.GetFields().size(), 1);

  single_fields.clear();
  single_fields.push_back("names");
  status = header_data_new.AddIndexFields(single_fields);
  EXPECT_FALSE(status.IsOk());
  ASSERT_EQ(header_data_new.GetFields().size(), 1);

  single_fields.clear();
  single_fields.push_back("box");
  status = header_data_new.AddIndexFields(single_fields);
  EXPECT_TRUE(status.IsOk());
  ASSERT_EQ(header_data_new.GetFields().size(), 2);
}
}  // namespace mindrecord
}  // namespace mindspore
