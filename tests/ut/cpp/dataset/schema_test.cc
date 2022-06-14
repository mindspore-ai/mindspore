/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <memory>
#include <string>
#include "common/common.h"
#include "utils/ms_utils.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/util/status.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;

class MindDataTestSchema : public UT::DatasetOpTesting {
 protected:
};

/// Feature: Schema
/// Description: Test DataSchema LoadSchemaFile using an old schema file
/// Expectation: Correct number of columns
TEST_F(MindDataTestSchema, TestOldSchema) {
  std::string schema_file = datasets_root_path_ + "/testDataset2/datasetSchema.json";
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  Status rc = schema->LoadSchemaFile(schema_file, {});
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during schema load: " << common::SafeCStr(rc.ToString()) << ".";
    EXPECT_TRUE(false);
  } else {
    int32_t num_cols = schema->NumColumns();
    EXPECT_TRUE(num_cols == 4);
  }
}

/// Feature: Schema
/// Description: Test DataSchema LoadSchemaFile using an Album Schema file
/// Expectation: Correct number of columns
TEST_F(MindDataTestSchema, TestAlbumSchema) {
  std::string schema_file = datasets_root_path_ + "/testAlbum/fullSchema.json";
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  Status rc = schema->LoadSchemaFile(schema_file, {});
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Return code error detected during schema load: " << common::SafeCStr(rc.ToString()) << ".";
    EXPECT_TRUE(false);
  } else {
    int32_t num_cols = schema->NumColumns();
    MS_LOG(INFO) << "num_cols: " << num_cols << ".";
    EXPECT_TRUE(num_cols == 8);
  }
}

