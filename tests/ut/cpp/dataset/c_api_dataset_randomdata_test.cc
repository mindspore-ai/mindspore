/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "common/common.h"
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/core/global_context.h"

#include "ir/dtype/type_id.h"

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestRandomDatasetBasic1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetBasic1.";

  // Create a RandomDataset
  std::shared_ptr<SchemaObj> schema = Schema();
  schema->add_column("image", mindspore::DataType::kNumberTypeUInt8, {2});
  schema->add_column("label", mindspore::DataType::kNumberTypeUInt8, {1});
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  ds = ds->Repeat(4);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Check if RandomData() read correct columns
  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 200);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomDatasetBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetBasicWithPipeline.";

  // Create two RandomDataset
  std::shared_ptr<SchemaObj> schema = Schema();
  schema->add_column("image", mindspore::DataType::kNumberTypeUInt8, {2});
  schema->add_column("label", mindspore::DataType::kNumberTypeUInt8, {1});
  std::shared_ptr<Dataset> ds1 = RandomData(50, schema);
  std::shared_ptr<Dataset> ds2 = RandomData(50, schema);
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds
  int32_t repeat_num = 2;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 2;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create two Project operation on ds
  std::vector<std::string> column_project = {"image", "label"};
  ds1 = ds1->Project(column_project);
  EXPECT_NE(ds1, nullptr);
  ds2 = ds2->Project(column_project);
  EXPECT_NE(ds2, nullptr);

  // Create a Concat operation on the ds
  ds1 = ds1->Concat({ds2});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Check if RandomData() read correct columns
  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor label shape: " << label.Shape();

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 200);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetGetters.";

  // Create a RandomDataset
  std::shared_ptr<SchemaObj> schema = Schema();
  schema->add_column("image", mindspore::DataType::kNumberTypeUInt8, {2});
  schema->add_column("label", mindspore::DataType::kNumberTypeUInt8, {1});
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  std::vector<std::string> column_names = {"image", "label"};
  EXPECT_EQ(ds->GetDatasetSize(), 50);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

TEST_F(MindDataTestPipeline, TestRandomDatasetBasic2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetBasic2.";

  // Create a RandomDataset
  std::shared_ptr<Dataset> ds = RandomData(10);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(1);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  ds = ds->Repeat(2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Check if RandomData() read correct columns
  uint64_t i = 0;
  while (row.size() != 0) {
    // If no schema specified, RandomData will generate random columns
    // So we don't check columns here
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomDatasetBasic3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetBasic3.";

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);

  std::string SCHEMA_FILE = datasets_root_path_ + "/testTFTestAllTypes/datasetSchema.json";
  std::shared_ptr<SchemaObj> schema = Schema(SCHEMA_FILE);
  std::shared_ptr<Dataset> ds = RandomData(0, schema);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  ds = ds->Repeat(2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<int64_t> expect_num = {1};
  std::vector<int64_t> expect_1d = {2};
  std::vector<int64_t> expect_2d = {2, 2};
  std::vector<int64_t> expect_3d = {2, 2, 2};

  // Check if RandomData() read correct columns
  uint64_t i = 0;
  while (row.size() != 0) {
    auto col_sint16 = row["col_sint16"];
    auto col_sint32 = row["col_sint32"];
    auto col_sint64 = row["col_sint64"];
    auto col_float = row["col_float"];
    auto col_1d = row["col_1d"];
    auto col_2d = row["col_2d"];
    auto col_3d = row["col_3d"];
    auto col_binary = row["col_binary"];

    // Validate shape
    ASSERT_EQ(col_sint16.Shape(), expect_num);
    ASSERT_EQ(col_sint32.Shape(), expect_num);
    ASSERT_EQ(col_sint64.Shape(), expect_num);
    ASSERT_EQ(col_float.Shape(), expect_num);
    ASSERT_EQ(col_1d.Shape(), expect_1d);
    ASSERT_EQ(col_2d.Shape(), expect_2d);
    ASSERT_EQ(col_3d.Shape(), expect_3d);
    ASSERT_EQ(col_binary.Shape(), expect_num);

    // Validate Rank
    ASSERT_EQ(col_sint16.Shape().size(), 1);
    ASSERT_EQ(col_sint32.Shape().size(), 1);
    ASSERT_EQ(col_sint64.Shape().size(), 1);
    ASSERT_EQ(col_float.Shape().size(), 1);
    ASSERT_EQ(col_1d.Shape().size(), 1);
    ASSERT_EQ(col_2d.Shape().size(), 2);
    ASSERT_EQ(col_3d.Shape().size(), 3);
    ASSERT_EQ(col_binary.Shape().size(), 1);

    // Validate type
    ASSERT_EQ(col_sint16.DataType(), mindspore::DataType::kNumberTypeInt16);
    ASSERT_EQ(col_sint32.DataType(), mindspore::DataType::kNumberTypeInt32);
    ASSERT_EQ(col_sint64.DataType(), mindspore::DataType::kNumberTypeInt64);
    ASSERT_EQ(col_float.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_EQ(col_1d.DataType(), mindspore::DataType::kNumberTypeInt64);
    ASSERT_EQ(col_2d.DataType(), mindspore::DataType::kNumberTypeInt64);
    ASSERT_EQ(col_3d.DataType(), mindspore::DataType::kNumberTypeInt64);
    ASSERT_EQ(col_binary.DataType(), mindspore::DataType::kNumberTypeUInt8);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 984);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

TEST_F(MindDataTestPipeline, TestRandomDatasetBasic4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetBasic4.";

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);

  std::string SCHEMA_FILE = datasets_root_path_ + "/testTFTestAllTypes/datasetSchema.json";
  std::shared_ptr<Dataset> ds = RandomData(0, SCHEMA_FILE);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  ds = ds->Repeat(2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<int64_t> expect_num = {1};
  std::vector<int64_t> expect_1d = {2};
  std::vector<int64_t> expect_2d = {2, 2};
  std::vector<int64_t> expect_3d = {2, 2, 2};

  // Check if RandomData() read correct columns
  uint64_t i = 0;
  while (row.size() != 0) {
    auto col_sint16 = row["col_sint16"];
    auto col_sint32 = row["col_sint32"];
    auto col_sint64 = row["col_sint64"];
    auto col_float = row["col_float"];
    auto col_1d = row["col_1d"];
    auto col_2d = row["col_2d"];
    auto col_3d = row["col_3d"];
    auto col_binary = row["col_binary"];

    // Validate shape
    ASSERT_EQ(col_sint16.Shape(), expect_num);
    ASSERT_EQ(col_sint32.Shape(), expect_num);
    ASSERT_EQ(col_sint64.Shape(), expect_num);
    ASSERT_EQ(col_float.Shape(), expect_num);
    ASSERT_EQ(col_1d.Shape(), expect_1d);
    ASSERT_EQ(col_2d.Shape(), expect_2d);
    ASSERT_EQ(col_3d.Shape(), expect_3d);
    ASSERT_EQ(col_binary.Shape(), expect_num);

    // Validate Rank
    ASSERT_EQ(col_sint16.Shape().size(), 1);
    ASSERT_EQ(col_sint32.Shape().size(), 1);
    ASSERT_EQ(col_sint64.Shape().size(), 1);
    ASSERT_EQ(col_float.Shape().size(), 1);
    ASSERT_EQ(col_1d.Shape().size(), 1);
    ASSERT_EQ(col_2d.Shape().size(), 2);
    ASSERT_EQ(col_3d.Shape().size(), 3);
    ASSERT_EQ(col_binary.Shape().size(), 1);

    // Validate type
    ASSERT_EQ(col_sint16.DataType(), mindspore::DataType::kNumberTypeInt16);
    ASSERT_EQ(col_sint32.DataType(), mindspore::DataType::kNumberTypeInt32);
    ASSERT_EQ(col_sint64.DataType(), mindspore::DataType::kNumberTypeInt64);
    ASSERT_EQ(col_float.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_EQ(col_1d.DataType(), mindspore::DataType::kNumberTypeInt64);
    ASSERT_EQ(col_2d.DataType(), mindspore::DataType::kNumberTypeInt64);
    ASSERT_EQ(col_3d.DataType(), mindspore::DataType::kNumberTypeInt64);
    ASSERT_EQ(col_binary.DataType(), mindspore::DataType::kNumberTypeUInt8);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 984);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

TEST_F(MindDataTestPipeline, TestRandomDatasetBasic5) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetBasic5.";

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);

  std::string SCHEMA_FILE = datasets_root_path_ + "/testTFTestAllTypes/datasetSchema.json";
  std::shared_ptr<Dataset> ds = RandomData(0, SCHEMA_FILE, {"col_sint32", "col_sint64", "col_1d"});
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  ds = ds->Repeat(2);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<int64_t> expect_num = {1};
  std::vector<int64_t> expect_1d = {2};

  // Check if RandomData() read correct columns
  uint64_t i = 0;
  while (row.size() != 0) {
    EXPECT_EQ(row.size(), 3);

    auto col_sint32 = row["col_sint32"];
    auto col_sint64 = row["col_sint64"];
    auto col_1d = row["col_1d"];

    // Validate shape
    ASSERT_EQ(col_sint32.Shape(), expect_num);
    ASSERT_EQ(col_sint64.Shape(), expect_num);
    ASSERT_EQ(col_1d.Shape(), expect_1d);

    // Validate Rank
    ASSERT_EQ(col_sint32.Shape().size(), 1);
    ASSERT_EQ(col_sint64.Shape().size(), 1);
    ASSERT_EQ(col_1d.Shape().size(), 1);

    // Validate type
    ASSERT_EQ(col_sint32.DataType(), mindspore::DataType::kNumberTypeInt32);
    ASSERT_EQ(col_sint64.DataType(), mindspore::DataType::kNumberTypeInt64);
    ASSERT_EQ(col_1d.DataType(), mindspore::DataType::kNumberTypeInt64);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 984);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

TEST_F(MindDataTestPipeline, TestRandomDatasetBasic6) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetBasic6.";

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);

  std::string SCHEMA_FILE = datasets_root_path_ + "/testTFTestAllTypes/datasetSchema.json";
  std::shared_ptr<Dataset> ds = RandomData(10, nullptr, {"col_sint32", "col_sint64", "col_1d"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Check if RandomData() read correct columns
  uint64_t i = 0;
  while (row.size() != 0) {
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

TEST_F(MindDataTestPipeline, TestRandomDatasetBasic7) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetBasic7.";

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);

  std::string SCHEMA_FILE = datasets_root_path_ + "/testTFTestAllTypes/datasetSchema.json";
  std::shared_ptr<Dataset> ds = RandomData(10, "", {"col_sint32", "col_sint64", "col_1d"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Check if RandomData() read correct columns
  uint64_t i = 0;
  while (row.size() != 0) {
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

TEST_F(MindDataTestPipeline, TestRandomDatasetDuplicateColumnName) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetDuplicateColumnName.";

  // Create a RandomDataset
  std::shared_ptr<SchemaObj> schema = Schema();
  schema->add_column("image", mindspore::DataType::kNumberTypeUInt8, {2});
  schema->add_column("label", mindspore::DataType::kNumberTypeUInt8, {1});
  std::shared_ptr<Dataset> ds = RandomData(50, schema, {"image", "image"});
  // Expect failure: duplicate column names
  EXPECT_EQ(ds->CreateIterator(), nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomDatasetFail.";
  // this will fail because num_workers is greater than num_rows
  std::shared_ptr<Dataset> ds = RandomData(3)->SetNumWorkers(5);
  EXPECT_EQ(ds->CreateIterator(), nullptr);
}
