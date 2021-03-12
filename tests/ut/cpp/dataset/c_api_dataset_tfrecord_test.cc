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
#include "minddata/dataset/include/vision.h"
#include "minddata/dataset/core/global_context.h"

using namespace mindspore::dataset;

using mindspore::dataset::DataType;
using mindspore::dataset::ShuffleMode;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestPipeline, TestTFRecordDatasetBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTFRecordDatasetBasic.";

  // Create a TFRecord Dataset
  std::string file_path = datasets_root_path_ + "/test_tf_file_3_images2/train-0000-of-0001.data";
  std::string schema_path = datasets_root_path_ + "/test_tf_file_3_images2/datasetSchema.json";
  std::shared_ptr<Dataset> ds = TFRecord({file_path}, schema_path, {"image"}, 0);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> decode_op = std::make_shared<vision::Decode>();
  std::shared_ptr<TensorTransform> random_horizontal_flip_op = std::make_shared<vision::RandomHorizontalFlip>(0.5);
  EXPECT_NE(random_horizontal_flip_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({decode_op, random_horizontal_flip_op}, {}, {}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Check column
  EXPECT_EQ(row.size(), 1);
  EXPECT_NE(row.find("image"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];

    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestTFRecordDatasetBasicGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTFRecordDatasetBasicGetters.";

  // Create a TFRecord Dataset
  std::string file_path = datasets_root_path_ + "/test_tf_file_3_images2/train-0000-of-0001.data";
  std::string schema_path = datasets_root_path_ + "/test_tf_file_3_images2/datasetSchema.json";
  std::shared_ptr<Dataset> ds = TFRecord({file_path}, schema_path, {"image"}, 0);
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_horizontal_flip_op = std::make_shared<vision::RandomHorizontalFlip>(0.5);
  EXPECT_NE(random_horizontal_flip_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_horizontal_flip_op}, {}, {}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 6);
  std::vector<std::string> column_names = {"image"};
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

TEST_F(MindDataTestPipeline, TestTFRecordDatasetShuffle) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTFRecordDatasetShuffle.";
  // This case is to verify if the list of datafiles are sorted in lexicographical order.
  // Set configuration
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a TFRecord Dataset
  std::string file1 = datasets_root_path_ + "/tf_file_dataset/test1.data";
  std::string file2 = datasets_root_path_ + "/tf_file_dataset/test2.data";
  std::string file3 = datasets_root_path_ + "/tf_file_dataset/test3.data";
  std::string file4 = datasets_root_path_ + "/tf_file_dataset/test4.data";
  std::shared_ptr<Dataset> ds1 = TFRecord({file4, file3, file2, file1}, "", {"scalars"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds1, nullptr);
  std::shared_ptr<Dataset> ds2 = TFRecord({file1}, "", {"scalars"}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  EXPECT_NE(iter1, nullptr);
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  EXPECT_NE(iter2, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row1;
  iter1->GetNextRow(&row1);
  std::unordered_map<std::string, mindspore::MSTensor> row2;
  iter2->GetNextRow(&row2);

  uint64_t i = 0;
  int64_t value1 = 0;
  int64_t value2 = 0;
  while (row1.size() != 0 && row2.size() != 0) {
    auto scalars1 = row1["scalars"];
    std::shared_ptr<Tensor> de_scalars1;
    ASSERT_OK(Tensor::CreateFromMSTensor(scalars1, &de_scalars1));
    de_scalars1->GetItemAt(&value1, {0});

    auto scalars2 = row2["scalars"];
    std::shared_ptr<Tensor> de_scalars2;
    ASSERT_OK(Tensor::CreateFromMSTensor(scalars2, &de_scalars2));
    de_scalars2->GetItemAt(&value2, {0});

    EXPECT_EQ(value1, value2);

    iter1->GetNextRow(&row1);
    iter2->GetNextRow(&row2);
    i++;
  }
  EXPECT_EQ(i, 10);
  // Manually terminate the pipeline
  iter1->Stop();
  iter2->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestTFRecordDatasetShuffle2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTFRecordDatasetShuffle2.";
  // This case is to verify the content of the data is indeed shuffled.
  // Set configuration
  uint32_t original_seed = GlobalContext::config_manager()->seed();
  uint32_t original_num_parallel_workers = GlobalContext::config_manager()->num_parallel_workers();
  MS_LOG(DEBUG) << "ORIGINAL seed: " << original_seed << ", num_parallel_workers: " << original_num_parallel_workers;
  GlobalContext::config_manager()->set_seed(155);
  GlobalContext::config_manager()->set_num_parallel_workers(1);

  // Create a TFRecord Dataset
  std::string file = datasets_root_path_ + "/tf_file_dataset/test1.data";
  std::shared_ptr<Dataset> ds = TFRecord({file}, nullptr, {"scalars"}, 0, ShuffleMode::kGlobal);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  std::vector<int> expect = {9, 3, 4, 7, 2, 1, 6, 8, 10, 5};
  std::vector<int> actual = {};
  int64_t value = 0;
  uint64_t i = 0;
  while (row.size() != 0) {
    auto scalars = row["scalars"];
    std::shared_ptr<Tensor> de_scalars;
    ASSERT_OK(Tensor::CreateFromMSTensor(scalars, &de_scalars));
    de_scalars->GetItemAt(&value, {0});
    actual.push_back(value);

    iter->GetNextRow(&row);
    i++;
  }
  ASSERT_EQ(actual, expect);
  EXPECT_EQ(i, 10);
  // Manually terminate the pipeline
  iter->Stop();

  // Restore configuration
  GlobalContext::config_manager()->set_seed(original_seed);
  GlobalContext::config_manager()->set_num_parallel_workers(original_num_parallel_workers);
}

TEST_F(MindDataTestPipeline, TestTFRecordDatasetSchemaPath) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTFRecordDatasetSchemaPath.";

  // Create a TFRecord Dataset
  std::string file_path1 = datasets_root_path_ + "/testTFTestAllTypes/test.data";
  std::string file_path2 = datasets_root_path_ + "/testTFTestAllTypes/test2.data";
  std::string schema_path = datasets_root_path_ + "/testTFTestAllTypes/datasetSchema.json";
  std::shared_ptr<Dataset> ds = TFRecord({file_path2, file_path1}, schema_path, {}, 9, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Check column
  EXPECT_EQ(row.size(), 8);
  EXPECT_NE(row.find("col_sint16"), row.end());
  EXPECT_NE(row.find("col_sint32"), row.end());
  EXPECT_NE(row.find("col_sint64"), row.end());
  EXPECT_NE(row.find("col_float"), row.end());
  EXPECT_NE(row.find("col_1d"), row.end());
  EXPECT_NE(row.find("col_2d"), row.end());
  EXPECT_NE(row.find("col_3d"), row.end());
  EXPECT_NE(row.find("col_binary"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 9);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestTFRecordDatasetSchemaObj) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTFRecordDatasetSchemaObj.";

  // Create a TFRecord Dataset
  std::string file_path = datasets_root_path_ + "/testTFTestAllTypes/test.data";
  std::shared_ptr<SchemaObj> schema = Schema();
  schema->add_column("col_sint16", "int16", {1});
  schema->add_column("col_float", "float32", {1});
  schema->add_column("col_2d", "int64", {2, 2});
  std::shared_ptr<Dataset> ds = TFRecord({file_path}, schema);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Check column
  EXPECT_EQ(row.size(), 3);
  EXPECT_NE(row.find("col_sint16"), row.end());
  EXPECT_NE(row.find("col_float"), row.end());
  EXPECT_NE(row.find("col_2d"), row.end());

  std::vector<int64_t> expect_num = {1};
  std::vector<int64_t> expect_2d = {2, 2};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto col_sint16 = row["col_sint16"];
    auto col_float = row["col_float"];
    auto col_2d = row["col_2d"];

    // Validate shape
    ASSERT_EQ(col_sint16.Shape(), expect_num);
    ASSERT_EQ(col_float.Shape(), expect_num);
    ASSERT_EQ(col_2d.Shape(), expect_2d);

    // Validate Rank
    ASSERT_EQ(col_sint16.Shape().size(), 1);
    ASSERT_EQ(col_float.Shape().size(), 1);
    ASSERT_EQ(col_2d.Shape().size(), 2);

    // Validate type
    ASSERT_EQ(col_sint16.DataType(), mindspore::DataType::kNumberTypeInt16);
    ASSERT_EQ(col_float.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_EQ(col_2d.DataType(), mindspore::DataType::kNumberTypeInt64);

    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 12);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestTFRecordDatasetNoSchema) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTFRecordDatasetNoSchema.";

  // Create a TFRecord Dataset
  std::string file_path = datasets_root_path_ + "/test_tf_file_3_images2/train-0000-of-0001.data";
  std::shared_ptr<SchemaObj> schema = nullptr;
  std::shared_ptr<Dataset> ds = TFRecord({file_path}, nullptr, {});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Check column
  EXPECT_EQ(row.size(), 2);
  EXPECT_NE(row.find("image"), row.end());
  EXPECT_NE(row.find("label"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    auto image = row["image"];
    auto label = row["label"];

    MS_LOG(INFO) << "Shape of column [image]:" << image.Shape();
    MS_LOG(INFO) << "Shape of column [label]:" << label.Shape();
    iter->GetNextRow(&row);
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestTFRecordDatasetColName) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTFRecordDatasetColName.";

  // Create a TFRecord Dataset
  // The dataset has two columns("image", "label") and 3 rows
  std::string file_path = datasets_root_path_ + "/test_tf_file_3_images2/train-0000-of-0001.data";
  std::shared_ptr<Dataset> ds = TFRecord({file_path}, "", {"image"}, 0);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  // Check column
  EXPECT_EQ(row.size(), 1);
  EXPECT_NE(row.find("image"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestTFRecordDatasetShard) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTFRecordDatasetShard.";

  // Create a TFRecord Dataset
  // Each file has two columns("image", "label") and 3 rows
  std::vector<std::string> files = {datasets_root_path_ + "/test_tf_file_3_images2/train-0000-of-0001.data",
                                    datasets_root_path_ + "/test_tf_file_3_images2/train-0000-of-0002.data",
                                    datasets_root_path_ + "/test_tf_file_3_images2/train-0000-of-0003.data"};
  std::shared_ptr<Dataset> ds1 = TFRecord(files, "", {}, 0, ShuffleMode::kFalse, 2, 1, true);
  EXPECT_NE(ds1, nullptr);
  std::shared_ptr<Dataset> ds2 = TFRecord(files, "", {}, 0, ShuffleMode::kFalse, 2, 1, false);
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  EXPECT_NE(iter1, nullptr);
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  EXPECT_NE(iter2, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row1;
  iter1->GetNextRow(&row1);
  std::unordered_map<std::string, mindspore::MSTensor> row2;
  iter2->GetNextRow(&row2);

  uint64_t i = 0;
  uint64_t j = 0;
  while (row1.size() != 0) {
    i++;
    iter1->GetNextRow(&row1);
  }

  while (row2.size() != 0) {
    j++;
    iter2->GetNextRow(&row2);
  }

  EXPECT_EQ(i, 5);
  EXPECT_EQ(j, 3);
  // Manually terminate the pipeline
  iter1->Stop();
  iter2->Stop();
}

TEST_F(MindDataTestPipeline, TestTFRecordDatasetExeception) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTFRecordDatasetExeception.";

  // This case expected to fail because the list of dir_path cannot be empty.
  std::shared_ptr<Dataset> ds1 = TFRecord({});
  EXPECT_EQ(ds1->CreateIterator(), nullptr);

  // This case expected to fail because the file in dir_path is not exist.
  std::string file_path = datasets_root_path_ + "/testTFTestAllTypes/test.data";
  std::shared_ptr<Dataset> ds2 = TFRecord({file_path, "noexist.data"});
  EXPECT_EQ(ds2->CreateIterator(), nullptr);

  // This case expected to fail because the file of schema is not exist.
  std::shared_ptr<Dataset> ds4 = TFRecord({file_path, "notexist.json"});
  EXPECT_EQ(ds4->CreateIterator(), nullptr);

  // This case expected to fail because num_samples is negative.
  std::shared_ptr<Dataset> ds5 = TFRecord({file_path}, "", {}, -1);
  EXPECT_EQ(ds5->CreateIterator(), nullptr);

  // This case expected to fail because num_shards is negative.
  std::shared_ptr<Dataset> ds6 = TFRecord({file_path}, "", {}, 10, ShuffleMode::kFalse, 0);
  EXPECT_EQ(ds6->CreateIterator(), nullptr);

  // This case expected to fail because shard_id is out_of_bound.
  std::shared_ptr<Dataset> ds7 = TFRecord({file_path}, "", {}, 10, ShuffleMode::kFalse, 3, 3);
  EXPECT_EQ(ds7->CreateIterator(), nullptr);

  // This case expected to fail because the provided number of files < num_shards in file-based sharding.
  std::string file_path1 = datasets_root_path_ + "/test_tf_file_3_images2/train-0000-of-0001.data";
  std::string file_path2 = datasets_root_path_ + "/test_tf_file_3_images2/train-0000-of-0002.data";
  std::shared_ptr<Dataset> ds8 = TFRecord({file_path1, file_path2}, "", {}, 0, ShuffleMode::kFalse, 3);
  EXPECT_EQ(ds8->CreateIterator(), nullptr);
}

TEST_F(MindDataTestPipeline, TestTFRecordDatasetExeception2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTFRecordDatasetExeception2.";
  // This case expected to fail because the input column name does not exist.

  std::string file_path1 = datasets_root_path_ + "/testTFTestAllTypes/test.data";
  std::string schema_path = datasets_root_path_ + "/testTFTestAllTypes/datasetSchema.json";
  // Create a TFRecord Dataset
  // Column "image" does not exist in the dataset
  std::shared_ptr<Dataset> ds = TFRecord({file_path1}, schema_path, {"image"}, 10);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This attempts to create Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestIncorrectTFSchemaObject) {
  std::string path = datasets_root_path_ + "/test_tf_file_3_images2/train-0000-of-0001.data";
  std::shared_ptr<SchemaObj> schema = Schema();
  schema->add_column("image", "uint8", {1});
  schema->add_column("label", "int64", {1});
  std::shared_ptr<Dataset> ds = TFRecord({path}, schema);
  EXPECT_NE(ds, nullptr);
  auto itr = ds->CreateIterator();
  EXPECT_NE(itr, nullptr);
  MSTensorMap mp;
  // This will fail due to the incorrect schema used
  EXPECT_ERROR(itr->GetNextRow(&mp));
}

TEST_F(MindDataTestPipeline, TestIncorrectTFrecordFile) {
  std::string path = datasets_root_path_ + "/test_tf_file_3_images2/datasetSchema.json";
  std::shared_ptr<Dataset> ds = TFRecord({path});
  EXPECT_NE(ds, nullptr);
  // The tf record file is incorrect, hence validate param will fail
  auto itr = ds->CreateIterator();
  EXPECT_EQ(itr, nullptr);
}
