/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <fstream>
#include <iostream>

#include "common/common.h"
#include "minddata/dataset/include/dataset/datasets.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: IMDBDataset
/// Description: Test IMBDDataset basic usage
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestIMDBBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIMDBBasic.";

  std::string dataset_path = datasets_root_path_ + "/testIMDBDataset";
  std::string usage = "all";  // 'train', 'test', 'all'

  // Create a IMDB Dataset
  std::shared_ptr<Dataset> ds = IMDB(dataset_path, usage);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto text = row["text"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape() << ", Tensor label shape: " << label.Shape() << "\n";
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: IMDBDataset
/// Description: Test IMBDDataset with train dataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestIMDBTrain) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIMDBTrain.";

  std::string dataset_path = datasets_root_path_ + "/testIMDBDataset";
  std::string usage = "train";  // 'train', 'test', 'all'

  // Create a IMDB Dataset
  std::shared_ptr<Dataset> ds = IMDB(dataset_path, usage);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto text = row["text"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape() << ", Tensor label shape: " << label.Shape() << "\n";
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: IMDBDataset
/// Description: Test IMBDDataset with test dataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestIMDBTest) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIMDBTest.";

  std::string dataset_path = datasets_root_path_ + "/testIMDBDataset";
  std::string usage = "test";  // 'train', 'test', 'all'

  // Create a IMDB Dataset
  std::shared_ptr<Dataset> ds = IMDB(dataset_path, usage);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto text = row["text"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape() << ", Tensor label shape: " << label.Shape() << "\n";
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: IMDBDataset
/// Description: Test IMBDDataset in pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestIMDBBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIMDBBasicWithPipeline.";

  std::string dataset_path = datasets_root_path_ + "/testIMDBDataset";
  std::string usage = "all";  // 'train', 'test', 'all'

  // Create two IMDB Dataset
  std::shared_ptr<Dataset> ds1 = IMDB(dataset_path, usage);
  std::shared_ptr<Dataset> ds2 = IMDB(dataset_path, usage);
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds
  int32_t repeat_num = 3;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 2;
  ds2 = ds2->Repeat(repeat_num);
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
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto text = row["text"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor text shape: " << text.Shape() << ", Tensor label shape: " << label.Shape() << "\n";
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 40);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: IMDBDataset
/// Description: Test iterator of IMDBDataset with only the text column.
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestIMDBIteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIMDBIteratorOneColumn.";
  std::string dataset_path = datasets_root_path_ + "/testIMDBDataset";
  std::string usage = "all";  // 'train', 'test', 'all'

  // Create a IMDB Dataset
  std::shared_ptr<Dataset> ds = IMDB(dataset_path, usage);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // Only select "text" column and drop others
  std::vector<std::string> columns = {"text"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    for (auto &v : row) {
      MS_LOG(INFO) << "text shape:" << v.Shape();
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: IMDBDataset
/// Description: Test iterator of IMDBDataset with wrong column
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestIMDBIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIMDBIteratorWrongColumn.";
  std::string dataset_path = datasets_root_path_ + "/testIMDBDataset";
  std::string usage = "all";  // 'train', 'test', 'all'

  // Create a IMDB Dataset
  std::shared_ptr<Dataset> ds = IMDB(dataset_path, usage);
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: IMDBDataset
/// Description: Test IMDBDataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestIMDBGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIMDBGetters.";

  std::string dataset_path = datasets_root_path_ + "/testIMDBDataset";
  std::string usage = "all";  // 'train', 'test', 'all'

  // Create a IMDB Dataset
  std::shared_ptr<Dataset> ds1 = IMDB(dataset_path, usage);
  std::vector<std::string> column_names = {"text", "label"};

  std::vector<DataType> types = ToDETypes(ds1->GetOutputTypes());
  EXPECT_EQ(types.size(), 2);
  EXPECT_EQ(types[0].ToString(), "string");
  EXPECT_EQ(types[1].ToString(), "int32");
  EXPECT_NE(ds1, nullptr);
  EXPECT_EQ(ds1->GetDatasetSize(), 8);
  EXPECT_EQ(ds1->GetColumnNames(), column_names);
  EXPECT_EQ(ds1->GetBatchSize(), 1);
}

/// Feature: IMDBDataset
/// Description: Test IMDBDataset with invalid inputs
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestIMDBError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIMDBError.";

  std::string dataset_path = datasets_root_path_ + "/testIMDBDataset";
  std::string usage = "all";  // 'train', 'test', 'all'

  // Create a IMDB Dataset with non-existing dataset dir
  std::shared_ptr<Dataset> ds0 = IMDB("NotExistDir", usage);
  EXPECT_NE(ds0, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid IMDB input
  EXPECT_EQ(iter0, nullptr);

  // Create a IMDB Dataset with err usage
  std::shared_ptr<Dataset> ds1 = IMDB(dataset_path, "val");
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid IMDB input
  EXPECT_EQ(iter1, nullptr);
}

/// Feature: IMDBDataset
/// Description: Test IMDBDataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestIMDBWithNullSamplerError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestIMDBWithNullSamplerError.";

  std::string dataset_path = datasets_root_path_ + "/testIMDBDataset";
  std::string usage = "all";

  // Create a IMDB Dataset
  std::shared_ptr<Dataset> ds = IMDB(dataset_path, usage, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid IMDB input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}
