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
#include "common/common.h"
#include "minddata/dataset/include/dataset/datasets.h"

using namespace mindspore::dataset;
using mindspore::dataset::DataType;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: YesNoDataset
/// Description: Test YesNoDataset using a single file
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestYesNoDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYesNoDataset.";
  // Create a YesNoDataset
  std::string folder_path = datasets_root_path_ + "/testYesNoData/";
  std::shared_ptr<Dataset> ds = YesNo(folder_path, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);
  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  MS_LOG(INFO) << "iter->GetNextRow(&row) OK";

  EXPECT_NE(row.find("waveform"), row.end());
  EXPECT_NE(row.find("sample_rate"), row.end());
  EXPECT_NE(row.find("label"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto waveform = row["waveform"];
    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: YesNoDataset
/// Description: Test YesNoDataset with pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, YesNoDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-YesNoDatasetWithPipeline.";

  std::string folder_path = datasets_root_path_ + "/testYesNoData/";
  std::shared_ptr<Dataset> ds1 = YesNo(folder_path, std::make_shared<RandomSampler>(false, 1));
  std::shared_ptr<Dataset> ds2 = YesNo(folder_path, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds
  int32_t repeat_num = 1;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 2;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create two Project operation on ds
  std::vector<std::string> column_project = {"waveform", "sample_rate", "label"};
  ds1 = ds1->Project(column_project);
  EXPECT_NE(ds1, nullptr);
  ds2 = ds2->Project(column_project);
  EXPECT_NE(ds2, nullptr);

  // Create a Concat operation on the ds
  ds1 = ds1->Concat({ds2});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("waveform"), row.end());
  EXPECT_NE(row.find("sample_rate"), row.end());
  EXPECT_NE(row.find("label"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto waveform = row["waveform"];
    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 5);
  iter->Stop();
}

/// Feature: YesNoDataset
/// Description: Test iterator of YesNoDataset with only the waveform column
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestYesNoDatasetIteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYesNoDatasetIteratorOneColumn.";
  // Create a YesNo dataset
  std::string folder_path = datasets_root_path_ + "/testYesNoData/";
  std::shared_ptr<Dataset> ds = YesNo(folder_path, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // Only select "waveform" column and drop others
  std::vector<std::string> columns = {"waveform"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::vector<int64_t> expect_image = {1, 1, 16000};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (auto &v : row) {
      MS_LOG(INFO) << "waveform shape:" << v.Shape();
      EXPECT_EQ(expect_image, v.Shape());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: YesNoDataset
/// Description: Test iterator of YesNoDataset with wrong column
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestYesNoGetDatasetIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYesNoGetDatasetIteratorWrongColumn.";
  // Create a YesNo dataset
  std::string folder_path = datasets_root_path_ + "/testYesNoData/";
  std::shared_ptr<Dataset> ds = YesNo(folder_path, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: YesNoDataset
/// Description: Test YesNoDataset GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestYesNoGetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYesNoGetDatasetSize.";

  // Create a YesNo Dataset
  std::string folder_path = datasets_root_path_ + "/testYesNoData/";
  std::shared_ptr<Dataset> ds = YesNo(folder_path);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 3);
}

/// Feature: YesNoDataset
/// Description: Test YesNoDataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestYesNoGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYesNoMixGetter.";
  // Create a YesNo Dataset
  std::string folder_path = datasets_root_path_ + "/testYesNoData/";
  std::shared_ptr<Dataset> ds = YesNo(folder_path);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 3);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  std::vector<std::string> column_names = {"waveform", "sample_rate", "label"};
  EXPECT_EQ(types.size(), 3);
  EXPECT_EQ(types[0].ToString(), "float32");
  EXPECT_EQ(types[1].ToString(), "int32");
  EXPECT_EQ(types[2].ToString(), "int32");
  EXPECT_EQ(shapes.size(), 3);
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(shapes[2].ToString(), "<8>");
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);

  EXPECT_EQ(ds->GetDatasetSize(), 3);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);

  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: Test YesNo dataset.
/// Description: DatasetFail tests.
/// Expectation: Throw error messages when certain errors occur.
TEST_F(MindDataTestPipeline, TestYesNoDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYesNoDatasetFail.";

  // Create a YesNo Dataset
  std::shared_ptr<Dataset> ds = YesNo("", std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid YesNo directory
  EXPECT_EQ(iter, nullptr);
}

/// Feature: YesNoDataset
/// Description: Test YesNoDataset using null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestYesNoDatasetWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestYesNo10DatasetWithNullSamplerFail.";

  // Create a YesNo Dataset
  std::string folder_path = datasets_root_path_ + "/testYesNoData/";
  std::shared_ptr<Dataset> ds = YesNo(folder_path, nullptr);
  // Expect failure: Null Sampler
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Null Sampler
  EXPECT_EQ(iter, nullptr);
}
