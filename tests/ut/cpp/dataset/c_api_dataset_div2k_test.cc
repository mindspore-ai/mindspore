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
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: DIV2KDataset
/// Description: Test basic usage of DIV2KDataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestDIV2KBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDIV2KBasic.";

  std::string dataset_path = datasets_root_path_ + "/testDIV2KData/div2k";
  std::string usage = "train";     // train valid, all
  std::string downgrade = "bicubic";  // bicubic, unknown, mild, difficult, wild
  int32_t scale = 2;               // 2, 3, 4, 8

  // Create a DIV2K Dataset
  std::shared_ptr<Dataset> ds = DIV2K(dataset_path, usage, downgrade, scale);
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
    auto hr_image = row["hr_image"];
    auto lr_image = row["lr_image"];
    MS_LOG(INFO) << "Tensor hr_image shape: " << hr_image.Shape();
    MS_LOG(INFO) << "Tensor lr_image shape: " << lr_image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: DIV2KDataset
/// Description: Test usage of DIV2KDataset with pipeline
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestDIV2KBasicWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDIV2KBasicWithPipeline.";

  std::string dataset_path = datasets_root_path_ + "/testDIV2KData/div2k";
  std::string usage = "train";        // train valid, all
  std::string downgrade = "bicubic";  // bicubic, unknown, mild, difficult, wild
  int32_t scale = 2;                  // 2, 3, 4, 8

  // Create two DIV2K Dataset
  std::shared_ptr<Dataset> ds1 =
    DIV2K(dataset_path, usage, downgrade, scale, false, std::make_shared<RandomSampler>(false, 2));
  std::shared_ptr<Dataset> ds2 =
    DIV2K(dataset_path, usage, downgrade, scale, false, std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds
  int32_t repeat_num = 3;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 2;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create two Project operation on ds
  std::vector<std::string> column_project = {"hr_image", "lr_image"};
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
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["hr_image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 12);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: DIV2KDataset
/// Description: Test iterator of DIV2KDataset with only the hr_image column
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestDIV2KIteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDIV2KIteratorOneColumn.";
  // Create a DIV2K Dataset
  std::string dataset_path = datasets_root_path_ + "/testDIV2KData/div2k";
  std::string usage = "train";     // train valid, all
  std::string downgrade = "bicubic";  // bicubic, unknown, mild, difficult, wild
  int32_t scale = 2;               // 2, 3, 4, 8
  std::shared_ptr<Dataset> ds = DIV2K(dataset_path, usage, downgrade, scale);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // Only select "image" column and drop others
  std::vector<std::string> columns = {"hr_image"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    for (auto &v : row) {
      MS_LOG(INFO) << "image shape:" << v.Shape();
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: DIV2KDataset
/// Description: Test iterator of DIV2KDataset with wrong column
/// Expectation: Error message is logged, and CreateIterator for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestDIV2KIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDIV2KIteratorWrongColumn.";
  // Create a DIV2K Dataset
  std::string dataset_path = datasets_root_path_ + "/testDIV2KData/div2k";
  std::string usage = "train";     // train valid, all
  std::string downgrade = "bicubic";  // bicubic, unknown, mild, difficult, wild
  int32_t scale = 2;               // 2, 3, 4, 8
  std::shared_ptr<Dataset> ds = DIV2K(dataset_path, usage, downgrade, scale);
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: DIV2KDataset
/// Description: Test usage of DIV2KDataset Getters method
/// Expectation: Get correct number of data and correct tensor shape
TEST_F(MindDataTestPipeline, TestDIV2KGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDIV2KGetters.";

  std::string dataset_path = datasets_root_path_ + "/testDIV2KData/div2k";
  std::string usage = "train";        // train valid, all
  std::string downgrade = "bicubic";  // bicubic, unknown, mild, difficult, wild
  int32_t scale = 2;                  // 2, 3, 4, 8

  // Create a DIV2K Dataset
  std::shared_ptr<Dataset> ds1 =
    DIV2K(dataset_path, usage, downgrade, scale, false, std::make_shared<RandomSampler>(false, 2));
  std::shared_ptr<Dataset> ds2 =
    DIV2K(dataset_path, usage, downgrade, scale, false, std::make_shared<RandomSampler>(false, 3));
  std::vector<std::string> column_names = {"hr_image", "lr_image"};

  EXPECT_NE(ds1, nullptr);
  EXPECT_EQ(ds1->GetDatasetSize(), 2);
  EXPECT_EQ(ds1->GetColumnNames(), column_names);

  EXPECT_NE(ds2, nullptr);
  EXPECT_EQ(ds2->GetDatasetSize(), 3);
  EXPECT_EQ(ds2->GetColumnNames(), column_names);
}

/// Feature: DIV2KDataset
/// Description: Test usage of DIV2KDataset with Decode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestDIV2KDecode) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDIV2KDecode.";

  std::string dataset_path = datasets_root_path_ + "/testDIV2KData/div2k";
  std::string usage = "train";          // train valid, all
  std::string downgrade = "bicubic";  // bicubic, unknown, mild, difficult, wild
  int32_t scale = 2;                    // 2, 3, 4, 8

  // Create a DIV2K Dataset
  std::shared_ptr<Dataset> ds = DIV2K(dataset_path, usage, downgrade, scale, true, std::make_shared<RandomSampler>());
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
    auto hr_image = row["hr_image"];
    auto lr_image = row["lr_image"];
    auto h_size = hr_image.Shape().size();
    auto l_size = lr_image.Shape().size();
    MS_LOG(INFO) << "Tensor hr_image shape size: " << h_size;
    MS_LOG(INFO) << "Tensor lr_image shape size: " << l_size;
    EXPECT_GT(h_size, 1);  // Verify decode=true took effect
    EXPECT_GT(l_size, 1);  // Verify decode=true took effect
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: DIV2KDataset
/// Description: Test usage of DIV2KDataset with num sampler
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestDIV2KNumSamplers) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDIV2KNumSamplers.";

  std::string dataset_path = datasets_root_path_ + "/testDIV2KData/div2k";
  std::string usage = "train";        // train valid, all
  std::string downgrade = "bicubic";  // bicubic, unknown, mild, difficult, wild
  int32_t scale = 2;                  // 2, 3, 4, 8

  // Create a DIV2K Dataset
  std::shared_ptr<Dataset> ds =
    DIV2K(dataset_path, usage, downgrade, scale, true, std::make_shared<SequentialSampler>(0, 1));
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
    auto hr_image = row["hr_image"];
    auto lr_image = row["lr_image"];

    MS_LOG(INFO) << "Tensor hr_image shape: " << hr_image.Shape();
    MS_LOG(INFO) << "Tensor lr_image shape: " << lr_image.Shape();

    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: DIV2KDataset
/// Description: Test DIV2KDataset with non-existing dataset directory and other invalid inputs
/// Expectation: Error message is logged, and CreateIterator for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestDIV2KError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDIV2KError.";

  std::string dataset_path = datasets_root_path_ + "/testDIV2KData/div2k";
  std::string usage = "train";        // train valid, all
  std::string downgrade = "unknown";  // bicubic, unknown, mild, difficult, wild
  int32_t scale = 2;                  // 2, 3, 4, 8

  // Create a DIV2K Dataset with non-existing dataset dir
  std::shared_ptr<Dataset> ds0 = DIV2K("NotExistFile", usage, downgrade, scale);
  EXPECT_NE(ds0, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter0 = ds0->CreateIterator();
  // Expect failure: invalid DIV2K input
  EXPECT_EQ(iter0, nullptr);

  // Create a DIV2K Dataset with err usage
  std::shared_ptr<Dataset> ds1 = DIV2K(dataset_path, "test", downgrade, scale);
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid DIV2K input
  EXPECT_EQ(iter1, nullptr);

  // Create a DIV2K Dataset with err scale
  std::shared_ptr<Dataset> ds2 = DIV2K(dataset_path, usage, downgrade, 16);
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid DIV2K input
  EXPECT_EQ(iter2, nullptr);

  // Create a DIV2K Dataset with err downgrade
  std::shared_ptr<Dataset> ds3 = DIV2K(dataset_path, usage, "downgrade", scale);
  EXPECT_NE(ds3, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid DIV2K input
  EXPECT_EQ(iter3, nullptr);

  // Create a DIV2K Dataset with scale 8 and downgrade unknown
  std::shared_ptr<Dataset> ds4 = DIV2K(dataset_path, usage, "unknown", 8);
  EXPECT_NE(ds4, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid DIV2K input
  EXPECT_EQ(iter4, nullptr);

  // Create a DIV2K Dataset with scale 2 and downgrade mild
  std::shared_ptr<Dataset> ds5 = DIV2K(dataset_path, usage, "mild", 2);
  EXPECT_NE(ds5, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter5 = ds5->CreateIterator();
  // Expect failure: invalid DIV2K input
  EXPECT_EQ(iter5, nullptr);
}

/// Feature: DIV2KDataset
/// Description: Test DIV2KDataset with null sampler
/// Expectation: Error message is logged, and CreateIterator for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestDIV2KWithNullSamplerError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDIV2KWithNullSamplerError.";

  std::string dataset_path = datasets_root_path_ + "/testDIV2KData/div2k";
  std::string usage = "train";        // train valid, all
  int32_t scale = 2;                  // 2, 3, 4, 8
  std::string downgrade = "unknown";  // bicubic, unknown, mild, difficult, wild

  // Create a DIV2K Dataset
  std::shared_ptr<Dataset> ds = DIV2K(dataset_path, usage, downgrade, scale, false, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid DIV2K input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}
