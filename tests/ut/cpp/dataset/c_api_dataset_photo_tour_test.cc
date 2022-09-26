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

/// Feature: PhotoTourDataset
/// Description: Test basic usage of PhotoTourDataset with train dataset
/// Expectation: Get correct number of data
TEST_F(MindDataTestPipeline, TestPhotoTourTrainDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPhotoTourTrainDataset.";

  // Create a PhotoTour Train Dataset
  std::string folder_path = datasets_root_path_ + "/testPhotoTourData";
  std::shared_ptr<Dataset> ds = PhotoTour(folder_path, "liberty", "train", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("image"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PhotoTourDataset
/// Description: Test basic usage of PhotoTourDataset with test dataset
/// Expectation: Get correct number of data
TEST_F(MindDataTestPipeline, TestPhotoTourTestDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPhotoTourTestDataset.";

  // Create a PhotoTour Test Dataset
  std::string folder_path = datasets_root_path_ + "/testPhotoTourData";
  std::shared_ptr<Dataset> ds = PhotoTour(folder_path, "liberty", "test", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("image1"), row.end());
  EXPECT_NE(row.find("image2"), row.end());
  EXPECT_NE(row.find("matches"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image1"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PhotoTourDataset
/// Description: Test usage of PhotoTourDataset in pipeline mode with train dataset
/// Expectation: Get correct number of data
TEST_F(MindDataTestPipeline, TestPhotoTourTrainDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPhotoTourTrainDatasetWithPipeline.";

  // Create two PhotoTour Train Dataset
  std::string folder_path = datasets_root_path_ + "/testPhotoTourData";
  std::shared_ptr<Dataset> ds1 = PhotoTour(folder_path, "liberty", "train", std::make_shared<RandomSampler>(false, 10));
  std::shared_ptr<Dataset> ds2 = PhotoTour(folder_path, "liberty", "train", std::make_shared<RandomSampler>(false, 10));
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
  std::vector<std::string> column_project = {"image"};
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

  EXPECT_NE(row.find("image"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 40);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PhotoTourDataset
/// Description: Test iterator of PhotoTourDataset with only the image column
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestPhotoTourIteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPhotoTourIteratorOneColumn.";
  // Create a PhotoTour Dataset
  std::string folder_path = datasets_root_path_ + "/testPhotoTourData";
  std::shared_ptr<Dataset> ds = PhotoTour(folder_path, "liberty", "train", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 2;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // Only select "image" column and drop others
  std::vector<std::string> columns = {"image"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::vector<int64_t> expect_image = {2, 64, 64, 1};

  uint64_t i = 0;
  while (row.size() != 0) {
    for (auto &v : row) {
      MS_LOG(INFO) << "image shape:" << v.Shape();
      EXPECT_EQ(expect_image, v.Shape());
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PhotoTourDataset
/// Description: Test iterator of PhotoTourDataset with wrong column
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestPhotoTourIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPhotoTourIteratorWrongColumn.";
  // Create a PhotoTour Dataset
  std::string folder_path = datasets_root_path_ + "/testPhotoTourData";
  std::shared_ptr<Dataset> ds = PhotoTour(folder_path, "liberty", "train", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);
  
  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: PhotoTourDataset
/// Description: Test PhotoTourDataset GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestGetPhotoTourTrainDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetPhotoTourTrainDatasetSize.";

  // Create a PhotoTour Train Dataset
  std::string folder_path = datasets_root_path_ + "/testPhotoTourData";
  std::shared_ptr<Dataset> ds = PhotoTour(folder_path, "liberty", "train");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 100);
}

/// Feature: PhotoTourDataset
/// Description: Test PhotoTourDataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestPhotoTourTrainDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPhotoTourTrainDatasetGetters.";

  // Create a PhotoTour Train Dataset
  std::string folder_path = datasets_root_path_ + "/testPhotoTourData";
  std::shared_ptr<Dataset> ds = PhotoTour(folder_path, "liberty", "train");
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 100);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  std::vector<std::string> column_names = {"image"};
  int64_t num_classes = ds->GetNumClasses();
  EXPECT_EQ(types.size(), 1);
  EXPECT_EQ(types[0].ToString(), "uint8");
  EXPECT_EQ(shapes.size(), 1);
  EXPECT_EQ(shapes[0].ToString(), "<64,64,1>");
  EXPECT_EQ(num_classes, -1);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);

  EXPECT_EQ(ds->GetDatasetSize(), 100);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetNumClasses(), -1);

  EXPECT_EQ(ds->GetColumnNames(), column_names);
  EXPECT_EQ(ds->GetDatasetSize(), 100);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetNumClasses(), -1);
  EXPECT_EQ(ds->GetDatasetSize(), 100);
}

/// Feature: PhotoTourDataset
/// Description: Test PhotoTourDataset with invalid folder path input
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestPhotoTourDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPhotoTourDatasetFail.";

  // Create a PhotoTour Dataset
  std::shared_ptr<Dataset> ds = PhotoTour("", "", "train", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid PhotoTour input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: PhotoTourDataset
/// Description: Test PhotoTourDataset with invalid usage
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestPhotoTourDatasetWithInvalidUsageFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPhotoTourDatasetWithInvalidUsageFail.";

  // Create a PhotoTour Dataset
  std::string folder_path = datasets_root_path_ + "/testPhotoTourData";
  std::shared_ptr<Dataset> ds = PhotoTour(folder_path, "liberty", "validation");
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid PhotoTour input, validation is not a valid usage
  EXPECT_EQ(iter, nullptr);
}

/// Feature: PhotoTourDataset
/// Description: Test PhotoTourDataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestPhotoTourDatasetWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPhotoTourDatasetWithNullSamplerFail.";

  // Create a PhotoTour Dataset
  std::string folder_path = datasets_root_path_ + "/testPhotoTourData";
  std::shared_ptr<Dataset> ds = PhotoTour(folder_path, "liberty", "train", nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid PhotoTour input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}
