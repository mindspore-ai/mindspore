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
#include "common/common.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/core/tensor.h"

using namespace mindspore::dataset;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

// Tests for datasets (in alphabetical order)

/// Feature: GetRepeatCount
/// Description: Test GetRepeatCount on ImageFolderDataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestGetRepeatCount) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetRepeatCount.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true);
  EXPECT_NE(ds, nullptr);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  ds = ds->Repeat(4);
  EXPECT_NE(ds, nullptr);
  EXPECT_EQ(ds->GetRepeatCount(), 4);
  ds = ds->Repeat(3);
  EXPECT_NE(ds, nullptr);
  EXPECT_EQ(ds->GetRepeatCount(), 3);
}

/// Feature: GetBatchSize
/// Description: Test GetBatchSize on ImageFolderDataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestGetBatchSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetRepeatCount.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true)->Project({"label"});
  EXPECT_NE(ds, nullptr);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  ds = ds->Batch(2);
  EXPECT_NE(ds, nullptr);
  EXPECT_EQ(ds->GetBatchSize(), 2);
  ds = ds->Batch(3);
  EXPECT_NE(ds, nullptr);
  EXPECT_EQ(ds->GetBatchSize(), 3);
}

/// Feature: ImageFolderDataset
/// Description: Test ImageFolderDataset with wrong dataset directory
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestImageFolderWithWrongDatasetDirFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderWithWrongDatasetDirFail.";

  // Create an ImageFolder Dataset
  std::shared_ptr<Dataset> ds = ImageFolder("", true, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid ImageFolder input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: ImageFolderDataset
/// Description: Test ImageFolderDataset with wrong extension
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestImageFolderFailWithWrongExtensionFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderFailWithWrongExtensionFail.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2), {".JGP"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  // Expect no data: cannot find files with specified extension
  EXPECT_ERROR(iter->GetNextRow(&row));
  EXPECT_EQ(row.size(), 0);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: ImageFolderDataset
/// Description: Test ImageFolderDataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestImageFolderGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderGetDatasetSize.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 44);
  EXPECT_EQ(ds->GetNumClasses(), 4);
  EXPECT_EQ(ds->GetNumClasses(), 4);
  EXPECT_EQ(ds->GetDatasetSize(), 44);
  EXPECT_EQ(ds->GetDatasetSize(), 44);
}

/// Feature: ImageFolderDataset
/// Description: Test ImageFolderDataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestImageFolderFailWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderFailWithNullSamplerFail.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid ImageFolder input, sampler cannot be nullptr
  EXPECT_EQ(iter, nullptr);
}

/// Feature: ImageFolderDataset
/// Description: Test ImageFolderDataset with wrong sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestImageFolderFailWithWrongSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderFailWithWrongSamplerFail.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<SequentialSampler>(-2, 5));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid ImageFolder input, sampler is not constructed correctly
  EXPECT_EQ(iter, nullptr);
}

/// Feature: ImageFolderDataset
/// Description: Test ImageFolderDataset with class_index
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestImageFolderClassIndexDatasetSize) {
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  std::map<std::string, int32_t> class_index;
  class_index["class1"] = 111;
  class_index["class2"] = 333;
  auto ds = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(), {}, class_index);
  EXPECT_EQ(ds->GetNumClasses(), 2);
}

/// Feature: ImageFolderDataset
/// Description: Test ImageFolderDataset with wrong class in class_index
/// Expectation: GetNumClasses should return -1
TEST_F(MindDataTestPipeline, TestImageFolderClassIndexDatasetSizeFail) {
  std::string folder_path = datasets_root_path_ + "/testPK/data";
  std::map<std::string, int32_t> class_index;
  class_index["class1"] = 111;
  class_index["wrong class"] = 333;
  auto ds = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(), {}, class_index);
  EXPECT_EQ(ds->GetNumClasses(), -1);
}

/// Feature: GetClassIndexing
/// Description: Test GetClassIndexing on ImageFolderDataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestImageFolderClassIndexing) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestImageFolderClassIndexing.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path);
  EXPECT_NE(ds, nullptr);
  std::vector<std::pair<std::string, std::vector<int32_t>>> indexing = ds->GetClassIndexing();

  EXPECT_EQ(indexing.size(), 4);
  EXPECT_EQ(indexing[0].first, "class1");
  EXPECT_EQ(indexing[0].second[0], 0);
  EXPECT_EQ(indexing[1].first, "class2");
  EXPECT_EQ(indexing[1].second[0], 1);
  EXPECT_EQ(indexing[2].first, "class3");
  EXPECT_EQ(indexing[2].second[0], 2);
  EXPECT_EQ(indexing[3].first, "class4");
  EXPECT_EQ(indexing[3].second[0], 3);

  std::map<std::string, int32_t> class_index;
  class_index["class1"] = 0;
  class_index["class2"] = 1;
  class_index["class3"] = 9;
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(), {}, class_index);
  std::vector<std::pair<std::string, std::vector<int32_t>>> indexing2 = ds2->GetClassIndexing();

  EXPECT_EQ(indexing2.size(), 3);
  EXPECT_EQ(indexing2[0].first, "class1");
  EXPECT_EQ(indexing2[0].second[0], 0);
  EXPECT_EQ(indexing2[1].first, "class2");
  EXPECT_EQ(indexing2[1].second[0], 1);
  EXPECT_EQ(indexing2[2].first, "class3");
  EXPECT_EQ(indexing2[2].second[0], 9);
}
