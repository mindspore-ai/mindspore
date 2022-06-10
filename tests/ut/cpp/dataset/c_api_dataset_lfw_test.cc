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

/// Feature: LFWDataset
/// Description: Test LFWDataset basic usage
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestLFWDataset) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLFWDataset.";

 // Create a LFW Dataset.
 std::string folder_path = datasets_root_path_ + "/testLFW";
 std::shared_ptr<Dataset> ds = LFW(folder_path, "people", "all", "original", false);
 EXPECT_NE(ds, nullptr);

 // Create an iterator over the result of the above dataset.
 // This will trigger the creation of the Execution Tree and launch it.
 std::shared_ptr<Iterator> iter = ds->CreateIterator();
 EXPECT_NE(iter, nullptr);

 // Iterate the dataset and get each row.
 std::unordered_map<std::string, mindspore::MSTensor> row;
 ASSERT_OK(iter->GetNextRow(&row));

 EXPECT_NE(row.find("image"), row.end());
 EXPECT_NE(row.find("label"), row.end());

 uint64_t i = 0;
 while (row.size() != 0) {
   i++;
   auto image = row["image"];
   MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
   ASSERT_OK(iter->GetNextRow(&row));
 }

 EXPECT_EQ(i, 4);

 // Manually terminate the pipeline.
 iter->Stop();
}

/// Feature: LFWDataset
/// Description: Test LFWDataset Getters method using people dataset and RandomSampler
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestLFWPeopleDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLFWPeopleDatasetGetters.";

  // Create a LFW People Dataset.
  std::string folder_path = datasets_root_path_ + "/testLFW";
  std::shared_ptr<Dataset> ds =
    LFW(folder_path, "people", "10fold", "original", false, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 2);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<std::string> column_names = {"image", "label"};
  int64_t num_classes = ds->GetNumClasses();
  EXPECT_EQ(types.size(), 2);
  EXPECT_EQ(types[0].ToString(), "uint8");
  EXPECT_EQ(types[1].ToString(), "uint32");
  EXPECT_EQ(num_classes, -1);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetDatasetSize(), 2);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ds->GetNumClasses(), -1);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: LFWDataset
/// Description: Test LFWDataset Getters method using pairs dataset and RandomSampler
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestLFWPairsDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLFWPairsDatasetGetters.";

  // Create a LFW Pairs Dataset.
  std::string folder_path = datasets_root_path_ + "/testLFW";
  std::shared_ptr<Dataset> ds =
    LFW(folder_path, "pairs", "10fold", "original", false, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 2);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<std::string> column_names = {"image1", "image2", "label"};
  int64_t num_classes = ds->GetNumClasses();
  EXPECT_EQ(types.size(), 3);
  EXPECT_EQ(types[0].ToString(), "uint8");
  EXPECT_EQ(types[1].ToString(), "uint8");
  EXPECT_EQ(types[2].ToString(), "uint32");
  EXPECT_EQ(num_classes, -1);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetDatasetSize(), 2);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ds->GetNumClasses(), -1);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: LFWDataset
/// Description: Test LFWDataset with pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestLFWDatasetWithPipeline) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLFWDatasetWithPipeline.";

 // Create two LFW Dataset.
 std::string folder_path = datasets_root_path_ + "/testLFW";
 std::shared_ptr<Dataset> ds1 =
   LFW(folder_path, "people", "test", "funneled", false);
 std::shared_ptr<Dataset> ds2 =
   LFW(folder_path, "people", "train", "deepfunneled", false);
 EXPECT_NE(ds1, nullptr);
 EXPECT_NE(ds2, nullptr);

 // Create two Repeat operation on ds.
 int32_t repeat_num = 1;
 ds1 = ds1->Repeat(repeat_num);
 EXPECT_NE(ds1, nullptr);
 repeat_num = 1;
 ds2 = ds2->Repeat(repeat_num);
 EXPECT_NE(ds2, nullptr);

 // Create two Project operation on ds.
 std::vector<std::string> column_project = {"image", "label"};
 ds1 = ds1->Project(column_project);
 EXPECT_NE(ds1, nullptr);
 ds2 = ds2->Project(column_project);
 EXPECT_NE(ds2, nullptr);

 // Create a Concat operation on the ds.
 ds1 = ds1->Concat({ds2});
 EXPECT_NE(ds1, nullptr);

 // Create an iterator over the result of the above dataset.
 // This will trigger the creation of the Execution Tree and launch it.
 std::shared_ptr<Iterator> iter = ds1->CreateIterator();
 EXPECT_NE(iter, nullptr);

 // Iterate the dataset and get each row.
 std::unordered_map<std::string, mindspore::MSTensor> row;
 ASSERT_OK(iter->GetNextRow(&row));

 EXPECT_NE(row.find("image"), row.end());
 EXPECT_NE(row.find("label"), row.end());

 uint64_t i = 0;
 while (row.size() != 0) {
   i++;
   auto image = row["image"];
   MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
   ASSERT_OK(iter->GetNextRow(&row));
 }

 EXPECT_EQ(i, 4);

 // Manually terminate the pipeline.
 iter->Stop();
}

/// Feature: LFWDataset
/// Description: Test LFWDataset GetDatasetSize
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestLFWGetDatasetSize) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGetLFWDatasetSize.";

 // Create a LFW Dataset.
 std::string folder_path = datasets_root_path_ + "/testLFW";
 std::shared_ptr<Dataset> ds = LFW(folder_path, "pairs", "10fold", "original", false);
 EXPECT_NE(ds, nullptr);

 EXPECT_EQ(ds->GetDatasetSize(), 4);
}

/// Feature: LFWDataset
/// Description: Test LFWDataset Getters method with people dataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestLFWDatasetPeopleGetters) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLFWPeopleDatasetGetters.";

 // Create a LFW Dataset.
 std::string folder_path = datasets_root_path_ + "/testLFW";
 std::shared_ptr<Dataset> ds = LFW(folder_path, "people", "test", "funneled", true);
 EXPECT_NE(ds, nullptr);

 EXPECT_EQ(ds->GetDatasetSize(), 3);
 std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
 std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
 std::vector<std::string> column_names = {"image", "label"};
 EXPECT_EQ(types.size(), 2);
 EXPECT_EQ(types[0].ToString(), "uint8");
 EXPECT_EQ(types[1].ToString(), "uint32");
 EXPECT_EQ(shapes.size(), 2);
 EXPECT_EQ(shapes[1].ToString(), "<>");
 EXPECT_EQ(ds->GetBatchSize(), 1);
 EXPECT_EQ(ds->GetRepeatCount(), 1);

 EXPECT_EQ(ds->GetDatasetSize(), 3);
 EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
 EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);

 EXPECT_EQ(ds->GetColumnNames(), column_names);
 EXPECT_EQ(ds->GetDatasetSize(), 3);
 EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
 EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
 EXPECT_EQ(ds->GetBatchSize(), 1);
 EXPECT_EQ(ds->GetRepeatCount(), 1);
 EXPECT_EQ(ds->GetDatasetSize(), 3);
}

/// Feature: LFWDataset
/// Description: Test LFWDataset Getters method with pairs dataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestLFWDatasetPairsGetters) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLFWDatasetPairsGetters.";

 // Create a LFW Dataset.
 std::string folder_path = datasets_root_path_ + "/testLFW";
 std::shared_ptr<Dataset> ds = LFW(folder_path, "pairs", "10fold", "original", true);
 EXPECT_NE(ds, nullptr);

 EXPECT_EQ(ds->GetDatasetSize(), 4);
 std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
 std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
 std::vector<std::string> column_names = {"image1", "image2", "label"};
 EXPECT_EQ(types.size(), 3);
 EXPECT_EQ(types[0].ToString(), "uint8");
 EXPECT_EQ(types[1].ToString(), "uint8");
 EXPECT_EQ(types[2].ToString(), "uint32");
 EXPECT_EQ(shapes.size(), 3);
 EXPECT_EQ(shapes[1].ToString(), "<2268,4032,3>");
 EXPECT_EQ(ds->GetBatchSize(), 1);
 EXPECT_EQ(ds->GetRepeatCount(), 1);

 EXPECT_EQ(ds->GetDatasetSize(), 4);
 EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
 EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);

 EXPECT_EQ(ds->GetColumnNames(), column_names);
 EXPECT_EQ(ds->GetDatasetSize(), 4);
 EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
 EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);
 EXPECT_EQ(ds->GetBatchSize(), 1);
 EXPECT_EQ(ds->GetRepeatCount(), 1);
 EXPECT_EQ(ds->GetDatasetSize(), 4);
}

/// Feature: LFWDataset
/// Description: Test LFWDataset with all usage
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestLFWDatasetUsage) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLFWDatasetUsage.";

 // Create a LFW Dataset.
 std::string folder_path = datasets_root_path_ + "/testLFW";
 std::shared_ptr<Dataset> ds = LFW(folder_path, "pairs", "all", "funneled", false,
                                   std::make_shared<RandomSampler>(false, 7));
 EXPECT_NE(ds, nullptr);

 // Create an iterator over the result of the above dataset.
 // This will trigger the creation of the Execution Tree and launch it.
 std::shared_ptr<Iterator> iter = ds->CreateIterator();
 EXPECT_NE(iter, nullptr);

 // Iterate the dataset and get each row.
 std::unordered_map<std::string, mindspore::MSTensor> row;
 ASSERT_OK(iter->GetNextRow(&row));

 EXPECT_NE(row.find("image1"), row.end());
 EXPECT_NE(row.find("image2"), row.end());
 EXPECT_NE(row.find("label"), row.end());

 uint64_t i = 0;
 while (row.size() != 0) {
   i++;
   auto image1 = row["image1"];
   auto image2 = row["image2"];
   MS_LOG(INFO) << "Tensor image1 shape: " << image1.Shape();
   MS_LOG(INFO) << "Tensor image2 shape: " << image2.Shape();
   ASSERT_OK(iter->GetNextRow(&row));
 }

 EXPECT_EQ(i, 7);

 // Manually terminate the pipeline.
 iter->Stop();
}

/// Feature: LFWDataset
/// Description: Test LFWDataset with deepfunneled as image_set
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestLFWDatasetImagSet) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLFWDatasetImageSet.";

 // Create a LFW Dataset.
 std::string folder_path = datasets_root_path_ + "/testLFW";
 std::shared_ptr<Dataset> ds = LFW(folder_path, "pairs", "train", "deepfunneled", false,
                                   std::make_shared<RandomSampler>(false, 7));
 EXPECT_NE(ds, nullptr);

 // Create an iterator over the result of the above dataset.
 // This will trigger the creation of the Execution Tree and launch it.
 std::shared_ptr<Iterator> iter = ds->CreateIterator();
 EXPECT_NE(iter, nullptr);

 // Iterate the dataset and get each row.
 std::unordered_map<std::string, mindspore::MSTensor> row;
 ASSERT_OK(iter->GetNextRow(&row));

 EXPECT_NE(row.find("image1"), row.end());
 EXPECT_NE(row.find("image2"), row.end());
 EXPECT_NE(row.find("label"), row.end());

 uint64_t i = 0;
 while (row.size() != 0) {
   i++;
   auto image1 = row["image1"];
   auto image2 = row["image2"];
   MS_LOG(INFO) << "Tensor image1 shape: " << image1.Shape();
   MS_LOG(INFO) << "Tensor image2 shape: " << image2.Shape();
   ASSERT_OK(iter->GetNextRow(&row));
 }

 EXPECT_EQ(i, 4);

 // Manually terminate the pipeline.
 iter->Stop();
}


/// Feature: LFWDataset
/// Description: Test LFWDataset with invalid folder path (empty string)
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestLFWDatasetWithNullFileDirFail) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLFWDatasetFail.";

 // Create a LFW Dataset in which th folder path is invalid.
 std::shared_ptr<Dataset> ds = LFW("", "people", "10fold", "original", false,
                                   std::make_shared<RandomSampler>(false, 5));
 EXPECT_NE(ds, nullptr);

 // Create an iterator over the result of the above dataset.
 std::shared_ptr<Iterator> iter = ds->CreateIterator();
 // Expect failure: invalid LFW input, state folder path is invalid.
 EXPECT_EQ(iter, nullptr);
}

/// Feature: LFWDataset
/// Description: Test LFWDataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestLFWDatasetWithNullSamplerFail) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLFWDatasetWithNullSamplerFail.";

 // Create a LFW Dataset in which th Sampler is not provided.
 std::string folder_path = datasets_root_path_ + "/testLFW";
 std::shared_ptr<Dataset> ds = LFW(folder_path, "people", "train", "original", false, nullptr);
 EXPECT_NE(ds, nullptr);

 // Create an iterator over the result of the above dataset.
 std::shared_ptr<Iterator> iter = ds->CreateIterator();
 // Expect failure: invalid LFW input, sampler cannot be nullptr.
 EXPECT_EQ(iter, nullptr);
}
