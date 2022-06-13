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
#include "minddata/dataset/core/tensor.h"

using namespace mindspore::dataset;
using mindspore::dataset::DataType;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
protected:
};

/// Feature: KITTIDataset
/// Description: Test KITTIDataset in pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestKITTIPipeline) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKITTIPipeline.";

 // Create a KITTI Dataset.
 std::string folder_path = datasets_root_path_ + "/testKITTI";
 std::shared_ptr<Dataset> ds = KITTI(folder_path, "train", false, std::make_shared<SequentialSampler>(0, 2));
 EXPECT_NE(ds, nullptr);

 // Create an iterator over the result of the above dataset.
 // This will trigger the creation of the Execution Tree and launch it.
 std::shared_ptr<Iterator> iter = ds->CreateIterator();
 EXPECT_NE(iter, nullptr);

 // Iterate the dataset and get each row.
 std::unordered_map<std::string, mindspore::MSTensor> row;
 ASSERT_OK(iter->GetNextRow(&row));

 // Check if KITTI() read correct images.
 std::string expect_file[] = {"000000", "000001", "000002"};

 uint64_t i = 0;
 while (row.size() != 0) {
   auto image = row["image"];
   auto label = row["label"];
   MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
   MS_LOG(INFO) << "Tensor label shape: " << label.Shape();

   mindspore::MSTensor expect_image =
     ReadFileToTensor(folder_path + "/data_object_image_2/training/image_2/" + expect_file[i] + ".png");
   EXPECT_MSTENSOR_EQ(image, expect_image);

   ASSERT_OK(iter->GetNextRow(&row));
   i++;
 }

 EXPECT_EQ(i, 2);

 // Manually terminate the pipeline.
 iter->Stop();
}

/// Feature: KITTIDataset
/// Description: Test KITTIDataset Getters method
/// Expectation: Get correct number of data and correct tensor shape
TEST_F(MindDataTestPipeline, TestKITTITrainDatasetGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKITTITrainDatasetGetters.";

  // Create a KITTI Train Dataset.
  std::string folder_path = datasets_root_path_ + "/testKITTI";
  std::shared_ptr<Dataset> ds = KITTI(folder_path, "train", true, std::make_shared<SequentialSampler>(0, 2));
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 2);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<std::string> column_names = {"image", "label", "truncated", "occluded", "alpha",
                                           "bbox", "dimensions", "location", "rotation_y"};
  int64_t num_classes = ds->GetNumClasses();
  EXPECT_EQ(types.size(), 9);
  EXPECT_EQ(types[0].ToString(), "uint8");
  EXPECT_EQ(types[1].ToString(), "uint32");
  EXPECT_EQ(types[2].ToString(), "float32");
  EXPECT_EQ(types[3].ToString(), "uint32");
  EXPECT_EQ(types[4].ToString(), "float32");
  EXPECT_EQ(types[5].ToString(), "float32");
  EXPECT_EQ(types[6].ToString(), "float32");
  EXPECT_EQ(types[7].ToString(), "float32");
  EXPECT_EQ(types[8].ToString(), "float32");
  EXPECT_EQ(num_classes, -1);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetDatasetSize(), 2);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ds->GetNumClasses(), -1);
  EXPECT_EQ(ds->GetColumnNames(), column_names);
  EXPECT_EQ(ds->GetDatasetSize(), 2);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);
  EXPECT_EQ(ds->GetNumClasses(), -1);
  EXPECT_EQ(ds->GetDatasetSize(), 2);
}

/// Feature: KITTIDataset
/// Description: Test KITTIDataset with train dataset and test decode
/// Expectation: Getters get the correct value
TEST_F(MindDataTestPipeline, TestKITTIUsageTrainDecodeFalse) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKITTIGettersTrainDecodeFalse.";

 // Create a KITTI Dataset.
 std::string folder_path = datasets_root_path_ + "/testKITTI";
 std::shared_ptr<Dataset> ds = KITTI(folder_path, "train", false, std::make_shared<SequentialSampler>(0, 2));
 EXPECT_NE(ds, nullptr);

 ds = ds->Batch(2);
 ds = ds->Repeat(2);

 EXPECT_EQ(ds->GetDatasetSize(), 2);
 std::vector<std::string> column_names = {"image", "label", "truncated", "occluded", "alpha",
                                          "bbox", "dimensions", "location", "rotation_y"};
 EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: KITTIDataset
/// Description: Test KITTIDataset with test dataset and test decode
/// Expectation: Getters of KITTI get the correct value
TEST_F(MindDataTestPipeline, TestKITTIUsageTestDecodeTrue) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKITTIGettersTestDecodeTrue.";

 // Create a KITTI Dataset.
 std::string folder_path = datasets_root_path_ + "/testKITTI";
 std::shared_ptr<Dataset> ds = KITTI(folder_path, "test", true, std::make_shared<SequentialSampler>(0, 2));
 EXPECT_NE(ds, nullptr);

 ds = ds->Batch(2);
 ds = ds->Repeat(2);

 EXPECT_EQ(ds->GetDatasetSize(), 2);
 std::vector<std::string> column_names = {"image"};
 EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: KITTIDataset
/// Description: Test KITTIDataset with RandomSampler
/// Expectation: Getters of KITTI get the correct value
TEST_F(MindDataTestPipeline, TestKITTIPipelineRandomSampler) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKITTIPipelineRandomSampler.";

 // Create a KITTI Dataset.
 std::string folder_path = datasets_root_path_ + "/testKITTI";
 std::shared_ptr<Dataset> ds = KITTI(folder_path, "test", true, std::make_shared<RandomSampler>(false, 2));
 EXPECT_NE(ds, nullptr);

 ds = ds->Batch(2);
 ds = ds->Repeat(2);

 EXPECT_EQ(ds->GetDatasetSize(), 2);
 std::vector<std::string> column_names = {"image"};
 EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: KITTIDataset
/// Description: Test KITTIDataset with DistributedSampler
/// Expectation: Getters of KITTI get the correct value
TEST_F(MindDataTestPipeline, TestKITTIPipelineDistributedSampler) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKITTIPipelineDistributedSampler.";

 // Create a KITTI Dataset.
 std::string folder_path = datasets_root_path_ + "/testKITTI";
 // num_shards=3, shard_id=0, shuffle=false, num_samplers=0, seed=0, offset=-1, even_dist=true
 DistributedSampler sampler = DistributedSampler(3, 0, false, 0, 0, -1, true);
 std::shared_ptr<Dataset> ds = KITTI(folder_path, "train", false, sampler);
 EXPECT_NE(ds, nullptr);

 // Iterate the dataset and get each row
 std::shared_ptr<Iterator> iter = ds->CreateIterator();
 EXPECT_NE(iter, nullptr);
 std::unordered_map<std::string, mindspore::MSTensor> row;
 ASSERT_OK(iter->GetNextRow(&row));

 uint64_t i = 0;
 while (row.size() != 0) {
   i++;
   auto label = row["image"];
   ASSERT_OK(iter->GetNextRow(&row));
 }

 EXPECT_EQ(i, 1);
 iter->Stop();
}

/// Feature: KITTIDataset
/// Description: Test KITTIDataset with null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestKITTIWithNullSamplerError) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKITTIWithNullSamplerError.";
 // Create a KITTI Dataset.
 std::string folder_path = datasets_root_path_ + "/testKITTI";
 std::shared_ptr<Dataset> ds = KITTI(folder_path, "train", false, nullptr);
 EXPECT_NE(ds, nullptr);

 // Create an iterator over the result of the above dataset.
 std::shared_ptr<Iterator> iter = ds->CreateIterator();
 // Expect failure: invalid KITTI input, sampler cannot be nullptr.
 EXPECT_EQ(iter, nullptr);
}

/// Feature: KITTIDataset
/// Description: Test KITTIDataset with empty string path
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestKITTIWithNullPath) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKITTIWithNullPath.";
 // Create a KITTI Dataset.
 std::string folder_path = "";
 std::shared_ptr<Dataset> ds = KITTI(folder_path, "train", false, nullptr);
 EXPECT_NE(ds, nullptr);

 // Create an iterator over the result of the above dataset.
 std::shared_ptr<Iterator> iter = ds->CreateIterator();
 // Expect failure: invalid KITTI input, path cannot be "".
 EXPECT_EQ(iter, nullptr);
}

/// Feature: KITTIDataset
/// Description: Test KITTIDataset with wrong usage
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestKITTIWithWrongUsage) {
 MS_LOG(INFO) << "Doing MindDataTestPipeline-TestKITTIWithWrongUsage.";
 // Create a KITTI Dataset.
 std::string folder_path = "";
 std::shared_ptr<Dataset> ds = KITTI(folder_path, "all", false, nullptr);
 EXPECT_NE(ds, nullptr);

 // Create an iterator over the result of the above dataset.
 std::shared_ptr<Iterator> iter = ds->CreateIterator();
 // Expect failure: invalid KITTI input, path cannot be "".
 EXPECT_EQ(iter, nullptr);
}
