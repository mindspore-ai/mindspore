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
#include "minddata/dataset/include/dataset/execute.h"
#include "minddata/dataset/include/dataset/vision.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestSlicePatches : public UT::DatasetOpTesting {
 protected:
};

/// Feature: SlicePatches op
/// Description: Test SlicePatches op with invalid inputs (num_height and num_width)
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestSlicePatches, TestSlicePacthesParamCheck) {
  MS_LOG(INFO) << "Doing TestSlicePatchesParamCheck with invalid parameters.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Case 1: num_height is not positive
  // Create objects for the tensor ops
  auto slice_patches_1 = std::make_shared<vision::SlicePatches>(-1);
  auto ds1 = ds->Map({slice_patches_1});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid num_height for SlicePatches
  EXPECT_EQ(iter1, nullptr);

  // Case 2: num_width is not positive
  // Create objects for the tensor ops
  auto slice_patches_2 = std::make_shared<vision::SlicePatches>(1, 0);
  auto ds2 = ds->Map({slice_patches_2});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid num_height for SlicePatches
  EXPECT_EQ(iter2, nullptr);
}

/// Feature: SlicePatches op
/// Description: Test SlicePatches op in pipeline mode
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestSlicePatches, TestSlicePatchesPipeline) {
  MS_LOG(INFO) << "Doing TestGaussianBlurPipeline.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto slice_patches = std::make_shared<vision::SlicePatches>(2, 2);

  // Create a Map operation on ds
  ds = ds->Map({slice_patches}, {"image"}, {"img0", "img1", "img2", "img3"});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<ProjectDataset> project_ds = ds->Project({"img0", "img1", "img2", "img3"});
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (!row.empty()) {
    i++;
    ASSERT_EQ(row.size(), 4);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: SlicePatches op
/// Description: Test SlicePatches op in eager mode
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestSlicePatches, TestSlicePatchesEager) {
  MS_LOG(INFO) << "Doing TestGaussianBlurEager.";

  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  std::vector<mindspore::MSTensor> input{image};
  std::vector<mindspore::MSTensor> output;

  // Transform params
  auto decode = vision::Decode();
  auto slice_patches = vision::SlicePatches(2, 2);

  auto transform = Execute({decode, slice_patches});
  Status rc = transform(input, &output);

  EXPECT_EQ(rc, Status::OK());
}
