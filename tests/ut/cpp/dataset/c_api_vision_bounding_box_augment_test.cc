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
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/vision.h"

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

// Tests for vision C++ API BoundingBoxAugment TensorTransform Operation

TEST_F(MindDataTestPipeline, TestBoundingBoxAugmentSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBoundingBoxAugmentSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  /* FIXME - Resolve BoundingBoxAugment to properly handle TensorTransform input
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> bound_box_augment = std::make_shared<vision::BoundingBoxAugment>(vision::RandomRotation({90.0}), 1.0);
  EXPECT_NE(bound_box_augment, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({bound_box_augment}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);
  // Manually terminate the pipeline
  iter->Stop();
  */
}

TEST_F(MindDataTestPipeline, TestBoundingBoxAugmentFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBoundingBoxAugmentFail with invalid params.";

  // FIXME: For error tests, need to check for failure from CreateIterator execution
  /*
  // Testing invalid ratio < 0.0
  std::shared_ptr<TensorTransform> bound_box_augment = std::make_shared<vision::BoundingBoxAugment>(vision::RandomRotation({90.0}), -1.0);
  EXPECT_EQ(bound_box_augment, nullptr);
  // Testing invalid ratio > 1.0
  std::shared_ptr<TensorTransform> bound_box_augment1 = std::make_shared<vision::BoundingBoxAugment>(vision::RandomRotation({90.0}), 2.0);
  EXPECT_EQ(bound_box_augment1, nullptr);
  // Testing invalid transform
  std::shared_ptr<TensorTransform> bound_box_augment2 = std::make_shared<vision::BoundingBoxAugment>(nullptr, 0.5);
  EXPECT_EQ(bound_box_augment2, nullptr);
  */
}
