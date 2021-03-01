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

TEST_F(MindDataTestPipeline, TestBoundingBoxAugmentSuccess1Shr) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBoundingBoxAugmentSuccess1Shr.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Use shared pointers
  std::shared_ptr<TensorTransform> random_rotation_op(new vision::RandomRotation({90.0}));
  std::shared_ptr<TensorTransform> bound_box_augment_op(new vision::BoundingBoxAugment({random_rotation_op}, 1.0));

  // Create a Map operation on ds
  ds = ds->Map({bound_box_augment_op}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);
  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestBoundingBoxAugmentSuccess2Auto) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBoundingBoxAugmentSuccess2Auto.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Use auto for raw pointers
  // Note that with auto and new, we have to explicitly delete the allocated object as shown below.
  auto random_rotation_op(new vision::RandomRotation({90.0}));
  auto bound_box_augment_op(new vision::BoundingBoxAugment({random_rotation_op}, 1.0));

  // Create a Map operation on ds
  ds = ds->Map({bound_box_augment_op}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);
  // Manually terminate the pipeline
  iter->Stop();

  // Delete allocated objects with raw pointers
  delete random_rotation_op;
  delete bound_box_augment_op;
}

TEST_F(MindDataTestPipeline, TestBoundingBoxAugmentSuccess3Obj) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBoundingBoxAugmentSuccess3Obj.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Use object references
  vision::RandomRotation random_rotation_op = vision::RandomRotation({90.0});
  vision::BoundingBoxAugment bound_box_augment_op = vision::BoundingBoxAugment({random_rotation_op}, 1.0);

  // Create a Map operation on ds
  ds = ds->Map({bound_box_augment_op}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);
  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestBoundingBoxAugmentFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBoundingBoxAugmentFail1 with invalid ratio parameter.";

  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_rotation_op(new vision::RandomRotation({90.0}));

  // Create BoundingBoxAugment op with invalid ratio < 0.0
  std::shared_ptr<TensorTransform> bound_box_augment_op(new vision::BoundingBoxAugment({random_rotation_op}, -1.0));

  // Create a Map operation on ds
  ds = ds->Map({bound_box_augment_op}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid BoundingBoxAugment input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestBoundingBoxAugmentFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBoundingBoxAugmentFail2 with invalid ratio parameter.";

  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_rotation_op(new vision::RandomRotation({90.0}));

  // Create BoundingBoxAugment op with invalid ratio > 1.0
  std::shared_ptr<TensorTransform> bound_box_augment_op(new vision::BoundingBoxAugment({random_rotation_op}, 2.0));

  // Create a Map operation on ds
  ds = ds->Map({bound_box_augment_op}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid BoundingBoxAugment input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestBoundingBoxAugmentFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBoundingBoxAugmentFail3 with invalid transform.";

  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create BoundingBoxAugment op with invalid nullptr transform
  std::shared_ptr<TensorTransform> bound_box_augment_op(new vision::BoundingBoxAugment(nullptr, 0.5));

  // Create a Map operation on ds
  ds = ds->Map({bound_box_augment_op}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid BoundingBoxAugment input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestBoundingBoxAugmentFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBoundingBoxAugmentFail4 with invalid transform input.";

  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // RandomRotation has invalid input, first column value of degrees is greater than the second column value
  std::shared_ptr<TensorTransform> random_rotation_op(new vision::RandomRotation({50.0, -50.0}));

  // Create BoundingBoxAugment op with invalid transform
  std::shared_ptr<TensorTransform> bound_box_augment_op(new vision::BoundingBoxAugment({random_rotation_op}, 0.25));

  // Create a Map operation on ds
  ds = ds->Map({bound_box_augment_op}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid BoundingBoxAugment input
  EXPECT_EQ(iter, nullptr);
}
