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

// Tests for vision UniformAugment
// Tests for vision C++ API UniformAugment TensorTransform Operations

TEST_F(MindDataTestPipeline, TestUniformAugmentFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUniformAugmentFail1 with invalid num_ops parameter.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  /*
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop_op(new vision::RandomCrop({28, 28}));
  EXPECT_NE(random_crop_op, nullptr);

  std::shared_ptr<TensorTransform> center_crop_op(new vision::CenterCrop({16, 16}));
  EXPECT_NE(center_crop_op, nullptr);

  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // UniformAug: num_ops must be greater than 0
  std::shared_ptr<TensorTransform> uniform_aug_op1(new vision::UniformAugment({random_crop_op, center_crop_op}, 0));
  EXPECT_EQ(uniform_aug_op1, nullptr);

  // UniformAug: num_ops must be greater than 0
  std::shared_ptr<TensorTransform> uniform_aug_op2(new vision::UniformAugment({random_crop_op, center_crop_op}, -1));
  EXPECT_EQ(uniform_aug_op2, nullptr);

  // UniformAug: num_ops is greater than transforms size
  std::shared_ptr<TensorTransform> uniform_aug_op3(new vision::UniformAugment({random_crop_op, center_crop_op}, 3));
  EXPECT_EQ(uniform_aug_op3, nullptr);
  */

}

TEST_F(MindDataTestPipeline, TestUniformAugmentFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUniformAugmentFail2 with invalid transform.";

  // FIXME: For error tests, need to check for failure from CreateIterator execution
  /*
  // UniformAug: transform ops must not be null
  std::shared_ptr<TensorTransform> uniform_aug_op1(new vision::UniformAugment({vision::RandomCrop({-28})}, 1));
  EXPECT_NE(uniform_aug_op1, nullptr);

  // UniformAug: transform ops must not be null
  std::shared_ptr<TensorTransform> uniform_aug_op2(new vision::UniformAugment({vision::RandomCrop({28}), nullptr}, 2));
  EXPECT_NE(uniform_aug_op2, nullptr);

  // UniformAug: transform list must not be empty
  std::shared_ptr<TensorTransform> uniform_aug_op3(new vision::UniformAugment({}, 1));
  EXPECT_NE(uniform_aug_op3, nullptr);
  */
}

TEST_F(MindDataTestPipeline, TestUniformAugWithOps) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUniformAugWithOps.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 20));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 1;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> resize_op(new vision::Resize({30, 30}));
  EXPECT_NE(resize_op, nullptr);

  std::shared_ptr<TensorTransform> random_crop_op(new vision::RandomCrop({28, 28}));
  EXPECT_NE(random_crop_op, nullptr);

  std::shared_ptr<TensorTransform> center_crop_op(new vision::CenterCrop({16, 16}));
  EXPECT_NE(center_crop_op, nullptr);

  std::shared_ptr<TensorTransform> uniform_aug_op(new vision::UniformAugment({random_crop_op, center_crop_op}, 2));
  EXPECT_NE(uniform_aug_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({resize_op, uniform_aug_op});
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

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}
