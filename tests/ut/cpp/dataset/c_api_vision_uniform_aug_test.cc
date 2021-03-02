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

TEST_F(MindDataTestPipeline, TestUniformAugWithOps1Shr) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUniformAugWithOps1Shr.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 20));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 1;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Use shared pointers
  std::shared_ptr<TensorTransform> resize_op(new vision::Resize({30, 30}));
  std::shared_ptr<TensorTransform> random_crop_op(new vision::RandomCrop({28, 28}));
  std::shared_ptr<TensorTransform> center_crop_op(new vision::CenterCrop({16, 16}));
  std::shared_ptr<TensorTransform> uniform_aug_op(new vision::UniformAugment({random_crop_op, center_crop_op}, 2));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestUniformAugWithOps2Auto) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUniformAugWithOps2Auto.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 20));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 1;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Use auto for raw pointers
  // Note that with auto and new, we have to explicitly delete the allocated object as shown below.
  auto resize_op(new vision::Resize({30, 30}));
  auto random_crop_op(new vision::RandomCrop({28, 28}));
  auto center_crop_op(new vision::CenterCrop({16, 16}));
  auto uniform_aug_op(new vision::UniformAugment({random_crop_op, center_crop_op}, 2));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();

  // Delete allocated objects with raw pointers
  delete resize_op;
  delete random_crop_op;
  delete center_crop_op;
  delete uniform_aug_op;
}

TEST_F(MindDataTestPipeline, TestUniformAugWithOps3Obj) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUniformAugWithOps3Obj.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 20));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 1;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Use object references
  vision::Resize resize_op = vision::Resize({30, 30});
  vision::RandomCrop random_crop_op = vision::RandomCrop({28, 28});
  vision::CenterCrop center_crop_op = vision::CenterCrop({16, 16});
  vision::UniformAugment uniform_aug_op = vision::UniformAugment({random_crop_op, center_crop_op}, 2);

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestUniformAugmentFail1num_ops) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUniformAugmentFail1num_ops with invalid num_ops parameter.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 20));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop_op(new vision::RandomCrop({28, 28}));
  std::shared_ptr<TensorTransform> center_crop_op(new vision::CenterCrop({16, 16}));

  // UniformAug: num_ops must be greater than 0
  std::shared_ptr<TensorTransform> uniform_aug_op(new vision::UniformAugment({random_crop_op, center_crop_op}, 0));

  // Create a Map operation on ds
  ds = ds->Map({uniform_aug_op}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid UniformAugment input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestUniformAugmentFail2num_ops) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUniformAugmentFail2num_ops with invalid num_ops parameter.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 20));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop_op(new vision::RandomCrop({28, 28}));
  std::shared_ptr<TensorTransform> center_crop_op(new vision::CenterCrop({16, 16}));

  // UniformAug: num_ops is greater than transforms size
  std::shared_ptr<TensorTransform> uniform_aug_op(new vision::UniformAugment({random_crop_op, center_crop_op}, 3));

  // Create a Map operation on ds
  ds = ds->Map({uniform_aug_op}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid UniformAugment input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestUniformAugmentFail3transforms) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUniformAugmentFail3transforms with invalid transform.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 20));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // RandomRotation has invalid input, negative size
  std::shared_ptr<TensorTransform> random_crop_op(new vision::RandomCrop({-28}));

  // Create UniformAug op with invalid transform op
  std::shared_ptr<TensorTransform> uniform_aug_op(new vision::UniformAugment({random_crop_op}, 1));

  // Create a Map operation on ds
  ds = ds->Map({uniform_aug_op}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid UniformAugment input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestUniformAugmentFail4transforms) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUniformAugmentFail4transforms with invalid transform.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 20));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop_op(new vision::RandomCrop({28}));

  // Create UniformAug op with invalid transform op, nullptr
  std::shared_ptr<TensorTransform> uniform_aug_op(new vision::UniformAugment({random_crop_op, nullptr}, 2));

  // Create a Map operation on ds
  ds = ds->Map({uniform_aug_op}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid UniformAugment input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestUniformAugmentFail5transforms) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestUniformAugmentFail5transforms with invalid transform.";

  // Create a Mnist Dataset
  std::string folder_path = datasets_root_path_ + "/testMnistData/";
  std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 20));
  EXPECT_NE(ds, nullptr);

  // Create UniformAug op with invalid transform op empty list
  std::vector<std::shared_ptr<TensorTransform>> list = {};
  std::shared_ptr<TensorTransform> uniform_aug_op(new vision::UniformAugment(list, 1));

  // Create a Map operation on ds
  ds = ds->Map({uniform_aug_op}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid UniformAugment input
  EXPECT_EQ(iter, nullptr);
}
