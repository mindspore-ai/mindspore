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

// Tests for vision C++ API RandomSelectSubpolicy TensorTransform Operations

TEST_F(MindDataTestPipeline, TestRandomSelectSubpolicySuccess1Shr) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSelectSubpolicySuccess1Shr.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 7));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Use shared pointers
  // Valid case: TensorTransform is not null and probability is between (0,1)
  std::shared_ptr<TensorTransform> invert_op(new vision::Invert());
  std::shared_ptr<TensorTransform> equalize_op(new vision::Equalize());
  std::shared_ptr<TensorTransform> resize_op(new vision::Resize({15, 15}));

  // Prepare input parameters for RandomSelectSubpolicy op
  auto invert_pair = std::make_pair(invert_op, 0.5);
  auto equalize_pair = std::make_pair(equalize_op, 0.5);
  auto resize_pair = std::make_pair(resize_op, 1);

  // Create RandomSelectSubpolicy op
  std::vector<std::pair<std::shared_ptr<TensorTransform>, double>> policy = {invert_pair, equalize_pair, resize_pair};
  std::shared_ptr<TensorTransform> random_select_subpolicy_op(new vision::RandomSelectSubpolicy({policy}));

  // Create a Map operation on ds
  ds = ds->Map({random_select_subpolicy_op});
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

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomSelectSubpolicySuccess2Auto) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSelectSubpolicySuccess2Auto.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 7));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Use auto for raw pointers
  // Note that with auto and new, we have to explicitly delete the allocated object as shown below.
  // Valid case: TensorTransform is not null and probability is between (0,1)
  auto invert_op(new vision::Invert());
  auto equalize_op(new vision::Equalize());
  auto resize_op(new vision::Resize({15, 15}));

  // Prepare input parameters for RandomSelectSubpolicy op
  auto invert_pair = std::make_pair(invert_op, 0.5);
  auto equalize_pair = std::make_pair(equalize_op, 0.5);
  auto resize_pair = std::make_pair(resize_op, 1);

  std::vector<std::pair<TensorTransform *, double>> policy = {invert_pair, equalize_pair, resize_pair};

  // Create RandomSelectSubpolicy op
  auto random_select_subpolicy_op(new vision::RandomSelectSubpolicy({policy}));

  // Create a Map operation on ds
  ds = ds->Map({random_select_subpolicy_op});
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

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();

  // Delete allocated objects with raw pointers
  delete invert_op;
  delete equalize_op;
  delete resize_op;
  delete random_select_subpolicy_op;
}

TEST_F(MindDataTestPipeline, TestRandomSelectSubpolicySuccess3Obj) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSelectSubpolicySuccess3Obj.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 7));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Use object references
  // Valid case: TensorTransform is not null and probability is between (0,1)
  vision::Invert invert_op = vision::Invert();
  vision::Equalize equalize_op = vision::Equalize();
  vision::Resize resize_op = vision::Resize({15, 15});

  // Prepare input parameters for RandomSelectSubpolicy op
  auto invert_pair = std::make_pair(std::ref(invert_op), 0.5);
  auto equalize_pair = std::make_pair(std::ref(equalize_op), 0.5);
  auto resize_pair = std::make_pair(std::ref(resize_op), 1);
  std::vector<std::pair<std::reference_wrapper<TensorTransform>, double>> policy = {invert_pair, equalize_pair,
                                                                                    resize_pair};
  // Create RandomSelectSubpolicy op
  vision::RandomSelectSubpolicy random_select_subpolicy_op = vision::RandomSelectSubpolicy({policy});

  // Create a Map operation on ds
  ds = ds->Map({random_select_subpolicy_op});
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

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomSelectSubpolicySuccess4MultiPolicy) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSelectSubpolicySuccess1MultiPolicy.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 7));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Tensor transform ops have shared pointers
  // Valid case: TensorTransform is not null and probability is between (0,1)
  std::shared_ptr<TensorTransform> invert_op(new vision::Invert());
  std::shared_ptr<TensorTransform> equalize_op(new vision::Equalize());
  std::shared_ptr<TensorTransform> resize_op(new vision::Resize({15, 15}));

  // Prepare input parameters for RandomSelectSubpolicy op
  auto invert_pair = std::make_pair(invert_op, 0.75);
  auto equalize_pair = std::make_pair(equalize_op, 0.25);
  auto resize_pair = std::make_pair(resize_op, 0.5);

  // Create RandomSelectSubpolicy op with 2 policies
  std::vector<std::pair<std::shared_ptr<TensorTransform>, double>> policy1 = {resize_pair, invert_pair};
  std::vector<std::pair<std::shared_ptr<TensorTransform>, double>> policy2 = {equalize_pair};
  std::shared_ptr<TensorTransform> random_select_subpolicy_op(new vision::RandomSelectSubpolicy({policy1, policy2}));

  // Create a Map operation on ds
  ds = ds->Map({random_select_subpolicy_op});
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

  EXPECT_EQ(i, 7);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomSelectSubpolicyFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSelectSubpolicyFail1.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 7));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> invert_op(new vision::Invert());
  std::shared_ptr<TensorTransform> equalize_op(new vision::Equalize());
  std::shared_ptr<TensorTransform> resize_op(new vision::Resize({15, 15}));

  // Prepare input parameters for RandomSelectSubpolicy op
  // For RandomSelectSubpolicy : probability of transform must be between 0.0 and 1.0
  // Equalize pair has invalid negative probability
  auto invert_pair = std::make_pair(invert_op, 0.5);
  auto equalize_pair = std::make_pair(equalize_op, -0.5);
  auto resize_pair = std::make_pair(resize_op, 1);

  // Create RandomSelectSubpolicy op
  std::vector<std::pair<std::shared_ptr<TensorTransform>, double>> policy = {invert_pair, equalize_pair, resize_pair};
  std::shared_ptr<TensorTransform> random_select_subpolicy_op(new vision::RandomSelectSubpolicy({policy}));

  // Create a Map operation on ds
  ds = ds->Map({random_select_subpolicy_op});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomSelectSubpolicy input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomSelectSubpolicyFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSelectSubpolicyFail2.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 7));
  EXPECT_NE(ds, nullptr);

  // Create RandomSelectSubpolicy op with invalid empty subpolicy
  std::vector<std::pair<std::shared_ptr<TensorTransform>, double>> policy = {};
  std::shared_ptr<TensorTransform> random_select_subpolicy_op(new vision::RandomSelectSubpolicy({policy}));

  // Create a Map operation on ds
  ds = ds->Map({random_select_subpolicy_op});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomSelectSubpolicy input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomSelectSubpolicyFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSelectSubpolicyFail3.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 7));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> invert_op(new vision::Invert());
  std::shared_ptr<TensorTransform> equalize_op(new vision::Equalize());
  std::shared_ptr<TensorTransform> resize_op(new vision::Resize({15, 15}));

  // Prepare input parameters for RandomSelectSubpolicy op
  auto invert_pair = std::make_pair(invert_op, 0.5);
  auto equalize_pair = std::make_pair(equalize_op, 0.5);
  auto resize_pair = std::make_pair(resize_op, 1);

  // Prepare pair with nullptr op
  auto dummy_pair = std::make_pair(nullptr, 0.25);

  // Create RandomSelectSubpolicy op with invalid nullptr pair
  std::vector<std::pair<std::shared_ptr<TensorTransform>, double>> policy = {invert_pair, dummy_pair, equalize_pair,
                                                                             resize_pair};
  std::shared_ptr<TensorTransform> random_select_subpolicy_op(new vision::RandomSelectSubpolicy({policy}));

  // Create a Map operation on ds
  ds = ds->Map({random_select_subpolicy_op});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomSelectSubpolicy input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomSelectSubpolicyFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSelectSubpolicyFail4.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 7));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Create RandomVerticalFlip op with invalid negative input
  std::shared_ptr<TensorTransform> vertflip_op(new vision::RandomVerticalFlip(-2.0));

  // Prepare input parameters for RandomSelectSubpolicy op
  auto vertflip_pair = std::make_pair(vertflip_op, 1);

  // Create RandomSelectSubpolicy op with invalid transform op within a subpolicy
  std::vector<std::pair<std::shared_ptr<TensorTransform>, double>> policy = {vertflip_pair};
  std::shared_ptr<TensorTransform> random_select_subpolicy_op(new vision::RandomSelectSubpolicy({policy}));

  // Create a Map operation on ds
  ds = ds->Map({random_select_subpolicy_op});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomSelectSubpolicy input
  EXPECT_EQ(iter, nullptr);
}
