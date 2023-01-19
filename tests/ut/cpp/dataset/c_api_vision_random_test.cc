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
#include "minddata/dataset/include/dataset/config.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/include/dataset/vision.h"

using namespace mindspore::dataset;
using mindspore::dataset::InterpolationMode;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

// Tests for vision C++ API Random* TensorTransform Operations (in alphabetical order)

/// Feature: RandomAffine op
/// Description: Test RandomAffine op with invalid parameters
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomAffineFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAffineFail with invalid parameters.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Case 1: Empty input for translate
  // Create objects for the tensor ops
  auto affine1 = std::make_shared<vision::RandomAffine>(std::vector<float_t>{0.0, 0.0},  std::vector<float_t>{});
  auto ds1 = ds->Map({affine1});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid input for RandomAffine
  EXPECT_EQ(iter1, nullptr);

  // Case 2: Invalid number of values for translate
  // Create objects for the tensor ops
  auto affine2 = std::make_shared<vision::RandomAffine>(
    std::vector<float_t>{0.0, 0.0}, std::vector<float_t>{1, 1, 1, 1, 1});
  auto ds2 = ds->Map({affine2});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid input for RandomAffine
  EXPECT_EQ(iter2, nullptr);

  // Case 3: Invalid number of values for shear
  // Create objects for the tensor ops
  auto affine3 = std::make_shared<vision::RandomAffine>(
    std::vector<float_t>{30.0, 30.0}, std::vector<float_t>{0.0, 0.0}, std::vector<float_t>{2.0, 2.0}, 
    std::vector<float_t>{10.0});
  auto ds3 = ds->Map({affine3});
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid input for RandomAffine
  EXPECT_EQ(iter3, nullptr);
}

/// Feature: RandomAffine op
/// Description: Test RandomAffine op with non-default parameters
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomAffineSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAffineSuccess1 with non-default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto affine = std::make_shared<vision::RandomAffine>(
    std::vector<float>{30.0, 30.0}, std::vector<float>{-1.0, 1.0, -1.0, 1.0}, std::vector<float>{2.0, 2.0}, 
    std::vector<float>{10.0, 10.0, 20.0, 20.0});

  // Create a Map operation on ds
  ds = ds->Map({affine});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomAffine op
/// Description: Test RandomAffine op with default parameters
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomAffineSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAffineSuccess2 with default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto affine = std::make_shared<vision::RandomAffine>(std::vector<float_t>{0.0, 0.0});

  // Create a Map operation on ds
  ds = ds->Map({affine});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomColor op
/// Description: Test RandomColor op with non-default parameters
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomColor) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomColor with non-default parameters.";

  std::shared_ptr<mindspore::dataset::ConfigManager> cfg = mindspore::dataset::GlobalContext::config_manager();
  uint32_t original_num_parallel_workers = cfg->num_parallel_workers();
  cfg->set_num_parallel_workers(1);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Valid case: Set lower bound and upper bound to be the same value zero
  std::shared_ptr<TensorTransform> random_color_op_1 = std::make_shared<vision::RandomColor>(0.0, 0.0);

  // Valid case: Set lower bound as zero and less than upper bound
  std::shared_ptr<TensorTransform> random_color_op_2 = std::make_shared<vision::RandomColor>(0.0, 1.1);

  // Create a Map operation on ds
  ds = ds->Map({random_color_op_1, random_color_op_2});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();

  cfg->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: RandomColorAdjust op
/// Description: Test RandomColorAdjust op with non-default parameters
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomColorAdjust) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomColorAdjust.";

  std::shared_ptr<mindspore::dataset::ConfigManager> cfg = mindspore::dataset::GlobalContext::config_manager();
  uint32_t original_num_parallel_workers = cfg->num_parallel_workers();
  cfg->set_num_parallel_workers(1);

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Use single value for vectors
  auto random_color_adjust1 = std::make_shared<vision::RandomColorAdjust>(
    std::vector<float>{1.0}, std::vector<float>{0.0}, std::vector<float>{0.5}, std::vector<float>{0.5});

  // Use same 2 values for vectors
  auto random_color_adjust2 = std::make_shared<vision::RandomColorAdjust>(
    std::vector<float>{1.0, 1.0}, std::vector<float>{0.0, 0.0}, 
    std::vector<float>{0.5, 0.5}, std::vector<float>{0.5, 0.5});

  // Use different 2 value for vectors
  auto random_color_adjust3 = std::make_shared<vision::RandomColorAdjust>(
    std::vector<float>{0.5, 1.0}, std::vector<float>{0.0, 0.5}, 
    std::vector<float>{0.25, 0.5}, std::vector<float>{0.25, 0.5});

  // Use default input values
  auto random_color_adjust4 = std::make_shared<vision::RandomColorAdjust>();

  // Use subset of explicitly set parameters
  auto random_color_adjust5 = std::make_shared<vision::RandomColorAdjust>(
    std::vector<float>{0.0, 0.5}, std::vector<float>{0.25});

  // Create a Map operation on ds
  ds = ds->Map(
    {random_color_adjust1, random_color_adjust2, random_color_adjust3, random_color_adjust4, random_color_adjust5});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();

  cfg->set_num_parallel_workers(original_num_parallel_workers);
}

/// Feature: RandomCrop op
/// Description: Test RandomCrop op with various size of size vector, padding vector, and fill_value vector
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomCropSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Testing size of size vector is 1
  auto random_crop = std::make_shared<vision::RandomCrop>(std::vector<int32_t>{20});

  // Testing size of size vector is 2
  auto random_crop1 = std::make_shared<vision::RandomCrop>(std::vector<int32_t>{20, 20});

  // Testing size of paddiing vector is 1
  auto random_crop2 = std::make_shared<vision::RandomCrop>(
    std::vector<int32_t>{20, 20}, std::vector<int32_t>{10});

  // Testing size of paddiing vector is 2
  auto random_crop3 = std::make_shared<vision::RandomCrop>(
    std::vector<int32_t>{20, 20}, std::vector<int32_t>{10, 20});

  // Testing size of paddiing vector is 2
  auto random_crop4 = std::make_shared<vision::RandomCrop>(
    std::vector<int32_t>{20, 20}, std::vector<int32_t>{10, 10, 10, 10});

  // Testing size of fill_value vector is 1
  auto random_crop5 = std::make_shared<vision::RandomCrop>(
    std::vector<int32_t>{20, 20}, std::vector<int32_t>{10, 10, 10, 10}, false, std::vector<uint8_t>{5});

  // Testing size of fill_value vector is 3
  auto random_crop6 = std::make_shared<vision::RandomCrop>(
    std::vector<int32_t>{20, 20}, std::vector<int32_t>{10, 10, 10, 10}, false, std::vector<uint8_t>{4, 4, 4});

  // Create a Map operation on ds
  ds = ds->Map({random_crop, random_crop1, random_crop2, random_crop3, random_crop4, random_crop5, random_crop6},
               {"image"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

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

/// Feature: RandomCrop op
/// Description: Test RandomCrop op with multiple fields
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomCropWithMultiField) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropWithMultiField.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Segmentation", "train", {}, true, std::make_shared<SequentialSampler>(0, 1));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  transforms::Duplicate duplicate = transforms::Duplicate();
  auto random_crop = std::make_shared<mindspore::dataset::vision::RandomCrop>(std::vector<int32_t>{500, 500});

  // Create a Map operation on ds
  ds = ds->Map({duplicate}, {"image"}, {"image", "image_copy"});
  EXPECT_NE(ds, nullptr);

  ds = ds->Map({random_crop}, {"image", "image_copy"}, {"image", "image_copy"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto image_copy = row["image_copy"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor image_copy shape: " << image_copy.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);
  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomCrop op
/// Description: Test RandomCrop op with invalid parameters
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomCropFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropFail with invalid parameters.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Case 1: Testing the size parameter is negative.
  // Create objects for the tensor ops
  auto random_crop1 = std::make_shared<vision::RandomCrop>(std::vector<int32_t>{-28, 28});
  auto ds1 = ds->Map({random_crop1});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter1, nullptr);

  // Case 2: Testing the size parameter is None.
  // Create objects for the tensor ops
  auto random_crop2 = std::make_shared<vision::RandomCrop>(std::vector<int32_t>{});
  auto ds2 = ds->Map({random_crop2});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter2, nullptr);

  // Case 3: Testing the size of size vector is 3.
  // Create objects for the tensor ops
  auto random_crop3 = std::make_shared<vision::RandomCrop>(std::vector<int32_t>{28, 28, 28});
  auto ds3 = ds->Map({random_crop3});
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter3, nullptr);

  // Case 4: Testing the padding parameter is negative.
  // Create objects for the tensor ops
  auto random_crop4 = std::make_shared<vision::RandomCrop>(
    std::vector<int32_t>{28, 28}, std::vector<int32_t>{-5});
  auto ds4 = ds->Map({random_crop4});
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter4, nullptr);

  // Case 5: Testing the size of padding vector is empty.
  // Create objects for the tensor ops
  auto random_crop5 = std::make_shared<vision::RandomCrop>(
    std::vector<int32_t>{28, 28}, std::vector<int32_t>{});
  auto ds5 = ds->Map({random_crop5});
  EXPECT_NE(ds5, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter5 = ds5->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter5, nullptr);

  // Case 6: Testing the size of padding vector is 3.
  // Create objects for the tensor ops
  auto random_crop6 = std::make_shared<vision::RandomCrop>(
    std::vector<int32_t>{28, 28}, std::vector<int32_t>{5, 5, 5});
  auto ds6 = ds->Map({random_crop6});
  EXPECT_NE(ds6, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter6 = ds6->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter6, nullptr);

  // Case 7: Testing the size of padding vector is 5.
  // Create objects for the tensor ops
  auto random_crop7 = std::make_shared<vision::RandomCrop>(
    std::vector<int32_t>{28, 28}, std::vector<int32_t>{5, 5, 5, 5, 5});
  auto ds7 = ds->Map({random_crop7});
  EXPECT_NE(ds7, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter7 = ds7->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter7, nullptr);

  // Case 8: Testing the size of fill_value vector is empty.
  // Create objects for the tensor ops
  auto random_crop8 = std::make_shared<vision::RandomCrop>(
    std::vector<int32_t>{28, 28}, std::vector<int32_t>{0, 0, 0, 0}, false, std::vector<uint8_t>{});
  auto ds8 = ds->Map({random_crop8});
  EXPECT_NE(ds8, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter8 = ds8->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter8, nullptr);

  // Case 9: Testing the size of fill_value vector is 2.
  // Create objects for the tensor ops
  auto random_crop9 = std::make_shared<vision::RandomCrop>(
    std::vector<int32_t>{28, 28}, std::vector<int32_t>{0, 0, 0, 0}, false, std::vector<uint8_t>{0, 0});
  auto ds9 = ds->Map({random_crop9});
  EXPECT_NE(ds9, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter9 = ds9->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter9, nullptr);

  // Case 10: Testing the size of fill_value vector is 4.
  // Create objects for the tensor ops
  auto random_crop10 = std::make_shared<vision::RandomCrop>(
    std::vector<int32_t>{28, 28}, std::vector<int32_t>{0, 0, 0, 0}, false, std::vector<uint8_t>{0, 0, 0, 0});
  auto ds10 = ds->Map({random_crop10});
  EXPECT_NE(ds10, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter10 = ds10->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter10, nullptr);
}

/// Feature: RandomCropWithBBox op
/// Description: Test RandomCropWithBBox op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomCropWithBboxSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropWithBboxSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_crop = std::make_shared<mindspore::dataset::vision::RandomCropWithBBox>(
    std::vector<int32_t>{128, 128});

  // Create a Map operation on ds
  ds = ds->Map({random_crop}, {"image", "bbox"}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0], 128);
    EXPECT_EQ(image.Shape()[1], 128);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);
  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomCropWithBBox op
/// Description: Test RandomCropWithBBox op with invalid parameters
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomCropWithBboxFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropWithBboxFail with invalid parameters.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Case 1: The size parameter is negative.
  // Create objects for the tensor ops
  auto random_crop1 = std::make_shared<vision::RandomCropWithBBox>(std::vector<int32_t>{-10});
  auto ds1 = ds->Map({random_crop1});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid input for RandomCropWithBBox
  EXPECT_EQ(iter1, nullptr);

  // Case 2: The parameter in the padding vector is negative.
  // Create objects for the tensor ops
  auto random_crop2 = std::make_shared<vision::RandomCropWithBBox>(
    std::vector<int32_t>{10, 10}, std::vector<int32_t>{-2, 2, 2, 2});
  auto ds2 = ds->Map({random_crop2});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid input for RandomCropWithBBox
  EXPECT_EQ(iter2, nullptr);

  // Case 3: The size container is empty.
  // Create objects for the tensor ops
  auto random_crop3 = std::make_shared<vision::RandomCropWithBBox>(std::vector<int32_t>{});
  auto ds3 = ds->Map({random_crop3});
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid input for RandomCropWithBBox
  EXPECT_EQ(iter3, nullptr);

  // Case 4: The size of the size container is too large.
  // Create objects for the tensor ops
  auto random_crop4 = std::make_shared<vision::RandomCropWithBBox>(std::vector<int32_t>{10, 10, 10});
  auto ds4 = ds->Map({random_crop4});
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid input for RandomCropWithBBox
  EXPECT_EQ(iter4, nullptr);

  // Case 5: The padding container is empty.
  // Create objects for the tensor ops
  auto random_crop5 = std::make_shared<vision::RandomCropWithBBox>(
    std::vector<int32_t>{10, 10}, std::vector<int32_t>{});
  auto ds5 = ds->Map({random_crop5});
  EXPECT_NE(ds5, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter5 = ds5->CreateIterator();
  // Expect failure: invalid input for RandomCropWithBBox
  EXPECT_EQ(iter5, nullptr);

  // Case 6: The size of the padding container is too large.
  // Create objects for the tensor ops
  auto random_crop6 = std::make_shared<vision::RandomCropWithBBox>(
    std::vector<int32_t>{10, 10}, std::vector<int32_t>{5, 5, 5, 5, 5});
  auto ds6 = ds->Map({random_crop6});
  EXPECT_NE(ds6, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter6 = ds6->CreateIterator();
  // Expect failure: invalid input for RandomCropWithBBox
  EXPECT_EQ(iter6, nullptr);

  // Case 7: The fill_value container is empty.
  // Create objects for the tensor ops
  auto random_crop7 = std::make_shared<vision::RandomCropWithBBox>(
    std::vector<int32_t>{10, 10}, std::vector<int32_t>{5, 5, 5, 5}, false, std::vector<uint8_t>{});
  auto ds7 = ds->Map({random_crop7});
  EXPECT_NE(ds7, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter7 = ds7->CreateIterator();
  // Expect failure: invalid input for RandomCropWithBBox
  EXPECT_EQ(iter7, nullptr);

  // Case 8: The size of the fill_value container is too large.
  // Create objects for the tensor ops
  auto random_crop8 = std::make_shared<vision::RandomCropWithBBox>(
    std::vector<int32_t>{10, 10}, std::vector<int32_t>{5, 5, 5, 5}, false, std::vector<uint8_t>{3, 3, 3, 3});
  auto ds8 = ds->Map({random_crop8});
  EXPECT_NE(ds8, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter8 = ds8->CreateIterator();
  // Expect failure: invalid input for RandomCropWithBBox
  EXPECT_EQ(iter8, nullptr);
}

/// Feature: RandomHorizontalFlipWithBBox op
/// Description: Test RandomHorizontalFlipWithBBox op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomHorizontalFlipWithBBoxSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomHorizontalFlipWithBBoxSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_horizontal_flip_op =
    std::make_shared<vision::RandomHorizontalFlipWithBBox>(0.5);

  // Create a Map operation on ds
  ds = ds->Map({random_horizontal_flip_op}, {"image", "bbox"}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);
  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomHorizontalFlip and RandomVerticalFlip ops
/// Description: Test RandomVerticalFlip op then RandomHorizontalFlip op on ImageFolderDataset
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomHorizontalAndVerticalFlip) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomHorizontalAndVerticalFlip for horizontal and vertical flips.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlip>(0.75);
  std::shared_ptr<TensorTransform> random_horizontal_flip_op = std::make_shared<vision::RandomHorizontalFlip>(0.5);

  // Create a Map operation on ds
  ds = ds->Map({random_vertical_flip_op, random_horizontal_flip_op});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomResize op
/// Description: Test RandomResize with single integer input with multiple fields
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomResizeWithMultiField) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeWithMultiField with single integer input.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  transforms::Duplicate duplicate = transforms::Duplicate();
  auto random_resize = std::make_shared<vision::RandomResize>(std::vector<int32_t>{100});

  // Create a Map operation on ds
  ds = ds->Map({duplicate}, {"image"}, {"image", "image_copy"});
  EXPECT_NE(ds, nullptr);

  ds = ds->Map({random_resize}, {"image", "image_copy"}, {"image", "image_copy"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto image_copy = row["image_copy"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor image_copy shape: " << image_copy.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomPosterize op
/// Description: Test RandomPosterize op with non-default parameters
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomPosterizeSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomPosterizeSuccess1 with non-default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto posterize = std::make_shared<vision::RandomPosterize>(std::vector<uint8_t>{1, 4});

  // Create a Map operation on ds
  ds = ds->Map({posterize});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomPosterize op
/// Description: Test RandomPosterize op with default parameters
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomPosterizeSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomPosterizeSuccess2 with default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto posterize = std::make_shared<vision::RandomPosterize>();

  // Create a Map operation on ds
  ds = ds->Map({posterize});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomResize op
/// Description: Test RandomResize op with single integer input
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomResizeSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeSuccess1 with single integer input.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_resize = std::make_shared<vision::RandomResize>(std::vector<int32_t>{66});

  // Create a Map operation on ds
  ds = ds->Map({random_resize}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0] == 66, true);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomResize op
/// Description: Test RandomResize op with (height, width) input
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomResizeSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeSuccess2 with (height, width) input.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_resize = std::make_shared<vision::RandomResize>(std::vector<int32_t>{66, 77});

  // Create a Map operation on ds
  ds = ds->Map({random_resize}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0] == 66 && image.Shape()[1] == 77, true);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomResizeWithBBox op
/// Description: Test RandomResizeWithBBox op with single integer input
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomResizeWithBBoxSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeWithBBoxSuccess1 with single integer input.";
  // setting seed here to prevent random core dump
  uint32_t current_seed = config::get_seed();
  config::set_seed(327362);

  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_resize = std::make_shared<vision::RandomResizeWithBBox>(std::vector<int32_t>{88});

  // Create a Map operation on ds
  ds = ds->Map({random_resize}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0] == 88, true);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
  config::set_seed(current_seed);
}

/// Feature: RandomResizeWithBBox op
/// Description: Test RandomResizeWithBBox op with (height, width) input
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomResizeWithBBoxSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeWithBBoxSuccess2 with (height, width) input.";
  uint32_t current_seed = config::get_seed();
  config::set_seed(327362);
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 4));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_resize = std::make_shared<vision::RandomResizeWithBBox>(std::vector<int32_t>{88, 99});

  // Create a Map operation on ds
  ds = ds->Map({random_resize}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0] == 88 && image.Shape()[1] == 99, true);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
  config::set_seed(current_seed);
}

/// Feature: RandomResizedCrop op
/// Description: Test RandomResizedCrop op with default values
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomResizedCropSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropSuccess1.";
  // Testing RandomResizedCrop with default values
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_resized_crop = std::make_shared<vision::RandomResizedCrop>(std::vector<int32_t>{5});

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0] == 5 && image.Shape()[1] == 5, true);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomResizedCrop op
/// Description: Test RandomResizedCrop op with non-default values
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomResizedCropSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropSuccess2.";
  // Testing RandomResizedCrop with non-default values
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_resized_crop = std::make_shared<vision::RandomResizedCrop>(
    std::vector<int32_t>{5, 10}, std::vector<float>{0.25, 0.75}, std::vector<float>{0.5, 1.25}, 
    mindspore::dataset::InterpolationMode::kArea, 20);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0] == 5 && image.Shape()[1] == 10, true);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomResizedCrop op
/// Description: Test RandomResizedCrop op with negative size
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomResizedCropFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropFail1 with negative size.";
  // This should fail because size has negative value
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_resized_crop = std::make_shared<vision::RandomResizedCrop>(std::vector<int32_t>{5, -10});

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomResizedCrop input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomResizedCrop op
/// Description: Test RandomResizedCrop op with invalid scale input
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomResizedCropFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropFail1 with invalid scale input.";
  // This should fail because scale isn't in {min, max} format
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_resized_crop = std::make_shared<vision::RandomResizedCrop>(
    std::vector<int32_t>{5, 10}, std::vector<float>{4, 3});

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomResizedCrop input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomResizedCrop op
/// Description: Test RandomResizedCrop op with invalid ratio input
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomResizedCropFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropFail1 with invalid ratio input.";
  // This should fail because ratio isn't in {min, max} format
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_resized_crop = std::make_shared<vision::RandomResizedCrop>(
    std::vector<int32_t>{5, 10}, std::vector<float>{4, 5}, std::vector<float>{7, 6});

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomResizedCrop input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomResizedCrop op
/// Description: Test RandomResizedCrop op with invalid scale size
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomResizedCropFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropFail1 with invalid scale size.";
  // This should fail because scale has a size of more than 2
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_resized_crop = std::make_shared<vision::RandomResizedCrop>(
    std::vector<int32_t>{5, 10, 20}, std::vector<float>{4, 5}, std::vector<float>{7, 6});

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomResizedCrop input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomResizedCropWithBBox op
/// Description: Test RandomResizedCropWithBBox op with default values
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropWithBBoxSuccess1.";
  // Testing RandomResizedCropWithBBox with default values
  // Create an VOC Dataset
  uint32_t current_seed = config::get_seed();
  config::set_seed(327362);
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 4));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_resized_crop = std::make_shared<vision::RandomResizedCropWithBBox>(std::vector<int32_t>{5});

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0] == 5 && image.Shape()[1] == 5, true);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 4);

  config::set_seed(current_seed);
  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomResizedCropWithBBox op
/// Description: Test RandomResizedCropWithBBox op with non-default values
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropWithBBoxSuccess2.";
  // Testing RandomResizedCropWithBBox with non-default values
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  uint32_t current_seed = config::get_seed();
  config::set_seed(327362);
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 4));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_resized_crop = std::make_shared<vision::RandomResizedCropWithBBox>(
    std::vector<int32_t>{5, 10}, std::vector<float>{0.25, 0.75}, std::vector<float>{0.5, 1.25}, 
    mindspore::dataset::InterpolationMode::kArea, 20);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0] == 5 && image.Shape()[1] == 10, true);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 4);
  config::set_seed(current_seed);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomResizedCropWithBBox op
/// Description: Test RandomResizedCropWithBBox op with negative size value
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropWithBBoxFail1 with negative size value.";
  // This should fail because size has negative value
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_resized_crop = std::make_shared<vision::RandomResizedCropWithBBox>(std::vector<int32_t>{5, -10});
  auto ds1 = ds->Map({random_resized_crop});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid input for RandomResizedCropWithBBox
  EXPECT_EQ(iter1, nullptr);
}

/// Feature: RandomResizedCropWithBBox op
/// Description: Test RandomResizedCropWithBBox op with invalid scale input
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropWithBBoxFail2 with invalid scale input.";
  // This should fail because scale isn't in {min, max} format
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_resized_crop = std::make_shared<vision::RandomResizedCropWithBBox>(
    std::vector<int32_t>{5, 10}, std::vector<float>{4, 3});
  auto ds1 = ds->Map({random_resized_crop});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid input for RandomResizedCropWithBBox
  EXPECT_EQ(iter1, nullptr);
}

/// Feature: RandomResizedCropWithBBox op
/// Description: Test RandomResizedCropWithBBox op with invalid ratio input
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropWithBBoxFail3 with invalid ratio input.";
  // This should fail because ratio isn't in {min, max} format
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_resized_crop = std::make_shared<vision::RandomResizedCropWithBBox>(
    std::vector<int32_t>{5, 10}, std::vector<float>{4, 5}, std::vector<float>{7, 6});
  auto ds1 = ds->Map({random_resized_crop});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid input for RandomResizedCropWithBBox
  EXPECT_EQ(iter1, nullptr);
}

/// Feature: RandomResizedCropWithBBox op
/// Description: Test RandomResizedCropWithBBox op with invalid scale size
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropWithBBoxFail4 with invalid scale size.";
  // This should fail because scale has a size of more than 2
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto random_resized_crop = std::make_shared<vision::RandomResizedCropWithBBox>(
    std::vector<int32_t>{5, 10, 20}, std::vector<float>{4, 5}, std::vector<float>{7, 6});
  auto ds1 = ds->Map({random_resized_crop});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid input for RandomResizedCropWithBBox
  EXPECT_EQ(iter1, nullptr);
}

/// Feature: RandomRotation op
/// Description: Test RandomRotation op with various size of degree and size of fill_value inputs
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomRotation) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomRotation.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Testing the size of degrees is 1
  auto random_rotation_op = std::make_shared<vision::RandomRotation>(std::vector<float>{180});
  // Testing the size of degrees is 2
  auto random_rotation_op1 = std::make_shared<vision::RandomRotation>(std::vector<float>{-180, 180});
  // Testing the size of fill_value is 1
  auto random_rotation_op2 = std::make_shared<vision::RandomRotation>(
    std::vector<float>{180}, InterpolationMode::kNearestNeighbour, false, std::vector<float>{-1, -1}, 
    std::vector<uint8_t>{2});
  // Testing the size of fill_value is 3
  auto random_rotation_op3 = std::make_shared<vision::RandomRotation>(
    std::vector<float>{180}, InterpolationMode::kNearestNeighbour, false, std::vector<float>{-1, -1}, 
    std::vector<uint8_t>{2, 2, 2});

  // Create a Map operation on ds
  ds = ds->Map({random_rotation_op, random_rotation_op1, random_rotation_op2, random_rotation_op3});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomRotation op
/// Description: Test RandomRotation op with invalid parameters
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomRotationFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomRotationFail with invalid parameters.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Case 1: Testing the size of degrees vector is 0
  // Create objects for the tensor ops
  auto random_rotation_op1 = std::make_shared<vision::RandomRotation>(std::vector<float>{});
  auto ds1 = ds->Map({random_rotation_op1});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid input for RandomRotation
  EXPECT_EQ(iter1, nullptr);

  // Case 2: Testing the size of degrees vector is 3
  // Create objects for the tensor ops
  auto random_rotation_op2 = std::make_shared<vision::RandomRotation>(std::vector<float>{-50.0, 50.0, 100.0});
  auto ds2 = ds->Map({random_rotation_op2});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid input for RandomRotation
  EXPECT_EQ(iter2, nullptr);

  // Case 3: Test the case where the first column value of degrees is greater than the second column value
  // Create objects for the tensor ops
  auto random_rotation_op3 = std::make_shared<vision::RandomRotation>(std::vector<float>{50.0, -50.0});
  auto ds3 = ds->Map({random_rotation_op3});
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid input for RandomRotation
  EXPECT_EQ(iter3, nullptr);

  // Case 4: Testing the size of center vector is 1
  // Create objects for the tensor ops
  auto random_rotation_op4 = std::make_shared<vision::RandomRotation>(
    std::vector<float>{-50.0, 50.0}, mindspore::dataset::InterpolationMode::kNearestNeighbour, false, 
    std::vector<float>{-1.0});
  auto ds4 = ds->Map({random_rotation_op4});
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid input for RandomRotation
  EXPECT_EQ(iter4, nullptr);

  // Case 5: Testing the size of center vector is 3
  // Create objects for the tensor ops
  auto random_rotation_op5 = std::make_shared<vision::RandomRotation>(
    std::vector<float>{-50.0, 50.0}, mindspore::dataset::InterpolationMode::kNearestNeighbour, false, 
    std::vector<float>{-1.0, -1.0, -1.0});
  auto ds5 = ds->Map({random_rotation_op5});
  EXPECT_NE(ds5, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter5 = ds5->CreateIterator();
  // Expect failure: invalid input for RandomRotation
  EXPECT_EQ(iter5, nullptr);

  // Case 6: Testing the size of fill_value vector is 2
  // Create objects for the tensor ops
  auto random_rotation_op6 = std::make_shared<vision::RandomRotation>(
    std::vector<float>{-50.0, 50.0}, mindspore::dataset::InterpolationMode::kNearestNeighbour, false, 
    std::vector<float>{-1.0, -1.0}, std::vector<uint8_t>{2, 2});
  auto ds6 = ds->Map({random_rotation_op6});
  EXPECT_NE(ds6, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter6 = ds6->CreateIterator();
  // Expect failure: invalid input for RandomRotation
  EXPECT_EQ(iter6, nullptr);

  // Case 7: Testing the size of fill_value vector is 4
  // Create objects for the tensor ops
  auto random_rotation_op7 = std::make_shared<vision::RandomRotation>(
    std::vector<float>{-50.0, 50.0}, mindspore::dataset::InterpolationMode::kNearestNeighbour, false, 
    std::vector<float>{-1.0, -1.0}, std::vector<uint8_t>{2, 2, 2, 2});
  auto ds7 = ds->Map({random_rotation_op7});
  EXPECT_NE(ds7, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter7 = ds7->CreateIterator();
  // Expect failure: invalid input for RandomRotation
  EXPECT_EQ(iter7, nullptr);
}

/// Feature: RandomSharpness op
/// Description: Test RandomSharpness op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomSharpness) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSharpness.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Valid case: Input start degree and end degree
  auto random_sharpness_op_1 = std::make_shared<vision::RandomSharpness>(std::vector<float>{0.4, 2.3});

  // Valid case: Use default input values
  auto random_sharpness_op_2 = std::make_shared<vision::RandomSharpness>();

  // Create a Map operation on ds
  ds = ds->Map({random_sharpness_op_1, random_sharpness_op_2});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomSolarize op
/// Description: Test RandomSolarize op with non-default parameters
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomSolarizeSucess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSolarizeSucess1.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::vector<uint8_t> threshold = {10, 100};
  std::shared_ptr<TensorTransform> random_solarize =
    std::make_shared<mindspore::dataset::vision::RandomSolarize>(threshold);

  // Create a Map operation on ds
  ds = ds->Map({random_solarize});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

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

/// Feature: RandomSolarize op
/// Description: Test RandomSolarize op with default parameters
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomSolarizeSucess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSolarizeSuccess2 with default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_solarize = std::make_shared<mindspore::dataset::vision::RandomSolarize>();

  // Create a Map operation on ds
  ds = ds->Map({random_solarize});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

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

/// Feature: RandomVerticalFlipWithBBox op
/// Description: Test RandomVerticalFlipWithBBox op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomVerticalFlipWithBBoxSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomVerticalFlipWithBBoxSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlipWithBBox>(0.4);

  // Create a Map operation on ds
  ds = ds->Map({random_vertical_flip_op}, {"image", "bbox"}, {"image", "bbox"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);
  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomHorizontalFlip and RandomVerticalFlip ops
/// Description: Test RandomVerticalFlip op and RandomHorizontalFlip op with multiple fields
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomHorizontalAndVerticalFlipWithMultiField) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomHorizontalAndVerticalFlipWithMultiField for horizontal and "
                  "vertical flips.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  transforms::Duplicate duplicate = transforms::Duplicate();
  std::shared_ptr<TensorTransform> random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlip>(1);
  std::shared_ptr<TensorTransform> random_horizontal_flip_op = std::make_shared<vision::RandomHorizontalFlip>(1);

  // Create a Map operation on ds
  ds = ds->Map({duplicate}, {"image"}, {"image", "image_copy"});
  EXPECT_NE(ds, nullptr);

  ds = ds->Map({random_vertical_flip_op}, {"image", "image_copy"}, {"image", "image_copy"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto image_copy = row["image_copy"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor image_copy shape: " << image_copy.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomCropDecodeResize op
/// Description: Test RandomCropDecodeResize op with multiple fields
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomCropDecodeResizeWithMultiField) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropDecodeResizeWithMultiField.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  transforms::Duplicate duplicate = transforms::Duplicate();
  auto random_crop_decode_resize = std::make_shared<vision::RandomCropDecodeResize>(std::vector<int32_t>{500, 500});

  // Create a Map operation on ds
   ds = ds->Map({duplicate}, {"image"}, {"image", "image_copy"});
   EXPECT_NE(ds, nullptr);

   ds = ds->Map({random_crop_decode_resize}, {"image", "image_copy"}, {"image", "image_copy"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto image_copy = row["image_copy"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor image_copy shape: " << image_copy.Shape();
    
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomCropResize op
/// Description: Test RandomCropResize op with multiple fields
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomCropResizeWithMultiField) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropResizeWithMultiField.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  transforms::Duplicate duplicate = transforms::Duplicate();
  auto random_crop_decode_resize = std::make_shared<vision::RandomResizedCrop>(std::vector<int32_t>{500, 500});

  // Create a Map operation on ds
  ds = ds->Map({duplicate}, {"image"}, {"image", "image_copy"});
  EXPECT_NE(ds, nullptr);

  ds = ds->Map({random_crop_decode_resize}, {"image", "image_copy"}, {"image", "image_copy"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto image_copy = row["image_copy"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Tensor image_copy shape: " << image_copy.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}
