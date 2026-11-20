/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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
#include <opencv2/opencv.hpp>

#include "common/common.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/include/dataset/vision.h"

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

// Tests for vision C++ API R to Z TensorTransform Operations (in alphabetical order)

/// Feature: Rescale op
/// Description: Test Rescale op with 1.0 rescale factor and 0.0 shift factor
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRescaleSucess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRescaleSucess1.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<SequentialSampler>(0, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  auto image = row["image"];

  // Create objects for the tensor ops
  auto rescale = std::make_shared<mindspore::dataset::vision::Rescale>(1.0, 0.0);
  // Note: No need to check for output after calling API class constructor

  // Convert to the same type
  auto type_cast = std::make_shared<transforms::TypeCast>(mindspore::DataType::kNumberTypeUInt8);
  // Note: No need to check for output after calling API class constructor

  ds = ds->Map({rescale, type_cast}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter1 = ds->CreateIterator();
  EXPECT_NE(iter1, nullptr);

  // Iterate the dataset and get each row1
  std::unordered_map<std::string, mindspore::MSTensor> row1;
  ASSERT_OK(iter1->GetNextRow(&row1));

  auto image1 = row1["image"];

  EXPECT_MSTENSOR_EQ(image, image1);

  // Manually terminate the pipeline
  iter1->Stop();
}

/// Feature: Rescale op
/// Description: Test Rescale op with 1.0 / 255 rescale factor and 1.0 shift factor
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRescaleSucess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRescaleSucess2 with different params.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto rescale = std::make_shared<mindspore::dataset::vision::Rescale>(1.0 / 255, 1.0);
  // Note: No need to check for output after calling API class constructor

  ds = ds->Map({rescale}, {"image"});
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

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Resize op
/// Description: Test Resize op with single integer input
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestResize1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResize1 with single integer input.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 6));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 4;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create resize object with single integer input
  auto resize_op = std::make_shared<vision::Resize>(std::vector<int32_t>{30});
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({resize_op});
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

  EXPECT_EQ(i, 24);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RGB2GRAY op
/// Description: Test RGB2GRAY op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRGB2GRAYSucess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRGB2GRAYSucess.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<SequentialSampler>(0, 1));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto convert = std::make_shared<mindspore::dataset::vision::RGB2GRAY>();

  ds = ds->Map({convert});
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

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RGB2BGR op
/// Description: Test RGB2BGR op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRGB2BGR) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRGB2BGR.";
  // create two imagenet dataset
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds1, nullptr);
  std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds2, nullptr);

  auto rgb2bgr_op = vision::RGB2BGR();

  ds1 = ds1->Map({rgb2bgr_op});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  EXPECT_NE(iter1, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row1;
  iter1->GetNextRow(&row1);

  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  EXPECT_NE(iter2, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row2;
  iter2->GetNextRow(&row2);

  uint64_t i = 0;
  while (row1.size() != 0) {
    i++;
    auto image = row1["image"];
    iter1->GetNextRow(&row1);
    iter2->GetNextRow(&row2);
  }
  EXPECT_EQ(i, 2);

  iter1->Stop();
  iter2->Stop();
}

/// Feature: RandomCrop
/// Description: Use batched dataset as video inputs
/// Expectation: The log will print correct shape
TEST_F(MindDataTestPipeline, TestRandomCropHighDimensions) {
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds, choose batch size 5 to test high dimension input
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create random crop object with square crop
  const std::vector<int32_t> crop_size{30};
  std::shared_ptr<TensorTransform> centre_out1 = std::make_shared<vision::RandomCrop>(crop_size);
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({centre_out1});
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

/// Feature: RandomEqualize op
/// Description: Test RandomEqualize op basic usage
/// Expectation: Output is equal to the expected output

/// Feature: RandomEqualize op
/// Description: Test RandomEqualize op with invalid prob
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr

/// Feature: RandomInvert op
/// Description: Test RandomInvert op basic usage
/// Expectation: Output is equal to the expected output

/// Feature: RandomInvert op
/// Description: Test RandomInvert op with invalid prob
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr

/// Feature: RandomAutoContrast op
/// Description: Test RandomAutoContrast op basic usage
/// Expectation: Output is equal to the expected output

/// Feature: RandomAutoContrast op
/// Description: Test RandomAutoContrast op with invalid prob
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr

/// Feature: RandomAutoContrast op
/// Description: Test RandomAutoContrast op with invalid cutoff
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr

/// Feature: RandomAutoContrast op
/// Description: Test RandomAutoContrast op with invalid ignore
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr

/// Feature: RandomAdjustSharpness op
/// Description: Test RandomAdjustSharpness op basic usage
/// Expectation: Output is equal to the expected output

/// Feature: RandomAdjustSharpness op
/// Description: Test RandomAdjustSharpness op with invalid prob
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr

/// Feature: RandomAdjustSharpness op
/// Description: Test RandomAdjustSharpness op with invalid degree
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr

/// Feature: ToTensor op
/// Description: Test ToTensor op with default float32 type
/// Expectation: Tensor type is changed to float32 and all rows iterated correctly
TEST_F(MindDataTestPipeline, TestToTensorOpDefault) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToTensorOpDefault.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto to_tensor_op = vision::ToTensor();
  ds = ds->Map({to_tensor_op}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image type: " << image.DataType();
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);
  iter->Stop();
}

/// Feature: ToTensor op
/// Description: Test ToTensor op with float64 type passed as string
/// Expectation: Tensor type is changed to float64 and all rows iterated correctly
TEST_F(MindDataTestPipeline, TestToTensorOpFloat64String) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToTensorOpFloat64String.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto to_tensor_op = vision::ToTensor("float64");
  ds = ds->Map({to_tensor_op}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image type: " << image.DataType();
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);
  iter->Stop();
}

/// Feature: ToTensor op
/// Description: Test ToTensor op with float64 type passed as DataType
/// Expectation: Tensor type is changed to float64 and all rows iterated correctly
TEST_F(MindDataTestPipeline, TestToTensorOpFloat64DataType) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToTensorOpFloat64DataType.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto to_tensor_op = vision::ToTensor(mindspore::DataType::kNumberTypeFloat64);
  ds = ds->Map({to_tensor_op}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image type: " << image.DataType();
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);
  iter->Stop();
}

/// Feature: ToTensor op
/// Description: Test ToTensor op with float64 type and invalid uint32 type input data
/// Expectation: Error is caught as given invalid input data type
TEST_F(MindDataTestPipeline, TestToTensorOpInvalidInput) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestToTensorOpInvalidInput.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto type_cast = transforms::TypeCast(mindspore::DataType::kNumberTypeUInt32);
  auto to_tensor_op = vision::ToTensor("float64");
  ds = ds->Map({type_cast, to_tensor_op}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_ERROR(iter->GetNextRow(&row));
}
