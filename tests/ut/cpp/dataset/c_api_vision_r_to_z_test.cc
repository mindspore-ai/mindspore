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
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/include/dataset/vision.h"

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

// Tests for vision C++ API R to Z TensorTransform Operations (in alphabetical order)

/// Feature: RandomLighting op
/// Description: Test RandomLighting Op on pipeline when alpha=0.1
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestRandomLightingPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomLightingPipeline.";
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
  auto randomlighting = std::make_shared<mindspore::dataset::vision::RandomLighting>(0.1);
  // Note: No need to check for output after calling API class constructor

  // Convert to the same type
  auto type_cast = std::make_shared<transforms::TypeCast>(mindspore::DataType::kNumberTypeUInt8);
  // Note: No need to check for output after calling API class constructor

  ds = ds->Map({randomlighting, type_cast}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter1 = ds->CreateIterator();
  EXPECT_NE(iter1, nullptr);

  // Iterate the dataset and get each row1
  std::unordered_map<std::string, mindspore::MSTensor> row1;
  ASSERT_OK(iter1->GetNextRow(&row1));

  auto image1 = row1["image"];

  // Manually terminate the pipeline
  iter1->Stop();
}

/// Feature: RandomLighting op
/// Description: Test param check for RandomLighting Op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline
///     returns nullptr when params are invalid
TEST_F(MindDataTestPipeline, TestRandomLightingParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomLightingParamCheck.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Case 1: negative alpha
  // Create objects for the tensor ops
  auto random_lighting_op = std::make_shared<mindspore::dataset::vision::RandomLighting>(-0.1);
  auto ds2 = ds->Map({random_lighting_op});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid value of alpha
  EXPECT_EQ(iter2, nullptr);
}

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

/// Feature: ResizeWithBBox op
/// Description: Test ResizeWithBBox op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestResizeWithBBoxSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResizeWithBBoxSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto resize_with_bbox_op = std::make_shared<vision::ResizeWithBBox>(std::vector<int32_t>{30});
  auto resize_with_bbox_op1 = std::make_shared<vision::ResizeWithBBox>(std::vector<int32_t>{30, 30});
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({resize_with_bbox_op, resize_with_bbox_op1}, {"image", "bbox"}, {"image", "bbox"});
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

/// Feature: Rotate op
/// Description: Test Rotate op with invalid parameters
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRotateParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRotateParamCheck with invalid parameters.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Case 1: Size of center is not 2
  // Create objects for the tensor ops
  auto rotate1 =
    std::make_shared<vision::Rotate>(90.0, InterpolationMode::kNearestNeighbour, false, std::vector<float>{0.});
  auto ds2 = ds->Map({rotate1});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid center for Rotate
  EXPECT_EQ(iter2, nullptr);

  // Case 2: Size of fill_value is not 1 or 3
  // Create objects for the tensor ops
  auto rotate2 = std::make_shared<vision::Rotate>(-30, InterpolationMode::kNearestNeighbour, false,
                                                  std::vector<float>{1.0, 1.0}, std::vector<uint8_t>{2, 2});
  auto ds3 = ds->Map({rotate2});
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid fill_value for Rotate
  EXPECT_EQ(iter3, nullptr);
}

/// Feature: Rotate op
/// Description: Test Rotate op by passing it to a Map op
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRotatePass) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRotatePass.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto resize = std::make_shared<vision::Resize>(std::vector<int32_t>{50, 25});

  auto rotate = std::make_shared<vision::Rotate>(90, InterpolationMode::kLinear, true, std::vector<float>{-1, -1},
                                                 std::vector<uint8_t>{255, 255, 255});

  // Resize the image to 50 * 25
  ds = ds->Map({resize});
  EXPECT_NE(ds, nullptr);

  // Rotate the image 90 degrees
  ds = ds->Map({rotate});
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
    // After rotation with expanding, the image size comes to 25 * 50
    EXPECT_EQ(image.Shape()[1], 25);
    EXPECT_EQ(image.Shape()[2], 50);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

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

/// Feature: RandomEqualize op
/// Description: Test RandomEqualize op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomEqualize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomEqualize.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_equalize_op = vision::RandomEqualize(0.5);

  ds = ds->Map({random_equalize_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);

  iter->Stop();
}

/// Feature: RandomEqualize op
/// Description: Test RandomEqualize op with invalid prob
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomEqualizeInvalidProb) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomEqualizeInvalidProb.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_equalize_op = vision::RandomEqualize(1.5);

  ds = ds->Map({random_equalize_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomInvert op
/// Description: Test RandomInvert op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomInvert) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomInvert.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_invert_op = vision::RandomInvert(0.5);

  ds = ds->Map({random_invert_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);

  iter->Stop();
}

/// Feature: RandomInvert op
/// Description: Test RandomInvert op with invalid prob
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomInvertInvalidProb) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomInvertInvalidProb.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_invert_op = vision::RandomInvert(1.5);

  ds = ds->Map({random_invert_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomAutoContrast op
/// Description: Test RandomAutoContrast op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomAutoContrast) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAutoContrast.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_auto_contrast_op = vision::RandomAutoContrast(1.0, {0, 255}, 0.5);

  ds = ds->Map({random_auto_contrast_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);

  iter->Stop();
}

/// Feature: RandomAutoContrast op
/// Description: Test RandomAutoContrast op with invalid prob
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomAutoContrastInvalidProb) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAutoContrastInvalidProb.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_auto_contrast_op = vision::RandomAutoContrast(0.0, {}, 1.5);

  ds = ds->Map({random_auto_contrast_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomAutoContrast op
/// Description: Test RandomAutoContrast op with invalid cutoff
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomAutoContrastInvalidCutoff) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAutoContrastInvalidCutoff.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_auto_contrast_op = vision::RandomAutoContrast(-2.0, {}, 0.5);

  ds = ds->Map({random_auto_contrast_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomAutoContrast op
/// Description: Test RandomAutoContrast op with invalid ignore
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomAutoContrastInvalidIgnore) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAutoContrastInvalidCutoff.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_auto_contrast_op = vision::RandomAutoContrast(1.0, {10, 256}, 0.5);

  ds = ds->Map({random_auto_contrast_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomAdjustSharpness op
/// Description: Test RandomAdjustSharpness op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomAdjustSharpness) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAdjustSharpness.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));

  EXPECT_NE(ds, nullptr);

  auto random_adjust_sharpness_op = vision::RandomAdjustSharpness(2.0, 0.5);

  ds = ds->Map({random_adjust_sharpness_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);

  iter->Stop();
}

/// Feature: RandomAdjustSharpness op
/// Description: Test RandomAdjustSharpness op with invalid prob
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomAdjustSharpnessInvalidProb) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAdjustSharpnessInvalidProb.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_adjust_sharpness_op = vision::RandomAdjustSharpness(2.0, 1.5);

  ds = ds->Map({random_adjust_sharpness_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomAdjustSharpness op
/// Description: Test RandomAdjustSharpness op with invalid degree
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomAdjustSharpnessInvalidDegree) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAdjustSharpnessInvalidProb.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto random_adjust_sharpness_op = vision::RandomAdjustSharpness(-2.0, 0.3);

  ds = ds->Map({random_adjust_sharpness_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

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

/// Feature: ResizedCrop op
/// Description: Test ResizedCrop pipeline
/// Expectation: Input is processed as expected and all rows iterated correctly
TEST_F(MindDataTestPipeline, TestResizedCrop) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResizedCrop.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));

  EXPECT_NE(ds, nullptr);

  unsigned int left = 256;
  unsigned int top = 256;
  unsigned int height = 256;
  unsigned int width = 256;
  std::vector<int32_t> size{128, 128};
  auto resized_crop_op = vision::ResizedCrop(top, left, height, width, size);

  ds = ds->Map({resized_crop_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);

  iter->Stop();
}

/// Feature: ResizedCrop op
/// Description: Test ResizedCrop with invalid fill_value
/// Expectation: Pipeline iteration failed with wrong argument fill_value
TEST_F(MindDataTestPipeline, TestResizedCropParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResizedCropParamCheck.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Case 1: Value of top/left/height/width out of boundary
  // Create objects for the tensor ops
  auto resized_crop_op1 = vision::ResizedCrop(-1, -1, -1, -1, {128, 128});
  auto ds1 = ds->Map({resized_crop_op1});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid coordinates for Crop
  EXPECT_EQ(iter1, nullptr);

  // Case 2: Value of size is negative
  // Create objects for the tensor ops
  auto resized_crop_op2 = vision::ResizedCrop(256, 256, 256, 256, {-128, -128});
  auto ds2 = ds->Map({resized_crop_op2});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid coordinates for Crop
  EXPECT_EQ(iter2, nullptr);

  // Case 3: Size is neither a single number nor a vector of size 2
  // Create objects for the tensor ops
  auto resized_crop_op3 = vision::ResizedCrop(256, 256, 256, 256, {128, 128, 128});
  auto ds3 = ds->Map({resized_crop_op3});
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid size for Crop
  EXPECT_EQ(iter3, nullptr);
}

/// Feature: RandAugment
/// Description: test RandAugment pipeline
/// Expectation: create an ImageFolder dataset then do rand augmentation on it
TEST_F(MindDataTestPipeline, TestRandAugment) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugment.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData2/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));

  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(3, 4, 5, InterpolationMode::kLinear, {0, 0, 0});

  ds = ds->Map({rand_augment_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 5);

  iter->Stop();
}

/// Feature: RandAugment
/// Description: test RandAugment with invalid fill_value
/// Expectation: pipeline iteration failed with wrong argument fill_value
TEST_F(MindDataTestPipeline, TestRandAugmentInvalidFillValue) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugmentInvalidFillValue.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(3, 4, 5, InterpolationMode::kNearestNeighbour, {20, 20});

  ds = ds->Map({rand_augment_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandAugment
/// Description: test RandAugment with invalid num_ops
/// Expectation: pipeline iteration failed with wrong argument num_ops
TEST_F(MindDataTestPipeline, TestRandAugmentInvalidNumOps) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugmentInvalidNumOps.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(-1, 4, 5);

  ds = ds->Map({rand_augment_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandAugment
/// Description: test RandAugment with num_ops equal to 0
/// Expectation: pipeline iteration success with num_ops equal to 0
TEST_F(MindDataTestPipeline, TestRandAugmentNumOpsZero) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugmentNumOpsZero.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData2/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));

  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(0, 4, 5, InterpolationMode::kLinear, {0, 0, 0});

  ds = ds->Map({rand_augment_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 2);

  iter->Stop();
}

/// Feature: RandAugment
/// Description: test RandAugment with invalid magnitude
/// Expectation: pipeline iteration failed with wrong argument magnitude
TEST_F(MindDataTestPipeline, TestRandAugmentInvalidMagnitude) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugmentInvalidMagnitude.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(2, -1, 2, InterpolationMode::kLinear, {0, 0, 0});

  ds = ds->Map({rand_augment_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandAugment
/// Description: test RandAugment with magnitude equal to 0
/// Expectation: pipeline iteration success with magnitude equal to 0
TEST_F(MindDataTestPipeline, TestRandAugmentMagnitudeZero) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugmentMagnitudeZero.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData2/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));

  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(3, 0, 2, InterpolationMode::kLinear, {0, 0, 0});

  ds = ds->Map({rand_augment_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 5);

  iter->Stop();
}

/// Feature: RandAugment
/// Description: test RandAugment with invalid num_magnitude_bins
/// Expectation: pipeline iteration failed with wrong argument num_magnitude_bins
TEST_F(MindDataTestPipeline, TestRandAugmentInvalidNumMagBins) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugmentInvalidNumMagBins.";

  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData2/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));

  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(3, 2, 1, InterpolationMode::kLinear, {0, 0, 0});

  ds = ds->Map({rand_augment_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandAugment
/// Description: test RandAugment with num_magnitude_bins equal to 2
/// Expectation: pipeline iteration success with num_magnitude_bins equal to 2
TEST_F(MindDataTestPipeline, TestRandAugmentNumMagBins) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugmentNumMagBins.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData2/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));

  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(3, 0, 2, InterpolationMode::kLinear, {0, 0, 0});

  ds = ds->Map({rand_augment_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();

  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    iter->GetNextRow(&row);
  }
  EXPECT_EQ(i, 5);

  iter->Stop();
}

/// Feature: RandAugment
/// Description: test RandAugment with magnitude greater than num_magnitude_bins
/// Expectation: pipeline iteration failed with invalid magnitude value
TEST_F(MindDataTestPipeline, TestRandAugmentMagGreNMBError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandAugmentMagGreNMBError.";
  std::string MindDataPath = "data/dataset";
  std::string folder_path = MindDataPath + "/testImageNetData2/train/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  auto rand_augment_op = vision::RandAugment(3, 5, 4);
  ds = ds->Map({rand_augment_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}
    
/// Feature: WriteFile
/// Description: Test WriteFile by writing the data into a file using binary mode
/// Expectation: The file should be writeen and removed successfully
TEST_F(MindDataTestPipeline, TestWriteFileNormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TesWriteFileNormal.";
  std::string folder_path = "./data/dataset/";
  std::string filename_1, filename_2;
  filename_1 = folder_path + "apple.jpg";
  filename_2 = filename_1 + ".test_write_file";

  std::shared_ptr<Tensor> de_tensor_1, de_tensor_2;
  Tensor::CreateFromFile(filename_1, &de_tensor_1);
  auto data_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor_1));

  ASSERT_OK(mindspore::dataset::vision::WriteFile(filename_2, data_tensor));
  Tensor::CreateFromFile(filename_2, &de_tensor_2);
  EXPECT_EQ(de_tensor_1->shape(), de_tensor_2->shape());
  remove(filename_2.c_str());
}

/// Feature: WriteFile
/// Description: Test WriFile with invalid parameter
/// Expectation: Error is caught when the parameter is invalid
TEST_F(MindDataTestPipeline, TestWriteFileException) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TesWriteFileException.";
  std::string folder_path = "./data/dataset/";
  std::string filename_1, filename_2;
  filename_1 = folder_path + "apple.jpg";
  filename_2 = filename_1 + ".test_write_file";

  std::shared_ptr<Tensor> de_tensor_1;
  Tensor::CreateFromFile(filename_1, &de_tensor_1);
  auto data_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor_1));

  // Test with a directory name
  ASSERT_ERROR(mindspore::dataset::vision::WriteFile(folder_path, data_tensor));

  // Test with an invalid filename
  ASSERT_ERROR(mindspore::dataset::vision::WriteFile("/dev/cdrom/0", data_tensor));

  // Test with invalid float elements
  std::shared_ptr<Tensor> input;
  std::vector<float> float_vector = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12};
  Tensor::CreateFromVector(float_vector, TensorShape({12}), &input);
  data_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  ASSERT_ERROR(mindspore::dataset::vision::WriteFile(filename_2, data_tensor));
}
