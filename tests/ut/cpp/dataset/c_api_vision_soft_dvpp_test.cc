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

// Tests for vision C++ API SoftDvpp* TensorTransform Operations (in alphabetical order)

TEST_F(MindDataTestPipeline, TestSoftDvppDecodeRandomCropResizeJpegSuccess1) {
  MS_LOG(INFO)
    << "Doing MindDataTestPipeline-TestSoftDvppDecodeRandomCropResizeJpegSuccess1 with single integer input.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(false, 4));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> soft_dvpp_decode_random_crop_resize_jpeg(
    new vision::SoftDvppDecodeRandomCropResizeJpeg({500}));
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({soft_dvpp_decode_random_crop_resize_jpeg}, {"image"});
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
    EXPECT_EQ(image.Shape()[0] == 500 && image.Shape()[1] == 500, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestSoftDvppDecodeRandomCropResizeJpegSuccess2) {
  MS_LOG(INFO)
    << "Doing MindDataTestPipeline-TestSoftDvppDecodeRandomCropResizeJpegSuccess2 with (height, width) input.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(false, 6));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> soft_dvpp_decode_random_crop_resize_jpeg(
    new vision::SoftDvppDecodeRandomCropResizeJpeg({500, 600}, {0.25, 0.75}, {0.5, 1.25}, 20));
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({soft_dvpp_decode_random_crop_resize_jpeg}, {"image"});
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
    EXPECT_EQ(image.Shape()[0] == 500 && image.Shape()[1] == 600, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestSoftDvppDecodeResizeJpegSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSoftDvppDecodeResizeJpegSuccess1 with single integer input.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(false, 4));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create SoftDvppDecodeResizeJpeg object with single integer input
  std::shared_ptr<TensorTransform> soft_dvpp_decode_resize_jpeg_op(new vision::SoftDvppDecodeResizeJpeg({1134}));
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({soft_dvpp_decode_resize_jpeg_op});
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

  EXPECT_EQ(i, 12);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestSoftDvppDecodeResizeJpegSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSoftDvppDecodeResizeJpegSuccess2 with (height, width) input.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create SoftDvppDecodeResizeJpeg object with single integer input
  std::shared_ptr<TensorTransform> soft_dvpp_decode_resize_jpeg_op(new vision::SoftDvppDecodeResizeJpeg({100, 200}));
  // Note: No need to check for output after calling API class constructor

  // Create a Map operation on ds
  ds = ds->Map({soft_dvpp_decode_resize_jpeg_op});
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

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}
