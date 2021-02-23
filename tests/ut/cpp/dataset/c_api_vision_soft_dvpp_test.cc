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
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, RandomSampler(false, 4));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> soft_dvpp_decode_random_crop_resize_jpeg(new
    vision::SoftDvppDecodeRandomCropResizeJpeg({500}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg, nullptr);

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
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    // EXPECT_EQ(image->shape()[0] == 500 && image->shape()[1] == 500, true);
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
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, RandomSampler(false, 6));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> soft_dvpp_decode_random_crop_resize_jpeg(new 
    vision::SoftDvppDecodeRandomCropResizeJpeg({500, 600}, {0.25, 0.75}, {0.5, 1.25}, 20));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg, nullptr);

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
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    // EXPECT_EQ(image->shape()[0] == 500 && image->shape()[1] == 600, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestSoftDvppDecodeRandomCropResizeJpegFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSoftDvppDecodeRandomCropResizeJpegFail with incorrect parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // SoftDvppDecodeRandomCropResizeJpeg: size must only contain positive integers
  auto soft_dvpp_decode_random_crop_resize_jpeg1(new vision::SoftDvppDecodeRandomCropResizeJpeg({-500, 600}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg1, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: size must only contain positive integers
  auto soft_dvpp_decode_random_crop_resize_jpeg2(new vision::SoftDvppDecodeRandomCropResizeJpeg({-500}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg2, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: size must be a vector of one or two values
  auto soft_dvpp_decode_random_crop_resize_jpeg3(new vision::SoftDvppDecodeRandomCropResizeJpeg({500, 600, 700}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg3, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: scale must be greater than or equal to 0
  auto soft_dvpp_decode_random_crop_resize_jpeg4(new vision::SoftDvppDecodeRandomCropResizeJpeg({500}, {-0.1, 0.9}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg4, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: scale must be in the format of (min, max)
  auto soft_dvpp_decode_random_crop_resize_jpeg5(new vision::SoftDvppDecodeRandomCropResizeJpeg({500}, {0.6, 0.2}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg5, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: scale must be a vector of two values
  auto soft_dvpp_decode_random_crop_resize_jpeg6(new vision::SoftDvppDecodeRandomCropResizeJpeg({500}, {0.5, 0.6, 0.7}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg6, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: ratio must be greater than or equal to 0
  auto soft_dvpp_decode_random_crop_resize_jpeg7(new vision::SoftDvppDecodeRandomCropResizeJpeg({500}, {0.5, 0.9}, {-0.2, 0.4}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg7, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: ratio must be in the format of (min, max)
  auto soft_dvpp_decode_random_crop_resize_jpeg8(new vision::SoftDvppDecodeRandomCropResizeJpeg({500}, {0.5, 0.9}, {0.4, 0.2}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg8, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: ratio must be a vector of two values
  auto soft_dvpp_decode_random_crop_resize_jpeg9(new vision::SoftDvppDecodeRandomCropResizeJpeg({500}, {0.5, 0.9}, {0.1, 0.2, 0.3}));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg9, nullptr);

  // SoftDvppDecodeRandomCropResizeJpeg: max_attempts must be greater than or equal to 1
  auto soft_dvpp_decode_random_crop_resize_jpeg10(new vision::SoftDvppDecodeRandomCropResizeJpeg({500}, {0.5, 0.9}, {0.1, 0.2}, 0));
  EXPECT_NE(soft_dvpp_decode_random_crop_resize_jpeg10, nullptr);
}

TEST_F(MindDataTestPipeline, TestSoftDvppDecodeResizeJpegSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSoftDvppDecodeResizeJpegSuccess1 with single integer input.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, RandomSampler(false, 4));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 3;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create SoftDvppDecodeResizeJpeg object with single integer input
  std::shared_ptr<TensorTransform> soft_dvpp_decode_resize_jpeg_op(new vision::SoftDvppDecodeResizeJpeg({1134}));
  EXPECT_NE(soft_dvpp_decode_resize_jpeg_op, nullptr);

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
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
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
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, RandomSampler(false, 2));
  EXPECT_NE(ds, nullptr);

  // Create SoftDvppDecodeResizeJpeg object with single integer input
  std::shared_ptr<TensorTransform> soft_dvpp_decode_resize_jpeg_op(new vision::SoftDvppDecodeResizeJpeg({100, 200}));
  EXPECT_NE(soft_dvpp_decode_resize_jpeg_op, nullptr);

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
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestSoftDvppDecodeResizeJpegFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSoftDvppDecodeResizeJpegFail with incorrect size.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // CSoftDvppDecodeResizeJpeg: size must be a vector of one or two values
  std::shared_ptr<TensorTransform> soft_dvpp_decode_resize_jpeg_op1(new vision::SoftDvppDecodeResizeJpeg({}));
  EXPECT_NE(soft_dvpp_decode_resize_jpeg_op1, nullptr);

  // SoftDvppDecodeResizeJpeg: size must be a vector of one or two values
  std::shared_ptr<TensorTransform> soft_dvpp_decode_resize_jpeg_op2(new vision::SoftDvppDecodeResizeJpeg({1, 2, 3}));
  EXPECT_NE(soft_dvpp_decode_resize_jpeg_op2, nullptr);

  // SoftDvppDecodeResizeJpeg: size must only contain positive integers
  std::shared_ptr<TensorTransform> soft_dvpp_decode_resize_jpeg_op3(new vision::SoftDvppDecodeResizeJpeg({20, -20}));
  EXPECT_NE(soft_dvpp_decode_resize_jpeg_op3, nullptr);

  // SoftDvppDecodeResizeJpeg: size must only contain positive integers
  std::shared_ptr<TensorTransform> soft_dvpp_decode_resize_jpeg_op4(new vision::SoftDvppDecodeResizeJpeg({0}));
  EXPECT_NE(soft_dvpp_decode_resize_jpeg_op4, nullptr);
}
