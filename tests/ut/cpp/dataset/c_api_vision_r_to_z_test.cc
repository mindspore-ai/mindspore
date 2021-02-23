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

// Tests for vision C++ API R to Z TensorTransform Operations (in alphabetical order)

TEST_F(MindDataTestPipeline, TestRescaleSucess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRescaleSucess1.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, SequentialSampler(0, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  iter->GetNextRow(&row);

  auto image = row["image"];

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> rescale(new mindspore::dataset::vision::Rescale(1.0, 0.0));
  EXPECT_NE(rescale, nullptr);

  // Convert to the same type
  std::shared_ptr<TensorTransform> type_cast(new transforms::TypeCast("uint8"));
  EXPECT_NE(type_cast, nullptr);

  ds = ds->Map({rescale, type_cast}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter1 = ds->CreateIterator();
  EXPECT_NE(iter1, nullptr);

  // Iterate the dataset and get each row1
  std::unordered_map<std::string, mindspore::MSTensor> row1;
  iter1->GetNextRow(&row1);

  auto image1 = row1["image"];

  // EXPECT_EQ(*image, *image1);

  // Manually terminate the pipeline
  iter1->Stop();
}

TEST_F(MindDataTestPipeline, TestRescaleSucess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRescaleSucess2 with different params.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> rescale(new mindspore::dataset::vision::Rescale(1.0 / 255, 1.0));
  EXPECT_NE(rescale, nullptr);

  ds = ds->Map({rescale}, {"image"});
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

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRescaleFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRescaleFail with invalid params.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // incorrect negative rescale parameter
  std::shared_ptr<TensorTransform> rescale(new mindspore::dataset::vision::Rescale(-1.0, 0.0));
  EXPECT_NE(rescale, nullptr);
}

TEST_F(MindDataTestPipeline, TestResize1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResize1 with single integer input.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 6));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 4;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create resize object with single integer input
  std::shared_ptr<TensorTransform> resize_op(new vision::Resize({30}));
  EXPECT_NE(resize_op, nullptr);

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    // auto image = row["image"];
    // MS_LOG(INFO) << "Tensor image shape: " << image->shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 24);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestResizeFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResize with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // negative resize value
  std::shared_ptr<TensorTransform> resize_op1(new mindspore::dataset::vision::Resize({30, -30}));
  EXPECT_NE(resize_op1, nullptr);
  // zero resize value
  std::shared_ptr<TensorTransform> resize_op2(new mindspore::dataset::vision::Resize({0, 30}));
  EXPECT_NE(resize_op2, nullptr);
  // resize with 3 values
  std::shared_ptr<TensorTransform> resize_op3(new mindspore::dataset::vision::Resize({30, 20, 10}));
  EXPECT_NE(resize_op3, nullptr);
}

TEST_F(MindDataTestPipeline, TestResizeWithBBoxSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResizeWithBBoxSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> resize_with_bbox_op(new vision::ResizeWithBBox({30}));
  EXPECT_NE(resize_with_bbox_op, nullptr);

  std::shared_ptr<TensorTransform> resize_with_bbox_op1(new vision::ResizeWithBBox({30, 30}));
  EXPECT_NE(resize_with_bbox_op1, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({resize_with_bbox_op, resize_with_bbox_op1}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
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
}

TEST_F(MindDataTestPipeline, TestResizeWithBBoxFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResizeWithBBoxFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Testing negative resize value
  std::shared_ptr<TensorTransform> resize_with_bbox_op(new vision::ResizeWithBBox({10, -10}));
  EXPECT_NE(resize_with_bbox_op, nullptr);
  // Testing negative resize value
  std::shared_ptr<TensorTransform> resize_with_bbox_op1(new vision::ResizeWithBBox({-10}));
  EXPECT_NE(resize_with_bbox_op1, nullptr);
  // Testinig zero resize value
  std::shared_ptr<TensorTransform> resize_with_bbox_op2(new vision::ResizeWithBBox({0, 10}));
  EXPECT_NE(resize_with_bbox_op2, nullptr);
  // Testing resize with 3 values
  std::shared_ptr<TensorTransform> resize_with_bbox_op3(new vision::ResizeWithBBox({10, 10, 10}));
  EXPECT_NE(resize_with_bbox_op3, nullptr);
}

TEST_F(MindDataTestPipeline, TestVisionOperationName) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVisionOperationName.";

  std::string correct_name;

  // Create object for the tensor op, and check the name
  /* FIXME - Update and move test to IR level
  std::shared_ptr<TensorOperation> random_vertical_flip_op = vision::RandomVerticalFlip(0.5);
  correct_name = "RandomVerticalFlip";
  EXPECT_EQ(correct_name, random_vertical_flip_op->Name());

  // Create object for the tensor op, and check the name
  std::shared_ptr<TensorOperation> softDvpp_decode_resize_jpeg_op = vision::SoftDvppDecodeResizeJpeg({1, 1});
  correct_name = "SoftDvppDecodeResizeJpeg";
  EXPECT_EQ(correct_name, softDvpp_decode_resize_jpeg_op->Name());
  */
}
