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
using mindspore::dataset::InterpolationMode;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

// Tests for vision C++ API Random* TensorTransform Operations (in alphabetical order)

TEST_F(MindDataTestPipeline, TestRandomAffineFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAffineFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> affine1(new vision::RandomAffine({0.0, 0.0}, {}));
  EXPECT_NE(affine1, nullptr);
  // Invalid number of values for translate
  std::shared_ptr<TensorTransform> affine2(new vision::RandomAffine({0.0, 0.0}, {1, 1, 1, 1, 1}));
  EXPECT_NE(affine2, nullptr);
  // Invalid number of values for shear
  std::shared_ptr<TensorTransform> affine3(new vision::RandomAffine({30.0, 30.0}, {0.0, 0.0}, {2.0, 2.0}, {10.0}));
  EXPECT_NE(affine3, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomAffineSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAffineSuccess1 with non-default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> affine(
    new vision::RandomAffine({30.0, 30.0}, {-1.0, 1.0, -1.0, 1.0}, {2.0, 2.0}, {10.0, 10.0, 20.0, 20.0}));
  EXPECT_NE(affine, nullptr);

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

TEST_F(MindDataTestPipeline, TestRandomAffineSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomAffineSuccess2 with default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> affine(new vision::RandomAffine({0.0, 0.0}));
  EXPECT_NE(affine, nullptr);

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

TEST_F(MindDataTestPipeline, TestRandomColor) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomColor with non-default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Valid case: Set lower bound and upper bound to be the same value zero
  std::shared_ptr<TensorTransform> random_color_op_1 = std::make_shared<vision::RandomColor>(0.0, 0.0);
  EXPECT_NE(random_color_op_1, nullptr);

  // Failure case: Set invalid lower bound greater than upper bound
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  std::shared_ptr<TensorTransform> random_color_op_2 = std::make_shared<vision::RandomColor>(1.0, 0.1);
  EXPECT_NE(random_color_op_2, nullptr);

  // Valid case: Set lower bound as zero and less than upper bound
  std::shared_ptr<TensorTransform> random_color_op_3 = std::make_shared<vision::RandomColor>(0.0, 1.1);
  EXPECT_NE(random_color_op_3, nullptr);

  // Failure case: Set invalid negative lower bound
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  std::shared_ptr<TensorTransform> random_color_op_4 = std::make_shared<vision::RandomColor>(-0.5, 0.5);
  EXPECT_NE(random_color_op_4, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_color_op_1, random_color_op_3});
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

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomColorAdjust) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomColorAdjust.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Use single value for vectors
  std::shared_ptr<TensorTransform> random_color_adjust1(new vision::RandomColorAdjust({1.0}, {0.0}, {0.5}, {0.5}));
  EXPECT_NE(random_color_adjust1, nullptr);

  // Use same 2 values for vectors
  std::shared_ptr<TensorTransform> random_color_adjust2(new
    vision::RandomColorAdjust({1.0, 1.0}, {0.0, 0.0}, {0.5, 0.5}, {0.5, 0.5}));
  EXPECT_NE(random_color_adjust2, nullptr);

  // Use different 2 value for vectors
  std::shared_ptr<TensorTransform> random_color_adjust3(new
    vision::RandomColorAdjust({0.5, 1.0}, {0.0, 0.5}, {0.25, 0.5}, {0.25, 0.5}));
  EXPECT_NE(random_color_adjust3, nullptr);

  // Use default input values
  std::shared_ptr<TensorTransform> random_color_adjust4(new vision::RandomColorAdjust());
  EXPECT_NE(random_color_adjust4, nullptr);

  // Use subset of explicitly set parameters
  std::shared_ptr<TensorTransform> random_color_adjust5(new vision::RandomColorAdjust({0.0, 0.5}, {0.25}));
  EXPECT_NE(random_color_adjust5, nullptr);

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

TEST_F(MindDataTestPipeline, TestRandomColorAdjustFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomColorAdjustFail.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // brightness out of range
  std::shared_ptr<TensorTransform> random_color_adjust1(new vision::RandomColorAdjust({-1.0}));
  EXPECT_NE(random_color_adjust1, nullptr);

  // contrast out of range
  std::shared_ptr<TensorTransform> random_color_adjust2(new vision::RandomColorAdjust({1.0}, {-0.1}));
  EXPECT_NE(random_color_adjust2, nullptr);

  // saturation out of range
  std::shared_ptr<TensorTransform> random_color_adjust3(new vision::RandomColorAdjust({0.0}, {0.0}, {-0.2}));
  EXPECT_NE(random_color_adjust3, nullptr);

  // hue out of range
  std::shared_ptr<TensorTransform> random_color_adjust4(new vision::RandomColorAdjust({0.0}, {0.0}, {0.0}, {-0.6}));
  EXPECT_NE(random_color_adjust4, nullptr);

  std::shared_ptr<TensorTransform> random_color_adjust5(new vision::RandomColorAdjust({0.0}, {0.0}, {0.0}, {-0.5, 0.6}));
  EXPECT_NE(random_color_adjust5, nullptr);

  std::shared_ptr<TensorTransform> random_color_adjust6(new vision::RandomColorAdjust({0.0}, {0.0}, {0.0}, {0.51}));
  EXPECT_NE(random_color_adjust6, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomCropSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Testing siez of size vector is 1
  std::shared_ptr<TensorTransform> random_crop(new vision::RandomCrop({20}));
  EXPECT_NE(random_crop, nullptr);

  // Testing siez of size vector is 2
  std::shared_ptr<TensorTransform> random_crop1(new vision::RandomCrop({20, 20}));
  EXPECT_NE(random_crop1, nullptr);

  // Testing siez of paddiing vector is 1
  std::shared_ptr<TensorTransform> random_crop2(new vision::RandomCrop({20, 20}, {10}));
  EXPECT_NE(random_crop2, nullptr);

  // Testing siez of paddiing vector is 2
  std::shared_ptr<TensorTransform> random_crop3(new vision::RandomCrop({20, 20}, {10, 20}));
  EXPECT_NE(random_crop3, nullptr);

  // Testing siez of paddiing vector is 2
  std::shared_ptr<TensorTransform> random_crop4(new vision::RandomCrop({20, 20}, {10, 10, 10, 10}));
  EXPECT_NE(random_crop4, nullptr);

  // Testing siez of fill_value vector is 1
  std::shared_ptr<TensorTransform> random_crop5(new vision::RandomCrop({20, 20}, {10, 10, 10, 10}, false, {5}));
  EXPECT_NE(random_crop5, nullptr);

  // Testing siez of fill_value vector is 3
  std::shared_ptr<TensorTransform> random_crop6(new vision::RandomCrop({20, 20}, {10, 10, 10, 10}, false, {4, 4, 4}));
  EXPECT_NE(random_crop6, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_crop, random_crop1, random_crop2, random_crop3, random_crop4, random_crop5, random_crop6});
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

  EXPECT_EQ(i, 10);
  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomCropFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Testing the size parameter is negative.
  std::shared_ptr<TensorTransform> random_crop(new vision::RandomCrop({-28, 28}));
  EXPECT_NE(random_crop, nullptr);
  // Testing the size parameter is None.
  std::shared_ptr<TensorTransform> random_crop1(new vision::RandomCrop({}));
  EXPECT_NE(random_crop1, nullptr);
  // Testing the size of size vector is 3.
  std::shared_ptr<TensorTransform> random_crop2(new vision::RandomCrop({28, 28, 28}));
  EXPECT_NE(random_crop2, nullptr);
  // Testing the padding parameter is negative.
  std::shared_ptr<TensorTransform> random_crop3(new vision::RandomCrop({28, 28}, {-5}));
  EXPECT_NE(random_crop3, nullptr);
  // Testing the size of padding vector is empty.
  std::shared_ptr<TensorTransform> random_crop4(new vision::RandomCrop({28, 28}, {}));
  EXPECT_NE(random_crop4, nullptr);
  // Testing the size of padding vector is 3.
  std::shared_ptr<TensorTransform> random_crop5(new vision::RandomCrop({28, 28}, {5, 5, 5}));
  EXPECT_NE(random_crop5, nullptr);
  // Testing the size of padding vector is 5.
  std::shared_ptr<TensorTransform> random_crop6(new vision::RandomCrop({28, 28}, {5, 5, 5, 5, 5}));
  EXPECT_NE(random_crop6, nullptr);
  // Testing the size of fill_value vector is empty.
  std::shared_ptr<TensorTransform> random_crop7(new vision::RandomCrop({28, 28}, {0, 0, 0, 0}, false, {}));
  EXPECT_NE(random_crop7, nullptr);
  // Testing the size of fill_value vector is 2.
  std::shared_ptr<TensorTransform> random_crop8(new vision::RandomCrop({28, 28}, {0, 0, 0, 0}, false, {0, 0}));
  EXPECT_NE(random_crop8, nullptr);
  // Testing the size of fill_value vector is 4.
  std::shared_ptr<TensorTransform> random_crop9(new vision::RandomCrop({28, 28}, {0, 0, 0, 0}, false, {0, 0, 0, 0}));
  EXPECT_NE(random_crop9, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomCropWithBboxSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropWithBboxSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop(new mindspore::dataset::vision::RandomCropWithBBox({128, 128}));
  EXPECT_NE(random_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_crop}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
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
    // EXPECT_EQ(image->shape()[0], 128);
    // EXPECT_EQ(image->shape()[1], 128);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);
  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomCropWithBboxFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropWithBboxFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // The size parameter is negative.
  std::shared_ptr<TensorTransform> random_crop0(new vision::RandomCropWithBBox({-10}));
  EXPECT_NE(random_crop0, nullptr);
  // The parameter in the padding vector is negative.
  std::shared_ptr<TensorTransform> random_crop1(new vision::RandomCropWithBBox({10, 10}, {-2, 2, 2, 2}));
  EXPECT_NE(random_crop1, nullptr);
  // The size container is empty.
  std::shared_ptr<TensorTransform> random_crop2(new vision::RandomCropWithBBox({}));
  EXPECT_NE(random_crop2, nullptr);
  // The size of the size container is too large.
  std::shared_ptr<TensorTransform> random_crop3(new vision::RandomCropWithBBox({10, 10, 10}));
  EXPECT_NE(random_crop3, nullptr);
  // The padding container is empty.
  std::shared_ptr<TensorTransform> random_crop4(new vision::RandomCropWithBBox({10, 10}, {}));
  EXPECT_NE(random_crop4, nullptr);
  // The size of the padding container is too large.
  std::shared_ptr<TensorTransform> random_crop5(new vision::RandomCropWithBBox({10, 10}, {5, 5, 5, 5, 5}));
  EXPECT_NE(random_crop5, nullptr);
  // The fill_value container is empty.
  std::shared_ptr<TensorTransform> random_crop6(new vision::RandomCropWithBBox({10, 10}, {5, 5, 5, 5}, false, {}));
  EXPECT_NE(random_crop6, nullptr);
  // The size of the fill_value container is too large.
  std::shared_ptr<TensorTransform> random_crop7(new
    vision::RandomCropWithBBox({10, 10}, {5, 5, 5, 5}, false, {3, 3, 3, 3}));
  EXPECT_NE(random_crop7, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomHorizontalFlipFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomHorizontalFlipFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create object for the tensor op
  // Invalid negative input
  std::shared_ptr<TensorTransform> random_horizontal_flip_op = std::make_shared<vision::RandomHorizontalFlip>(-0.5);
  EXPECT_NE(random_horizontal_flip_op, nullptr);
  // Invalid >1 input
  random_horizontal_flip_op = std::make_shared<vision::RandomHorizontalFlip>(2);
  EXPECT_NE(random_horizontal_flip_op, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomHorizontalFlipWithBBoxSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomHorizontalFlipWithBBoxSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_horizontal_flip_op = std::make_shared<vision::RandomHorizontalFlipWithBBox>(0.5);
  EXPECT_NE(random_horizontal_flip_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_horizontal_flip_op}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
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

TEST_F(MindDataTestPipeline, TestRandomHorizontalFlipWithBBoxFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomHorizontalFlipWithBBoxFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Incorrect prob parameter.
  std::shared_ptr<TensorTransform> random_horizontal_flip_op = std::make_shared<vision::RandomHorizontalFlipWithBBox>(-1.0);
  EXPECT_NE(random_horizontal_flip_op, nullptr);
  // Incorrect prob parameter.
  std::shared_ptr<TensorTransform> random_horizontal_flip_op1 = std::make_shared<vision::RandomHorizontalFlipWithBBox>(2.0);
  EXPECT_NE(random_horizontal_flip_op1, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomHorizontalAndVerticalFlip) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomHorizontalAndVerticalFlip for horizontal and vertical flips.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlip>(0.75);
  EXPECT_NE(random_vertical_flip_op, nullptr);

  std::shared_ptr<TensorTransform> random_horizontal_flip_op = std::make_shared<vision::RandomHorizontalFlip>(0.5);
  EXPECT_NE(random_horizontal_flip_op, nullptr);

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

TEST_F(MindDataTestPipeline, TestRandomPosterizeFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomPosterizeFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create objects for the tensor ops
  // Invalid max > 8
  std::shared_ptr<TensorTransform> posterize1(new vision::RandomPosterize({1, 9}));
  EXPECT_NE(posterize1, nullptr);
  // Invalid min < 1
  std::shared_ptr<TensorTransform> posterize2(new vision::RandomPosterize({0, 8}));
  EXPECT_NE(posterize2, nullptr);
  // min > max
  std::shared_ptr<TensorTransform> posterize3(new vision::RandomPosterize({8, 1}));
  EXPECT_NE(posterize3, nullptr);
  // empty
  //std::shared_ptr<TensorTransform> posterize4(new vision::RandomPosterize({}));
  // EXPECT_NE(posterize4, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomPosterizeSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomPosterizeSuccess1 with non-default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> posterize(new vision::RandomPosterize({1, 4}));
  EXPECT_NE(posterize, nullptr);

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

TEST_F(MindDataTestPipeline, TestRandomPosterizeSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomPosterizeSuccess2 with default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> posterize(new vision::RandomPosterize());
  EXPECT_NE(posterize, nullptr);

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

TEST_F(MindDataTestPipeline, TestRandomResizeSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeSuccess1 with single integer input.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resize(new vision::RandomResize({66}));
  EXPECT_NE(random_resize, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resize}, {"image"});
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
    // EXPECT_EQ(image->shape()[0] == 66, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizeSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeSuccess2 with (height, width) input.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 3));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resize(new vision::RandomResize({66, 77}));
  EXPECT_NE(random_resize, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resize}, {"image"});
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
    // EXPECT_EQ(image->shape()[0] == 66 && image->shape()[1] == 77, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizeFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeFail incorrect size.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // RandomResize : size must only contain positive integers
  std::shared_ptr<TensorTransform> random_resize1(new vision::RandomResize({-66, 77}));
  EXPECT_NE(random_resize1, nullptr);

  // RandomResize : size must only contain positive integers
  std::shared_ptr<TensorTransform> random_resize2(new vision::RandomResize({0, 77}));
  EXPECT_NE(random_resize2, nullptr);

  // RandomResize : size must be a vector of one or two values
  std::shared_ptr<TensorTransform> random_resize3(new vision::RandomResize({1, 2, 3}));
  EXPECT_NE(random_resize3, nullptr);

  // RandomResize : size must be a vector of one or two values
  std::shared_ptr<TensorTransform> random_resize4(new vision::RandomResize({}));
  EXPECT_NE(random_resize4, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizeWithBBoxSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeWithBBoxSuccess1 with single integer input.";

  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resize(new vision::RandomResizeWithBBox({88}));
  EXPECT_NE(random_resize, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resize}, {"image", "bbox"});
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
    // EXPECT_EQ(image->shape()[0] == 88, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizeWithBBoxSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeWithBBoxSuccess2 with (height, width) input.";

  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 4));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resize(new vision::RandomResizeWithBBox({88, 99}));
  EXPECT_NE(random_resize, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resize}, {"image", "bbox"});
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
    // EXPECT_EQ(image->shape()[0] == 88 && image->shape()[1] == 99, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizeWithBBoxFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizeWithBBoxFail incorrect size.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // RandomResizeWithBBox : size must only contain positive integers
  std::shared_ptr<TensorTransform> random_resize_with_bbox1(new vision::RandomResizeWithBBox({-66, 77}));
  EXPECT_NE(random_resize_with_bbox1, nullptr);

  // RandomResizeWithBBox : size must be a vector of one or two values
  std::shared_ptr<TensorTransform> random_resize_with_bbox2(new vision::RandomResizeWithBBox({1, 2, 3}));
  EXPECT_NE(random_resize_with_bbox2, nullptr);

  // RandomResizeWithBBox : size must be a vector of one or two values
  std::shared_ptr<TensorTransform> random_resize_with_bbox3(new vision::RandomResizeWithBBox({}));
  EXPECT_NE(random_resize_with_bbox3, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropSuccess1) {
  // Testing RandomResizedCrop with default values
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5}));
  EXPECT_NE(random_resized_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop}, {"image"});
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
    // EXPECT_EQ(image->shape()[0] == 5 && image->shape()[1] == 5, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropSuccess2) {
  // Testing RandomResizedCrop with non-default values
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new
    vision::RandomResizedCrop({5, 10}, {0.25, 0.75}, {0.5, 1.25}, mindspore::dataset::InterpolationMode::kArea, 20));
  EXPECT_NE(random_resized_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop}, {"image"});
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
    // EXPECT_EQ(image->shape()[0] == 5 && image->shape()[1] == 10, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropFail1) {
  // This should fail because size has negative value
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5, -10}));
  EXPECT_NE(random_resized_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomResizedCrop input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropFail2) {
  // This should fail because scale isn't in {min, max} format
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5, 10}, {4, 3}));
  EXPECT_NE(random_resized_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomResizedCrop input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropFail3) {
  // This should fail because ratio isn't in {min, max} format
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5, 10}, {4, 5}, {7, 6}));
  EXPECT_NE(random_resized_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomResizedCrop input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropFail4) {
  // This should fail because scale has a size of more than 2
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5, 10, 20}, {4, 5}, {7, 6}));
  EXPECT_NE(random_resized_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomResizedCrop input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxSuccess1) {
  // Testing RandomResizedCropWithBBox with default values
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 4));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox({5}));
  EXPECT_NE(random_resized_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop}, {"image", "bbox"});
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
    // EXPECT_EQ(image->shape()[0] == 5 && image->shape()[1] == 5, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxSuccess2) {
  // Testing RandomResizedCropWithBBox with non-default values
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 4));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox(
    {5, 10}, {0.25, 0.75}, {0.5, 1.25}, mindspore::dataset::InterpolationMode::kArea, 20));
  EXPECT_NE(random_resized_crop, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop}, {"image", "bbox"});
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
    // EXPECT_EQ(image->shape()[0] == 5 && image->shape()[1] == 10, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxFail1) {
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // This should fail because size has negative value
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox({5, -10}));
  EXPECT_NE(random_resized_crop, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxFail2) {
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // This should fail because scale isn't in {min, max} format
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox({5, 10}, {4, 3}));
  EXPECT_NE(random_resized_crop, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxFail3) {
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // This should fail because ratio isn't in {min, max} format
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox({5, 10}, {4, 5}, {7, 6}));
  EXPECT_NE(random_resized_crop, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxFail4) {
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // This should fail because scale has a size of more than 2
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox({5, 10, 20}, {4, 5}, {7, 6}));
  EXPECT_NE(random_resized_crop, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomRotation) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomRotation.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Testing the size of degrees is 1
  std::shared_ptr<TensorTransform> random_rotation_op(new vision::RandomRotation({180}));
  EXPECT_NE(random_rotation_op, nullptr);
  // Testing the size of degrees is 2
  std::shared_ptr<TensorTransform> random_rotation_op1(new vision::RandomRotation({-180, 180}));
  EXPECT_NE(random_rotation_op1, nullptr);
  // Testing the size of fill_value is 1
  std::shared_ptr<TensorTransform> random_rotation_op2(new
    vision::RandomRotation({180}, InterpolationMode::kNearestNeighbour, false, {-1, -1}, {2}));
  EXPECT_NE(random_rotation_op2, nullptr);
  // Testing the size of fill_value is 3
  std::shared_ptr<TensorTransform> random_rotation_op3(new
    vision::RandomRotation({180}, InterpolationMode::kNearestNeighbour, false, {-1, -1}, {2, 2, 2}));
  EXPECT_NE(random_rotation_op3, nullptr);

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

TEST_F(MindDataTestPipeline, TestRandomRotationFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomRotationFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Testing the size of degrees vector is 0
  std::shared_ptr<TensorTransform> random_rotation_op(new vision::RandomRotation({}));
  EXPECT_NE(random_rotation_op, nullptr);
  // Testing the size of degrees vector is 3
  std::shared_ptr<TensorTransform> random_rotation_op1(new vision::RandomRotation({-50.0, 50.0, 100.0}));
  EXPECT_NE(random_rotation_op1, nullptr);
  // Test the case where the first column value of degrees is greater than the second column value
  std::shared_ptr<TensorTransform> random_rotation_op2(new vision::RandomRotation({50.0, -50.0}));
  EXPECT_NE(random_rotation_op2, nullptr);
  // Testing the size of center vector is 1
  std::shared_ptr<TensorTransform> random_rotation_op3(new vision::RandomRotation(
    {-50.0, 50.0}, mindspore::dataset::InterpolationMode::kNearestNeighbour, false, {-1.0}));
  EXPECT_NE(random_rotation_op3, nullptr);
  // Testing the size of center vector is 3
  std::shared_ptr<TensorTransform> random_rotation_op4(new vision::RandomRotation(
    {-50.0, 50.0}, mindspore::dataset::InterpolationMode::kNearestNeighbour, false, {-1.0, -1.0, -1.0}));
  EXPECT_NE(random_rotation_op4, nullptr);
  // Testing the size of fill_value vector is 2
  std::shared_ptr<TensorTransform> random_rotation_op5(new vision::RandomRotation(
    {-50.0, 50.0}, mindspore::dataset::InterpolationMode::kNearestNeighbour, false, {-1.0, -1.0}, {2, 2}));
  EXPECT_NE(random_rotation_op5, nullptr);
  // Testing the size of fill_value vector is 4
  std::shared_ptr<TensorTransform> random_rotation_op6(new vision::RandomRotation(
    {-50.0, 50.0}, mindspore::dataset::InterpolationMode::kNearestNeighbour, false, {-1.0, -1.0}, {2, 2, 2, 2}));
  EXPECT_NE(random_rotation_op6, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomSharpness) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSharpness.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Valid case: Input start degree and end degree
  std::shared_ptr<TensorTransform> random_sharpness_op_1(new vision::RandomSharpness({0.4, 2.3}));
  EXPECT_NE(random_sharpness_op_1, nullptr);

  // Failure case: Empty degrees vector
  //
  // std::shared_ptr<TensorTransform> random_sharpness_op_2(new vision::RandomSharpness({}));
  //
  // EXPECT_NE(random_sharpness_op_2, nullptr);

  // Valid case: Use default input values
  std::shared_ptr<TensorTransform> random_sharpness_op_3(new vision::RandomSharpness());
  EXPECT_NE(random_sharpness_op_3, nullptr);

  // Failure case: Single degree value
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  std::shared_ptr<TensorTransform> random_sharpness_op_4(new vision::RandomSharpness({0.1}));
  EXPECT_NE(random_sharpness_op_4, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_sharpness_op_1, random_sharpness_op_3});
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

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomSolarizeSucess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSolarizeSucess1.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::vector<uint8_t> threshold = {10, 100};
  std::shared_ptr<TensorTransform> random_solarize = std::make_shared<mindspore::dataset::vision::RandomSolarize>(threshold);
  EXPECT_NE(random_solarize, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_solarize});
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

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomSolarizeSucess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSolarizeSuccess2 with default parameters.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, RandomSampler(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_solarize = std::make_shared<mindspore::dataset::vision::RandomSolarize>();
  EXPECT_NE(random_solarize, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_solarize});
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

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomSolarizeFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomSolarizeFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  std::vector<uint8_t> threshold = {13, 1};
  std::shared_ptr<TensorTransform> random_solarize = std::make_shared<mindspore::dataset::vision::RandomSolarize>(threshold);
  EXPECT_NE(random_solarize, nullptr);

  threshold = {1, 2, 3};
  random_solarize = std::make_shared<mindspore::dataset::vision::RandomSolarize>(threshold);
  EXPECT_NE(random_solarize, nullptr);

  threshold = {1};
  random_solarize = std::make_shared<mindspore::dataset::vision::RandomSolarize>(threshold);
  EXPECT_NE(random_solarize, nullptr);

  threshold = {};
  random_solarize = std::make_shared<mindspore::dataset::vision::RandomSolarize>(threshold);
  EXPECT_NE(random_solarize, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomVerticalFlipFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomVerticalFlipFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create object for the tensor op
  // Invalid negative input
  std::shared_ptr<TensorTransform> random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlip>(-0.5);
  EXPECT_NE(random_vertical_flip_op, nullptr);
  // Invalid >1 input
  random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlip>(1.1);
  EXPECT_NE(random_vertical_flip_op, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomVerticalFlipWithBBoxSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomVerticalFlipWithBBoxSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlipWithBBox>(0.4);
  EXPECT_NE(random_vertical_flip_op, nullptr);

  // Create a Map operation on ds
  ds = ds->Map({random_vertical_flip_op}, {"image", "bbox"}, {"image", "bbox"}, {"image", "bbox"});
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

TEST_F(MindDataTestPipeline, TestRandomVerticalFlipWithBBoxFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomVerticalFlipWithBBoxFail with invalid parameters.";
  // FIXME: For error tests, need to check for failure from CreateIterator execution
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, true, SequentialSampler(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Incorrect prob parameter.
  std::shared_ptr<TensorTransform> random_vertical_flip_op = std::make_shared<vision::RandomVerticalFlipWithBBox>(-0.5);
  EXPECT_NE(random_vertical_flip_op, nullptr);
  // Incorrect prob parameter.
  std::shared_ptr<TensorTransform> random_vertical_flip_op1 = std::make_shared<vision::RandomVerticalFlipWithBBox>(3.0);
  EXPECT_NE(random_vertical_flip_op1, nullptr);
}
