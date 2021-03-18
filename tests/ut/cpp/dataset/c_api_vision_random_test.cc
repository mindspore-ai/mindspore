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
#include "minddata/dataset/include/config.h"
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
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Case 1: Empty input for translate
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> affine1(new vision::RandomAffine({0.0, 0.0}, {}));
  auto ds1 = ds->Map({affine1});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid input for RandomAffine
  EXPECT_EQ(iter1, nullptr);

  // Case 2: Invalid number of values for translate
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> affine2(new vision::RandomAffine({0.0, 0.0}, {1, 1, 1, 1, 1}));
  auto ds2 = ds->Map({affine2});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid input for RandomAffine
  EXPECT_EQ(iter2, nullptr);

  // Case 3: Invalid number of values for shear
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> affine3(new vision::RandomAffine({30.0, 30.0}, {0.0, 0.0}, {2.0, 2.0}, {10.0}));
  auto ds3 = ds->Map({affine3});
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid input for RandomAffine
  EXPECT_EQ(iter3, nullptr);
}

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
  std::shared_ptr<TensorTransform> affine(
    new vision::RandomAffine({30.0, 30.0}, {-1.0, 1.0, -1.0, 1.0}, {2.0, 2.0}, {10.0, 10.0, 20.0, 20.0}));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
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
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> affine(new vision::RandomAffine({0.0, 0.0}));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
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

TEST_F(MindDataTestPipeline, TestRandomColorAdjust) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomColorAdjust.";

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
  std::shared_ptr<TensorTransform> random_color_adjust1(new vision::RandomColorAdjust({1.0}, {0.0}, {0.5}, {0.5}));

  // Use same 2 values for vectors
  std::shared_ptr<TensorTransform> random_color_adjust2(
    new vision::RandomColorAdjust({1.0, 1.0}, {0.0, 0.0}, {0.5, 0.5}, {0.5, 0.5}));

  // Use different 2 value for vectors
  std::shared_ptr<TensorTransform> random_color_adjust3(
    new vision::RandomColorAdjust({0.5, 1.0}, {0.0, 0.5}, {0.25, 0.5}, {0.25, 0.5}));

  // Use default input values
  std::shared_ptr<TensorTransform> random_color_adjust4(new vision::RandomColorAdjust());

  // Use subset of explicitly set parameters
  std::shared_ptr<TensorTransform> random_color_adjust5(new vision::RandomColorAdjust({0.0, 0.5}, {0.25}));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomCropSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Testing size of size vector is 1
  std::shared_ptr<TensorTransform> random_crop(new vision::RandomCrop({20}));

  // Testing size of size vector is 2
  std::shared_ptr<TensorTransform> random_crop1(new vision::RandomCrop({20, 20}));

  // Testing size of paddiing vector is 1
  std::shared_ptr<TensorTransform> random_crop2(new vision::RandomCrop({20, 20}, {10}));

  // Testing size of paddiing vector is 2
  std::shared_ptr<TensorTransform> random_crop3(new vision::RandomCrop({20, 20}, {10, 20}));

  // Testing size of paddiing vector is 2
  std::shared_ptr<TensorTransform> random_crop4(new vision::RandomCrop({20, 20}, {10, 10, 10, 10}));

  // Testing size of fill_value vector is 1
  std::shared_ptr<TensorTransform> random_crop5(new vision::RandomCrop({20, 20}, {10, 10, 10, 10}, false, {5}));

  // Testing size of fill_value vector is 3
  std::shared_ptr<TensorTransform> random_crop6(new vision::RandomCrop({20, 20}, {10, 10, 10, 10}, false, {4, 4, 4}));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);
  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomCropFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropFail with invalid parameters.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Case 1: Testing the size parameter is negative.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop1(new vision::RandomCrop({-28, 28}));
  auto ds1 = ds->Map({random_crop1});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter1, nullptr);

  // Case 2: Testing the size parameter is None.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop2(new vision::RandomCrop({}));
  auto ds2 = ds->Map({random_crop2});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter2, nullptr);

  // Case 3: Testing the size of size vector is 3.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop3(new vision::RandomCrop({28, 28, 28}));
  auto ds3 = ds->Map({random_crop3});
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter3, nullptr);

  // Case 4: Testing the padding parameter is negative.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop4(new vision::RandomCrop({28, 28}, {-5}));
  auto ds4 = ds->Map({random_crop4});
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter4, nullptr);

  // Case 5: Testing the size of padding vector is empty.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop5(new vision::RandomCrop({28, 28}, {}));
  auto ds5 = ds->Map({random_crop5});
  EXPECT_NE(ds5, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter5 = ds5->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter5, nullptr);

  // Case 6: Testing the size of padding vector is 3.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop6(new vision::RandomCrop({28, 28}, {5, 5, 5}));
  auto ds6 = ds->Map({random_crop6});
  EXPECT_NE(ds6, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter6 = ds6->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter6, nullptr);

  // Case 7: Testing the size of padding vector is 5.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop7(new vision::RandomCrop({28, 28}, {5, 5, 5, 5, 5}));
  auto ds7 = ds->Map({random_crop7});
  EXPECT_NE(ds7, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter7 = ds7->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter7, nullptr);

  // Case 8: Testing the size of fill_value vector is empty.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop8(new vision::RandomCrop({28, 28}, {0, 0, 0, 0}, false, {}));
  auto ds8 = ds->Map({random_crop8});
  EXPECT_NE(ds8, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter8 = ds8->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter8, nullptr);

  // Case 9: Testing the size of fill_value vector is 2.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop9(new vision::RandomCrop({28, 28}, {0, 0, 0, 0}, false, {0, 0}));
  auto ds9 = ds->Map({random_crop9});
  EXPECT_NE(ds9, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter9 = ds9->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter9, nullptr);

  // Case 10: Testing the size of fill_value vector is 4.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop10(new vision::RandomCrop({28, 28}, {0, 0, 0, 0}, false, {0, 0, 0, 0}));
  auto ds10 = ds->Map({random_crop10});
  EXPECT_NE(ds10, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter10 = ds10->CreateIterator();
  // Expect failure: invalid input for RandomCrop
  EXPECT_EQ(iter10, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomCropWithBboxSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropWithBboxSuccess.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop(new mindspore::dataset::vision::RandomCropWithBBox({128, 128}));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0], 128);
    EXPECT_EQ(image.Shape()[1], 128);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);
  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomCropWithBboxFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomCropWithBboxFail with invalid parameters.";
  // Create an VOC Dataset
  std::string folder_path = datasets_root_path_ + "/testVOC2012_2";
  std::shared_ptr<Dataset> ds =
    VOC(folder_path, "Detection", "train", {}, true, std::make_shared<SequentialSampler>(0, 3));
  EXPECT_NE(ds, nullptr);

  // Case 1: The size parameter is negative.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop1(new vision::RandomCropWithBBox({-10}));
  auto ds1 = ds->Map({random_crop1});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid input for RandomCropWithBBox
  EXPECT_EQ(iter1, nullptr);

  // Case 2: The parameter in the padding vector is negative.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop2(new vision::RandomCropWithBBox({10, 10}, {-2, 2, 2, 2}));
  auto ds2 = ds->Map({random_crop2});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid input for RandomCropWithBBox
  EXPECT_EQ(iter2, nullptr);

  // Case 3: The size container is empty.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop3(new vision::RandomCropWithBBox({}));
  auto ds3 = ds->Map({random_crop3});
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid input for RandomCropWithBBox
  EXPECT_EQ(iter3, nullptr);

  // Case 4: The size of the size container is too large.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop4(new vision::RandomCropWithBBox({10, 10, 10}));
  auto ds4 = ds->Map({random_crop4});
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid input for RandomCropWithBBox
  EXPECT_EQ(iter4, nullptr);

  // Case 5: The padding container is empty.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop5(new vision::RandomCropWithBBox({10, 10}, {}));
  auto ds5 = ds->Map({random_crop5});
  EXPECT_NE(ds5, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter5 = ds5->CreateIterator();
  // Expect failure: invalid input for RandomCropWithBBox
  EXPECT_EQ(iter5, nullptr);

  // Case 6: The size of the padding container is too large.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop6(new vision::RandomCropWithBBox({10, 10}, {5, 5, 5, 5, 5}));
  auto ds6 = ds->Map({random_crop6});
  EXPECT_NE(ds6, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter6 = ds6->CreateIterator();
  // Expect failure: invalid input for RandomCropWithBBox
  EXPECT_EQ(iter6, nullptr);

  // Case 7: The fill_value container is empty.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop7(new vision::RandomCropWithBBox({10, 10}, {5, 5, 5, 5}, false, {}));
  auto ds7 = ds->Map({random_crop7});
  EXPECT_NE(ds7, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter7 = ds7->CreateIterator();
  // Expect failure: invalid input for RandomCropWithBBox
  EXPECT_EQ(iter7, nullptr);

  // Case 8: The size of the fill_value container is too large.
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_crop8(
    new vision::RandomCropWithBBox({10, 10}, {5, 5, 5, 5}, false, {3, 3, 3, 3}));
  auto ds8 = ds->Map({random_crop8});
  EXPECT_NE(ds8, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter8 = ds8->CreateIterator();
  // Expect failure: invalid input for RandomCropWithBBox
  EXPECT_EQ(iter8, nullptr);
}

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);
  // Manually terminate the pipeline
  iter->Stop();
}

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
  std::shared_ptr<TensorTransform> posterize(new vision::RandomPosterize({1, 4}));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
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
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> posterize(new vision::RandomPosterize());

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
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
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resize(new vision::RandomResize({66}));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0] == 66, true);
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
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds, nullptr);

  // Create a Repeat operation on ds
  int32_t repeat_num = 2;
  ds = ds->Repeat(repeat_num);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resize(new vision::RandomResize({66, 77}));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0] == 66 && image.Shape()[1] == 77, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline
  iter->Stop();
}

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
  std::shared_ptr<TensorTransform> random_resize(new vision::RandomResizeWithBBox({88}));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0] == 88, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);
  
  // Manually terminate the pipeline
  iter->Stop();
  config::set_seed(current_seed);
}

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
  std::shared_ptr<TensorTransform> random_resize(new vision::RandomResizeWithBBox({88, 99}));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0] == 88 && image.Shape()[1] == 99, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 8);

  // Manually terminate the pipeline
  iter->Stop();
  config::set_seed(current_seed);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropSuccess1.";
  // Testing RandomResizedCrop with default values
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5}));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0] == 5 && image.Shape()[1] == 5, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropSuccess2.";
  // Testing RandomResizedCrop with non-default values
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop(
    {5, 10}, {0.25, 0.75}, {0.5, 1.25}, mindspore::dataset::InterpolationMode::kArea, 20));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0] == 5 && image.Shape()[1] == 10, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropFail1 with negative size.";
  // This should fail because size has negative value
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5, -10}));

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomResizedCrop input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropFail1 with invalid scale input.";
  // This should fail because scale isn't in {min, max} format
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5, 10}, {4, 3}));

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomResizedCrop input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropFail1 with invalid ratio input.";
  // This should fail because ratio isn't in {min, max} format
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5, 10}, {4, 5}, {7, 6}));

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomResizedCrop input
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropFail1 with invalid scale size.";
  // This should fail because scale has a size of more than 2
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCrop({5, 10, 20}, {4, 5}, {7, 6}));

  // Create a Map operation on ds
  ds = ds->Map({random_resized_crop});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: Invalid RandomResizedCrop input
  EXPECT_EQ(iter, nullptr);
}

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
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox({5}));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0] == 5 && image.Shape()[1] == 5, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 4);

  config::set_seed(current_seed);
  // Manually terminate the pipeline
  iter->Stop();
}

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
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox(
    {5, 10}, {0.25, 0.75}, {0.5, 1.25}, mindspore::dataset::InterpolationMode::kArea, 20));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_EQ(image.Shape()[0] == 5 && image.Shape()[1] == 10, true);
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 4);
  config::set_seed(current_seed);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropWithBBoxFail1 with negative size value.";
  // This should fail because size has negative value
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox({5, -10}));
  auto ds1 = ds->Map({random_resized_crop});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid input for RandomResizedCropWithBBox
  EXPECT_EQ(iter1, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropWithBBoxFail2 with invalid scale input.";
  // This should fail because scale isn't in {min, max} format
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox({5, 10}, {4, 3}));
  auto ds1 = ds->Map({random_resized_crop});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid input for RandomResizedCropWithBBox
  EXPECT_EQ(iter1, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropWithBBoxFail3 with invalid ratio input.";
  // This should fail because ratio isn't in {min, max} format
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(new vision::RandomResizedCropWithBBox({5, 10}, {4, 5}, {7, 6}));
  auto ds1 = ds->Map({random_resized_crop});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid input for RandomResizedCropWithBBox
  EXPECT_EQ(iter1, nullptr);
}

TEST_F(MindDataTestPipeline, TestRandomResizedCropWithBBoxFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomResizedCropWithBBoxFail4 with invalid scale size.";
  // This should fail because scale has a size of more than 2
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_resized_crop(
    new vision::RandomResizedCropWithBBox({5, 10, 20}, {4, 5}, {7, 6}));
  auto ds1 = ds->Map({random_resized_crop});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid input for RandomResizedCropWithBBox
  EXPECT_EQ(iter1, nullptr);
}

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
  std::shared_ptr<TensorTransform> random_rotation_op(new vision::RandomRotation({180}));
  // Testing the size of degrees is 2
  std::shared_ptr<TensorTransform> random_rotation_op1(new vision::RandomRotation({-180, 180}));
  // Testing the size of fill_value is 1
  std::shared_ptr<TensorTransform> random_rotation_op2(
    new vision::RandomRotation({180}, InterpolationMode::kNearestNeighbour, false, {-1, -1}, {2}));
  // Testing the size of fill_value is 3
  std::shared_ptr<TensorTransform> random_rotation_op3(
    new vision::RandomRotation({180}, InterpolationMode::kNearestNeighbour, false, {-1, -1}, {2, 2, 2}));

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 20);

  // Manually terminate the pipeline
  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestRandomRotationFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomRotationFail with invalid parameters.";
  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Case 1: Testing the size of degrees vector is 0
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_rotation_op1(new vision::RandomRotation({}));
  auto ds1 = ds->Map({random_rotation_op1});
  EXPECT_NE(ds1, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  // Expect failure: invalid input for RandomRotation
  EXPECT_EQ(iter1, nullptr);

  // Case 2: Testing the size of degrees vector is 3
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_rotation_op2(new vision::RandomRotation({-50.0, 50.0, 100.0}));
  auto ds2 = ds->Map({random_rotation_op2});
  EXPECT_NE(ds2, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  // Expect failure: invalid input for RandomRotation
  EXPECT_EQ(iter2, nullptr);

  // Case 3: Test the case where the first column value of degrees is greater than the second column value
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_rotation_op3(new vision::RandomRotation({50.0, -50.0}));
  auto ds3 = ds->Map({random_rotation_op3});
  EXPECT_NE(ds3, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  // Expect failure: invalid input for RandomRotation
  EXPECT_EQ(iter3, nullptr);

  // Case 4: Testing the size of center vector is 1
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_rotation_op4(
    new vision::RandomRotation({-50.0, 50.0}, mindspore::dataset::InterpolationMode::kNearestNeighbour, false, {-1.0}));
  auto ds4 = ds->Map({random_rotation_op4});
  EXPECT_NE(ds4, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  // Expect failure: invalid input for RandomRotation
  EXPECT_EQ(iter4, nullptr);

  // Case 5: Testing the size of center vector is 3
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_rotation_op5(new vision::RandomRotation(
    {-50.0, 50.0}, mindspore::dataset::InterpolationMode::kNearestNeighbour, false, {-1.0, -1.0, -1.0}));
  auto ds5 = ds->Map({random_rotation_op5});
  EXPECT_NE(ds5, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter5 = ds5->CreateIterator();
  // Expect failure: invalid input for RandomRotation
  EXPECT_EQ(iter5, nullptr);

  // Case 6: Testing the size of fill_value vector is 2
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_rotation_op6(new vision::RandomRotation(
    {-50.0, 50.0}, mindspore::dataset::InterpolationMode::kNearestNeighbour, false, {-1.0, -1.0}, {2, 2}));
  auto ds6 = ds->Map({random_rotation_op6});
  EXPECT_NE(ds6, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter6 = ds6->CreateIterator();
  // Expect failure: invalid input for RandomRotation
  EXPECT_EQ(iter6, nullptr);

  // Case 7: Testing the size of fill_value vector is 4
  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> random_rotation_op7(new vision::RandomRotation(
    {-50.0, 50.0}, mindspore::dataset::InterpolationMode::kNearestNeighbour, false, {-1.0, -1.0}, {2, 2, 2, 2}));
  auto ds7 = ds->Map({random_rotation_op7});
  EXPECT_NE(ds7, nullptr);
  // Create an iterator over the result of the above dataset
  std::shared_ptr<Iterator> iter7 = ds7->CreateIterator();
  // Expect failure: invalid input for RandomRotation
  EXPECT_EQ(iter7, nullptr);
}

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
  std::shared_ptr<TensorTransform> random_sharpness_op_1(new vision::RandomSharpness({0.4, 2.3}));

  // Valid case: Use default input values
  std::shared_ptr<TensorTransform> random_sharpness_op_2(new vision::RandomSharpness());

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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
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
  iter->GetNextRow(&row);

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

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
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    iter->GetNextRow(&row);
  }

  EXPECT_EQ(i, 3);
  // Manually terminate the pipeline
  iter->Stop();
}
