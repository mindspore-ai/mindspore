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
#include "include/api/types.h"
#include "minddata/dataset/core/de_tensor.h"
#include "minddata/dataset/include/execute.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/include/vision.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestExecute : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestExecute, TestComposeTransforms) {
  MS_LOG(INFO) << "Doing TestComposeTransforms.";

  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  std::shared_ptr<TensorTransform> decode = std::make_shared<vision::Decode>();
  std::shared_ptr<TensorTransform> center_crop(new vision::CenterCrop({30}));
  std::shared_ptr<TensorTransform> rescale = std::make_shared<vision::Rescale>(1. / 3, 0.5);

  auto transform = Execute({decode, center_crop, rescale});
  Status rc = transform(image, &image);

  EXPECT_EQ(rc, Status::OK());
  EXPECT_EQ(30, image.Shape()[0]);
  EXPECT_EQ(30, image.Shape()[1]);
}

TEST_F(MindDataTestExecute, TestTransformInput1) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestTransformInput1.";
  // Test Execute with transform op input using API constructors, with std::shared_ptr<TensorTransform pointers,
  // instantiated via mix of make_shared and new

  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Define transform operations
  std::shared_ptr<TensorTransform> decode = std::make_shared<vision::Decode>();
  std::shared_ptr<TensorTransform> resize(new vision::Resize({224, 224}));
  std::shared_ptr<TensorTransform> normalize(
    new vision::Normalize({0.485 * 255, 0.456 * 255, 0.406 * 255}, {0.229 * 255, 0.224 * 255, 0.225 * 255}));
  std::shared_ptr<TensorTransform> hwc2chw = std::make_shared<vision::HWC2CHW>();

  mindspore::dataset::Execute Transform({decode, resize, normalize, hwc2chw});

  // Apply transform on image
  Status rc = Transform(image, &image);

  // Check image info
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(image.Shape().size(), 3);
  ASSERT_EQ(image.Shape()[0], 3);
  ASSERT_EQ(image.Shape()[1], 224);
  ASSERT_EQ(image.Shape()[2], 224);
}

TEST_F(MindDataTestExecute, TestTransformInput2) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestTransformInput2.";
  // Test Execute with transform op input using API constructors, with std::shared_ptr<TensorTransform pointers,
  // instantiated via new
  // With this way of creating TensorTransforms, we don't need to explicitly delete the object created with the
  // "new" keyword. When the shared pointer goes out of scope the object destructor will be called.

  // Read image, construct MSTensor from dataset tensor
  std::shared_ptr<mindspore::dataset::Tensor> de_tensor;
  mindspore::dataset::Tensor::CreateFromFile("data/dataset/apple.jpg", &de_tensor);
  auto image = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));

  // Define transform operations
  std::shared_ptr<TensorTransform> decode(new vision::Decode());
  std::shared_ptr<TensorTransform> resize(new vision::Resize({224, 224}));
  std::shared_ptr<TensorTransform> normalize(
    new vision::Normalize({0.485 * 255, 0.456 * 255, 0.406 * 255}, {0.229 * 255, 0.224 * 255, 0.225 * 255}));
  std::shared_ptr<TensorTransform> hwc2chw(new vision::HWC2CHW());

  mindspore::dataset::Execute Transform({decode, resize, normalize, hwc2chw});

  // Apply transform on image
  Status rc = Transform(image, &image);

  // Check image info
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(image.Shape().size(), 3);
  ASSERT_EQ(image.Shape()[0], 3);
  ASSERT_EQ(image.Shape()[1], 224);
  ASSERT_EQ(image.Shape()[2], 224);
}

TEST_F(MindDataTestExecute, TestTransformInput3) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestTransformInput3.";
  // Test Execute with transform op input using API constructors, with auto pointers

  // Read image, construct MSTensor from dataset tensor
  std::shared_ptr<mindspore::dataset::Tensor> de_tensor;
  mindspore::dataset::Tensor::CreateFromFile("data/dataset/apple.jpg", &de_tensor);
  auto image = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));

  // Define transform operations
  auto decode = vision::Decode();
  mindspore::dataset::Execute Transform1(decode);

  auto resize = vision::Resize({224, 224});
  mindspore::dataset::Execute Transform2(resize);

  // Apply transform on image
  Status rc;
  rc = Transform1(image, &image);
  ASSERT_TRUE(rc.IsOk());
  rc = Transform2(image, &image);
  ASSERT_TRUE(rc.IsOk());

  // Check image info
  ASSERT_EQ(image.Shape().size(), 3);
  ASSERT_EQ(image.Shape()[0], 224);
  ASSERT_EQ(image.Shape()[1], 224);
  ASSERT_EQ(image.Shape()[2], 3);
}

TEST_F(MindDataTestExecute, TestTransformInputSequential) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestTransformInputSequential.";
  // Test Execute with transform op input using API constructors, with auto pointers;
  // Apply 2 transformations sequentially, including single non-vector Transform op input

  // Read image, construct MSTensor from dataset tensor
  std::shared_ptr<mindspore::dataset::Tensor> de_tensor;
  mindspore::dataset::Tensor::CreateFromFile("data/dataset/apple.jpg", &de_tensor);
  auto image = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));

  // Define transform#1 operations
  std::shared_ptr<TensorTransform> decode(new vision::Decode());
  std::shared_ptr<TensorTransform> resize(new vision::Resize({224, 224}));
  std::shared_ptr<TensorTransform> normalize(
    new vision::Normalize({0.485 * 255, 0.456 * 255, 0.406 * 255}, {0.229 * 255, 0.224 * 255, 0.225 * 255}));

  std::vector<std::shared_ptr<TensorTransform>> op_list = {decode, resize, normalize};
  mindspore::dataset::Execute Transform(op_list);

  // Apply transform#1 on image
  Status rc = Transform(image, &image);

  // Define transform#2 operations
  std::shared_ptr<TensorTransform> hwc2chw(new vision::HWC2CHW());
  mindspore::dataset::Execute Transform2(hwc2chw);

  // Apply transform#2 on image
  rc = Transform2(image, &image);

  // Check image info
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(image.Shape().size(), 3);
  ASSERT_EQ(image.Shape()[0], 3);
  ASSERT_EQ(image.Shape()[1], 224);
  ASSERT_EQ(image.Shape()[2], 224);
}

TEST_F(MindDataTestExecute, TestTransformDecodeResizeCenterCrop1) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestTransformDecodeResizeCenterCrop1.";
  // Test Execute with Decode, Resize and CenterCrop transform ops input using API constructors, with shared pointers

  // Read image, construct MSTensor from dataset tensor
  std::shared_ptr<mindspore::dataset::Tensor> de_tensor;
  mindspore::dataset::Tensor::CreateFromFile("data/dataset/apple.jpg", &de_tensor);
  auto image = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));

  // Define transform operations
  std::vector<int32_t> resize_paras = {256, 256};
  std::vector<int32_t> crop_paras = {224, 224};
  std::shared_ptr<TensorTransform> decode(new vision::Decode());
  std::shared_ptr<TensorTransform> resize(new vision::Resize(resize_paras));
  std::shared_ptr<TensorTransform> centercrop(new vision::CenterCrop(crop_paras));
  std::shared_ptr<TensorTransform> hwc2chw(new vision::HWC2CHW());

  std::vector<std::shared_ptr<TensorTransform>> op_list = {decode, resize, centercrop, hwc2chw};
  mindspore::dataset::Execute Transform(op_list, MapTargetDevice::kCpu);

  // Apply transform on image
  Status rc = Transform(image, &image);

  // Check image info
  ASSERT_TRUE(rc.IsOk());
  ASSERT_EQ(image.Shape().size(), 3);
  ASSERT_EQ(image.Shape()[0], 3);
  ASSERT_EQ(image.Shape()[1], 224);
  ASSERT_EQ(image.Shape()[2], 224);
}
