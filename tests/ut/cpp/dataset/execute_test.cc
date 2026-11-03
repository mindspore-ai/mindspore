/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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
#include "include/dataset/ms_tensor.h"
#include "minddata/dataset/core/de_tensor.h"
#include "minddata/dataset/include/dataset/audio.h"
#include "minddata/dataset/include/dataset/execute.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/include/dataset/vision.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestExecute : public UT::DatasetOpTesting {
 protected:
};

/// Feature: Execute composed transform ops
/// Description: Test executing composed multiple transform ops
/// Expectation: Output is equal to the expected output and status is okay
TEST_F(MindDataTestExecute, TestComposeTransforms) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestComposeTransforms.";

  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  std::shared_ptr<TensorTransform> decode = std::make_shared<vision::Decode>();
  auto center_crop = std::make_shared<vision::CenterCrop>(std::vector<int32_t>{30});
  std::shared_ptr<TensorTransform> rescale = std::make_shared<vision::Rescale>(1. / 3, 0.5);

  auto transform = Execute({decode, center_crop, rescale});
  Status rc = transform(image, &image);

  EXPECT_EQ(rc, Status::OK());
  EXPECT_EQ(30, image.Shape()[0]);
  EXPECT_EQ(30, image.Shape()[1]);
}

/// Feature: Execute Transform op
/// Description: Test Execute with transform op input using API constructors with
///     std::shared_ptr<TensorTransform pointers, instantiated via mix of make_shared and new
/// Expectation: Output is equal to the expected output and status is okay
TEST_F(MindDataTestExecute, TestTransformInput1) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestTransformInput1.";
  // Test Execute with transform op input using API constructors, with std::shared_ptr<TensorTransform pointers,
  // instantiated via mix of make_shared and new

  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Define transform operations
  std::shared_ptr<TensorTransform> decode = std::make_shared<vision::Decode>();
  auto resize = std::make_shared<vision::Resize>(std::vector<int32_t>{224, 224});
  auto normalize = std::make_shared<vision::Normalize>(std::vector<float>{0.485 * 255, 0.456 * 255, 0.406 * 255},
                                                       std::vector<float>{0.229 * 255, 0.224 * 255, 0.225 * 255});
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

/// Feature: Execute Transform op
/// Description: Test Execute with transform op input using API constructors with
///     std::shared_ptr<TensorTransform pointers, instantiated via new
/// Expectation: Output is equal to the expected output and status is okay
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
  auto decode = std::make_shared<vision::Decode>();
  auto resize = std::make_shared<vision::Resize>(std::vector<int32_t>{224, 224});
  auto normalize = std::make_shared<vision::Normalize>(std::vector<float>{0.485 * 255, 0.456 * 255, 0.406 * 255},
                                                       std::vector<float>{0.229 * 255, 0.224 * 255, 0.225 * 255});
  auto hwc2chw = std::make_shared<vision::HWC2CHW>();

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

/// Feature: Execute Transform op
/// Description: Test Execute with transform op input using API constructors with auto pointers
/// Expectation: Output is equal to the expected output and status is okay
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

/// Feature: Execute Transform op
/// Description: Test Execute with transform op input using API constructors with auto pointers
///     then apply 2 transformations sequentially, including single non-vector transform op input
/// Expectation: Output is equal to the expected output and status is okay
TEST_F(MindDataTestExecute, TestTransformInputSequential) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestTransformInputSequential.";
  // Test Execute with transform op input using API constructors, with auto pointers;
  // Apply 2 transformations sequentially, including single non-vector Transform op input

  // Read image, construct MSTensor from dataset tensor
  std::shared_ptr<mindspore::dataset::Tensor> de_tensor;
  mindspore::dataset::Tensor::CreateFromFile("data/dataset/apple.jpg", &de_tensor);
  auto image = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));

  // Define transform#1 operations
  auto decode = std::make_shared<vision::Decode>();
  auto resize = std::make_shared<vision::Resize>(std::vector<int32_t>{224, 224});
  auto normalize = std::make_shared<vision::Normalize>(std::vector<float>{0.485 * 255, 0.456 * 255, 0.406 * 255},
                                                       std::vector<float>{0.229 * 255, 0.224 * 255, 0.225 * 255});

  std::vector<std::shared_ptr<TensorTransform>> op_list = {decode, resize, normalize};
  mindspore::dataset::Execute Transform(op_list);

  // Apply transform#1 on image
  Status rc = Transform(image, &image);

  // Define transform#2 operations
  auto hwc2chw = std::make_shared<vision::HWC2CHW>();
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

/// Feature: Execute Transform op
/// Description: Test Execute with Decode, Resize, and CenterCrop transform ops input using API constructors,
///     with shared pointers
/// Expectation: Output is equal to the expected output and status is okay
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
  auto decode = std::make_shared<vision::Decode>();
  auto resize = std::make_shared<vision::Resize>(resize_paras);
  auto centercrop = std::make_shared<vision::CenterCrop>(crop_paras);
  auto hwc2chw = std::make_shared<vision::HWC2CHW>();

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

/// Feature: Execute Transform op
/// Description: Test executing Decode op then RGB2BGR op with eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestRGB2BGREager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestRGB2BGREager.";

  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto rgb2bgr_op = vision::RGB2BGR();

  auto transform = Execute({decode, rgb2bgr_op});
  Status rc = transform(image, &image);

  EXPECT_EQ(rc, Status::OK());
}

/// Feature: Execute Construct Demo1
/// Description: Demonstrate how to construct a Execute
/// Expectation: Construct Execute object and run
TEST_F(MindDataTestExecute, TestConstructorDemo1) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestConstructorDemo1.";
  // Read images
  mindspore::MSTensor image = ReadFileToTensor("data/dataset/apple.jpg");
  mindspore::MSTensor aug_image;

  // Pass Transformation object
  auto decode = vision::Decode();
  auto resize = vision::Resize({66, 77});
  auto transform = Execute({decode, resize});

  Status rc = transform(image, &aug_image);
  EXPECT_EQ(rc, Status::OK());

  EXPECT_EQ(aug_image.Shape()[0], 66);
  EXPECT_EQ(aug_image.Shape()[1], 77);
}

/// Feature: Execute Construct Demo2
/// Description: Demonstrate how to construct a Execute
/// Expectation: Construct Execute object and run
TEST_F(MindDataTestExecute, TestConstructorDemo2) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestConstructorDemo2.";
  // Read images
  mindspore::MSTensor image = ReadFileToTensor("data/dataset/apple.jpg");
  mindspore::MSTensor aug_image;

  // Pass address of Transformation object
  auto decode = vision::Decode();
  auto resize = vision::Resize({66, 77});
  std::vector<TensorTransform *> ops = {&decode};
  bool use_resize = true;
  if (use_resize) {
    ops.push_back(&resize);
  }

  auto transform = Execute(ops);
  Status rc = transform(image, &aug_image);
  EXPECT_EQ(rc, Status::OK());

  EXPECT_EQ(aug_image.Shape()[0], 66);
  EXPECT_EQ(aug_image.Shape()[1], 77);
}

/// Feature: Execute Construct Demo3
/// Description: Demonstrate how to construct a Execute
/// Expectation: Construct Execute object and run
TEST_F(MindDataTestExecute, TestConstructorDemo3) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestConstructorDemo3.";
  // Read images
  mindspore::MSTensor image = ReadFileToTensor("data/dataset/apple.jpg");
  std::vector<mindspore::MSTensor> images = {image};
  std::vector<mindspore::MSTensor> aug_images;

  // Pass smart pointers of Transformation object
  auto decode = std::make_shared<vision::Decode>();
  auto resize = std::make_shared<vision::Resize>(std::vector<int32_t>{66, 77});
  std::vector<std::shared_ptr<TensorTransform>> ops = {decode, resize};

  auto transform = Execute(ops);
  Status rc = transform(images, &aug_images);
  EXPECT_EQ(rc, Status::OK());

  EXPECT_EQ(aug_images[0].Shape()[0], 66);
  EXPECT_EQ(aug_images[0].Shape()[1], 77);
}

/// Feature: Execute Construct Demo4
/// Description: Demonstrate how to construct a Execute
/// Expectation: Construct Execute object and run
TEST_F(MindDataTestExecute, TestConstructorDemo4) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestConstructorDemo4.";
  // Read images
  mindspore::MSTensor image = ReadFileToTensor("data/dataset/apple.jpg");
  std::vector<mindspore::MSTensor> images = {image};
  std::vector<mindspore::MSTensor> aug_images;

  // Pass raw pointers of Transformation object
  auto decode = new vision::Decode();
  auto resize = new vision::Resize({66, 77});
  std::vector<TensorTransform *> ops = {decode, resize};

  auto transform = Execute(ops);
  Status rc = transform(images, &aug_images);
  EXPECT_EQ(rc, Status::OK());

  EXPECT_EQ(aug_images[0].Shape()[0], 66);
  EXPECT_EQ(aug_images[0].Shape()[1], 77);

  delete decode;
  delete resize;
}

#ifdef ENABLE_FFMPEG

#endif
