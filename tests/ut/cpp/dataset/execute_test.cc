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
#include "minddata/dataset/include/dataset/audio.h"
#include "minddata/dataset/include/dataset/execute.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/include/dataset/vision.h"
#include "minddata/dataset/include/dataset/text.h"
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

TEST_F(MindDataTestExecute, TestCrop) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestCrop.";

  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto crop = vision::Crop({10, 30}, {10, 15});

  auto transform = Execute({decode, crop});
  Status rc = transform(image, &image);

  EXPECT_EQ(rc, Status::OK());
  EXPECT_EQ(image.Shape()[0], 10);
  EXPECT_EQ(image.Shape()[1], 15);
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

TEST_F(MindDataTestExecute, TestUniformAugment) {
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");
  std::vector<mindspore::MSTensor> image2;

  // Transform params
  std::shared_ptr<TensorTransform> decode = std::make_shared<vision::Decode>();
  std::shared_ptr<TensorTransform> resize_op(new vision::Resize({16, 16}));
  std::shared_ptr<TensorTransform> vertical = std::make_shared<vision::RandomVerticalFlip>();
  std::shared_ptr<TensorTransform> horizontal = std::make_shared<vision::RandomHorizontalFlip>();

  std::shared_ptr<TensorTransform> uniform_op(new vision::UniformAugment({resize_op, vertical, horizontal}, 3));

  auto transform1 = Execute({decode});
  Status rc = transform1(image, &image);
  ASSERT_TRUE(rc.IsOk());

  auto transform2 = Execute({uniform_op});
  rc = transform2({image}, &image2);
  ASSERT_TRUE(rc.IsOk());
}

TEST_F(MindDataTestExecute, TestBasicTokenizer) {
  std::shared_ptr<Tensor> de_tensor;
  Tensor::CreateScalar<std::string>("Welcome to China.", &de_tensor);
  auto txt = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));
  std::vector<mindspore::MSTensor> txt_result;

  // Transform params
  std::shared_ptr<TensorTransform> tokenizer =
    std::make_shared<text::BasicTokenizer>(false, false, NormalizeForm::kNone, false, true);

  // BasicTokenizer has 3 outputs so we need a vector to receive its result
  auto transform1 = Execute({tokenizer});
  Status rc = transform1({txt}, &txt_result);
  ASSERT_EQ(txt_result.size(), 3);
  ASSERT_TRUE(rc.IsOk());
}

TEST_F(MindDataTestExecute, TestRotate) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestRotate.";

  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto rotate = vision::Rotate(10.5);

  auto transform = Execute({decode, rotate});
  Status rc = transform(image, &image);

  EXPECT_EQ(rc, Status::OK());
}

TEST_F(MindDataTestExecute, TestResizeWithBBox) {
  auto image = ReadFileToTensor("data/dataset/apple.jpg");
  std::shared_ptr<TensorTransform> decode_op = std::make_shared<vision::Decode>();
  std::shared_ptr<TensorTransform> resizewithbbox_op =
    std::make_shared<vision::ResizeWithBBox>(std::vector<int32_t>{250, 500});

  // Test Compute(Tensor, Tensor) method of ResizeWithBBox
  auto transform = Execute({decode_op, resizewithbbox_op});

  // Expect fail since Compute(Tensor, Tensor) is not a valid behaviour for this Op,
  // while Compute(TensorRow, TensorRow) is the correct one.
  Status rc = transform(image, &image);
  EXPECT_FALSE(rc.IsOk());
}

TEST_F(MindDataTestExecute, TestBandBiquadWithEager) {
  MS_LOG(INFO) << "Basic Function Test With Eager.";
  // Original waveform
  std::vector<float> labels = {
    2.716064453125000000e-03, 6.347656250000000000e-03, 9.246826171875000000e-03, 1.089477539062500000e-02,
    1.138305664062500000e-02, 1.156616210937500000e-02, 1.394653320312500000e-02, 1.550292968750000000e-02,
    1.614379882812500000e-02, 1.840209960937500000e-02, 1.718139648437500000e-02, 1.599121093750000000e-02,
    1.647949218750000000e-02, 1.510620117187500000e-02, 1.385498046875000000e-02, 1.345825195312500000e-02,
    1.419067382812500000e-02, 1.284790039062500000e-02, 1.052856445312500000e-02, 9.368896484375000000e-03};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({2, 10}), &input));
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> band_biquad_01 = std::make_shared<audio::BandBiquad>(44100, 200);
  mindspore::dataset::Execute Transform01({band_biquad_01});
  // Filtered waveform by bandbiquad
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestBandBiquadWithWrongArg) {
  MS_LOG(INFO) << "Wrong Arg.";
  std::vector<double> labels = {
    2.716064453125000000e-03, 6.347656250000000000e-03, 9.246826171875000000e-03, 1.089477539062500000e-02,
    1.138305664062500000e-02, 1.156616210937500000e-02, 1.394653320312500000e-02, 1.550292968750000000e-02,
    1.614379882812500000e-02, 1.840209960937500000e-02, 1.718139648437500000e-02, 1.599121093750000000e-02,
    1.647949218750000000e-02, 1.510620117187500000e-02, 1.385498046875000000e-02, 1.345825195312500000e-02,
    1.419067382812500000e-02, 1.284790039062500000e-02, 1.052856445312500000e-02, 9.368896484375000000e-03};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({2, 10}), &input));
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  // Check Q
  MS_LOG(INFO) << "Q is zero.";
  std::shared_ptr<TensorTransform> band_biquad_op = std::make_shared<audio::BandBiquad>(44100, 200, 0);
  mindspore::dataset::Execute Transform01({band_biquad_op});
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_FALSE(s01.IsOk());
}
