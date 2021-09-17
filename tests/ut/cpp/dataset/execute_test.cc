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
#include "minddata/dataset/include/dataset/audio.h"
#include "minddata/dataset/include/dataset/vision.h"
#include "minddata/dataset/include/dataset/audio.h"
#include "minddata/dataset/include/dataset/text.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestExecute : public UT::DatasetOpTesting {
 protected:
};

TEST_F(MindDataTestExecute, TestAllpassBiquadWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestAllpassBiquadWithEager.";
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
  std::shared_ptr<TensorTransform> allpass_biquad_01 = std::make_shared<audio::AllpassBiquad>(44100, 200);
  mindspore::dataset::Execute Transform01({allpass_biquad_01});
  // Filtered waveform by allpassbiquad
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestAllpassBiquadWithWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestAllpassBiquadWithWrongArg.";
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
  std::shared_ptr<TensorTransform> allpass_biquad_op = std::make_shared<audio::AllpassBiquad>(44100, 200, 0);
  mindspore::dataset::Execute Transform01({allpass_biquad_op});
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_FALSE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestAdjustGammaEager3Channel) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestAdjustGammaEager3Channel.";
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto adjust_gamma_op = vision::AdjustGamma(0.1, 1.0);

  auto transform = Execute({decode, adjust_gamma_op});
  Status rc = transform(image, &image);
  EXPECT_EQ(rc, Status::OK());
}

TEST_F(MindDataTestExecute, TestAdjustGammaEager1Channel) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestAdjustGammaEager1Channel.";
  auto m1 = ReadFileToTensor("data/dataset/apple.jpg");
  // Transform params
  auto decode = vision::Decode();
  auto rgb2gray = vision::RGB2GRAY();
  auto adjust_gamma_op = vision::AdjustGamma(0.1, 1.0);

  auto transform = Execute({decode, rgb2gray, adjust_gamma_op});
  Status rc = transform(m1, &m1);
  EXPECT_EQ(rc, Status::OK());
}

TEST_F(MindDataTestExecute, TestAmplitudeToDB) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestAmplitudeToDB.";
  // Original waveform
  std::vector<float> labels = {
    2.716064453125000000e-03, 6.347656250000000000e-03, 9.246826171875000000e-03, 1.089477539062500000e-02,
    1.138305664062500000e-02, 1.156616210937500000e-02, 1.394653320312500000e-02, 1.550292968750000000e-02,
    1.614379882812500000e-02, 1.840209960937500000e-02, 1.718139648437500000e-02, 1.599121093750000000e-02,
    1.647949218750000000e-02, 1.510620117187500000e-02, 1.385498046875000000e-02, 1.345825195312500000e-02,
    1.419067382812500000e-02, 1.284790039062500000e-02, 1.052856445312500000e-02, 9.368896484375000000e-03,
    1.419067382812500000e-02, 1.284790039062500000e-02, 1.052856445312500000e-02, 9.368896484375000000e-03};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({2, 2, 2, 3}), &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> amplitude_to_db_op = std::make_shared<audio::AmplitudeToDB>();
  // apply amplitude_to_db
  mindspore::dataset::Execute trans({amplitude_to_db_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_TRUE(status.IsOk());
}

TEST_F(MindDataTestExecute, TestAmplitudeToDBWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestAmplitudeToDBWrongArgs.";
  // Original waveform
  std::vector<float> labels = {
    2.716064453125000000e-03, 6.347656250000000000e-03, 9.246826171875000000e-03, 1.089477539062500000e-02,
    1.138305664062500000e-02, 1.156616210937500000e-02, 1.394653320312500000e-02, 1.550292968750000000e-02,
    1.614379882812500000e-02, 1.840209960937500000e-02, 1.718139648437500000e-02, 1.599121093750000000e-02,
    1.647949218750000000e-02, 1.510620117187500000e-02, 1.385498046875000000e-02, 1.345825195312500000e-02,
    1.419067382812500000e-02, 1.284790039062500000e-02, 1.052856445312500000e-02, 9.368896484375000000e-03};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({2, 10}), &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> amplitude_to_db_op =
    std::make_shared<audio::AmplitudeToDB>(ScaleType::kPower, 1.0, -1e-10, 80.0);
  // apply amplitude_to_db
  mindspore::dataset::Execute trans({amplitude_to_db_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_FALSE(status.IsOk());
}

TEST_F(MindDataTestExecute, TestAmplitudeToDBWrongInput) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestAmplitudeToDBWrongInput.";
  // Original waveform
  std::vector<float> labels = {
    2.716064453125000000e-03, 6.347656250000000000e-03, 9.246826171875000000e-03, 1.089477539062500000e-02,
    1.138305664062500000e-02, 1.156616210937500000e-02, 1.394653320312500000e-02, 1.550292968750000000e-02,
    1.614379882812500000e-02, 1.840209960937500000e-02, 1.718139648437500000e-02, 1.599121093750000000e-02,
    1.647949218750000000e-02, 1.510620117187500000e-02, 1.385498046875000000e-02, 1.345825195312500000e-02,
    1.419067382812500000e-02, 1.284790039062500000e-02, 1.052856445312500000e-02, 9.368896484375000000e-03};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({20}), &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> amplitude_to_db_op = std::make_shared<audio::AmplitudeToDB>();
  // apply amplitude_to_db
  mindspore::dataset::Execute trans({amplitude_to_db_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_FALSE(status.IsOk());
}

TEST_F(MindDataTestExecute, TestComposeTransforms) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestComposeTransforms.";

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

TEST_F(MindDataTestExecute, TestFrequencyMasking) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestFrequencyMasking.";
  std::shared_ptr<Tensor> input_tensor_;
  TensorShape s = TensorShape({6, 2});
  ASSERT_OK(Tensor::CreateFromVector(
    std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}), s, &input_tensor_));
  auto input_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor_));
  std::shared_ptr<TensorTransform> frequency_masking_op = std::make_shared<audio::FrequencyMasking>(true, 2);
  mindspore::dataset::Execute transform({frequency_masking_op});
  Status status = transform(input_tensor, &input_tensor);
  EXPECT_TRUE(status.IsOk());
}

TEST_F(MindDataTestExecute, TestTimeMasking) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestTimeMasking.";
  std::shared_ptr<Tensor> input_tensor_;
  TensorShape s = TensorShape({2, 6});
  ASSERT_OK(Tensor::CreateFromVector(
    std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}), s, &input_tensor_));
  auto input_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor_));
  std::shared_ptr<TensorTransform> time_masking_op = std::make_shared<audio::TimeMasking>(true, 2);
  mindspore::dataset::Execute transform({time_masking_op});
  Status status = transform(input_tensor, &input_tensor);
  EXPECT_TRUE(status.IsOk());
}

TEST_F(MindDataTestExecute, TestTimeStretchEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestTimeStretchEager.";
  std::shared_ptr<Tensor> input_tensor_;
  // op param
  int freq = 4;
  int hop_length = 20;
  float rate = 1.3;
  int frame_num = 10;
  // create tensor
  TensorShape s = TensorShape({2, freq, frame_num, 2});
  // init input vec
  std::vector<float> input_vec(2 * freq * frame_num * 2);
  for (int ind = 0; ind < input_vec.size(); ind++) {
    input_vec[ind] = std::rand() % (1000) / (1000.0f);
  }
  ASSERT_OK(Tensor::CreateFromVector(input_vec, s, &input_tensor_));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor_));
  std::shared_ptr<TensorTransform> time_stretch_op = std::make_shared<audio::TimeStretch>(hop_length, freq, rate);

  // apply timestretch
  mindspore::dataset::Execute Transform({time_stretch_op});
  Status status = Transform(input_ms, &input_ms);
  EXPECT_TRUE(status.IsOk());
}

TEST_F(MindDataTestExecute, TestTimeStretchParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestTimeStretch-TestTimeStretchParamCheck.";
  // Create an input
  std::shared_ptr<Tensor> input_tensor_;
  std::shared_ptr<Tensor> output_tensor;
  TensorShape s = TensorShape({1, 4, 3, 2});
  ASSERT_OK(Tensor::CreateFromVector(
    std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}),
    s, &input_tensor_));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor_));

  std::shared_ptr<TensorTransform> time_stretch1 = std::make_shared<audio::TimeStretch>(4, 512, -2);
  mindspore::dataset::Execute Transform1({time_stretch1});
  Status status = Transform1(input_ms, &input_ms);
  EXPECT_FALSE(status.IsOk());

  std::shared_ptr<TensorTransform> time_stretch2 = std::make_shared<audio::TimeStretch>(4, -512, 2);
  mindspore::dataset::Execute Transform2({time_stretch2});
  status = Transform2(input_ms, &input_ms);
  EXPECT_FALSE(status.IsOk());
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
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestBandBiquadWithEager.";
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
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestBandBiquadWithWrongArg.";
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

TEST_F(MindDataTestExecute, TestBandpassBiquadWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestBandpassBiquadWithEager.";
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
  std::shared_ptr<TensorTransform> bandpass_biquad_01 = std::make_shared<audio::BandpassBiquad>(44100, 200);
  mindspore::dataset::Execute Transform01({bandpass_biquad_01});
  // Filtered waveform by bandpassbiquad
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestBandpassBiquadWithWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestBandpassBiquadWithWrongArg.";
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
  std::shared_ptr<TensorTransform> bandpass_biquad_op = std::make_shared<audio::BandpassBiquad>(44100, 200, 0);
  mindspore::dataset::Execute Transform01({bandpass_biquad_op});
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_FALSE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestBandrejectBiquadWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestBandrejectBiquadWithEager.";
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
  std::shared_ptr<TensorTransform> bandreject_biquad_01 = std::make_shared<audio::BandrejectBiquad>(44100, 200);
  mindspore::dataset::Execute Transform01({bandreject_biquad_01});
  // Filtered waveform by bandrejectbiquad
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestBandrejectBiquadWithWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestBandrejectBiquadWithWrongArg.";
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
  std::shared_ptr<TensorTransform> bandreject_biquad_op = std::make_shared<audio::BandrejectBiquad>(44100, 200, 0);
  mindspore::dataset::Execute Transform01({bandreject_biquad_op});
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_FALSE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestAngleEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestAngleEager.";
  std::vector<double> origin = {1.143, 1.3123, 2.632, 2.554, -1.213, 1.3, 0.456, 3.563};
  TensorShape input_shape({4, 2});
  std::shared_ptr<Tensor> de_tensor;
  Tensor::CreateFromVector(origin, input_shape, &de_tensor);

  std::shared_ptr<TensorTransform> angle = std::make_shared<audio::Angle>();
  auto input = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));
  mindspore::dataset::Execute Transform({angle});
  Status s = Transform(input, &input);

  ASSERT_TRUE(s.IsOk());
}

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

TEST_F(MindDataTestExecute, TestEqualizerBiquadEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestEqualizerBiquadEager.";
  int sample_rate = 44100;
  float center_freq = 3.5;
  float gain = 5.5;
  float Q = 0.707;
  std::vector<mindspore::MSTensor> output;
  std::shared_ptr<Tensor> test;
  std::vector<double> test_vector = {0.8236, 0.2049, 0.3335, 0.5933, 0.9911, 0.2482, 0.3007, 0.9054,
                                     0.7598, 0.5394, 0.2842, 0.5634, 0.6363, 0.2226, 0.2288};
  Tensor::CreateFromVector(test_vector, TensorShape({5, 3}), &test);
  auto input = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(test));
  std::shared_ptr<TensorTransform> equalizer_biquad(new audio::EqualizerBiquad({sample_rate, center_freq, gain, Q}));
  auto transform = Execute({equalizer_biquad});
  Status rc = transform({input}, &output);
  ASSERT_TRUE(rc.IsOk());
}

TEST_F(MindDataTestExecute, TestEqualizerBiquadParamCheckQ) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestEqualizerBiquadParamCheckQ.";
  std::vector<mindspore::MSTensor> output;
  std::shared_ptr<Tensor> test;
  std::vector<double> test_vector = {0.1129, 0.3899, 0.7762, 0.2437, 0.9911, 0.8764, 0.4524, 0.9034,
                                     0.3277, 0.8904, 0.1852, 0.6721, 0.1325, 0.2345, 0.5538};
  Tensor::CreateFromVector(test_vector, TensorShape({3, 5}), &test);
  auto input = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(test));
  // Check Q
  std::shared_ptr<TensorTransform> equalizer_biquad_op = std::make_shared<audio::EqualizerBiquad>(44100, 3.5, 5.5, 0);
  mindspore::dataset::Execute transform({equalizer_biquad_op});
  Status rc = transform({input}, &output);
  ASSERT_FALSE(rc.IsOk());
}

TEST_F(MindDataTestExecute, TestEqualizerBiquadParamCheckSampleRate) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestEqualizerBiquadParamCheckSampleRate.";
  std::vector<mindspore::MSTensor> output;
  std::shared_ptr<Tensor> test;
  std::vector<double> test_vector = {0.5236, 0.7049, 0.4335, 0.4533, 0.0911, 0.3482, 0.3407, 0.9054,
                                     0.7598, 0.5394, 0.2842, 0.5634, 0.6363, 0.2226, 0.2288, 0.6743};
  Tensor::CreateFromVector(test_vector, TensorShape({4, 4}), &test);
  auto input = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(test));
  // Check sample_rate
  std::shared_ptr<TensorTransform> equalizer_biquad_op = std::make_shared<audio::EqualizerBiquad>(0, 3.5, 5.5, 0.7);
  mindspore::dataset::Execute transform({equalizer_biquad_op});
  Status rc = transform({input}, &output);
  ASSERT_FALSE(rc.IsOk());
}

TEST_F(MindDataTestExecute, TestLowpassBiquadEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestLowpassBiquadEager.";
  int sample_rate = 44100;
  float cutoff_freq = 2000.0;
  float Q = 0.6;
  std::vector<mindspore::MSTensor> output;
  std::shared_ptr<Tensor> test;
  std::vector<double> test_vector = {23.5, 13.2, 62.5, 27.1, 15.5, 30.3, 44.9, 25.0,
                                     11.3, 37.4, 67.1, 33.8, 73.4, 53.3, 93.7, 31.1};
  Tensor::CreateFromVector(test_vector, TensorShape({4, 4}), &test);
  auto input = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(test));
  std::shared_ptr<TensorTransform> lowpass_biquad(new audio::LowpassBiquad({sample_rate, cutoff_freq, Q}));
  auto transform = Execute({lowpass_biquad});
  Status rc = transform({input}, &output);
  ASSERT_TRUE(rc.IsOk());
}

TEST_F(MindDataTestExecute, TestLowpassBiuqadParamCheckQ) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestLowpassBiuqadParamCheckQ.";

  std::vector<mindspore::MSTensor> output;
  std::shared_ptr<Tensor> test;
  std::vector<double> test_vector = {0.8236, 0.2049, 0.3335, 0.5933, 0.9911, 0.2482, 0.3007, 0.9054,
                                     0.7598, 0.5394, 0.2842, 0.5634, 0.6363, 0.2226, 0.2288};
  Tensor::CreateFromVector(test_vector, TensorShape({5, 3}), &test);
  auto input = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(test));
  // Check Q
  std::shared_ptr<TensorTransform> lowpass_biquad_op = std::make_shared<audio::LowpassBiquad>(44100, 3000.5, 0);
  mindspore::dataset::Execute transform({lowpass_biquad_op});
  Status rc = transform({input}, &output);
  ASSERT_FALSE(rc.IsOk());
}

TEST_F(MindDataTestExecute, TestLowpassBiuqadParamCheckSampleRate) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestLowpassBiuqadParamCheckSampleRate.";

  std::vector<mindspore::MSTensor> output;
  std::shared_ptr<Tensor> test;
  std::vector<double> test_vector = {0.5, 4.6, 2.2, 0.6, 1.9, 4.7, 2.3, 4.9, 4.7, 0.5, 0.8, 0.9};
  Tensor::CreateFromVector(test_vector, TensorShape({6, 2}), &test);
  auto input = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(test));
  // Check sample_rate
  std::shared_ptr<TensorTransform> lowpass_biquad_op = std::make_shared<audio::LowpassBiquad>(0, 2000.5, 0.7);
  mindspore::dataset::Execute transform({lowpass_biquad_op});
  Status rc = transform({input}, &output);
  ASSERT_FALSE(rc.IsOk());
}

TEST_F(MindDataTestExecute, TestComplexNormEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestComplexNormEager.";
  // testing
  std::shared_ptr<Tensor> input_tensor_;
  Tensor::CreateFromVector(std::vector<float>({1.0, 1.0, 2.0, 3.0, 4.0, 4.0}), TensorShape({3, 2}), &input_tensor_);

  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor_));
  std::shared_ptr<TensorTransform> complex_norm_01 = std::make_shared<audio::ComplexNorm>(4.0);

  // Filtered waveform by complexnorm
  mindspore::dataset::Execute Transform01({complex_norm_01});
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestContrastWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestContrastWithEager.";
  // Original waveform
  std::vector<float> labels = {4.11, 5.37, 5.85, 5.4, 4.27, 1.861, -1.1291, -4.76, 1.495};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({3, 3}), &input));
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> contrast_01 = std::make_shared<audio::Contrast>();
  mindspore::dataset::Execute Transform01({contrast_01});
  // Filtered waveform by contrast
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestContrastWithWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestContrastWithWrongArg.";
  std::vector<double> labels = {-1.007, -5.06, 7.934, 6.683, 1.312, 1.84, 2.246, 2.597};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({2, 4}), &input));
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  // Check enhancement_amount
  MS_LOG(INFO) << "enhancement_amount is negative.";
  std::shared_ptr<TensorTransform> contrast_op = std::make_shared<audio::Contrast>(-10);
  mindspore::dataset::Execute Transform01({contrast_op});
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_FALSE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestDeemphBiquadWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestDeemphBiquadWithEager";
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
  std::shared_ptr<TensorTransform> deemph_biquad_01 = std::make_shared<audio::DeemphBiquad>(44100);
  mindspore::dataset::Execute Transform01({deemph_biquad_01});
  // Filtered waveform by deemphbiquad
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestDeemphBiquadWithWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestDeemphBiquadWithWrongArg.";
  std::vector<double> labels = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({1, 6}), &input));
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  std::shared_ptr<TensorTransform> deemph_biquad_op = std::make_shared<audio::DeemphBiquad>(0);
  mindspore::dataset::Execute Transform01({deemph_biquad_op});
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_FALSE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestHighpassBiquadEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestHighpassBiquadEager.";
  int sample_rate = 44100;
  float cutoff_freq = 3000.5;
  float Q = 0.707;
  std::vector<mindspore::MSTensor> output;
  std::shared_ptr<Tensor> test;
  std::vector<double> test_vector = {0.8236, 0.2049, 0.3335, 0.5933, 0.9911, 0.2482, 0.3007, 0.9054,
                                     0.7598, 0.5394, 0.2842, 0.5634, 0.6363, 0.2226, 0.2288};
  Tensor::CreateFromVector(test_vector, TensorShape({5, 3}), &test);
  auto input = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(test));
  std::shared_ptr<TensorTransform> highpass_biquad(new audio::HighpassBiquad({sample_rate, cutoff_freq, Q}));
  auto transform = Execute({highpass_biquad});
  Status rc = transform({input}, &output);
  ASSERT_TRUE(rc.IsOk());
}

TEST_F(MindDataTestExecute, TestHighpassBiquadParamCheckQ) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestHighpassBiquadParamCheckQ.";
  std::vector<mindspore::MSTensor> output;
  std::shared_ptr<Tensor> test;
  std::vector<float> test_vector = {0.6013, 0.8081, 0.6600, 0.4278, 0.4049, 0.0541, 0.8800, 0.7143, 0.0926, 0.3502,
                                    0.6148, 0.8738, 0.1869, 0.9023, 0.4293, 0.2175, 0.5132, 0.2622, 0.6490, 0.0741,
                                    0.7903, 0.3428, 0.1598, 0.4841, 0.8128, 0.7409, 0.7226, 0.4951, 0.5589, 0.9210};
  Tensor::CreateFromVector(test_vector, TensorShape({5, 3, 2}), &test);
  auto input = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(test));
  // Check Q
  std::shared_ptr<TensorTransform> highpass_biquad_op = std::make_shared<audio::HighpassBiquad>(44100, 3000.5, 0);
  mindspore::dataset::Execute transform({highpass_biquad_op});
  Status rc = transform({input}, &output);
  ASSERT_FALSE(rc.IsOk());
}

TEST_F(MindDataTestExecute, TestHighpassBiquadParamCheckSampleRate) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestHighpassBiquadParamCheckSampleRate.";
  std::vector<mindspore::MSTensor> output;
  std::shared_ptr<Tensor> test;
  std::vector<double> test_vector = {0.0237, 0.6026, 0.3801, 0.1978, 0.8672, 0.0095, 0.5166, 0.2641, 0.5485, 0.5144};
  Tensor::CreateFromVector(test_vector, TensorShape({1, 10}), &test);
  auto input = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(test));
  // Check sample_rate
  std::shared_ptr<TensorTransform> highpass_biquad_op = std::make_shared<audio::HighpassBiquad>(0, 3000.5, 0.7);
  mindspore::dataset::Execute transform({highpass_biquad_op});
  Status rc = transform({input}, &output);
  ASSERT_FALSE(rc.IsOk());
}

TEST_F(MindDataTestExecute, TestMuLawDecodingEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestMuLawDecodingEager.";
  // testing
  std::shared_ptr<Tensor> input_tensor_;
  Tensor::CreateFromVector(std::vector<float>({1, 254, 231, 155, 101, 77}), TensorShape({1, 6}), &input_tensor_);

  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor_));
  std::shared_ptr<TensorTransform> mu_law_encoding_01 = std::make_shared<audio::MuLawDecoding>(255);

  // Filtered waveform by mulawencoding
  mindspore::dataset::Execute Transform01({mu_law_encoding_01});
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestLFilterWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestLFilterWithEager.";
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
  std::vector<float> a_coeffs = {0.1, 0.2, 0.3};
  std::vector<float> b_coeffs = {0.1, 0.2, 0.3};
  std::shared_ptr<TensorTransform> lfilter_01 = std::make_shared<audio::LFilter>(a_coeffs, b_coeffs);
  mindspore::dataset::Execute Transform01({lfilter_01});
  // Filtered waveform by lfilter
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestLFilterWithWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestLFilterWithWrongArg.";
  std::vector<double> labels = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({1, 6}), &input));
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));

  // Check a_coeffs size equal to b_coeffs
  MS_LOG(INFO) << "a_coeffs size not equal to b_coeffs";
  std::vector<float> a_coeffs = {0.1, 0.2, 0.3};
  std::vector<float> b_coeffs = {0.1, 0.2};
  std::shared_ptr<TensorTransform> lfilter_op = std::make_shared<audio::LFilter>(a_coeffs, b_coeffs);
  mindspore::dataset::Execute Transform01({lfilter_op});
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_FALSE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestDCShiftEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestDCShiftEager.";

  std::vector<float> origin = {0.67443, 1.87523, 0.73465, -0.74553, -1.54346, 1.54093, -1.23453};
  std::shared_ptr<Tensor> de_tensor;
  Tensor::CreateFromVector(origin, &de_tensor);

  std::shared_ptr<TensorTransform> dc_shift = std::make_shared<audio::DCShift>(0.5, 0.02);
  auto input = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));
  mindspore::dataset::Execute Transform({dc_shift});
  Status s = Transform(input, &input);
  ASSERT_TRUE(s.IsOk());
}

TEST_F(MindDataTestExecute, TestBiquadWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestBiquadWithEager.";
  // Original waveform
  std::vector<float> labels = {3.716064453125,  12.34765625,     5.246826171875,  1.0894775390625,
                               1.1383056640625, 2.1566162109375, 1.3946533203125, 3.55029296875};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({2, 4}), &input));
  auto input_01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> biquad_01 = std::make_shared<audio::Biquad>(1, 0.02, 0.13, 1, 0.12, 0.3);
  mindspore::dataset::Execute Transform01({biquad_01});
  // Filtered waveform by biquad
  Status s01 = Transform01(input_01, &input_01);
  EXPECT_TRUE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestBiquadWithWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestBiquadWithWrongArg.";
  std::vector<double> labels = {
    2.716064453125000000e-03,
    6.347656250000000000e-03,
    9.246826171875000000e-03,
    1.089477539062500000e-02,
  };
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({1, 4}), &input));
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  // Check a0
  MS_LOG(INFO) << "a0 is zero.";
  std::shared_ptr<TensorTransform> biquad_op = std::make_shared<audio::Biquad>(1, 0.02, 0.13, 0, 0.12, 0.3);
  mindspore::dataset::Execute Transform01({biquad_op});
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_FALSE(s01.IsOk());
}

TEST_F(MindDataTestExecute, TestFade) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestFade.";
  std::vector<float> waveform = {
    2.716064453125000000e-03, 6.347656250000000000e-03, 9.246826171875000000e-03, 1.089477539062500000e-02,
    1.138305664062500000e-02, 1.156616210937500000e-02, 1.394653320312500000e-02, 1.550292968750000000e-02,
    1.614379882812500000e-02, 1.840209960937500000e-02, 1.718139648437500000e-02, 1.599121093750000000e-02,
    1.647949218750000000e-02, 1.510620117187500000e-02, 1.385498046875000000e-02, 1.345825195312500000e-02,
    1.419067382812500000e-02, 1.284790039062500000e-02, 1.052856445312500000e-02, 9.368896484375000000e-03};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(waveform, TensorShape({1, 20}), &input));
  auto input_01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> fade01 = std::make_shared<audio::Fade>(5, 6, FadeShape::kLinear);
  mindspore::dataset::Execute Transform01({fade01});
  Status s01 = Transform01(input_01, &input_01);
  EXPECT_TRUE(s01.IsOk());
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> fade02 = std::make_shared<audio::Fade>(5, 6, FadeShape::kQuarterSine);
  mindspore::dataset::Execute Transform02({fade02});
  Status s02 = Transform02(input_02, &input_02);
  EXPECT_TRUE(s02.IsOk());
  auto input_03 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> fade03 = std::make_shared<audio::Fade>(5, 6, FadeShape::kExponential);
  mindspore::dataset::Execute Transform03({fade03});
  Status s03 = Transform03(input_03, &input_03);
  EXPECT_TRUE(s03.IsOk());
  auto input_04 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> fade04 = std::make_shared<audio::Fade>(5, 6, FadeShape::kHalfSine);
  mindspore::dataset::Execute Transform04({fade04});
  Status s04 = Transform01(input_04, &input_04);
  EXPECT_TRUE(s04.IsOk());
  auto input_05 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> fade05 = std::make_shared<audio::Fade>(5, 6, FadeShape::kLogarithmic);
  mindspore::dataset::Execute Transform05({fade05});
  Status s05 = Transform01(input_05, &input_05);
  EXPECT_TRUE(s05.IsOk());
}

TEST_F(MindDataTestExecute, TestFadeDefaultArg) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestFadeDefaultArg.";
  std::vector<double> waveform = {
    1.573897564868000000e-03, 5.462374385400000000e-03, 3.584989689205400000e-03, 2.035667767462500000e-02,
    2.353543454062500000e-02, 1.256616210937500000e-02, 2.394653320312500000e-02, 5.243553968750000000e-02,
    2.434554533002500000e-02, 3.454566960937500000e-02, 2.343545454437500000e-02, 2.534343093750000000e-02,
    2.354465654550000000e-02, 1.453545517187500000e-02, 1.454645535875000000e-02, 1.433243195312500000e-02,
    1.434354554812500000e-02, 3.343435276865400000e-02, 1.234257687312500000e-02, 5.368896484375000000e-03};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(waveform, TensorShape({2, 10}), &input));
  auto input_01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> fade01 = std::make_shared<audio::Fade>();
  mindspore::dataset::Execute Transform01({fade01});
  Status s01 = Transform01(input_01, &input_01);
  EXPECT_TRUE(s01.IsOk());
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> fade02 = std::make_shared<audio::Fade>(5);
  mindspore::dataset::Execute Transform02({fade02});
  Status s02 = Transform02(input_02, &input_02);
  EXPECT_TRUE(s02.IsOk());
  auto input_03 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> fade03 = std::make_shared<audio::Fade>(5, 6);
  mindspore::dataset::Execute Transform03({fade03});
  Status s03 = Transform03(input_03, &input_03);
  EXPECT_TRUE(s03.IsOk());
}

TEST_F(MindDataTestExecute, TestFadeWithInvalidArg) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestFadeWithInvalidArg.";
  std::vector<float> waveform = {
    2.716064453125000000e-03, 6.347656250000000000e-03, 9.246826171875000000e-03, 1.089477539062500000e-02,
    1.138305664062500000e-02, 1.156616210937500000e-02, 1.394653320312500000e-02, 1.550292968750000000e-02,
    1.614379882812500000e-02, 1.840209960937500000e-02, 1.718139648437500000e-02, 1.599121093750000000e-02,
    1.647949218750000000e-02, 1.510620117187500000e-02, 1.385498046875000000e-02, 1.345825195312500000e-02,
    1.419067382812500000e-02, 1.284790039062500000e-02, 1.052856445312500000e-02, 9.368896484375000000e-03};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(waveform, TensorShape({1, 20}), &input));
  auto input_01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> fade1 = std::make_shared<audio::Fade>(-5, 6);
  mindspore::dataset::Execute Transform01({fade1});
  Status s01 = Transform01(input_01, &input_01);
  EXPECT_FALSE(s01.IsOk());
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> fade2 = std::make_shared<audio::Fade>(0, -1);
  mindspore::dataset::Execute Transform02({fade2});
  Status s02 = Transform02(input_02, &input_02);
  EXPECT_FALSE(s02.IsOk());
  auto input_03 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> fade3 = std::make_shared<audio::Fade>(30, 10);
  mindspore::dataset::Execute Transform03({fade3});
  Status s03 = Transform03(input_03, &input_03);
  EXPECT_FALSE(s03.IsOk());
  auto input_04 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> fade4 = std::make_shared<audio::Fade>(10, 30);
  mindspore::dataset::Execute Transform04({fade4});
  Status s04 = Transform04(input_04, &input_04);
  EXPECT_FALSE(s04.IsOk());
}
TEST_F(MindDataTestExecute, TestVolDefalutValue) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestVolDefalutValue.";
  std::shared_ptr<Tensor> input_tensor_;
  TensorShape s = TensorShape({2, 6});
  ASSERT_OK(Tensor::CreateFromVector(
    std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}), s, &input_tensor_));
  auto input_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor_));
  std::shared_ptr<TensorTransform> vol_op = std::make_shared<audio::Vol>(0.333);
  mindspore::dataset::Execute transform({vol_op});
  Status status = transform(input_tensor, &input_tensor);
  EXPECT_TRUE(status.IsOk());
}

TEST_F(MindDataTestExecute, TestVolGainTypePower) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestVolGainTypePower.";
  std::shared_ptr<Tensor> input_tensor_;
  TensorShape s = TensorShape({4, 3});
  ASSERT_OK(Tensor::CreateFromVector(
    std::vector<double>({4.0f, 5.0f, 3.0f, 5.0f, 4.0f, 6.0f, 6.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f}), s, &input_tensor_));
  auto input_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor_));
  std::shared_ptr<TensorTransform> vol_op = std::make_shared<audio::Vol>(0.2, GainType::kPower);
  mindspore::dataset::Execute transform({vol_op});
  Status status = transform(input_tensor, &input_tensor);
  EXPECT_TRUE(status.IsOk());
}

TEST_F(MindDataTestExecute, TestMagphaseEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestMagphaseEager.";
  float power = 1.0;
  std::vector<mindspore::MSTensor> output_tensor;
  std::shared_ptr<Tensor> test;
  std::vector<float> test_vector = {3, 4, -3, 4, 3, -4, -3, -4,
                                    5, 12, -5, 12, 5, -12, -5, -12};
  Tensor::CreateFromVector(test_vector, TensorShape({2, 4, 2}), &test);
  auto input_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(test));
  std::shared_ptr<TensorTransform> magphase(new audio::Magphase({power}));
  auto transform = Execute({magphase});
  Status rc = transform({input_tensor}, &output_tensor);
  ASSERT_TRUE(rc.IsOk());
}
