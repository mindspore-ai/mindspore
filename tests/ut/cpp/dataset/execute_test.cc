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
#include "include/api/types.h"
#include "minddata/dataset/core/de_tensor.h"
#include "minddata/dataset/include/dataset/audio.h"
#include "minddata/dataset/include/dataset/execute.h"
#include "minddata/dataset/include/dataset/text.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/include/dataset/vision.h"
#include "minddata/dataset/text/char_n_gram.h"
#include "minddata/dataset/text/fast_text.h"
#include "minddata/dataset/text/glove.h"
#include "minddata/dataset/text/vectors.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::dataset::CharNGram;
using mindspore::dataset::FastText;
using mindspore::dataset::GloVe;
using mindspore::dataset::Vectors;

class MindDataTestExecute : public UT::DatasetOpTesting {
 protected:
};

/// Feature: Execute Transform op
/// Description: Test executing AllpassBiquad op in eager mode
/// Expectation: The data is processed successfully
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

/// Feature: Execute Transform op
/// Description: Test executing AllpassBiquad op with wrong arguments
/// Expectation: Throw correct error and message
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

/// Feature: Execute Transform op
/// Description: Test executing AdjustGamma op in eager mode with dataset with 3 channels
/// Expectation: The data is processed successfully
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

/// Feature: Execute Transform op
/// Description: Test executing AdjustGamma op in eager mode with dataset transformed to 1 channel only
/// Expectation: The data is processed successfully
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

/// Feature: Execute Transform op
/// Description: Test executing AmplitudeToDB with basic usage
/// Expectation: The data is processed successfully
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

/// Feature: Execute Transform op
/// Description: Test executing AmplitudeToDB with wrong arguments
/// Expectation: Throw correct error and message
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

/// Feature: Execute Transform op
/// Description: Test executing AmplitudeToDB with no arguments
/// Expectation: Throw correct error and message
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

/// Feature: ComputeDeltas op
/// Description: Test basic function of ComputeDeltas op
/// Expectation: Get correct number of data
TEST_F(MindDataTestExecute, TestComputeDeltas) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestComputeDeltas.";
  std::shared_ptr<Tensor> input_tensor;

  int win_length = 5;

  // create tensor
  TensorShape s = TensorShape({2, 15, 7});
  // init input vec
  std::vector<float> input_vec(s.NumOfElements());
  for (int ind = 0; ind < input_vec.size(); ind++) {
    input_vec[ind] = std::rand() % (1000) / (1000.0f);
  }
  ASSERT_OK(Tensor::CreateFromVector(input_vec, s, &input_tensor));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor));
  std::shared_ptr<TensorTransform> compute_deltas_op = std::make_shared<audio::ComputeDeltas>(win_length);

  // apply compute_deltas
  mindspore::dataset::Execute Transform({compute_deltas_op});
  Status status = Transform(input_ms, &input_ms);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: ComputeDeltas op
/// Description: Test wrong input args of ComputeDeltas op
/// Expectation: Get nullptr of iterator
TEST_F(MindDataTestExecute, TestComputeDeltasWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestComputeDeltasWrongArgs.";
  std::shared_ptr<Tensor> input_tensor;
  // win_length is less than minimum of 3
  int win_length = 2;

  // create tensor
  TensorShape s = TensorShape({2, 15, 7});
  // init input vec
  std::vector<float> input_vec(s.NumOfElements());
  for (int ind = 0; ind < input_vec.size(); ind++) {
    input_vec[ind] = std::rand() % (1000) / (1000.0f);
  }
  ASSERT_OK(Tensor::CreateFromVector(input_vec, s, &input_tensor));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor));

  std::shared_ptr<TensorTransform> compute_deltas_op = std::make_shared<audio::ComputeDeltas>(win_length);
  mindspore::dataset::Execute Transform({compute_deltas_op});
  Status status = Transform(input_ms, &input_ms);
  EXPECT_FALSE(status.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing Crop op after Decode op
/// Expectation: Output is equal to the expected output and status is okay
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

/// Feature: FilterWikipediaXMLEager op
/// Description: Test FilterWikipediaXML's Eager mode
/// Expectation: Run successfully
TEST_F(MindDataTestExecute, TestFilterWikipediaXMLEager) {
  // Test FilterWikipediaXML's Eager mode
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestFilterWikipediaXMLEager.";
  std::vector<std::string> origin = {"中国","Wcdma","Pang","Yuchao"};
  TensorShape input_shape({2, 2});
  std::shared_ptr<Tensor> de_tensor;
  Tensor::CreateFromVector(origin, input_shape, &de_tensor);
  std::shared_ptr<TensorTransform> filter = std::make_shared<text::FilterWikipediaXML>();
  auto input = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));
  mindspore::dataset::Execute Transform({filter});
  Status s = Transform(input, &input);

  ASSERT_TRUE(s.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing FrequencyMasking op basic usage
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestFrequencyMasking) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestFrequencyMasking.";
  std::shared_ptr<Tensor> input;
  TensorShape s = TensorShape({6, 2});
  ASSERT_OK(Tensor::CreateFromVector(
    std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}), s, &input));
  auto input_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> frequency_masking_op = std::make_shared<audio::FrequencyMasking>(true, 2);
  mindspore::dataset::Execute transform({frequency_masking_op});
  Status status = transform(input_tensor, &input_tensor);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: RandomLighting op
/// Description: Test RandomLighting Op when alpha=0.1
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestRandomLighting) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestRandomLighting.";
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto random_lighting_op = vision::RandomLighting(0.1);

  auto transform = Execute({decode, random_lighting_op});
  Status rc = transform(image, &image);
  EXPECT_EQ(rc, Status::OK());
}

/// Feature: Execute Transform op
/// Description: Test executing TimeMasking op basic usage
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestTimeMasking) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestTimeMasking.";
  std::shared_ptr<Tensor> input;
  TensorShape s = TensorShape({2, 6});
  ASSERT_OK(Tensor::CreateFromVector(
    std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}), s, &input));
  auto input_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> time_masking_op = std::make_shared<audio::TimeMasking>(true, 2);
  mindspore::dataset::Execute transform({time_masking_op});
  Status status = transform(input_tensor, &input_tensor);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing TimeStretch op basic usage in eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestTimeStretchEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestTimeStretchEager.";
  std::shared_ptr<Tensor> input_tensor;
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
  ASSERT_OK(Tensor::CreateFromVector(input_vec, s, &input_tensor));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor));
  std::shared_ptr<TensorTransform> time_stretch_op = std::make_shared<audio::TimeStretch>(hop_length, freq, rate);

  // apply timestretch
  mindspore::dataset::Execute Transform({time_stretch_op});
  Status status = Transform(input_ms, &input_ms);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing TimeStretch op with invalid parameters
/// Expectation: Throw correct error and message
TEST_F(MindDataTestExecute, TestTimeStretchParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestTimeStretch-TestTimeStretchParamCheck.";
  // Create an input
  std::shared_ptr<Tensor> input_tensor;
  std::shared_ptr<Tensor> output_tensor;
  TensorShape s = TensorShape({1, 4, 3, 2});
  ASSERT_OK(Tensor::CreateFromVector(
    std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
                        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}),
    s, &input_tensor));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor));

  std::shared_ptr<TensorTransform> time_stretch1 = std::make_shared<audio::TimeStretch>(4, 512, -2);
  mindspore::dataset::Execute Transform1({time_stretch1});
  Status status = Transform1(input_ms, &input_ms);
  EXPECT_FALSE(status.IsOk());

  std::shared_ptr<TensorTransform> time_stretch2 = std::make_shared<audio::TimeStretch>(4, -512, 2);
  mindspore::dataset::Execute Transform2({time_stretch2});
  status = Transform2(input_ms, &input_ms);
  EXPECT_FALSE(status.IsOk());
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
  auto normalize = std::make_shared<vision::Normalize>(
    std::vector<float>{0.485 * 255, 0.456 * 255, 0.406 * 255}, 
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
  auto normalize = std::make_shared<vision::Normalize>(
    std::vector<float>{0.485 * 255, 0.456 * 255, 0.406 * 255}, 
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
  auto normalize = std::make_shared<vision::Normalize>(
    std::vector<float>{0.485 * 255, 0.456 * 255, 0.406 * 255}, 
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
/// Description: Test executing UniformAugment op basic usage
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestUniformAugment) {
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");
  std::vector<mindspore::MSTensor> image2;

  // Transform params
  std::shared_ptr<TensorTransform> decode = std::make_shared<vision::Decode>();
  auto resize_op = std::make_shared<vision::Resize>(std::vector<int32_t>{16, 16});
  std::shared_ptr<TensorTransform> vertical = std::make_shared<vision::RandomVerticalFlip>();
  std::shared_ptr<TensorTransform> horizontal = std::make_shared<vision::RandomHorizontalFlip>();

  auto uniform_op = std::make_shared<vision::UniformAugment>(
    std::vector<std::shared_ptr<TensorTransform>>{resize_op, vertical, horizontal}, 3);

  auto transform1 = Execute({decode});
  Status rc = transform1(image, &image);
  ASSERT_TRUE(rc.IsOk());

  auto transform2 = Execute(std::vector<std::shared_ptr<TensorTransform>>{uniform_op});
  rc = transform2({image}, &image2);
  ASSERT_TRUE(rc.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing BasicTokenizer op basic usage
/// Expectation: Status is okay and output is equal to the expected output
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

/// Feature: Execute Transform op
/// Description: Test executing Decode op then Rotate op
/// Expectation: The data is processed successfully
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

/// Feature: Execute Transform op
/// Description: Test executing Decode op then ResizeWithBBox op
/// Expectation: Throw correct error and message
TEST_F(MindDataTestExecute, TestResizeWithBBox) {
  auto image = ReadFileToTensor("data/dataset/apple.jpg");
  std::shared_ptr<TensorTransform> decode_op = std::make_shared<vision::Decode>();
  std::shared_ptr<TensorTransform> resizewithbbox_op =
    std::make_shared<vision::ResizeWithBBox>(std::vector<int32_t>{250, 500});

  // Test Compute(Tensor, Tensor) method of ResizeWithBBox
  auto transform = Execute({decode_op, resizewithbbox_op});

  // Expect fail since Compute(Tensor, Tensor) is not a valid behavior for this Op,
  // while Compute(TensorRow, TensorRow) is the correct one.
  Status rc = transform(image, &image);
  EXPECT_FALSE(rc.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing BandBiquad op with eager mode
/// Expectation: The data is processed successfully
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

/// Feature: Execute Transform op
/// Description: Test executing BandBiquad op with wrong arguments
/// Expectation: Throw correct error and message
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

/// Feature: Execute Transform op
/// Description: Test executing BandpassBiquad op with eager mode
/// Expectation: The data is processed successfully
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

/// Feature: Execute Transform op
/// Description: Test executing BandpassBiquad op with wrong arguments
/// Expectation: Throw correct error and message
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

/// Feature: Execute Transform op
/// Description: Test executing BandrejectBiquad op with eager mode
/// Expectation: The data is processed successfully
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

/// Feature: Execute Transform op
/// Description: Test executing BandrejectBiquad op with wrong arguments
/// Expectation: Throw correct error and message
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

/// Feature: Execute Transform op
/// Description: Test executing Angle op with eager mode
/// Expectation: The data is processed successfully
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

/// Feature: MelScale op
/// Description: Test basic usage of MelScale op
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestMelScale) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestMelScale.";
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
  std::shared_ptr<TensorTransform> mel_scale_op = std::make_shared<audio::MelScale>(2, 10, -50, 100, 2);
  // apply melscale
  mindspore::dataset::Execute trans({mel_scale_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_TRUE(status.IsOk());
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

/// Feature: Execute Transform op
/// Description: Test executing EqualizerBiquad op with eager mode
/// Expectation: The data is processed successfully
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
  auto equalizer_biquad = std::make_shared<audio::EqualizerBiquad>(sample_rate, center_freq, gain, Q);
  auto transform = Execute(std::vector<std::shared_ptr<TensorTransform>>{equalizer_biquad});
  Status rc = transform({input}, &output);
  ASSERT_TRUE(rc.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing EqualizerBiquad op with invalid Q parameter
/// Expectation: Throw correct error and message
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

/// Feature: Execute Transform op
/// Description: Test executing EqualizerBiquad op with invalid sample_rate
/// Expectation: Throw correct error and message
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

/// Feature: Execute Transform op
/// Description: Test executing LowpassBiquad op with eager mode
/// Expectation: The data is processed successfully
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
  auto lowpass_biquad = std::make_shared<audio::LowpassBiquad>(sample_rate, cutoff_freq, Q);
  auto transform = Execute(std::vector<std::shared_ptr<TensorTransform>>{lowpass_biquad});
  Status rc = transform({input}, &output);
  ASSERT_TRUE(rc.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing LowpassBiquad op with invalid Q
/// Expectation: Throw correct error and message
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

/// Feature: Execute Transform op
/// Description: Test executing LowpassBiquad with invalid sample_rate
/// Expectation: Throw correct error and message
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

/// Feature: Execute Transform op
/// Description: Test executing ComplexNorm op with eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestComplexNormEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestComplexNormEager.";
  // testing
  std::shared_ptr<Tensor> input_tensor;
  Tensor::CreateFromVector(std::vector<float>({1.0, 1.0, 2.0, 3.0, 4.0, 4.0}), TensorShape({3, 2}), &input_tensor);

  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor));
  std::shared_ptr<TensorTransform> complex_norm_01 = std::make_shared<audio::ComplexNorm>(4.0);

  // Filtered waveform by complexnorm
  mindspore::dataset::Execute Transform01({complex_norm_01});
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing Contrast op with eager mode
/// Expectation: The data is processed successfully
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

/// Feature: Execute Transform op
/// Description: Test executing Contrast op with wrong arguments
/// Expectation: Throw correct error and message
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

/// Feature: Execute Transform op
/// Description: Test executing DeemphBiquad op with eager mode
/// Expectation: The data is processed successfully
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

/// Feature: Execute Transform op
/// Description: Test executing DeemphBiquad op with wrong arguments
/// Expectation: Throw correct error and message
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

// Feature: Gain op
// Description: Test Gain op in eager mode
// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestGainWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestGainWithEager.";
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
  std::shared_ptr<TensorTransform> Gain_01 = std::make_shared<audio::Gain>();
  mindspore::dataset::Execute Transform01({Gain_01});
  // Filtered waveform by Gain
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing HighpassBiquad op with eager mode
/// Expectation: The data is processed successfully
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
  auto highpass_biquad = std::make_shared<audio::HighpassBiquad>(sample_rate, cutoff_freq, Q);
  auto transform = Execute(std::vector<std::shared_ptr<TensorTransform>>{highpass_biquad});
  Status rc = transform({input}, &output);
  ASSERT_TRUE(rc.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing HighpassBiquad with invalid Q
/// Expectation: Throw correct error and message
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

/// Feature: Execute Transform op
/// Description: Test executing HighpassBiquad op with invalid sample_rate
/// Expectation: Throw correct error and message
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

// Feature: InverseMelScale op
// Description: Test InverseMelScale op in eager mode
// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestInverseMelScale) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestInverseMelScale.";
  // Original waveform
  std::vector<float> labels = {
    2.716064453125000000e-03, 6.347656250000000000e-03, 9.246826171875000000e-03, 1.089477539062500000e-02,
    1.138305664062500000e-02, 1.156616210937500000e-02, 1.394653320312500000e-02, 1.550292968750000000e-02,
    1.614379882812500000e-02, 1.840209960937500000e-02, 1.718139648437500000e-02, 1.599121093750000000e-02,
    1.647949218750000000e-02, 1.510620117187500000e-02, 1.385498046875000000e-02, 1.345825195312500000e-02,
    1.419067382812500000e-02, 1.284790039062500000e-02, 1.052856445312500000e-02, 9.368896484375000000e-03,
    1.419067382812500000e-02, 1.284790039062500000e-02, 1.052856445312500000e-02, 9.368896484375000000e-03};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({2, 2, 3, 2}), &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> inverse_mel_op = std::make_shared<audio::InverseMelScale>(20, 3, 16000, 0, 8000, 10);
  // apply inverse mel scale
  mindspore::dataset::Execute trans({inverse_mel_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing MuLawDecoding op with eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestMuLawDecodingEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestMuLawDecodingEager.";
  // testing
  std::shared_ptr<Tensor> input_tensor;
  Tensor::CreateFromVector(std::vector<float>({1, 254, 231, 155, 101, 77}), TensorShape({1, 6}), &input_tensor);

  auto input_01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor));
  std::shared_ptr<TensorTransform> mu_law_encoding_01 = std::make_shared<audio::MuLawDecoding>(255);

  // Filtered waveform by mulawencoding
  mindspore::dataset::Execute Transform01({mu_law_encoding_01});
  Status s01 = Transform01(input_01, &input_01);
  EXPECT_TRUE(s01.IsOk());
}

/// Feature: MuLawEncoding op
/// Description: Test MuLawEncoding op in eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestMuLawEncodingEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestMuLawEncodingEager.";
  // testing
  std::shared_ptr<Tensor> input_tensor;
  Tensor::CreateFromVector(std::vector<float>({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}), TensorShape({1, 6}), &input_tensor);

  auto input_01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor));
  std::shared_ptr<TensorTransform> mu_law_encoding_01 = std::make_shared<audio::MuLawEncoding>(255);

  // Filtered waveform by mulawencoding
  mindspore::dataset::Execute Transform01({mu_law_encoding_01});
  Status s01 = Transform01(input_01, &input_01);
  EXPECT_TRUE(s01.IsOk());
}

/// Feature: Overdrive op
/// Description: Test basic usage of Overdrive op
/// Expectation: Get correct number of data
TEST_F(MindDataTestExecute, TestOverdriveBasicWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestOverdriveBasicWithEager.";
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
  std::shared_ptr<TensorTransform> phaser_op_01 = std::make_shared<audio::Overdrive>(5.0, 3.0);
  mindspore::dataset::Execute Transform01({phaser_op_01});
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

/// Feature: MaskAlongAxisIID op
/// Description: Test MaskAlongAxisIID op
/// Expectation: The returned result is as expected
TEST_F(MindDataTestExecute, TestMaskAlongAxisIID) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestMaskAlongAxisIID.";
  // testing
  std::shared_ptr<Tensor> input;
  TensorShape s = TensorShape({1, 1, 4, 4});
  ASSERT_OK(Tensor::CreateFromVector(std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f, 4.0f, 3.0f,
                                                         2.0f, 1.0f, 4.0f, 3.0f, 2.0f, 1.0f}), s, &input));
  auto input_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> mask_along_axisiid_op = std::make_shared<audio::MaskAlongAxisIID>(3, 9.0, 2);
  mindspore::dataset::Execute transform({mask_along_axisiid_op});
  Status status = transform(input_tensor, &input_tensor);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: Overdrive op
/// Description: Test invalid parameter of Overdrive op
/// Expectation: Throw exception correctly
TEST_F(MindDataTestExecute, TestOverdriveWrongArgWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestOverdriveWrongArgWithEager";
  std::vector<double> labels = {0.271, 1.634, 9.246,  0.108, 1.138, 1.156, 3.394,
                                1.55,  3.614, 1.8402, 0.718, 4.599, 5.64,  2.510620117187500000e-02,
                                1.38,  5.825, 4.1906, 5.28,  1.052, 9.36};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({4, 5}), &input));

  // verify the gain range from 0 to 100
  auto input_01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> overdrive_op1 = std::make_shared<audio::Overdrive>(100.1);
  mindspore::dataset::Execute Transform01({overdrive_op1});
  Status s01 = Transform01(input_01, &input_01);
  EXPECT_FALSE(s01.IsOk());

  // verify the color range from 0 to 100
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> overdrive_op2 = std::make_shared<audio::Overdrive>(5.0, 100.1);
  mindspore::dataset::Execute Transform02({overdrive_op2});
  Status s02 = Transform02(input_02, &input_02);
  EXPECT_FALSE(s02.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing RiaaBiquad op with eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestRiaaBiquadWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestRiaaBiquadWithEager.";
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
  std::shared_ptr<TensorTransform> riaa_biquad_01 = std::make_shared<audio::RiaaBiquad>(44100);
  mindspore::dataset::Execute Transform01({riaa_biquad_01});
  // Filtered waveform by riaabiquad
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing RiaaBiquad op with wrong arguments
/// Expectation: Throw correct error and message
TEST_F(MindDataTestExecute, TestRiaaBiquadWithWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestRiaaBiquadWithWrongArg.";
  std::vector<float> labels = {3.156, 5.690, 1.362, 1.093, 5.782, 6.381, 5.982, 3.098, 1.222, 6.027,
                               3.909, 7.993, 4.324, 1.092, 5.093, 0.991, 1.099, 4.092, 8.111, 6.666};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({4, 5}), &input));
  auto input01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  std::shared_ptr<TensorTransform> riaa_biquad_op01 = std::make_shared<audio::RiaaBiquad>(0);
  mindspore::dataset::Execute Transform01({riaa_biquad_op01});
  Status s01 = Transform01(input01, &input01);
  EXPECT_FALSE(s01.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing TrebleBiquad op with eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestTrebleBiquadWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestTrebleBiquadWithEager.";
  // Original waveform
  std::vector<float> labels = {3.156, 5.690, 1.362, 1.093, 5.782, 6.381, 5.982, 3.098, 1.222, 6.027,
                               3.909, 7.993, 4.324, 1.092, 5.093, 0.991, 1.099, 4.092, 8.111, 6.666};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({2, 10}), &input));
  auto input_01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> treble_biquad_01 = std::make_shared<audio::TrebleBiquad>(44100, 200);
  mindspore::dataset::Execute Transform01({treble_biquad_01});
  // Filtered waveform by treblebiquad
  EXPECT_OK(Transform01(input_01, &input_01));
}

/// Feature: Execute Transform op
/// Description: Test executing TrebleBiquad op with wrong arguments
/// Expectation: Throw correct error and message
TEST_F(MindDataTestExecute, TestTrebleBiquadWithWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestTrebleBiquadWithWrongArg.";
  std::vector<double> labels = {
    2.716064453125000000e-03, 6.347656250000000000e-03, 9.246826171875000000e-03, 1.089477539062500000e-02,
    1.138305664062500000e-02, 1.156616210937500000e-02, 1.394653320312500000e-02, 1.550292968750000000e-02,
    1.614379882812500000e-02, 1.840209960937500000e-02, 1.718139648437500000e-02, 1.599121093750000000e-02,
    1.647949218750000000e-02, 1.510620117187500000e-02, 1.385498046875000000e-02, 1.345825195312500000e-02,
    1.419067382812500000e-02, 1.284790039062500000e-02, 1.052856445312500000e-02, 9.368896484375000000e-03};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({2, 10}), &input));
  auto input01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  auto input02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  std::shared_ptr<TensorTransform> treble_biquad_op01 = std::make_shared<audio::TrebleBiquad>(0.0, 200.0);
  mindspore::dataset::Execute Transform01({treble_biquad_op01});
  EXPECT_ERROR(Transform01(input01, &input01));
  // Check Q
  MS_LOG(INFO) << "Q is zero.";
  std::shared_ptr<TensorTransform> treble_biquad_op02 =
    std::make_shared<audio::TrebleBiquad>(44100, 200.0, 3000.0, 0.0);
  mindspore::dataset::Execute Transform02({treble_biquad_op02});
  EXPECT_ERROR(Transform02(input02, &input02));
}

/// Feature: Execute Transform op
/// Description: Test executing LFilter op with eager mode
/// Expectation: The data is processed successfully
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

/// Feature: Execute Transform op
/// Description: Test executing LFilter op with wrong arguments
/// Expectation: Throw correct error and message
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

/// Feature: Phaser op
/// Description: Test basic usage of Phaser op
/// Expectation: Get correct number of data
TEST_F(MindDataTestExecute, TestPhaserBasicWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestPhaserBasicWithEager.";
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
  std::shared_ptr<TensorTransform> phaser_op_01 = std::make_shared<audio::Phaser>(44100);
  mindspore::dataset::Execute Transform01({phaser_op_01});
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

/// Feature: Phaser op
/// Description: Test invalid parameter of Phaser op
/// Expectation: Throw exception correctly
TEST_F(MindDataTestExecute, TestPhaserInputArgWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestPhaserInputArgWithEager";
  std::vector<double> labels = {0.271, 1.634, 9.246,  0.108, 1.138, 1.156, 3.394,
                                1.55,  3.614, 1.8402, 0.718, 4.599, 5.64,  2.510620117187500000e-02,
                                1.38,  5.825, 4.1906, 5.28,  1.052, 9.36};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({4, 5}), &input));

  // check gain_in rang [0.0,1.0]
  auto input_01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> phaser_op1 = std::make_shared<audio::Phaser>(44100, 2.0);
  mindspore::dataset::Execute Transform01({phaser_op1});
  Status s01 = Transform01(input_01, &input_01);
  EXPECT_FALSE(s01.IsOk());

  // check gain_out range [0.0,1e9]
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> phaser_op2 = std::make_shared<audio::Phaser>(44100, 0.2, -0.1);
  mindspore::dataset::Execute Transform02({phaser_op2});
  Status s02 = Transform02(input_02, &input_02);
  EXPECT_FALSE(s02.IsOk());

  // check delay_ms range [0.0,5.0]
  auto input_03 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> phaser_op3 = std::make_shared<audio::Phaser>(44100, 0.2, 0.2, 6.0);
  mindspore::dataset::Execute Transform03({phaser_op3});
  Status s03 = Transform03(input_03, &input_03);
  EXPECT_FALSE(s03.IsOk());

  // check decay range [0.0,0.99]
  auto input_04 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> phaser_op4 = std::make_shared<audio::Phaser>(44100, 0.2, 0.2, 4.0, 1.0);
  mindspore::dataset::Execute Transform04({phaser_op4});
  Status s04 = Transform04(input_04, &input_04);
  EXPECT_FALSE(s04.IsOk());

  // check mod_speed range [0.1, 2]
  auto input_05 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> phaser_op5 = std::make_shared<audio::Phaser>(44100, 0.2, 0.2, 4.0, 0.8, 3.0);
  mindspore::dataset::Execute Transform05({phaser_op5});
  Status s05 = Transform05(input_05, &input_05);
  EXPECT_FALSE(s05.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing DCShift op with eager mode
/// Expectation: The data is processed successfully
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

/// Feature: Execute Transform op
/// Description: Test executing Biquad op with eager mode
/// Expectation: The data is processed successfully
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

/// Feature: Execute Transform op
/// Description: Test executing Biquad op with wrong arguments
/// Expectation: Throw correct error and message
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

/// Feature: Execute Transform op
/// Description: Test executing Fade op with various FadeShape
/// Expectation: The data is processed successfully
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

/// Feature: Execute Transform op
/// Description: Test executing Fade op with default arguments
/// Expectation: The data is processed successfully
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

/// Feature: Execute Transform op
/// Description: Test executing Fade op with invalid arguments
/// Expectation: Throw correct error and message
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

/// Feature: Fade op
/// Description: Test Fade op with bool type
/// Expectation: The dataset is processed successfully
TEST_F(MindDataTestExecute, TestFadeWithBool) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestFadeWithBool.";
  std::vector<bool> waveform = {1, 0, 1, 1, 1, 1, 1, 1};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(waveform, TensorShape({1, 8}), &input));
  auto input_01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> fade1 = std::make_shared<audio::Fade>(5, 6, FadeShape::kLinear);
  mindspore::dataset::Execute Transform01({fade1});
  Status s01 = Transform01(input_01, &input_01);
  EXPECT_TRUE(s01.IsOk());
}

/// Feature: GriffinLim op
/// Description: Test basic usage of GriffinLim op
/// Expectation: The dataset is processed successfully
TEST_F(MindDataTestExecute, TestGriffinLimDefaultValue) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestGriffinLimDefaultValue.";
  // Random waveform
  std::mt19937 gen;
  std::normal_distribution<float> distribution(1.0, 0.5);
  std::vector<float> vec;
  for (int i = 0; i < 1206; ++i) {
    vec.push_back(distribution(gen));
  }
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(vec, TensorShape({1, 201, 6}), &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> griffin_lim_op = std::make_shared<audio::GriffinLim>();
  // apply griffinlim
  mindspore::dataset::Execute trans({griffin_lim_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: Vad op
/// Description: Test basic usage of Vad op
/// Expectation: The dataset is processed successfully
TEST_F(MindDataTestExecute, TestVadDefaultValue) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestVadDefaultValue.";
  // Random waveform
  std::mt19937 gen;
  std::normal_distribution<float> distribution(1.0, 0.5);
  std::vector<float> vec;
  for (int i = 0; i < 1000; ++i){
    vec.push_back(distribution(gen));
  }
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(vec, TensorShape({1, 1000}), &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> vad_op = std::make_shared<audio::Vad>(1600);
  // apply vad
  mindspore::dataset::Execute trans({vad_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: Execute Vol op
/// Description: Test executing Vol op with default values
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestVolDefalutValue) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestVolDefalutValue.";
  std::shared_ptr<Tensor> input;
  TensorShape s = TensorShape({2, 6});
  ASSERT_OK(Tensor::CreateFromVector(
    std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}), s, &input));
  auto input_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> vol_op = std::make_shared<audio::Vol>(0.333);
  mindspore::dataset::Execute transform({vol_op});
  Status status = transform(input_tensor, &input_tensor);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing Vol op with GainType::kPower
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestVolGainTypePower) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestVolGainTypePower.";
  std::shared_ptr<Tensor> input;
  TensorShape s = TensorShape({4, 3});
  ASSERT_OK(Tensor::CreateFromVector(
    std::vector<double>({4.0f, 5.0f, 3.0f, 5.0f, 4.0f, 6.0f, 6.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f}), s, &input));
  auto input_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> vol_op = std::make_shared<audio::Vol>(0.2, GainType::kPower);
  mindspore::dataset::Execute transform({vol_op});
  Status status = transform(input_tensor, &input_tensor);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing Magphase op with eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestMagphaseEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestMagphaseEager.";
  float power = 1.0;
  std::vector<mindspore::MSTensor> output_tensor;
  std::shared_ptr<Tensor> test;
  std::vector<float> test_vector = {3, 4, -3, 4, 3, -4, -3, -4, 5, 12, -5, 12, 5, -12, -5, -12};
  Tensor::CreateFromVector(test_vector, TensorShape({2, 4, 2}), &test);
  auto input_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(test));
  auto magphase = std::make_shared<audio::Magphase>(power);
  auto transform = Execute(std::vector<std::shared_ptr<TensorTransform>>{magphase});
  Status rc = transform({input_tensor}, &output_tensor);
  ASSERT_TRUE(rc.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing Decode then RandomInvert op with eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestRandomInvertEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestRandomInvertEager.";
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto random_invert_op = vision::RandomInvert(0.6);

  auto transform = Execute({decode, random_invert_op});
  Status rc = transform(image, &image);
  EXPECT_EQ(rc, Status::OK());
}

/// Feature: Execute Transform op
/// Description: Test executing Decode then RandomAutoContrast op with eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestRandomAutoContrastEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestRandomAutoContrastEager.";
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto random_auto_contrast_op = vision::RandomAutoContrast(0.6);

  auto transform = Execute({decode, random_auto_contrast_op});
  Status rc = transform(image, &image);
  EXPECT_EQ(rc, Status::OK());
}

/// Feature: Execute Transform op
/// Description: Test executing Decode then RandomEqualize op with eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestRandomEqualizeEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestRandomEqualizeEager.";
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto random_equalize_op = vision::RandomEqualize(0.6);

  auto transform = Execute({decode, random_equalize_op});
  Status rc = transform(image, &image);
  EXPECT_EQ(rc, Status::OK());
}

/// Feature: Execute Transform op
/// Description: Test executing Decode then RandomAdjustSharpness op with eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestRandomAdjustSharpnessEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestRandomAdjustSharpnessEager.";
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto random_adjust_sharpness_op = vision::RandomAdjustSharpness(2.0, 0.6);

  auto transform = Execute({decode, random_adjust_sharpness_op});
  Status rc = transform(image, &image);
  EXPECT_EQ(rc, Status::OK());
}

/// Feature: Execute Transform op
/// Description: Test executing DetectPitchFrequency op with eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestDetectPitchFrequencyWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestDetectPitchFrequencyWithEager.";
  // Original waveform
  std::vector<double> labels = {
    3.716064453125000000e-03, 2.347656250000000000e-03, 9.246826171875000000e-03, 4.089477539062500000e-02,
    3.138305664062500000e-02, 1.156616210937500000e-02, 0.394653320312500000e-02, 1.550292968750000000e-02,
    1.614379882812500000e-02, 0.840209960937500000e-02, 1.718139648437500000e-02, 2.599121093750000000e-02,
    5.647949218750000000e-02, 1.510620117187500000e-02, 2.385498046875000000e-02, 1.345825195312500000e-02,
    1.419067382812500000e-02, 3.284790039062500000e-02, 9.052856445312500000e-02, 2.368896484375000000e-03};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({2, 10}), &input));
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> detect_pitch_frequency_01 =
    std::make_shared<audio::DetectPitchFrequency>(30, 0.1, 3, 5, 25);
  mindspore::dataset::Execute Transform01({detect_pitch_frequency_01});
  // Detect pitch frequence
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing DetectPitchFrequency with wrong arguments
/// Expectation: Throw correct error and message
TEST_F(MindDataTestExecute, TestDetectPitchFrequencyWithWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestDetectPitchFrequencyWithWrongArg.";
  std::vector<float> labels = {
    0.716064e-03, 5.347656e-03, 6.246826e-03, 2.089477e-02, 7.138305e-02,
    4.156616e-02, 1.394653e-02, 3.550292e-02, 0.614379e-02, 3.840209e-02,
  };
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({2, 5}), &input));
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  // Check frame_time
  MS_LOG(INFO) << "frame_time is zero.";
  std::shared_ptr<TensorTransform> detect_pitch_frequency_01 =
    std::make_shared<audio::DetectPitchFrequency>(40, 0, 3, 3, 20);
  mindspore::dataset::Execute Transform01({detect_pitch_frequency_01});
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_FALSE(s01.IsOk());
  // Check win_length
  MS_LOG(INFO) << "win_length is zero.";
  std::shared_ptr<TensorTransform> detect_pitch_frequency_02 =
    std::make_shared<audio::DetectPitchFrequency>(40, 0.1, 0, 3, 20);
  mindspore::dataset::Execute Transform02({detect_pitch_frequency_02});
  Status s02 = Transform02(input_02, &input_02);
  EXPECT_FALSE(s02.IsOk());
  // Check freq_low
  MS_LOG(INFO) << "freq_low is zero.";
  std::shared_ptr<TensorTransform> detect_pitch_frequency_03 =
    std::make_shared<audio::DetectPitchFrequency>(40, 0.1, 3, 0, 20);
  mindspore::dataset::Execute Transform03({detect_pitch_frequency_03});
  Status s03 = Transform03(input_02, &input_02);
  EXPECT_FALSE(s03.IsOk());
  // Check freq_high
  MS_LOG(INFO) << "freq_high is zero.";
  std::shared_ptr<TensorTransform> detect_pitch_frequency_04 =
    std::make_shared<audio::DetectPitchFrequency>(40, 0.1, 3, 3, 0);
  mindspore::dataset::Execute Transform04({detect_pitch_frequency_04});
  Status s04 = Transform04(input_02, &input_02);
  EXPECT_FALSE(s04.IsOk());
  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  std::shared_ptr<TensorTransform> detect_pitch_frequency_05 = std::make_shared<audio::DetectPitchFrequency>(0);
  mindspore::dataset::Execute Transform05({detect_pitch_frequency_05});
  Status s05 = Transform05(input_02, &input_02);
  EXPECT_FALSE(s05.IsOk());
}

/// Feature: Dither op
/// Description: Test Dither op in eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestDitherWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestDitherWithEager.";
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
  std::shared_ptr<TensorTransform> dither_01 = std::make_shared<audio::Dither>();
  mindspore::dataset::Execute Transform01({dither_01});
  // Filtered waveform by Dither
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing Flanger op with eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestFlangerWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestFlangerWithEager.";
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
  std::shared_ptr<TensorTransform> flanger_01 = std::make_shared<audio::Flanger>(44100);
  mindspore::dataset::Execute Transform01({flanger_01});
  // Filtered waveform by flanger
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing Flanger op with wrong arguments
/// Expectation: Throw correct error and message
TEST_F(MindDataTestExecute, TestFlangerWithWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestFlangerWithWrongArg.";
  std::vector<double> labels = {1.143, 1.3123, 2.632, 2.554, 1.213, 1.3, 0.456, 3.563};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({4, 2}), &input));
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  std::shared_ptr<TensorTransform> flanger_op = std::make_shared<audio::Flanger>(0);
  mindspore::dataset::Execute Transform01({flanger_op});
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_FALSE(s01.IsOk());
}

/// Feature: Vectors
/// Description: Test basic usage of Vectors and the ToVectors with default parameter
/// Expectation: Get correct MSTensor
TEST_F(MindDataTestExecute, TestVectorsParam) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestVectorsParam.";
  std::shared_ptr<Tensor> de_tensor;
  Tensor::CreateScalar<std::string>("ok", &de_tensor);
  auto token = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));
  mindspore::MSTensor lookup_result;

  // Create expected output.
  std::shared_ptr<Tensor> de_expected;
  std::vector<float> expected = {0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411};
  dsize_t dim = 6;
  ASSERT_OK(Tensor::CreateFromVector(expected, TensorShape({dim}), &de_expected));
  auto ms_expected = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected));

  // Transform params.
  std::string vectors_dir = "data/dataset/testVectors/vectors.txt";
  std::shared_ptr<Vectors> vectors01;
  Status s01 = Vectors::BuildFromFile(&vectors01, vectors_dir);
  EXPECT_EQ(s01, Status::OK());
  std::shared_ptr<TensorTransform> to_vectors01 = std::make_shared<text::ToVectors>(vectors01);
  auto transform01 = Execute({to_vectors01});
  Status status01 = transform01(token, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected);
  EXPECT_TRUE(status01.IsOk());

  std::shared_ptr<Vectors> vectors02;
  Status s02 = Vectors::BuildFromFile(&vectors02, vectors_dir, 100);
  EXPECT_EQ(s02, Status::OK());
  std::shared_ptr<TensorTransform> to_vectors02 = std::make_shared<text::ToVectors>(vectors02);
  auto transform02 = Execute({to_vectors02});
  Status status02 = transform02(token, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected);
  EXPECT_TRUE(status02.IsOk());

  std::shared_ptr<Vectors> vectors03;
  Status s03 = Vectors::BuildFromFile(&vectors03, vectors_dir, 3);
  EXPECT_EQ(s03, Status::OK());
  std::shared_ptr<TensorTransform> to_vectors03 = std::make_shared<text::ToVectors>(vectors03);
  auto transform03 = Execute({to_vectors03});
  Status status03 = transform03(token, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected);
  EXPECT_TRUE(status03.IsOk());
}

/// Feature: ToVectors op
/// Description: Test basic usage of ToVectors op and the Vectors with default parameter
/// Expectation: Get correct MSTensor
TEST_F(MindDataTestExecute, TestToVectorsParam) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestToVectorsParam.";
  std::shared_ptr<Tensor> de_tensor01;
  Tensor::CreateScalar<std::string>("none", &de_tensor01);
  auto token01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor01));
  std::shared_ptr<Tensor> de_tensor02;
  Tensor::CreateScalar<std::string>("ok", &de_tensor02);
  auto token02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor02));
  std::shared_ptr<Tensor> de_tensor03;
  Tensor::CreateScalar<std::string>("OK", &de_tensor03);
  auto token03 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor03));
  mindspore::MSTensor lookup_result;

  // Create expected output.
  dsize_t dim = 6;
  std::shared_ptr<Tensor> de_expected01;
  std::vector<float> expected01 = {0, 0, 0, 0, 0, 0};
  ASSERT_OK(Tensor::CreateFromVector(expected01, TensorShape({dim}), &de_expected01));
  auto ms_expected01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected01));
  std::shared_ptr<Tensor> de_expected02;
  std::vector<float> expected02 = {-1, -1, -1, -1, -1, -1};
  ASSERT_OK(Tensor::CreateFromVector(expected02, TensorShape({dim}), &de_expected02));
  auto ms_expected02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected02));
  std::shared_ptr<Tensor> de_expected03;
  std::vector<float> expected03 = {0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411};
  ASSERT_OK(Tensor::CreateFromVector(expected03, TensorShape({dim}), &de_expected03));
  auto ms_expected03 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected03));

  // Transform params.
  std::string vectors_dir = "data/dataset/testVectors/vectors.txt";
  std::shared_ptr<Vectors> vectors;
  Status s = Vectors::BuildFromFile(&vectors, vectors_dir);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<TensorTransform> to_vectors01 = std::make_shared<text::ToVectors>(vectors);
  auto transform01 = Execute({to_vectors01});
  Status status01 = transform01(token01, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected01);
  EXPECT_TRUE(status01.IsOk());
  std::vector<float> unknown_init = {-1, -1, -1, -1, -1, -1};
  std::shared_ptr<TensorTransform> to_vectors02 = std::make_shared<text::ToVectors>(vectors, unknown_init);
  auto transform02 = Execute({to_vectors02});
  Status status02 = transform02(token01, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected02);
  EXPECT_TRUE(status02.IsOk());
  std::shared_ptr<TensorTransform> to_vectors03 = std::make_shared<text::ToVectors>(vectors, unknown_init);
  auto transform03 = Execute({to_vectors03});
  Status status03 = transform03(token02, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected03);
  EXPECT_TRUE(status03.IsOk());
  std::shared_ptr<TensorTransform> to_vectors04 = std::make_shared<text::ToVectors>(vectors, unknown_init, true);
  auto transform04 = Execute({to_vectors04});
  Status status04 = transform04(token03, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected03);
  EXPECT_TRUE(status04.IsOk());
}

/// Feature: ToVectors op
/// Description: Test invalid parameter of ToVectors op
/// Expectation: Throw exception correctly
TEST_F(MindDataTestExecute, TestToVectorsWithInvalidParam) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestToVectorsWithInvalidParam.";
  std::shared_ptr<Tensor> de_tensor;
  Tensor::CreateScalar<std::string>("none", &de_tensor);
  auto token = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));
  mindspore::MSTensor lookup_result;

  // Transform params.
  std::string vectors_dir = "data/dataset/testVectors/vectors.txt";
  std::shared_ptr<Vectors> vectors01;
  Status s = Vectors::BuildFromFile(&vectors01, vectors_dir);
  EXPECT_EQ(s, Status::OK());
  std::vector<float> unknown_init = {-1, -1, -1, -1};
  std::shared_ptr<TensorTransform> to_vectors01 = std::make_shared<text::ToVectors>(vectors01, unknown_init);
  auto transform01 = Execute({to_vectors01});
  Status status01 = transform01(token, &lookup_result);
  EXPECT_FALSE(status01.IsOk());
  std::shared_ptr<Vectors> vectors02 = nullptr;
  std::shared_ptr<TensorTransform> to_vectors02 = std::make_shared<text::ToVectors>(vectors02);
  auto transform02 = Execute({to_vectors02});
  Status status02 = transform02(token, &lookup_result);
  EXPECT_FALSE(status02.IsOk());
}

/// Feature: FastText
/// Description: Test basic usage of FastText and the ToVectors with default parameter
/// Expectation: Get correct MSTensor
TEST_F(MindDataTestExecute, TestFastTextParam) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestFastTextParam.";
  std::shared_ptr<Tensor> de_tensor;
  Tensor::CreateScalar<std::string>("ok", &de_tensor);
  auto token = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));
  mindspore::MSTensor lookup_result;

  // Create expected output.
  std::shared_ptr<Tensor> de_expected;
  std::vector<float> expected = {0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411};
  dsize_t dim = 6;
  ASSERT_OK(Tensor::CreateFromVector(expected, TensorShape({dim}), &de_expected));
  auto ms_expected = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected));

  // Transform params.
  std::string vectors_dir = "data/dataset/test_fast_text/fast_text.vec";
  std::shared_ptr<FastText> fast_text01;
  Status s01 = FastText::BuildFromFile(&fast_text01, vectors_dir);
  EXPECT_EQ(s01, Status::OK());
  std::shared_ptr<TensorTransform> to_vectors01 = std::make_shared<text::ToVectors>(fast_text01);
  auto transform01 = Execute({to_vectors01});
  Status status01 = transform01(token, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected);
  EXPECT_TRUE(status01.IsOk());

  std::shared_ptr<FastText> fast_text02;
  Status s02 = FastText::BuildFromFile(&fast_text02, vectors_dir, 100);
  EXPECT_EQ(s02, Status::OK());
  std::shared_ptr<TensorTransform> to_vectors02 = std::make_shared<text::ToVectors>(fast_text02);
  auto transform02 = Execute({to_vectors02});
  Status status02 = transform02(token, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected);
  EXPECT_TRUE(status02.IsOk());

  std::shared_ptr<FastText> fast_text03;
  Status s03 = FastText::BuildFromFile(&fast_text03, vectors_dir, 3);
  EXPECT_EQ(s03, Status::OK());
  std::shared_ptr<TensorTransform> to_vectors03 = std::make_shared<text::ToVectors>(fast_text03);
  auto transform03 = Execute({to_vectors03});
  Status status03 = transform03(token, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected);
  EXPECT_TRUE(status03.IsOk());
}

/// Feature: ToVectors op
/// Description: Test basic usage of ToVectors op and the FastText with default parameter
/// Expectation: Get correct MSTensor
TEST_F(MindDataTestExecute, TestToVectorsParamForFastText) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestToVectorsParamForFastText.";
  std::shared_ptr<Tensor> de_tensor01;
  Tensor::CreateScalar<std::string>("none", &de_tensor01);
  auto token01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor01));
  std::shared_ptr<Tensor> de_tensor02;
  Tensor::CreateScalar<std::string>("ok", &de_tensor02);
  auto token02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor02));
  std::shared_ptr<Tensor> de_tensor03;
  Tensor::CreateScalar<std::string>("OK", &de_tensor03);
  auto token03 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor03));
  mindspore::MSTensor lookup_result;

  // Create expected output.
  dsize_t dim = 6;
  std::shared_ptr<Tensor> de_expected01;
  std::vector<float> expected01 = {0, 0, 0, 0, 0, 0};
  ASSERT_OK(Tensor::CreateFromVector(expected01, TensorShape({dim}), &de_expected01));
  auto ms_expected01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected01));
  std::shared_ptr<Tensor> de_expected02;
  std::vector<float> expected02 = {-1, -1, -1, -1, -1, -1};
  ASSERT_OK(Tensor::CreateFromVector(expected02, TensorShape({dim}), &de_expected02));
  auto ms_expected02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected02));
  std::shared_ptr<Tensor> de_expected03;
  std::vector<float> expected03 = {0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411};
  ASSERT_OK(Tensor::CreateFromVector(expected03, TensorShape({dim}), &de_expected03));
  auto ms_expected03 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected03));

  // Transform params.
  std::string vectors_dir = "data/dataset/test_fast_text/fast_text.vec";
  std::shared_ptr<FastText> fast_text;
  Status s = FastText::BuildFromFile(&fast_text, vectors_dir);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<TensorTransform> to_vectors01 = std::make_shared<text::ToVectors>(fast_text);
  auto transform01 = Execute({to_vectors01});
  Status status01 = transform01(token01, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected01);
  EXPECT_TRUE(status01.IsOk());
  std::vector<float> unknown_init = {-1, -1, -1, -1, -1, -1};
  std::shared_ptr<TensorTransform> to_vectors02 = std::make_shared<text::ToVectors>(fast_text, unknown_init);
  auto transform02 = Execute({to_vectors02});
  Status status02 = transform02(token01, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected02);
  EXPECT_TRUE(status02.IsOk());
  std::shared_ptr<TensorTransform> to_vectors03 = std::make_shared<text::ToVectors>(fast_text, unknown_init);
  auto transform03 = Execute({to_vectors03});
  Status status03 = transform03(token02, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected03);
  EXPECT_TRUE(status03.IsOk());
  std::shared_ptr<TensorTransform> to_vectors04 = std::make_shared<text::ToVectors>(fast_text, unknown_init, true);
  auto transform04 = Execute({to_vectors04});
  Status status04 = transform04(token03, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected03);
  EXPECT_TRUE(status04.IsOk());
}

/// Feature: ToVectors op
/// Description: Test invalid parameter of ToVectors op for FastText
/// Expectation: Throw exception correctly
TEST_F(MindDataTestExecute, TestToVectorsWithInvalidParamForFastText) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestToVectorsWithInvalidParamForFastText.";
  std::shared_ptr<Tensor> de_tensor;
  Tensor::CreateScalar<std::string>("none", &de_tensor);
  auto token = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));
  mindspore::MSTensor lookup_result;

  // Transform params.
  std::string vectors_dir = "data/dataset/test_fast_text/fast_text.vec";
  std::shared_ptr<FastText> fast_text01;
  Status s = FastText::BuildFromFile(&fast_text01, vectors_dir);
  EXPECT_EQ(s, Status::OK());
  std::vector<float> unknown_init = {-1, -1, -1, -1};
  std::shared_ptr<TensorTransform> to_vectors01 = std::make_shared<text::ToVectors>(fast_text01, unknown_init);
  auto transform01 = Execute({to_vectors01});
  Status status01 = transform01(token, &lookup_result);
  EXPECT_FALSE(status01.IsOk());
  std::shared_ptr<FastText> fast_text02 = nullptr;
  std::shared_ptr<TensorTransform> to_vectors02 = std::make_shared<text::ToVectors>(fast_text02);
  auto transform02 = Execute({to_vectors02});
  Status status02 = transform02(token, &lookup_result);
  EXPECT_FALSE(status02.IsOk());
}

/// Feature: GloVe
/// Description: Test basic usage of GloVe and the ToVectors with default parameter
/// Expectation: Get correct MSTensor
TEST_F(MindDataTestExecute, TestGloVeParam) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestGloVeParam.";
  std::shared_ptr<Tensor> de_tensor;
  Tensor::CreateScalar<std::string>("ok", &de_tensor);
  auto token = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));
  mindspore::MSTensor lookup_result;

  // Create expected output.
  std::shared_ptr<Tensor> de_expected;
  std::vector<float> expected = {0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411};
  dsize_t dim = 6;
  ASSERT_OK(Tensor::CreateFromVector(expected, TensorShape({dim}), &de_expected));
  auto ms_expected = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected));

  // Transform params.
  std::string vectors_dir = "data/dataset/testGloVe/glove.6B.test.txt";
  std::shared_ptr<GloVe> glove01;
  Status s01 = GloVe::BuildFromFile(&glove01, vectors_dir);
  EXPECT_EQ(s01, Status::OK());
  std::shared_ptr<TensorTransform> to_vectors01 = std::make_shared<text::ToVectors>(glove01);
  auto transform01 = Execute({to_vectors01});
  Status status01 = transform01(token, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected);
  EXPECT_TRUE(status01.IsOk());

  std::shared_ptr<GloVe> glove02;
  Status s02 = GloVe::BuildFromFile(&glove02, vectors_dir, 100);
  EXPECT_EQ(s02, Status::OK());
  std::shared_ptr<TensorTransform> to_vectors02 = std::make_shared<text::ToVectors>(glove02);
  auto transform02 = Execute({to_vectors02});
  Status status02 = transform02(token, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected);
  EXPECT_TRUE(status02.IsOk());

  std::shared_ptr<GloVe> glove03;
  Status s03 = GloVe::BuildFromFile(&glove03, vectors_dir, 3);
  EXPECT_EQ(s03, Status::OK());
  std::shared_ptr<TensorTransform> to_vectors03 = std::make_shared<text::ToVectors>(glove03);
  auto transform03 = Execute({to_vectors03});
  Status status03 = transform03(token, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected);
  EXPECT_TRUE(status03.IsOk());
}

/// Feature: ToVectors op
/// Description: Test basic usage of ToVectors op and the GloVe with default parameter
/// Expectation: Get correct MSTensor
TEST_F(MindDataTestExecute, TestToVectorsParamForGloVe) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestToVectorsParamForGloVe.";
  std::shared_ptr<Tensor> de_tensor01;
  Tensor::CreateScalar<std::string>("none", &de_tensor01);
  auto token01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor01));
  std::shared_ptr<Tensor> de_tensor02;
  Tensor::CreateScalar<std::string>("ok", &de_tensor02);
  auto token02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor02));
  std::shared_ptr<Tensor> de_tensor03;
  Tensor::CreateScalar<std::string>("OK", &de_tensor03);
  auto token03 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor03));
  mindspore::MSTensor lookup_result;

  // Create expected output.
  dsize_t dim = 6;
  std::shared_ptr<Tensor> de_expected01;
  std::vector<float> expected01 = {0, 0, 0, 0, 0, 0};
  ASSERT_OK(Tensor::CreateFromVector(expected01, TensorShape({dim}), &de_expected01));
  auto ms_expected01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected01));
  std::shared_ptr<Tensor> de_expected02;
  std::vector<float> expected02 = {-1, -1, -1, -1, -1, -1};
  ASSERT_OK(Tensor::CreateFromVector(expected02, TensorShape({dim}), &de_expected02));
  auto ms_expected02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected02));
  std::shared_ptr<Tensor> de_expected03;
  std::vector<float> expected03 = {0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411};
  ASSERT_OK(Tensor::CreateFromVector(expected03, TensorShape({dim}), &de_expected03));
  auto ms_expected03 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected03));

  // Transform params.
  std::string vectors_dir = "data/dataset/testGloVe/glove.6B.test.txt";
  std::shared_ptr<GloVe> glove;
  Status s = GloVe::BuildFromFile(&glove, vectors_dir);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<TensorTransform> to_vectors01 = std::make_shared<text::ToVectors>(glove);
  auto transform01 = Execute({to_vectors01});
  Status status01 = transform01(token01, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected01);
  EXPECT_TRUE(status01.IsOk());
  std::vector<float> unknown_init = {-1, -1, -1, -1, -1, -1};
  std::shared_ptr<TensorTransform> to_vectors02 = std::make_shared<text::ToVectors>(glove, unknown_init);
  auto transform02 = Execute({to_vectors02});
  Status status02 = transform02(token01, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected02);
  EXPECT_TRUE(status02.IsOk());
  std::shared_ptr<TensorTransform> to_vectors03 = std::make_shared<text::ToVectors>(glove, unknown_init);
  auto transform03 = Execute({to_vectors03});
  Status status03 = transform03(token02, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected03);
  EXPECT_TRUE(status03.IsOk());
  std::shared_ptr<TensorTransform> to_vectors04 = std::make_shared<text::ToVectors>(glove, unknown_init, true);
  auto transform04 = Execute({to_vectors04});
  Status status04 = transform04(token03, &lookup_result);
  EXPECT_MSTENSOR_EQ(lookup_result, ms_expected03);
  EXPECT_TRUE(status04.IsOk());
}

/// Feature: ToVectors op
/// Description: Test invalid parameter of ToVectors for GloVe
/// Expectation: Throw exception correctly
TEST_F(MindDataTestExecute, TestToVectorsWithInvalidParamForGloVe) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestToVectorsWithInvalidParamForGloVe.";
  std::shared_ptr<Tensor> de_tensor;
  Tensor::CreateScalar<std::string>("none", &de_tensor);
  auto token = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));
  mindspore::MSTensor lookup_result;

  // Transform params.
  std::string vectors_dir = "data/dataset/testGloVe/glove.6B.test.txt";
  std::shared_ptr<GloVe> glove01;
  Status s = GloVe::BuildFromFile(&glove01, vectors_dir);
  EXPECT_EQ(s, Status::OK());
  std::vector<float> unknown_init = {-1, -1, -1, -1};
  std::shared_ptr<TensorTransform> to_vectors01 = std::make_shared<text::ToVectors>(glove01, unknown_init);
  auto transform01 = Execute({to_vectors01});
  Status status01 = transform01(token, &lookup_result);
  EXPECT_FALSE(status01.IsOk());
  std::shared_ptr<GloVe> glove02 = nullptr;
  std::shared_ptr<TensorTransform> to_vectors02 = std::make_shared<text::ToVectors>(glove02);
  auto transform02 = Execute({to_vectors02});
  Status status02 = transform02(token, &lookup_result);
  EXPECT_FALSE(status02.IsOk());
}

/// Feature: CharNGram
/// Description: Test basic usage of CharNGram and the ToVectors with default parameter
/// Expectation: Get correct MSTensor
TEST_F(MindDataTestExecute, TestCharNGramParam) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestCharNGramParam.";
  std::shared_ptr<Tensor> de_tensor;
  Tensor::CreateScalar<std::string>("the", &de_tensor);
  auto token = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));
  mindspore::MSTensor lookup_result;

  // Create expected output.
  std::shared_ptr<Tensor> de_expected01;
  std::vector<float> expected01 = {-0.840079, -0.0270003, -0.833472, 0.588367, -0.210012};
  ASSERT_OK(Tensor::CreateFromVector(expected01, &de_expected01));
  auto ms_expected01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected01));
  std::shared_ptr<Tensor> de_expected02;
  std::vector<float> expected02 = {-1.34122, 0.0442693, -0.48697, 0.662939, -0.367669};
  ASSERT_OK(Tensor::CreateFromVector(expected02, &de_expected02));
  auto ms_expected02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected02));

  // Transform params.
  std::string vectors_dir = "data/dataset/testVectors/char_n_gram_20.txt";
  std::shared_ptr<CharNGram> char_n_gram01;
  Status s01 = CharNGram::BuildFromFile(&char_n_gram01, vectors_dir);
  EXPECT_EQ(s01, Status::OK());
  std::shared_ptr<TensorTransform> to_vectors01 = std::make_shared<text::ToVectors>(char_n_gram01);
  auto transform01 = Execute({to_vectors01});
  Status status01 = transform01(token, &lookup_result);
  EXPECT_EQ(lookup_result.Shape(), ms_expected01.Shape());
  EXPECT_TRUE(status01.IsOk());

  std::shared_ptr<CharNGram> char_n_gram02;
  Status s02 = CharNGram::BuildFromFile(&char_n_gram02, vectors_dir, 100);
  EXPECT_EQ(s02, Status::OK());
  std::shared_ptr<TensorTransform> to_vectors02 = std::make_shared<text::ToVectors>(char_n_gram02);
  auto transform02 = Execute({to_vectors02});
  Status status02 = transform02(token, &lookup_result);
  EXPECT_EQ(lookup_result.Shape(), ms_expected01.Shape());
  EXPECT_TRUE(status02.IsOk());

  std::shared_ptr<CharNGram> char_n_gram03;
  Status s03 = CharNGram::BuildFromFile(&char_n_gram03, vectors_dir, 18);
  EXPECT_EQ(s03, Status::OK());
  std::shared_ptr<TensorTransform> to_vectors03 = std::make_shared<text::ToVectors>(char_n_gram03);
  auto transform03 = Execute({to_vectors03});
  Status status03 = transform03(token, &lookup_result);
  EXPECT_EQ(lookup_result.Shape(), ms_expected02.Shape());
  EXPECT_TRUE(status03.IsOk());
}

/// Feature: CharNGram
/// Description: Test basic usage of ToVectors and the CharNGram with default parameter
/// Expectation: Get correct MSTensor
TEST_F(MindDataTestExecute, TestToVectorsParamForCharNGram) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestToVectorsParamForCharNGram.";
  std::shared_ptr<Tensor> de_tensor01;
  Tensor::CreateScalar<std::string>("none", &de_tensor01);
  auto token01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor01));
  std::shared_ptr<Tensor> de_tensor02;
  Tensor::CreateScalar<std::string>("the", &de_tensor02);
  auto token02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor02));
  std::shared_ptr<Tensor> de_tensor03;
  Tensor::CreateScalar<std::string>("The", &de_tensor03);
  auto token03 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor03));
  mindspore::MSTensor lookup_result;

  // Create expected output.
  std::shared_ptr<Tensor> de_expected01;
  std::vector<float> expected01(5, 0);
  ASSERT_OK(Tensor::CreateFromVector(expected01, &de_expected01));
  auto ms_expected01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected01));
  std::shared_ptr<Tensor> de_expected02;
  std::vector<float> expected02(5, -1);
  ASSERT_OK(Tensor::CreateFromVector(expected02, &de_expected02));
  auto ms_expected02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected02));
  std::shared_ptr<Tensor> de_expected03;
  std::vector<float> expected03 = {-0.840079, -0.0270003, -0.833472, 0.588367, -0.210012};
  ASSERT_OK(Tensor::CreateFromVector(expected03, &de_expected03));
  auto ms_expected03 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected03));

  // Transform params.
  std::string vectors_dir = "data/dataset/testVectors/char_n_gram_20.txt";
  std::shared_ptr<CharNGram> char_n_gram;
  Status s = CharNGram::BuildFromFile(&char_n_gram, vectors_dir);
  EXPECT_EQ(s, Status::OK());

  std::shared_ptr<TensorTransform> to_vectors01 = std::make_shared<text::ToVectors>(char_n_gram);
  auto transform01 = Execute({to_vectors01});
  Status status01 = transform01(token01, &lookup_result);
  EXPECT_EQ(lookup_result.Shape(), ms_expected01.Shape());
  EXPECT_TRUE(status01.IsOk());
  std::vector<float> unknown_init(5, -1);
  std::shared_ptr<TensorTransform> to_vectors02 = std::make_shared<text::ToVectors>(char_n_gram, unknown_init);
  auto transform02 = Execute({to_vectors02});
  Status status02 = transform02(token01, &lookup_result);
  EXPECT_EQ(lookup_result.Shape(), ms_expected02.Shape());
  EXPECT_TRUE(status02.IsOk());
  std::shared_ptr<TensorTransform> to_vectors03 = std::make_shared<text::ToVectors>(char_n_gram, unknown_init);
  auto transform03 = Execute({to_vectors03});
  Status status03 = transform03(token02, &lookup_result);
  EXPECT_EQ(lookup_result.Shape(), ms_expected03.Shape());
  EXPECT_TRUE(status03.IsOk());
  std::shared_ptr<TensorTransform> to_vectors04 = std::make_shared<text::ToVectors>(char_n_gram, unknown_init, true);
  auto transform04 = Execute({to_vectors04});
  Status status04 = transform04(token03, &lookup_result);
  EXPECT_EQ(lookup_result.Shape(), ms_expected03.Shape());
  EXPECT_TRUE(status04.IsOk());
}

/// Feature: CharNGram
/// Description: Test invalid parameter of ToVectors op
/// Expectation: Throw exception correctly
TEST_F(MindDataTestExecute, TestToVectorsWithInvalidParamForCharNGram) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestToVectorsWithInvalidParamForCharNGram.";
  std::shared_ptr<Tensor> de_tensor;
  Tensor::CreateScalar<std::string>("none", &de_tensor);
  auto token = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_tensor));
  mindspore::MSTensor lookup_result;

  // Transform params.
  std::string vectors_dir = "data/dataset/testVectors/char_n_gram_20.txt";
  std::shared_ptr<CharNGram> char_n_gram01;
  Status s = CharNGram::BuildFromFile(&char_n_gram01, vectors_dir);
  EXPECT_EQ(s, Status::OK());
  std::vector<float> unknown_init(4, -1);
  std::shared_ptr<TensorTransform> to_vectors01 = std::make_shared<text::ToVectors>(char_n_gram01, unknown_init);
  auto transform01 = Execute({to_vectors01});
  Status status01 = transform01(token, &lookup_result);
  EXPECT_FALSE(status01.IsOk());
  std::shared_ptr<CharNGram> char_n_gram02 = nullptr;
  std::shared_ptr<TensorTransform> to_vectors02 = std::make_shared<text::ToVectors>(char_n_gram02);
  auto transform02 = Execute({to_vectors02});
  Status status02 = transform02(token, &lookup_result);
  EXPECT_FALSE(status02.IsOk());
}

// Feature: DBToAmplitude op
// Description: Test DBToAmplitude op in eager mode
// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestDBToAmplitudeWithEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestDBToAmplitudeWithEager.";
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
  std::shared_ptr<TensorTransform> DBToAmplitude_01 = std::make_shared<audio::DBToAmplitude>(2, 2);
  mindspore::dataset::Execute Transform01({DBToAmplitude_01});
  // Filtered waveform by DBToAmplitude
  Status s01 = Transform01(input_02, &input_02);
  EXPECT_TRUE(s01.IsOk());
}

/// Feature: PhaseVocoder op
/// Description: Test PhaseVocoder op in eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestPhaseVocoderEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestPhaseVocoderEager.";
  // testing
  std::shared_ptr<Tensor> input_tensor, input_phase_advance_tensor;
  Tensor::CreateFromVector(
    std::vector<float>({0.1468,  -1.1094, 0.0525,  -0.3742, -0.7729, -0.7138, 0.3253,  0.0419,  0.8433,  -0.5313,
                        -0.0988, -0.0927, -0.7071, -0.7740, -1.1087, -1.1925, -1.2749, -0.0862, 0.0693,  0.2937,
                        0.1676,  0.2356,  2.7333,  2.5171,  0.8055,  0.7380,  -0.4437, -0.7257, -0.7154, 0.1801,
                        -1.9323, 1.8184,  0.8196,  0.1371,  -0.0677, -2.2315, 0.0662,  -0.0071, -0.8639, 0.6215,
                        -0.5144, 0.8373,  -0.1072, 0.6184,  0.1985,  -0.7692, -0.5879, -0.0029, 0.0676,  -0.5520}),
    TensorShape({1, 5, 5, 2}), &input_tensor);
  auto input_tensor_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor));
  float rate = 2;
  std::vector<float> phase_advance{0.0000, 1.5708, 3.1416, 4.7124, 6.2832};
  Tensor::CreateFromVector(phase_advance, TensorShape({5, 1}), &input_phase_advance_tensor);
  auto phase_advance_ms =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_phase_advance_tensor));
  std::shared_ptr<TensorTransform> pv = std::make_shared<audio::PhaseVocoder>(rate, phase_advance_ms);
  mindspore::dataset::Execute transform({pv});
  Status status = transform(input_tensor_ms, &input_tensor_ms);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: SlidingWindowCmn op
/// Description: Test basic function of SlidingWindowCmn op
/// Expectation: Get correct number of data
TEST_F(MindDataTestExecute, TestSlidingWindowCmn) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestSlidingWindowCmn.";

  std::shared_ptr<Tensor> input_tensor;
  int32_t cmn_window = 500;
  int32_t min_cmn_window = 50;
  bool center = false;
  bool norm_vars = false;

  // create tensor shape
  TensorShape s = TensorShape({2, 2, 500});
  // init input vector
  std::vector<float> input_vec(s.NumOfElements());
  for (int idx = 0; idx < input_vec.size(); ++idx) {
    input_vec[idx] = std::rand() % (1000) / (1000.0f);
  }
  ASSERT_OK(Tensor::CreateFromVector(input_vec, s, &input_tensor));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor));
  std::shared_ptr<TensorTransform> sliding_window_cmn_op =
    std::make_shared<audio::SlidingWindowCmn>(cmn_window, min_cmn_window, center, norm_vars);

  // apply sliding_window_cmn
  mindspore::dataset::Execute Transform({sliding_window_cmn_op});
  Status status = Transform(input_ms, &input_ms);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: SlidingWindowCmn op
/// Description: Test wrong input args of SlidingWindowCmn op
/// Expectation: Get nullptr of iterator
TEST_F(MindDataTestExecute, TestSlidingWindowCmnWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestSlidingWindowCmnWrongArgs.";

  std::shared_ptr<Tensor> input_tensor;
  // create tensor shape
  TensorShape s = TensorShape({2, 2, 500});
  // init input vector
  std::vector<float> input_vec(s.NumOfElements());
  for (int idx = 0; idx < input_vec.size(); ++idx) {
    input_vec[idx] = std::rand() % (1000) / (1000.0f);
  }
  ASSERT_OK(Tensor::CreateFromVector(input_vec, s, &input_tensor));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input_tensor));

  // SlidingWindowCmn: cmn_window must be greater than or equal to 0.
  std::shared_ptr<TensorTransform> sliding_window_cmn_op_1 =
    std::make_shared<audio::SlidingWindowCmn>(-1, 100, false, false);
  mindspore::dataset::Execute Transform_1({sliding_window_cmn_op_1});
  Status status_1 = Transform_1(input_ms, &input_ms);
  EXPECT_FALSE(status_1.IsOk());

  // SlidingWindowCmn: min_cmn_window must be greater than or equal to 0.
  std::shared_ptr<TensorTransform> sliding_window_cmn_op_2 =
    std::make_shared<audio::SlidingWindowCmn>(500, -1, false, false);
  mindspore::dataset::Execute Transform_2({sliding_window_cmn_op_2});
  Status status_2 = Transform_2(input_ms, &input_ms);
  EXPECT_FALSE(status_2.IsOk());
}

/// Feature: AutoAugment op
/// Description: Test AutoAugment op eager
/// Expectation: Load one image data and process auto augmentation with given policy on it
TEST_F(MindDataTestExecute, TestAutoAugmentEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestAutoAugmentEager.";
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto auto_augment_op = vision::AutoAugment(AutoAugmentPolicy::kImageNet, InterpolationMode::kLinear, {0, 0, 0});

  auto transform = Execute({decode, auto_augment_op});
  Status rc = transform(image, &image);
  EXPECT_EQ(rc, Status::OK());
}

/// Feature: Spectrogram op
/// Description: Test Spectrogram op in eager mode.
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestSpectrogramEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-SpectrogramEager.";
  std::shared_ptr<Tensor> test_input_tensor;
  std::vector<double> waveform = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1};
  ASSERT_OK(Tensor::CreateFromVector(waveform, TensorShape({1, (long)waveform.size()}), &test_input_tensor));
  auto input_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(test_input_tensor));
  std::shared_ptr<TensorTransform> spectrogram =
    std::make_shared<audio::Spectrogram>(8, 8, 4, 0, WindowType::kHann, 2., false, true, BorderType::kReflect, true);
  auto transform = Execute({spectrogram});
  Status rc = transform({input_tensor}, &input_tensor);
  ASSERT_TRUE(rc.IsOk());
}

/// Feature: SpectralCentroid op
/// Description: Test SpectralCentroid op in eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestSpectralCentroidEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-SpectralCentroidEager.";
  std::shared_ptr<Tensor> test_input_tensor;
  std::vector<double> waveform = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1};
  ASSERT_OK(Tensor::CreateFromVector(waveform, TensorShape({1, (long)waveform.size()}), &test_input_tensor));
  auto input_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(test_input_tensor));
  std::shared_ptr<TensorTransform> spectral_centroid =
    std::make_shared<audio::SpectralCentroid>(44100, 8, 8, 4, 1, WindowType::kHann);
  auto transform = Execute({spectral_centroid});
  Status rc = transform({input_tensor}, &input_tensor);
  ASSERT_TRUE(rc.IsOk());
}

/// Feature: SpectralCentroid op
/// Description: Test wrong input args of SpectralCentroid in eager mode
/// Expectation: Throw exception correctly
TEST_F(MindDataTestExecute, TestSpectralCentroidWithWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestSpectralCentroidWithWrongArg.";
  std::shared_ptr<Tensor> test_input_tensor;
  std::vector<double> waveform = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1};
  ASSERT_OK(Tensor::CreateFromVector(waveform, TensorShape({1, (long)waveform.size()}), &test_input_tensor));
  auto input_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(test_input_tensor));
  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  std::shared_ptr<TensorTransform> spectral_centroid =
    std::make_shared<audio::SpectralCentroid>(0, 8, 8, 4, 1, WindowType::kHann);
  auto transform = Execute({spectral_centroid});
  Status rc = transform({input_tensor}, &input_tensor);
  EXPECT_FALSE(rc.IsOk());
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

/// Feature: MaskAlongAxis op
/// Description: Test MaskAlongAxis op
/// Expectation: The returned result is as expected
TEST_F(MindDataTestExecute, TestMaskAlongAxis) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestMaskAlongAxis.";
  std::shared_ptr<Tensor> input;
  TensorShape s = TensorShape({1, 4, 3});
  ASSERT_OK(Tensor::CreateFromVector(
    std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f}), s, &input));
  auto input_tensor = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> mask_along_axis_op = std::make_shared<audio::MaskAlongAxis>(0, 2, 9.0, 2);
  mindspore::dataset::Execute transform({mask_along_axis_op});
  Status status = transform(input_tensor, &input_tensor);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: Resample.
/// Description: test Resample in eager mode.
/// Expectation: the data is processed successfully.
TEST_F(MindDataTestExecute, TestResampleEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestResampleEager.";
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(
    std::vector<double>({9.1553e-05, 6.1035e-05, 6.1035e-05, -2.1362e-04, -1.8311e-04, -1.8311e-04}),
    TensorShape({1, 6}), &input));
  auto input_01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  int orig_freq = 6;
  int new_freq = 2;
  ResampleMethod resample_method = ResampleMethod::kSincInterpolation;
  std::shared_ptr<TensorTransform> resample_01 =
    std::make_shared<audio::Resample>(orig_freq, new_freq, resample_method);
  mindspore::dataset::Execute Transform_01({resample_01});
  Status res_01 = Transform_01(input_01, &input_01);
  EXPECT_TRUE(res_01.IsOk());

  ASSERT_OK(Tensor::CreateFromVector(std::vector<double>({9.1553e-05, 6.1035e-05, 6.1035e-05, -2.1362e-04, -1.8311e-04,
                                                          -1.8311e-04, 3.6597e-03, 5.5943e-05, 2.3598e-4, 3.5948e-4}),
                                     TensorShape({2, 5}), &input));
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  resample_method = ResampleMethod::kKaiserWindow;
  std::shared_ptr<TensorTransform> resample_02 =
    std::make_shared<audio::Resample>(orig_freq, new_freq, resample_method);
  mindspore::dataset::Execute Transform_02({resample_02});
  Status res_02 = Transform_02(input_02, &input_02);
  EXPECT_TRUE(res_02.IsOk());
}

/// Feature: Resample
/// Description: test wrong input args of Resample
/// Expectation: get false status
TEST_F(MindDataTestExecute, TestResampleWithInvalidArg) {
   MS_LOG(INFO) << "Doing MindDataTestExecute-TestResampleInvalidArgs.";
  std::shared_ptr<Tensor> input;
  TensorShape s = TensorShape({1, 6});
  ASSERT_OK(Tensor::CreateFromVector(
    std::vector<double>({1.4695e-05, 6.1035e-05, 1.4266e-05, -2.1362e-04, -1.8311e-04, -2.3511e-04}), s, &input));
  auto input_01 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  int orig_freq = 0;
  int new_freq = 2;
  ResampleMethod resample_method = ResampleMethod::kSincInterpolation;
  std::shared_ptr<TensorTransform> resample_01 =
    std::make_shared<audio::Resample>(orig_freq, new_freq, resample_method);
  mindspore::dataset::Execute Transform_01({resample_01});
  Status res_01 = Transform_01(input_01, &input_01);
  // Expect failure, orig_freq can not be zero.
  EXPECT_FALSE(res_01.IsOk());

  s = TensorShape({1, 7});
  ASSERT_OK(Tensor::CreateFromVector(
    std::vector<double>({1.2207e-04, 0.8090e-05, 6.1035e-05, 0.5878e-05, -2.1362e-04, -1.8311e-04, -1.8311e-04}), s,
    &input));
  auto input_02 = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  orig_freq = 1;
  new_freq = -1;
  std::shared_ptr<TensorTransform> resample_02 =
    std::make_shared<audio::Resample>(orig_freq, new_freq, resample_method);
  mindspore::dataset::Execute Transform_02({resample_02});
  Status res_02 = Transform_02(input_02, &input_02);
  // Expect failure, new_freq can not be negative.
  EXPECT_FALSE(res_02.IsOk());
}

/// Feature: Execute Transform op
/// Description: Test executing Decode then Erase op with eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestEraseEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestEraseEager.";
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto erase_op = vision::Erase(10, 10, 10, 10);

  auto transform = Execute({decode, erase_op});
  Status rc = transform(image, &image);
  EXPECT_EQ(rc, Status::OK());
}

/// Feature: AdjustBrightness
/// Description: Test executing AdjustBrightness op in eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestAdjustBrightness) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestAdjustBrightness.";
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto adjust_brightness_op = vision::AdjustBrightness(1);

  auto transform = Execute({decode, adjust_brightness_op});
  Status rc = transform(image, &image);
  EXPECT_EQ(rc, Status::OK());
}

/// Feature: AdjustSharpness
/// Description: Test executing Decode then AdjustSharpness op with eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestAdjustSharpnessEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestAdjustSharpnessEager.";
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto adjust_sharpness_op = vision::AdjustSharpness(1.0);

  auto transform = Execute({decode, adjust_sharpness_op});
  Status rc = transform(image, &image);
  EXPECT_EQ(rc, Status::OK());
}

/// Feature: AdjustSaturation
/// Description: Test executing Decode then AdjustSaturation op with eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestAdjustSaturationEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestAdjustSaturationEager.";
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto adjust_saturation_op = vision::AdjustSaturation(1.0);

  auto transform = Execute({decode, adjust_saturation_op});
  Status rc = transform(image, &image);
  EXPECT_EQ(rc, Status::OK());
}

/// Feature: Posterize
/// Description: Test executing Posterize op in eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestPosterizeEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestPosterizeEager.";
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto posterize_op = vision::Posterize(1);

  auto transform = Execute({decode, posterize_op});
  Status rc = transform(image, &image);
  EXPECT_EQ(rc, Status::OK());
}

/// Feature: AdjustHue
/// Description: Test executing AdjustHue op in eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestAdjustHue) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestAdjustHue.";
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto adjust_hue_op = vision::AdjustHue(0.2);

  auto transform = Execute({decode, adjust_hue_op});
  Status rc = transform(image, &image);
  EXPECT_EQ(rc, Status::OK());
}

/// Feature: AdjustContrast
/// Description: Test executing AdjustContrast op in eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestAdjustContrast) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestAdjustContrast.";
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  auto adjust_contrast_op = vision::AdjustContrast(1);

  auto transform = Execute({decode, adjust_contrast_op});
  Status rc = transform(image, &image);
  EXPECT_EQ(rc, Status::OK());
}

/// Feature: ResizedCrop
/// Description: Test executing Decode op then ResizedCrop op
/// Expectation: Throw correct error and message
TEST_F(MindDataTestExecute, TestResizedCrop) {
  auto image = ReadFileToTensor("data/dataset/apple.jpg");
  std::shared_ptr<TensorTransform> decode_op = std::make_shared<vision::Decode>();
  std::shared_ptr<TensorTransform> resizedCrop_op =
    std::make_shared<vision::ResizedCrop>(0, 0, 128, 128, std::vector<int32_t>{128, 128});

  // Test Compute(Tensor, Tensor) method of ResizedCrop
  auto transform = Execute({decode_op, resizedCrop_op});
  Status rc = transform(image, &image);

  EXPECT_EQ(rc, Status::OK());
}

/// Feature: RandAugment
/// Description: test RandAugment eager
/// Expectation: load one image data and process rand augmentation on it
TEST_F(MindDataTestExecute, TestRandAugmentEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestRandAugmentEager.";
  auto image = ReadFileToTensor("data/dataset/apple.jpg");
  auto decode = vision::Decode();
  auto rand_augment_op = vision::RandAugment(3, 4, 5, InterpolationMode::kLinear, {0, 0, 0});
  auto transform = Execute({decode, rand_augment_op});
  Status rc = transform(image, &image);
  EXPECT_EQ(rc, Status::OK());
}

/// Feature: Perspective
/// Description: Test executing Perspective op in eager mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestPerspective) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestPerspective.";
  // Read images
  auto image = ReadFileToTensor("data/dataset/apple.jpg");

  // Transform params
  auto decode = vision::Decode();
  std::vector<std::vector<int32_t>> src = {{0, 200}, {400, 200}, {400, 0}, {0, 0}};
  std::vector<std::vector<int32_t>> dst = {{0, 180}, {400, 180}, {400, 0}, {0, 0}};
  auto perspective_op = vision::Perspective(src, dst, InterpolationMode::kLinear);

  auto transform = Execute({decode, perspective_op});
  Status rc = transform(image, &image);
  EXPECT_EQ(rc, Status::OK());
}

/// Feature: AddToken op
/// Description: Test basic usage of AddToken op
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestAddToken) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestAddToken.";
  std::vector<std::string> input_vectors = {"a", "b", "c", "d", "e"};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(input_vectors, &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> add_token_op = std::make_shared<text::AddToken>("Token", true);
  // apply AddToken
  mindspore::dataset::Execute trans({add_token_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: LFCC op
/// Description: Test basic usage of LFCC op
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestLFCCEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestLFCC.";
  // Original waveform
  std::vector<float> labels = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2,
                               2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({1, 1, 30}), &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> lfcc_op =
    std::make_shared<audio::LFCC>(16000, 128, 4, 0.0, 10000.0, 2, NormMode::kOrtho, true);
  // apply LFCC
  mindspore::dataset::Execute trans({lfcc_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: LFCC op
/// Description: Wrong dct_type of LFCC op
/// Expectation: Get false status
TEST_F(MindDataTestExecute, TestLFCCWrongArgsDctType) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestLFCCWrongArgsDctType.";
  // Original waveform
  std::vector<float> labels = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2,
                               2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({1, 1, 30}), &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> lfcc_op =
    std::make_shared<audio::LFCC>(16000, 128, 4, 0.0, 10000.0, -2, NormMode::kOrtho, true);
  // apply LFCC
  mindspore::dataset::Execute trans({lfcc_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_FALSE(status.IsOk());
}

/// Feature: Truncate
/// Description: Test basic usage of Truncate op
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestTruncateOpStr) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestTruncateOpStr.";
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(std::vector<std::string>({"hello", "hhx", "hyx", "world", "this", "is"}), &input);
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> truncate_op = std::make_shared<text::Truncate>(3);
  // apply Truncate
  mindspore::dataset::Execute trans({truncate_op});
   Status status = trans(input_ms, &input_ms);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: MFCC op
/// Description: Test basic usage of MFCC op
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestMFCCEager) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestMFCC.";
  // Original waveform
  std::vector<float> labels = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2,
                               2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({1, 1, 30}), &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> mfcc_op =
    std::make_shared<audio::MFCC>(16000, 4, 2, NormMode::kOrtho, true, 10);
  // apply MFCC
  mindspore::dataset::Execute trans({mfcc_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: MelSpectrogram op
/// Description: Test basic usage of MelSpectrogram op
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestMelSpectrogram) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestMelSpectrogram.";
  // Original waveform
  std::vector<float> labels = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2,
                               2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({1, 1, 30}), &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> mel_spectrogram_op =
    std::make_shared<audio::MelSpectrogram>(16000, 16, 16, 8, 0.0, 10000.0, 0, 8, WindowType::kHann, 2.0, false, true,
                                            BorderType::kReflect, true, NormType::kNone, MelType::kHtk);
  // apply MelSpectrogram
  mindspore::dataset::Execute trans({mel_spectrogram_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: MelSpectrogram op
/// Description: First test wrong args for MelSpectrogram
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestMelSpectrogramWrongArgs1) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestMelSpectrogramWrongArgs1.";
  std::vector<float> labels = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2,
                               2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({1, 1, 30}), &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> mel_spectrogram_op =
    std::make_shared<audio::MelSpectrogram>(16000, -16, 16, 8, 0.0, 10000.0, 0, 8, WindowType::kHann, 2.0, false, true,
                                            BorderType::kReflect, true, NormType::kNone, MelType::kHtk);
  mindspore::dataset::Execute trans({mel_spectrogram_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());

  mel_spectrogram_op =
    std::make_shared<audio::MelSpectrogram>(16000, 16, -16, 8, 0.0, 10000.0, 0, 8, WindowType::kHann, 2.0, false, true,
                                            BorderType::kReflect, true, NormType::kNone, MelType::kHtk);
  mindspore::dataset::Execute trans1({mel_spectrogram_op});
  status = trans1(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());
  mel_spectrogram_op =
    std::make_shared<audio::MelSpectrogram>(16000, 16, 16, -8, 0.0, 10000.0, 0, 8, WindowType::kHann, 2.0, false, true,
                                            BorderType::kReflect, true, NormType::kNone, MelType::kHtk);
  mindspore::dataset::Execute trans2({mel_spectrogram_op});
  status = trans2(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());
  mel_spectrogram_op =
    std::make_shared<audio::MelSpectrogram>(16000, 16, 16, -8, 0.0, 10000.0, 0, 8, WindowType::kHann, 2.0, false, true,
                                            BorderType::kReflect, true, NormType::kNone, MelType::kHtk);
  mindspore::dataset::Execute trans3({mel_spectrogram_op});
  status = trans3(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());
  mel_spectrogram_op =
    std::make_shared<audio::MelSpectrogram>(16000, 16, 16, 8, 10000.0, 1.0, 0, 8, WindowType::kHann, 2.0, false, true,
                                            BorderType::kReflect, true, NormType::kNone, MelType::kHtk);
  mindspore::dataset::Execute trans4({mel_spectrogram_op});
  status = trans4(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());
  mel_spectrogram_op =
    std::make_shared<audio::MelSpectrogram>(16000, 16, 16, 8, 0.0, -10000.0, 0, 8, WindowType::kHann, 2.0, false, true,
                                            BorderType::kReflect, true, NormType::kNone, MelType::kHtk);
  mindspore::dataset::Execute trans5({mel_spectrogram_op});
  status = trans5(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());
  mel_spectrogram_op =
    std::make_shared<audio::MelSpectrogram>(16000, 16, 16, 8, 0.0, 10000.0, -1, 8, WindowType::kHann, 2.0, false, true,
                                            BorderType::kReflect, true, NormType::kNone, MelType::kHtk);
  mindspore::dataset::Execute trans6({mel_spectrogram_op});
  status = trans6(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());
}

/// Feature: MelSpectrogram op
/// Description: Second test wrong args for MelSpectrogram
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestMelSpectrogramWrongArgs2) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestMelSpectrogramWrongArgs2.";
  std::vector<float> labels = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2,
                               2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({1, 1, 30}), &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> mel_spectrogram_op =
    std::make_shared<audio::MelSpectrogram>(16000, 16, 16, 8, 0.0, 10000.0, 0, -8, WindowType::kHann, 2.0, false, true,
                                            BorderType::kReflect, true, NormType::kNone, MelType::kHtk);
  mindspore::dataset::Execute trans({mel_spectrogram_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());
  mel_spectrogram_op =
    std::make_shared<audio::MelSpectrogram>(16000, 16, 16, 8, 0.0, 10000.0, 0, 8, WindowType::kHann, -2.0, false, true,
                                            BorderType::kReflect, true, NormType::kNone, MelType::kHtk);
  mindspore::dataset::Execute trans2({mel_spectrogram_op});
  status = trans2(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());
}

/// Feature: InverseSpectrogram op
/// Description: Test basic usage of InverseSpectrogram op
/// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestInverseSpectrogram) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestInverseSpectrogram.";
  // Original spectrogram
  std::vector<float> labels = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 5, 5, 4, 4, 3, 3, 
                               2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({2, 9, 1, 2}), &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> inverse_spectrogram_op =
    std::make_shared<audio::InverseSpectrogram>(1, 16, 16, 8, 0, WindowType::kHann, false, true,
                                                BorderType::kReflect, true);
  // apply InverseSpectrogram
  mindspore::dataset::Execute trans({inverse_spectrogram_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_TRUE(status.IsOk());
}

/// Feature: InverseSpectrogram op
/// Description: Test wrong args for InverseSpectrogram
/// Expectation: Throw correct error and message
TEST_F(MindDataTestExecute, TestInverseSpectrogramWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestInverseSpectrogramWrongArgs.";
  std::vector<float> labels = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 5, 5, 4, 4, 3, 3, 
                               2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({2, 9, 1, 2}), &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> inverse_spectrogram_op =
    std::make_shared<audio::InverseSpectrogram>(1, -16, 16, 8, 0, WindowType::kHann, false, true,
                                                BorderType::kReflect, true);
  mindspore::dataset::Execute trans({inverse_spectrogram_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());

  inverse_spectrogram_op =
    std::make_shared<audio::InverseSpectrogram>(1, 16, -16, 8, 0, WindowType::kHann, false, true,
                                                BorderType::kReflect, true);
  mindspore::dataset::Execute trans1({inverse_spectrogram_op});
  status = trans1(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());

  inverse_spectrogram_op =
    std::make_shared<audio::InverseSpectrogram>(1, 16, 16, -8, 0, WindowType::kHann, false, true,
                                                BorderType::kReflect, true);
  mindspore::dataset::Execute trans2({inverse_spectrogram_op});
  status = trans2(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());

  inverse_spectrogram_op =
    std::make_shared<audio::InverseSpectrogram>(1, 16, 16, 8, -1, WindowType::kHann, false, true,
                                                BorderType::kReflect, true);
  mindspore::dataset::Execute trans3({inverse_spectrogram_op});
  status = trans3(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());
}

// Feature: PitchShift op
// Description: Test basic usage of PitchShift op
// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestAdjustPitchShift) {
  MS_LOG(INFO) << "Doing MindDataExecute-TestAdjustPitchShift.";
  // Original waveform
  std::vector<float> labels = {1, 1, 2, 3, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3,
                               2, 1, 2, 3, 0, 1, 0, 2, 4, 5, 3, 1, 2, 3, 4};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({1, 1, 30}), &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> PitchShift_op =
    std::make_shared<audio::PitchShift>(16000, 4, 12, 16, 16, 4, WindowType::kHann);
  // apply PitchShift
  mindspore::dataset::Execute trans({PitchShift_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_TRUE(status.IsOk());
}

// Feature: PitchShift op
// Description: First test wrong args of PitchShift
// Expectation: The data is processed successfully
TEST_F(MindDataTestExecute, TestPitchShiftWrongArgs1) {
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestPitchShiftWrongArgs1.";
  std::vector<float> labels = {1, 1, 2, 3, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3,
                               2, 1, 2, 3, 0, 1, 0, 2, 4, 5, 3, 1, 2, 3, 4};
  std::shared_ptr<Tensor> input;
  ASSERT_OK(Tensor::CreateFromVector(labels, TensorShape({1, 1, 30}), &input));
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(input));
  std::shared_ptr<TensorTransform> PitchShift_op =
    std::make_shared<audio::PitchShift>(16000, 4, 12, -16, 16, 4, WindowType::kHann);
  mindspore::dataset::Execute trans({PitchShift_op});
  Status status = trans(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());

  PitchShift_op = std::make_shared<audio::PitchShift>(16000, 4, 0, 16, 16, 4, WindowType::kHann);
  mindspore::dataset::Execute trans1({PitchShift_op});
  status = trans1(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());

  PitchShift_op = std::make_shared<audio::PitchShift>();
  mindspore::dataset::Execute trans2({PitchShift_op});
  status = trans2(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());

  PitchShift_op = std::make_shared<audio::PitchShift>();
  mindspore::dataset::Execute trans4({PitchShift_op});
  status = trans4(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());

  PitchShift_op = std::make_shared<audio::PitchShift>(16000, 4, 12, 16, -16, 4, WindowType::kHann);
  mindspore::dataset::Execute trans5({PitchShift_op});
  status = trans5(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());
  
  PitchShift_op = std::make_shared<audio::PitchShift>(16000, 4, 12, 16, 16, -4, WindowType::kHann);
  mindspore::dataset::Execute trans6({PitchShift_op});
  status = trans6(input_ms, &input_ms);
  EXPECT_TRUE(status.IsError());
}
