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
#include "minddata/dataset/include/dataset/text.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/include/dataset/vision.h"
#include "minddata/dataset/text/char_n_gram.h"
#include "minddata/dataset/text/fast_text.h"
#include "minddata/dataset/text/glove.h"
#include "minddata/dataset/text/vectors.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::dataset::CharNGram;
using mindspore::dataset::FastText;
using mindspore::dataset::GloVe;
using mindspore::dataset::Vectors;

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

/// Feature: FilterWikipediaXMLEager op
/// Description: Test FilterWikipediaXML's Eager mode
/// Expectation: Run successfully
TEST_F(MindDataTestExecute, TestFilterWikipediaXMLEager) {
  // Test FilterWikipediaXML's Eager mode
  MS_LOG(INFO) << "Doing MindDataTestExecute-TestFilterWikipediaXMLEager.";
  std::vector<std::string> origin = {"中国", "Wcdma", "Pang", "Yuchao"};
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

#ifdef ENABLE_FFMPEG

#endif
