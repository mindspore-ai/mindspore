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
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/include/dataset/vision.h"
#include "mindspore/ccsrc/minddata/dataset/core/tensor.h"
#include "mindspore/ccsrc/minddata/dataset/core/data_type.h"

using namespace mindspore::dataset;
using mindspore::dataset::BorderType;
using mindspore::dataset::Tensor;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

// Tests for data transforms ops (in alphabetical order)

/// Feature: Compose op
/// Description: Test Compose op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestComposeSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComposeSuccess.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto decode_op = std::make_shared<vision::Decode>();
  auto resize_op = std::make_shared<vision::Resize>(std::vector<int32_t>{777, 777});
  transforms::Compose compose({decode_op, resize_op});

  // Create a Map operation on ds
  ds = ds->Map({compose}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Label shape: " << label.Shape();
    EXPECT_EQ(image.Shape()[0], 777);
    EXPECT_EQ(image.Shape()[1], 777);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Compose op
/// Description: Test Compose op with invalid transform op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestComposeFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComposeFail1 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Resize: Non-positive size value: -1 at element: 0
  // Compose: transform ops must not be null
  auto decode_op = vision::Decode();
  auto resize_op = vision::Resize({-1});
  auto compose = transforms::Compose({decode_op, resize_op});

  // Create a Map operation on ds
  ds = ds->Map({compose}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Compose parameter(invalid transform op)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Compose op
/// Description: Test Compose op with null transform op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestComposeFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComposeFail2 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Compose: transform ops must not be null
  std::shared_ptr<TensorTransform> decode_op = std::make_shared<vision::Decode>();
  auto compose =
    std::make_shared<transforms::Compose>(std::vector<std::shared_ptr<TensorTransform>>{decode_op, nullptr});

  // Create a Map operation on ds
  ds = ds->Map({compose}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Compose parameter (transform ops must not be null)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Compose op
/// Description: Test Compose op with empty transform list
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestComposeFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComposeFail3 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Compose: transform list must not be empty
  std::vector<std::shared_ptr<TensorTransform>> list = {};
  auto compose = transforms::Compose(list);

  // Create a Map operation on ds
  ds = ds->Map({compose}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid Compose parameter (transform list must not be empty)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Concatenate op
/// Description: Test basic Concatenate op with prepend and append
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestConcatenateSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConcatenateSuccess1.";
  // Test basic concatenate with prepend and append

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {1}));
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Create Concatenate op
  std::vector<int16_t> prepend_vector = {1, 2};
  std::shared_ptr<Tensor> prepend_tensor;
  ASSERT_OK(Tensor::CreateFromVector(prepend_vector, &prepend_tensor));
  mindspore::MSTensor prepend_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(prepend_tensor));

  std::vector<int16_t> append_vector = {3};
  std::shared_ptr<Tensor> append_tensor;
  ASSERT_OK(Tensor::CreateFromVector(append_vector, &append_tensor));
  mindspore::MSTensor append_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(append_tensor));

  transforms::Concatenate concatenate = transforms::Concatenate(0, prepend_MSTensor, append_MSTensor);

  // Create a Map operation on ds
  ds = ds->Map({concatenate}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<int16_t>> expected = {
    {1, 2, 31354, 3}, {1, 2, -17734, 3}, {1, 2, -5655, 3}, {1, 2, -17220, 3}};

  // Check concatenate results
  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Concatenate op
/// Description: Test Concatenate op with no input
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestConcatenateSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConcatenateSuccess2.";
  // Test concatenate with no input

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {1}));
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  transforms::Concatenate concatenate = transforms::Concatenate();

  // Create a Map operation on ds
  ds = ds->Map({concatenate}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  // The data generated by RandomData
  std::vector<std::vector<int16_t>> expected = {{31354}, {-17734}, {-5655}, {-17220}};

  // Check concatenate results
  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Concatenate op
/// Description: Test Concatenate op with strings
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestConcatenateSuccess3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConcatenateSuccess3.";
  // Test concatenate of string

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(1);
  EXPECT_NE(ds, nullptr);

  // Create BasicTokenizer operation on ds
  std::shared_ptr<TensorTransform> basic_tokenizer = std::make_shared<text::BasicTokenizer>(true);
  EXPECT_NE(basic_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({basic_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create Concatenate op
  std::vector<std::string> prepend_vector = {"1", "2"};
  std::shared_ptr<Tensor> prepend_tensor;
  ASSERT_OK(Tensor::CreateFromVector(prepend_vector, &prepend_tensor));
  mindspore::MSTensor prepend_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(prepend_tensor));

  std::vector<std::string> append_vector = {"3"};
  std::shared_ptr<Tensor> append_tensor;
  ASSERT_OK(Tensor::CreateFromVector(append_vector, &append_tensor));
  mindspore::MSTensor append_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(append_tensor));

  transforms::Concatenate concatenate = transforms::Concatenate(0, prepend_MSTensor, append_MSTensor);

  // Create a Map operation on ds
  ds = ds->Map({concatenate}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<std::string>> expected = {{"1", "2", "welcome", "to", "beijing", "!", "3"}};

  // Check concatenate results
  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
  // GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Concatenate op
/// Description: Test Concatenate op with negative axis
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestConcatenateSuccess4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConcatenateSuccess4.";
  // Test concatenate with negative axis

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {1}));
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Create Concatenate op
  std::vector<int16_t> prepend_vector = {1, 2};
  std::shared_ptr<Tensor> prepend_tensor;
  ASSERT_OK(Tensor::CreateFromVector(prepend_vector, &prepend_tensor));
  mindspore::MSTensor prepend_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(prepend_tensor));

  std::vector<int16_t> append_vector = {3};
  std::shared_ptr<Tensor> append_tensor;
  ASSERT_OK(Tensor::CreateFromVector(append_vector, &append_tensor));
  mindspore::MSTensor append_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(append_tensor));

  transforms::Concatenate concatenate = transforms::Concatenate(-1, prepend_MSTensor, append_MSTensor);

  // Create a Map operation on ds
  ds = ds->Map({concatenate}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<int16_t>> expected = {
    {1, 2, 31354, 3}, {1, 2, -17734, 3}, {1, 2, -5655, 3}, {1, 2, -17220, 3}};

  // Check concatenate results
  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Concatenate op
/// Description: Test Concatenate op with type mismatch
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestConcatenateFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConcatenateFail1.";
  // Test concatenate with type mismatch

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {1}));
  std::shared_ptr<Dataset> ds = RandomData(1, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(1);
  EXPECT_NE(ds, nullptr);

  // Create Concatenate op
  std::vector<std::string> prepend_vector = {"1", "2"};
  std::shared_ptr<Tensor> prepend_tensor;
  ASSERT_OK(Tensor::CreateFromVector(prepend_vector, &prepend_tensor));
  mindspore::MSTensor prepend_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(prepend_tensor));

  std::vector<std::string> append_vector = {"3"};
  std::shared_ptr<Tensor> append_tensor;
  ASSERT_OK(Tensor::CreateFromVector(append_vector, &append_tensor));
  mindspore::MSTensor append_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(append_tensor));

  transforms::Concatenate concatenate = transforms::Concatenate(0, prepend_MSTensor, append_MSTensor);

  // Create a Map operation on ds
  ds = ds->Map({concatenate}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // EXPECT_EQ(iter, nullptr);
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  // Expect failure: type mismatch, concatenate string tensor to dataset of Int16
  EXPECT_ERROR(iter->GetNextRow(&row));

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Concatenate op
/// Description: Test Concatenate op with incorrect dimension
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestConcatenateFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConcatenateFail2.";
  // Test concatenate with incorrect dimension

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {1, 2}));
  std::shared_ptr<Dataset> ds = RandomData(1, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(1);
  EXPECT_NE(ds, nullptr);

  // Create Concatenate op
  std::vector<int16_t> prepend_vector = {1, 2};
  std::shared_ptr<Tensor> prepend_tensor;
  ASSERT_OK(Tensor::CreateFromVector(prepend_vector, &prepend_tensor));
  mindspore::MSTensor prepend_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(prepend_tensor));

  std::vector<int16_t> append_vector = {3};
  std::shared_ptr<Tensor> append_tensor;
  ASSERT_OK(Tensor::CreateFromVector(append_vector, &append_tensor));
  mindspore::MSTensor append_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(append_tensor));

  transforms::Concatenate concatenate = transforms::Concatenate(0, prepend_MSTensor, append_MSTensor);

  // Create a Map operation on ds
  ds = ds->Map({concatenate}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  // Expect failure: concatenate on 2D dataset, only support 1D concatenate so far
  EXPECT_ERROR(iter->GetNextRow(&row));

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Concatenate op
/// Description: Test Concatenate op with wrong axis
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestConcatenateFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestConcatenateFail3.";
  // Test concatenate with wrong axis

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {1}));
  std::shared_ptr<Dataset> ds = RandomData(1, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(1);
  EXPECT_NE(ds, nullptr);

  // Create Concatenate op
  std::vector<int16_t> prepend_vector = {1, 2};
  std::shared_ptr<Tensor> prepend_tensor;
  ASSERT_OK(Tensor::CreateFromVector(prepend_vector, &prepend_tensor));
  mindspore::MSTensor prepend_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(prepend_tensor));

  std::vector<int16_t> append_vector = {3};
  std::shared_ptr<Tensor> append_tensor;
  ASSERT_OK(Tensor::CreateFromVector(append_vector, &append_tensor));
  mindspore::MSTensor append_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(append_tensor));
  // The parameter axis support 0 or -1 only for now
  transforms::Concatenate concatenate = transforms::Concatenate(2, prepend_MSTensor, append_MSTensor);

  // Create a Map operation on ds
  ds = ds->Map({concatenate}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: wrong axis, axis can only be 0 or -1
  EXPECT_EQ(iter, nullptr);

  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Duplicate op
/// Description: Test Duplicate op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestDuplicateSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDuplicateSuccess.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  transforms::Duplicate duplicate = transforms::Duplicate();

  // Create a Map operation on ds
  ds = ds->Map({duplicate}, {"image"}, {"image", "image_copy"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto image_copy = row["image_copy"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    EXPECT_MSTENSOR_EQ(image, image_copy);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 10);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Fill op
/// Description: Test Fill op basic usage on RandomDataset with Int32 numbers for given shape
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFillSuccessInt) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFillSuccessInt.";

  // Create a RandomDataset with Int32 numbers for given shape
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(864);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt32, {6}));
  std::shared_ptr<Dataset> ds = RandomData(5, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(3);
  EXPECT_NE(ds, nullptr);

  // Create Fill op - to fill with 3
  std::shared_ptr<Tensor> fill_value_tensor;
  ASSERT_OK(Tensor::CreateScalar(3, &fill_value_tensor));
  mindspore::MSTensor fill_value_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(fill_value_tensor));
  transforms::Fill mask = transforms::Fill(fill_value_MSTensor);
  ds = ds->Map({mask}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<int32_t>> expected = {
    {3, 3, 3, 3, 3, 3}, {3, 3, 3, 3, 3, 3}, {3, 3, 3, 3, 3, 3}, {3, 3, 3, 3, 3, 3}, {3, 3, 3, 3, 3, 3}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Fill op
/// Description: Test Fill op basic usage on RandomDataset with bool values for given shape
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFillSuccessBool) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFillSuccessBool.";

  // Create a RandomDataset with bool values for given shape
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(963);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeBool, {4}));
  std::shared_ptr<Dataset> ds = RandomData(3, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Create Fill op - to fill with zero
  std::shared_ptr<Tensor> fill_value_tensor;
  ASSERT_OK(Tensor::CreateScalar((bool)true, &fill_value_tensor));
  mindspore::MSTensor fill_value_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(fill_value_tensor));
  transforms::Fill mask = transforms::Fill(fill_value_MSTensor);
  ds = ds->Map({mask}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<bool>> expected = {
    {true, true, true, true}, {true, true, true, true}, {true, true, true, true}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Fill op
/// Description: Test Fill op using negative numbers on RandomDataset with UInt8 numbers for given shape,
///     so there will be down typecasting
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFillSuccessDownTypecast) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFillSuccessDownTypecast.";

  // Create a RandomDataset with UInt8 numbers for given shape
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(963);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeUInt8, {4}));
  std::shared_ptr<Dataset> ds = RandomData(3, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Create Fill op - to fill with -3
  std::shared_ptr<Tensor> fill_value_tensor;
  ASSERT_OK(Tensor::CreateScalar(-3, &fill_value_tensor));
  mindspore::MSTensor fill_value_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(fill_value_tensor));
  transforms::Fill mask = transforms::Fill(fill_value_MSTensor);
  ds = ds->Map({mask}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  // Note: 2**8 -3 = 256 -3 = 253
  std::vector<std::vector<uint8_t>> expected = {{253, 253, 253, 253}, {253, 253, 253, 253}, {253, 253, 253, 253}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Fill op
/// Description: Test Fill op using 0 on RandomDataset with UInt8 numbers for given shape,
///     so there will be down typecasting to 0
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFillSuccessDownTypecastZero) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFillSuccessDownTypecastZero.";

  // Create a RandomDataset with UInt8 numbers for given shape
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(963);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeUInt8, {4}));
  std::shared_ptr<Dataset> ds = RandomData(3, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Create Fill op - to fill with zero
  std::shared_ptr<Tensor> fill_value_tensor;
  ASSERT_OK(Tensor::CreateScalar(0, &fill_value_tensor));
  mindspore::MSTensor fill_value_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(fill_value_tensor));
  transforms::Fill mask = transforms::Fill(fill_value_MSTensor);
  ds = ds->Map({mask}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  // Note: 2**8 = 256
  std::vector<std::vector<uint8_t>> expected = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Fill op
/// Description: Test Fill op using negative numbers on RandomDataset with UInt16 numbers for given shape,
///     so there will be down typecasting
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFillSuccessDownTypecast16) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFillSuccessDownTypecast16.";

  // Create a RandomDataset with UInt16 numbers for given shape
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(963);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeUInt16, {4}));
  std::shared_ptr<Dataset> ds = RandomData(3, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Create Fill op - to fill with -3
  std::shared_ptr<Tensor> fill_value_tensor;
  ASSERT_OK(Tensor::CreateScalar(-3, &fill_value_tensor));
  mindspore::MSTensor fill_value_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(fill_value_tensor));
  transforms::Fill mask = transforms::Fill(fill_value_MSTensor);
  ds = ds->Map({mask}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  // Note: 2**16 -3 = 65536 -3 = 65533
  std::vector<std::vector<uint16_t>> expected = {
    {65533, 65533, 65533, 65533}, {65533, 65533, 65533, 65533}, {65533, 65533, 65533, 65533}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Fill op
/// Description: Test Fill op using 0 on RandomDataset with Float numbers for given shape,
///     so there will be up typecasting to 0 scalar value
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFillSuccessUpTypecast) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFillSuccessUpTypecast.";

  // Create a RandomDataset with Float numbers for given shape
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(963);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeFloat32, {2}));
  std::shared_ptr<Dataset> ds = RandomData((float)4.0, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Create Fill op - to fill with zeroes
  std::shared_ptr<Tensor> fill_value_tensor;
  ASSERT_OK(Tensor::CreateScalar(0, &fill_value_tensor));
  mindspore::MSTensor fill_value_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(fill_value_tensor));
  transforms::Fill mask = transforms::Fill(fill_value_MSTensor);
  ds = ds->Map({mask}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<float_t>> expected = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);

    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Fill op
/// Description: Test Fill op on TextFileDataset which contains strings
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFillSuccessString) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFillSuccessString.";

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/basic_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Skip operation on ds
  ds = ds->Skip(6);
  EXPECT_NE(ds, nullptr);

  // Create BasicTokenizer operation on ds
  std::shared_ptr<TensorTransform> basic_tokenizer = std::make_shared<text::BasicTokenizer>(true);
  EXPECT_NE(basic_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({basic_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create Fill op - to fill with string
  std::shared_ptr<Tensor> fill_value_tensor;
  ASSERT_OK(Tensor::CreateScalar<std::string>("Hello", &fill_value_tensor));
  mindspore::MSTensor fill_value_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(fill_value_tensor));
  transforms::Fill mask = transforms::Fill(fill_value_MSTensor);
  ds = ds->Map({mask}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::string> expected = {"Hello", "Hello", "Hello", "Hello", "Hello"};
  std::shared_ptr<Tensor> de_expected_tensor;
  ASSERT_OK(Tensor::CreateFromVector(expected, &de_expected_tensor));
  mindspore::MSTensor expected_tensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    TEST_MS_LOG_MSTENSOR(INFO, "ind: ", ind);
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Fill op
/// Description: Test Fill op with wrongful vector shape instead of scalar
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestFillFailFillValueNotScalar) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFillFailFillValueNotScalar.";
  // Test BasicTokenizer with lower_case true

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/basic_tokenizer.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Skip operation on ds
  ds = ds->Skip(6);
  EXPECT_NE(ds, nullptr);

  // Create BasicTokenizer operation on ds
  std::shared_ptr<TensorTransform> basic_tokenizer = std::make_shared<text::BasicTokenizer>(true);
  EXPECT_NE(basic_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({basic_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create Fill op - with wrongful vector shape instead of scalar
  std::vector<std::string> fill_string = {"ERROR"};
  std::shared_ptr<Tensor> fill_value_tensor;
  ASSERT_OK(Tensor::CreateFromVector(fill_string, &fill_value_tensor));
  mindspore::MSTensor fill_value_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(fill_value_tensor));
  transforms::Fill mask = transforms::Fill(fill_value_MSTensor);
  ds = ds->Map({mask}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();

  // Expect failure: invalid Fill parameter (the shape of fill_value is not a scalar)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Mask op
/// Description: Test Mask op on RandomDataset with Int16 data type with int
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestMaskSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMaskSuccess1.";
  // Test Mask random int dataset with int

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {4}));
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Create an int Mask op
  std::shared_ptr<Tensor> constant_tensor;
  ASSERT_OK(Tensor::CreateScalar(0, &constant_tensor));
  mindspore::MSTensor constant_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(constant_tensor));
  transforms::Mask mask = transforms::Mask(RelationalOp::kGreater, constant_MSTensor);
  ds = ds->Map({mask}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<bool>> expected = {
    {true, true, true, true}, {false, false, false, false}, {false, false, false, false}, {false, false, false, false}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Mask op
/// Description: Test Mask op on RandomDataset with Float16 data type with float
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestMaskSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMaskSuccess2.";
  // Test Mask random float dataset with float

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeFloat16, {4}));
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Create a float Mask op
  std::shared_ptr<Tensor> constant_tensor;
  ASSERT_OK(Tensor::CreateScalar(-1.1, &constant_tensor));
  mindspore::MSTensor constant_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(constant_tensor));
  // Use explicit input ms_type(kNumberTypeBool) as the mask return type
  transforms::Mask mask =
    transforms::Mask(RelationalOp::kLessEqual, constant_MSTensor, mindspore::DataType::kNumberTypeBool);
  ds = ds->Map({mask}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<bool>> expected = {
    {false, false, false, false}, {false, false, false, false}, {true, true, true, true}, {true, true, true, true}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();

  // Test Mask result boolean dataset with boolean

  // Create another boolean Mask op
  std::shared_ptr<Tensor> constant_tensor2;
  ASSERT_OK(Tensor::CreateScalar<bool>(false, &constant_tensor2));
  mindspore::MSTensor constant_MSTensor2 =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(constant_tensor2));
  transforms::Mask mask2 = transforms::Mask(RelationalOp::kLessEqual, constant_MSTensor2);
  ds = ds->Map({mask2}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<bool>> expected2 = {
    {true, true, true, true}, {true, true, true, true}, {false, false, false, false}, {false, false, false, false}};

  i = 0;
  while (row.size() != 0) {
    auto ind2 = row["col1"];
    std::shared_ptr<Tensor> de_expected_tensor2;
    ASSERT_OK(Tensor::CreateFromVector(expected2[i], &de_expected_tensor2));
    mindspore::MSTensor expected_tensor2 =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor2));
    EXPECT_MSTENSOR_EQ(ind2, expected_tensor2);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();

  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Mask op
/// Description: Test Mask op on TextFileDataset with strings
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestMaskSuccess3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMaskSuccess3.";
  // Test Mask random text dataset with string

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(1);
  EXPECT_NE(ds, nullptr);

  // Create BasicTokenizer operation on ds
  std::shared_ptr<TensorTransform> basic_tokenizer = std::make_shared<text::BasicTokenizer>(true);
  EXPECT_NE(basic_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({basic_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create a string Mask op
  std::shared_ptr<Tensor> constant_tensor;
  ASSERT_OK(Tensor::CreateScalar<std::string>("to", &constant_tensor));
  mindspore::MSTensor constant_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(constant_tensor));
  // Use kNumberTypeInt16 as an explicit ms_type parameter for the mask return type,
  // instead of using default kNumberTypeBool.
  transforms::Mask mask =
    transforms::Mask(RelationalOp::kEqual, constant_MSTensor, mindspore::DataType::kNumberTypeInt16);
  ds = ds->Map({mask}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<int16_t>> expected = {{0, 1, 0, 0}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Mask op
/// Description: Test Mask op with nun-numeric datatype as output result
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMaskFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMaskFail1.";
  // Test Mask with nun-numeric datatype as output result.

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {4}));
  std::shared_ptr<Dataset> ds = RandomData(1, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(1);
  EXPECT_NE(ds, nullptr);

  // Create an int Mask op
  std::shared_ptr<Tensor> constant_tensor;
  ASSERT_OK(Tensor::CreateScalar(0, &constant_tensor));
  mindspore::MSTensor constant_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(constant_tensor));
  transforms::Mask mask =
    transforms::Mask(RelationalOp::kGreater, constant_MSTensor, mindspore::DataType::kObjectTypeString);
  ds = ds->Map({mask}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: using string as output datatype which is invalid
  EXPECT_EQ(iter, nullptr);

  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Mask op
/// Description: Test Mask op with mismatched datatype
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestMaskFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMaskFail2.";
  // Test Mask with mismatched datatype.

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {4}));
  std::shared_ptr<Dataset> ds = RandomData(1, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(1);
  EXPECT_NE(ds, nullptr);

  // Create a string Mask op
  std::shared_ptr<Tensor> constant_tensor;
  ASSERT_OK(Tensor::CreateScalar<std::string>("0", &constant_tensor));
  mindspore::MSTensor constant_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(constant_tensor));
  transforms::Mask mask = transforms::Mask(RelationalOp::kGreater, constant_MSTensor);
  ds = ds->Map({mask}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  // Expect failure: mismatched datatype, mask Int16 with string
  EXPECT_ERROR(iter->GetNextRow(&row));

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: OneHot op
/// Description: Test OneHot op basic usage with default smoothing value
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestOneHotSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOneHotSuccess1.";
  // Testing CutMixBatch on a batch of CHW images
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  int number_of_classes = 10;
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> hwc_to_chw = std::make_shared<vision::HWC2CHW>();

  // Create a Map operation on ds
  ds = ds->Map({hwc_to_chw}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(number_of_classes);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorTransform> cutmix_batch_op =
    std::make_shared<vision::CutMixBatch>(mindspore::dataset::ImageBatchFormat::kNCHW, 1.0, 1.0);

  // Create a Map operation on ds
  ds = ds->Map({cutmix_batch_op}, {"image", "label"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Label shape: " << label.Shape();
    EXPECT_EQ(image.Shape().size() == 4 && batch_size == image.Shape()[0] && 3 == image.Shape()[1] &&
                32 == image.Shape()[2] && 32 == image.Shape()[3],
              true);
    EXPECT_EQ(label.Shape().size() == 2 && batch_size == label.Shape()[0] && number_of_classes == label.Shape()[1],
              true);
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: OneHot op
/// Description: Test OneHot op followed by MixUpBatch op
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestOneHotSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOneHotSuccess2.";
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<TensorTransform> mixup_batch_op = std::make_shared<vision::MixUpBatch>(2.0);

  // Create a Map operation on ds
  ds = ds->Map({mixup_batch_op}, {"image", "label"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: OneHot op
/// Description: Test OneHot op with non-default smoothing rate value
/// Expectation: Rows in the dataset are iterated without failure
TEST_F(MindDataTestPipeline, TestOneHotSuccess3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOneHotSuccess3.";
  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 5;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  // Create OneHot op with non-default smoothing_rate
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(10, 0.2);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 2);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: OneHot op
/// Description: Test OneHot op with invalid num_class=0
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestOneHotFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOneHotFail1 with invalid params.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // incorrect num_class
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(0);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid OneHot input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: OneHot op
/// Description: Test OneHot op with invalid num_class < 0
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestOneHotFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOneHotFail2 with invalid params.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // incorrect num_class
  std::shared_ptr<TensorTransform> one_hot_op = std::make_shared<transforms::OneHot>(-5);

  // Create a Map operation on ds
  ds = ds->Map({one_hot_op}, {"label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid OneHot input
  EXPECT_EQ(iter, nullptr);
}

/// Feature: PadEnd op
/// Description: Test PadEnd op basic usage with int as pad_value
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestPadEndSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPadEndSuccess1.";
  // Test PadEnd basic with int as pad_value

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {1}));
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Create PadEnd op
  std::shared_ptr<Tensor> pad_value;
  ASSERT_OK(Tensor::CreateScalar(0, &pad_value));
  mindspore::MSTensor pad_value_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(pad_value));

  transforms::PadEnd pad_end = transforms::PadEnd({3}, pad_value_MSTensor);
  ds = ds->Map({pad_end}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<int16_t>> expected = {{31354, 0, 0}, {-17734, 0, 0}, {-5655, 0, 0}, {-17220, 0, 0}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: PadEnd op
/// Description: Test PadEnd op with pad_shape equals to the current shape, nothing padded
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestPadEndSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPadEndSuccess2.";
  // Test PadEnd with pad_shape equals to current shape, nothing padded

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {2}));
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Create PadEnd op
  std::shared_ptr<Tensor> pad_value;
  ASSERT_OK(Tensor::CreateScalar(0, &pad_value));
  mindspore::MSTensor pad_value_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(pad_value));

  transforms::PadEnd pad_end = transforms::PadEnd({2}, pad_value_MSTensor);
  ds = ds->Map({pad_end}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<int16_t>> expected = {{31354, 31354}, {-17734, -17734}, {-5655, -5655}, {-17220, -17220}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: PadEnd op
/// Description: Test PadEnd op without pad_value (using default pad_value)
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestPadEndSuccess3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPadEndSuccess3.";
  // Test PadEnd without pad_value (using default pad_value)

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {1}));
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Create PadEnd op
  transforms::PadEnd pad_end = transforms::PadEnd({3});
  ds = ds->Map({pad_end}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<int16_t>> expected = {{31354, 0, 0}, {-17734, 0, 0}, {-5655, 0, 0}, {-17220, 0, 0}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: PadEnd op
/// Description: Test PadEnd op with pad_shape less than current shape, will truncate the values
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestPadEndSuccess4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPadEndSuccess4.";
  // Test PadEnd with pad_shape less than current shape, will truncate the values

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {4}));
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Create PadEnd op
  std::shared_ptr<Tensor> pad_value;
  ASSERT_OK(Tensor::CreateScalar(0, &pad_value));
  mindspore::MSTensor pad_value_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(pad_value));

  transforms::PadEnd pad_end = transforms::PadEnd({2}, pad_value_MSTensor);
  ds = ds->Map({pad_end}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<int16_t>> expected = {{31354, 31354}, {-17734, -17734}, {-5655, -5655}, {-17220, -17220}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: PadEnd op
/// Description: Test PadEnd op with string as pad_value
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestPadEndSuccess5) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPadEndSuccess5.";
  // Test PadEnd with string as pad_value

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(1);
  EXPECT_NE(ds, nullptr);

  // Create BasicTokenizer operation on ds
  std::shared_ptr<TensorTransform> basic_tokenizer = std::make_shared<text::BasicTokenizer>(true);
  EXPECT_NE(basic_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({basic_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create PadEnd op
  std::shared_ptr<Tensor> pad_value;
  ASSERT_OK(Tensor::CreateScalar<std::string>("pad_string", &pad_value));
  mindspore::MSTensor pad_value_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(pad_value));

  transforms::PadEnd pad_end = transforms::PadEnd({5}, pad_value_MSTensor);
  ds = ds->Map({pad_end}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<std::string>> expected = {{"welcome", "to", "beijing", "!", "pad_string"}};

  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: PadEnd op
/// Description: Test PadEnd op with type mismatch, source and pad_value are not of the same type
/// Expectation: Throw correct error and message
TEST_F(MindDataTestPipeline, TestPadEndFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPadEndFail.";
  // Test PadEnd with type mismatch, source and pad_value are not of the same type.

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {1}));
  std::shared_ptr<Dataset> ds = RandomData(1, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(1);
  EXPECT_NE(ds, nullptr);

  // Create PadEnd op
  std::shared_ptr<Tensor> pad_value;
  ASSERT_OK(Tensor::CreateScalar<std::string>("0", &pad_value));
  mindspore::MSTensor pad_value_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(pad_value));

  transforms::PadEnd pad_end = transforms::PadEnd({3}, pad_value_MSTensor);
  ds = ds->Map({pad_end}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  // Expect failure: type mismatch, pad a string to Int16 dataset
  EXPECT_ERROR(iter->GetNextRow(&row));

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: RandomApply op
/// Description: Test RandomApply op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomApplySuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomApplySuccess.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 5));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto resize_op = vision::Resize({777, 777});
  auto random_apply = transforms::RandomApply({resize_op}, 0.8);

  // Create a Map operation on ds
  ds = ds->Map({random_apply}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 5);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomApply op
/// Description: Test RandomApply op with invalid transform op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomApplyFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomApplyFail1 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // Resize: Non-positive size value: -1 at element: 0
  // RandomApply: transform ops must not be null
  auto decode_op = vision::Decode();
  auto resize_op = vision::Resize({-1});
  auto random_apply = transforms::RandomApply({decode_op, resize_op});

  // Create a Map operation on ds
  ds = ds->Map({random_apply}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid RandomApply parameter (transform ops must not be null)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomApply op
/// Description: Test RandomApply op with null transform op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomApplyFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomApplyFail2 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // RandomApply: transform ops must not be null
  std::shared_ptr<TensorTransform> decode_op = std::make_shared<vision::Decode>();
  auto random_apply =
    std::make_shared<transforms::RandomApply>(std::vector<std::shared_ptr<TensorTransform>>{decode_op, nullptr});

  // Create a Map operation on ds
  ds = ds->Map({random_apply}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid RandomApply parameter (transform ops must not be null)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomApply op
/// Description: Test RandomApply op probability out of range
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomApplyFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomApplyFail3 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // RandomApply: Probability has to be between 0 and 1
  auto resize_op = vision::Resize({100});
  auto random_apply = transforms::RandomApply({resize_op}, -1);

  // Create a Map operation on ds
  ds = ds->Map({random_apply}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid RandomApply parameter (Probability has to be between 0 and 1)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomApply op
/// Description: Test RandomApply op with empty transform list
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomApplyFail4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomApplyFail4 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // RandomApply: transform list must not be empty
  std::vector<std::shared_ptr<TensorTransform>> list = {};
  auto random_apply = transforms::RandomApply(list);

  // Create a Map operation on ds
  ds = ds->Map({random_apply}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid RandomApply parameter (transform list must not be empty)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomChoice op
/// Description: Test RandomChoice op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRandomChoiceSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomChoiceSuccess.";

  // Create an ImageFolder Dataset
  std::string folder_path = datasets_root_path_ + "/testPK/data/";
  std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds, nullptr);

  // Create objects for the tensor ops
  auto resize_op1 = std::make_shared<vision::Resize>(std::vector<int32_t>{777, 777});
  auto resize_op2 = std::make_shared<vision::Resize>(std::vector<int32_t>{888, 888});
  auto random_choice = transforms::RandomChoice({resize_op1, resize_op2});

  // Create a Map operation on ds
  ds = ds->Map({random_choice}, {"image"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto image = row["image"];
    auto label = row["label"];
    MS_LOG(INFO) << "Tensor image shape: " << image.Shape();
    MS_LOG(INFO) << "Label shape: " << label.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: RandomChoice op
/// Description: Test RandomChoice op with invalid transform op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomChoiceFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomChoiceFail1 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  RandomSampler sampler = RandomSampler(false, 10);
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", sampler);
  EXPECT_NE(ds, nullptr);

  // Resize: Non-positive size value: -1 at element: 0
  // RandomChoice: transform ops must not be null
  auto decode_op = vision::Decode();
  auto resize_op = vision::Resize({-1});
  auto random_choice = transforms::RandomChoice({decode_op, resize_op});

  // Create a Map operation on ds
  ds = ds->Map({random_choice}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid RandomApply parameter (transform ops must not be null)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomChoice op
/// Description: Test RandomChoice op with null transform op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomChoiceFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomChoiceFail2 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // RandomChoice: transform ops must not be null
  std::shared_ptr<TensorTransform> decode_op = std::make_shared<vision::Decode>();
  auto random_choice =
    std::make_shared<transforms::RandomApply>(std::vector<std::shared_ptr<TensorTransform>>{decode_op, nullptr});

  // Create a Map operation on ds
  ds = ds->Map({random_choice}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid RandomApply parameter (transform ops must not be null)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: RandomChoice op
/// Description: Test RandomChoice op with empty transform list
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRandomChoiceFail3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRandomChoiceFail3 with invalid transform.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // RandomChoice: transform list must not be empty
  std::vector<std::shared_ptr<TensorTransform>> list = {};
  auto random_choice = transforms::RandomChoice(list);

  // Create a Map operation on ds
  ds = ds->Map({random_choice}, {"image"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid RandomApply parameter (transform list must not be empty)
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Slice op
/// Description: Test Slice op with user defined slice object
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSliceSuccess1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSliceSuccess1.";
  // Test Slice int with user defined slice object.

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {1}));
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Create concatenate op
  std::vector<int16_t> prepend_vector = {1, 2, 3};
  std::shared_ptr<Tensor> prepend_tensor;
  ASSERT_OK(Tensor::CreateFromVector(prepend_vector, &prepend_tensor));
  mindspore::MSTensor prepend_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(prepend_tensor));

  transforms::Concatenate concatenate = transforms::Concatenate(0, prepend_MSTensor);

  // Create a Map operation on ds
  ds = ds->Map({concatenate}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Apply Slice op on ds, get the first and third elements in each row.
  SliceOption slice_option = SliceOption(Slice(0, 3, 2));
  transforms::Slice slice = transforms::Slice({slice_option});
  ds = ds->Map({slice}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<int16_t>> expected = {{1, 3}, {1, 3}, {1, 3}, {1, 3}};

  // Check slice results
  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Slice op
/// Description: Test Slice op on int dataset with bool true (slice all)
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSliceSuccess2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSliceSuccess2.";
  // Test Slice int with bool true (slice all).

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {1}));
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Create concatenate op
  std::vector<int16_t> prepend_vector = {1, 2, 3};
  std::shared_ptr<Tensor> prepend_tensor;
  ASSERT_OK(Tensor::CreateFromVector(prepend_vector, &prepend_tensor));
  mindspore::MSTensor prepend_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(prepend_tensor));

  transforms::Concatenate concatenate = transforms::Concatenate(0, prepend_MSTensor);

  // Create a Map operation on ds
  ds = ds->Map({concatenate}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Apply Slice op on ds, get the first and third elements in each row.
  SliceOption slice_option = SliceOption(true);
  transforms::Slice slice = transforms::Slice({slice_option});
  ds = ds->Map({slice}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<int16_t>> expected = {
    {1, 2, 3, 31354}, {1, 2, 3, -17734}, {1, 2, 3, -5655}, {1, 2, 3, -17220}};

  // Check slice results
  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Slice op
/// Description: Test Slice op on int dataset with list of indices including negative
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSliceSuccess3) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSliceSuccess3.";
  // Test Slice int with list of indices including negative.

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {1}));
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Create concatenate op
  std::vector<int16_t> prepend_vector = {1, 2, 3};
  std::shared_ptr<Tensor> prepend_tensor;
  ASSERT_OK(Tensor::CreateFromVector(prepend_vector, &prepend_tensor));
  mindspore::MSTensor prepend_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(prepend_tensor));

  transforms::Concatenate concatenate = transforms::Concatenate(0, prepend_MSTensor);

  // Create a Map operation on ds
  ds = ds->Map({concatenate}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Apply Slice op on ds, get the first and third elements in each row.
  std::vector<dsize_t> indices = {-1, 2};
  SliceOption slice_option = SliceOption(indices);
  transforms::Slice slice = transforms::Slice({slice_option});
  ds = ds->Map({slice}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<int16_t>> expected = {{31354, 3}, {-17734, 3}, {-5655, 3}, {-17220, 3}};

  // Check slice results
  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Slice op
/// Description: Test Slice op on string dataset with list of indices
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSliceSuccess4) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSliceSuccess4.";
  // Test Slice string with list of indices.

  // Create a TextFile dataset
  std::string data_file = datasets_root_path_ + "/testTokenizerData/1.txt";
  std::shared_ptr<Dataset> ds = TextFile({data_file}, 0, ShuffleMode::kFalse);
  EXPECT_NE(ds, nullptr);

  // Create Take operation on ds
  ds = ds->Take(1);
  EXPECT_NE(ds, nullptr);

  // Create BasicTokenizer operation on ds
  std::shared_ptr<TensorTransform> basic_tokenizer = std::make_shared<text::BasicTokenizer>(true);
  EXPECT_NE(basic_tokenizer, nullptr);

  // Create Map operation on ds
  ds = ds->Map({basic_tokenizer}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Apply Slice op on ds, get the first and third elements in each row.
  std::vector<dsize_t> indices = {-1, -2, 1, 0};
  SliceOption slice_option = SliceOption(indices);
  transforms::Slice slice = transforms::Slice({slice_option});
  ds = ds->Map({slice}, {"text"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<std::string>> expected = {{"!", "beijing", "to", "welcome"}};

  // Check slice results
  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["text"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 1);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: Slice op
/// Description: Test Slice op on int dataset on multi-dimension
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSliceSuccess5) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSliceSuccess5.";
  // Test Slice int on multi-dimension.

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  // Generate a ds of 4 tensors, each tensor has 3 rows and 2 columns
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {3, 2}));
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  // Apply two SliceOptions on ds which includes 4 tensors with shape 3*2
  // The first SliceOption is to slice the first and the second row of each tensor
  // The shape of result tensor changes to 2*2
  std::vector<dsize_t> indices1 = {0, 1};
  SliceOption slice_option1 = SliceOption(indices1);
  // The second SliceOption is to slice the last column of each tensor
  // The shape of result tensor changes to 2*1
  std::vector<dsize_t> indices2 = {-1};
  SliceOption slice_option2 = SliceOption(indices2);

  transforms::Slice slice = transforms::Slice({slice_option1, slice_option2});
  ds = ds->Map({slice}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<std::vector<int16_t>> expected = {{31354, 31354}, {-17734, -17734}, {-5655, -5655}, {-17220, -17220}};

  // Check slice results
  uint64_t i = 0;
  while (row.size() != 0) {
    auto ind = row["col1"];
    std::shared_ptr<Tensor> de_expected_tensor;
    ASSERT_OK(Tensor::CreateFromVector(expected[i], TensorShape({2, 1}), &de_expected_tensor));
    mindspore::MSTensor expected_tensor =
      mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(de_expected_tensor));
    EXPECT_MSTENSOR_EQ(ind, expected_tensor);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 4);

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Slice op
/// Description: Test Slice op with index out of bounds
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestSliceFail1) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSliceFail.";
  // Test Slice with index out of bounds.

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {2}));
  std::shared_ptr<Dataset> ds = RandomData(1, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(1);
  EXPECT_NE(ds, nullptr);

  // Apply Slice op on ds, get the first and third elements in each row.
  std::vector<dsize_t> indices = {0, 2};  // index 2 is out of bounds
  SliceOption slice_option = SliceOption(indices);
  transforms::Slice slice = transforms::Slice({slice_option});
  ds = ds->Map({slice}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  // Expect failure: the index 2 is out of the bounds
  EXPECT_ERROR(iter->GetNextRow(&row));

  // Manually terminate the pipeline
  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: Slice op
/// Description: Test Slice op with false as input for SliceOption only (no other index nor slice list provided)
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestSliceFail2) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSliceFail2.";
  // Test Slice with false as input SliceOption only (no other index nor slice list provided)

  // Create a RandomDataset
  u_int32_t curr_seed = GlobalContext::config_manager()->seed();
  GlobalContext::config_manager()->set_seed(246);
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeInt16, {1}));
  std::shared_ptr<Dataset> ds = RandomData(1, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(1);
  EXPECT_NE(ds, nullptr);

  // Create concatenate op
  std::vector<int16_t> prepend_vector = {1, 2, 3};
  std::shared_ptr<Tensor> prepend_tensor;
  ASSERT_OK(Tensor::CreateFromVector(prepend_vector, &prepend_tensor));
  mindspore::MSTensor prepend_MSTensor =
    mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(prepend_tensor));

  transforms::Concatenate concatenate = transforms::Concatenate(0, prepend_MSTensor);

  // Create a Map operation on ds
  ds = ds->Map({concatenate}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Apply Slice op on ds, get the first and third elements in each row.
  SliceOption slice_option = SliceOption(false);
  transforms::Slice slice = transforms::Slice({slice_option});
  ds = ds->Map({slice}, {"col1"});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  // Expect failure: SliceOption is false and no other index nor slice list provided
  EXPECT_ERROR(iter->GetNextRow(&row));

  iter->Stop();
  GlobalContext::config_manager()->set_seed(curr_seed);
}

/// Feature: TypeCast op
/// Description: Test TypeCast op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestTypeCastSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTypeCastSuccess.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 1));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  // Check original data type of dataset
  auto image = row["image"];
  auto ori_type = image.DataType();
  MS_LOG(INFO) << "Original data type id: " << ori_type;
  EXPECT_EQ(ori_type, mindspore::DataType(mindspore::TypeId::kNumberTypeUInt8));

  // Manually terminate the pipeline
  iter->Stop();

  // Create objects for the tensor ops
  std::shared_ptr<TensorTransform> type_cast =
    std::make_shared<transforms::TypeCast>(mindspore::DataType::kNumberTypeUInt16);

  // Create a Map operation on ds
  std::shared_ptr<Dataset> ds2 = ds->Map({type_cast}, {"image"});
  EXPECT_NE(ds2, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  EXPECT_NE(iter2, nullptr);

  // Check current data type of dataset
  ASSERT_OK(iter2->GetNextRow(&row));
  auto image2 = row["image"];
  auto cur_type = image2.DataType();
  MS_LOG(INFO) << "Current data type id: " << cur_type;
  EXPECT_EQ(cur_type, mindspore::DataType(mindspore::TypeId::kNumberTypeUInt16));

  // Manually terminate the pipeline
  iter2->Stop();
}

/// Feature: TypeCast op
/// Description: Test TypeCast op with incorrect data type
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestTypeCastFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTypeCastFail with invalid param.";

  // Create a Cifar10 Dataset
  std::string folder_path = datasets_root_path_ + "/testCifar10Data/";
  std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  EXPECT_NE(ds, nullptr);

  // incorrect data type
  std::shared_ptr<TensorTransform> type_cast =
    std::make_shared<transforms::TypeCast>(mindspore::DataType::kTypeUnknown);

  // Create a Map operation on ds
  ds = ds->Map({type_cast}, {"image", "label"});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid TypeCast input
  EXPECT_EQ(iter, nullptr);
}
