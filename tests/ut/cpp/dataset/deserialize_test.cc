/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/engine/serdes.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/vision.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/general/transform/transforms_ir.h"

using namespace mindspore::dataset;
using mindspore::dataset::DatasetNode;

using mindspore::dataset::ShuffleMode;
using mindspore::dataset::Tensor;

class MindDataTestDeserialize : public UT::DatasetOpTesting {
 protected:
};

void compare_dataset(std::shared_ptr<DatasetNode> ds) {
  nlohmann::json out_json;
  ASSERT_OK(Serdes::SaveToJSON(ds, "dataset_pipeline.json", &out_json));
  // output the deserialized out_json to ds1 and then out_json1
  std::shared_ptr<DatasetNode> ds1;
  ASSERT_OK(Serdes::Deserialize("dataset_pipeline.json", &ds1));
  EXPECT_NE(ds1, nullptr);

  // check original and deserialized dataset are the same
  nlohmann::json out_json1;
  ASSERT_OK(Serdes::SaveToJSON(ds1, "dataset_pipeline_1.json", &out_json1));
  std::stringstream json_ss;
  json_ss << out_json;
  std::stringstream json_ss1;
  json_ss1 << out_json1;
  EXPECT_EQ(json_ss.str(), json_ss1.str());
  return;
}

/// Feature: Deserialize
/// Description: Test Deserialize on CSVDataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestDeserialize, TestDeserializeCSV) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-CSV.";
  std::string data_file = "./data/dataset/testCSV/1.csv";
  std::vector<std::string> dataset_files = {data_file};
  char field_delim = ',';
  std::vector<std::string> column_names = {"col1", "col2", "col3", "col4"};
  std::vector<std::string> columns = {"col1", "col4", "col2"};
  std::vector<std::shared_ptr<CsvBase>> column_defaults = {};
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::shared_ptr<DatasetNode> ds = std::make_shared<CSVNode>(dataset_files, field_delim, column_defaults, column_names,
                                                              3, ShuffleMode::kGlobal, 1, 0, cache);
  ds = std::make_shared<ProjectNode>(ds, columns);
  compare_dataset(ds);
}

/// Feature: Deserialize
/// Description: Test Deserialize on ImageFolderDataset and some random tensor operations
/// Expectation: The data is processed successfully
TEST_F(MindDataTestDeserialize, TestDeserializeImageFolder) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-ImageFolder.";
  std::string dataset_dir = "./data/dataset/testPK/data";
  std::shared_ptr<SamplerObj> child_sampler = std::make_shared<PKSamplerObj>(3, true, 1);
  std::vector<double> weights = {1.0, 0.1, 0.02, 0.3, 0.4, 0.05, 1.2, 0.13, 0.14, 0.015, 0.16, 1.1};
  std::set<std::string> extensions = {};
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::map<std::string, int32_t> class_indexing = {};
  std::shared_ptr<SamplerObj> sampler = std::make_shared<WeightedRandomSamplerObj>(weights, 11);
  sampler->AddChildSampler(child_sampler);
  std::shared_ptr<DatasetNode> ds =
    std::make_shared<ImageFolderNode>(dataset_dir, false, sampler, false, extensions, class_indexing, cache);
  ds = std::make_shared<RepeatNode>(ds, 1);
  std::vector<int32_t> size = {224, 224};
  std::vector<float> scale = {0.5, 0.5};
  std::vector<float> ratio = {0.5, 0.5};
  std::vector<float> center = {50.0, 50.0};
  std::vector<uint8_t> fill_value = {150, 150, 150};
  InterpolationMode interpolation = InterpolationMode::kLinear;
  std::vector<std::shared_ptr<TensorOperation>> operations;
  std::shared_ptr<TensorOperation> operation1 =
    std::make_shared<vision::RandomCropDecodeResizeOperation>(size, scale, ratio, interpolation, 2);
  std::shared_ptr<TensorOperation> operation2 =
    std::make_shared<vision::RotateOperation>(0.5, interpolation, true, center, fill_value);
  operations.push_back(operation1);
  operations.push_back(operation2);
  ds = std::make_shared<MapNode>(ds, operations);
  ds = std::make_shared<BatchNode>(ds, 2, true);
  compare_dataset(ds);
}

/// Feature: Deserialize
/// Description: Test Deserialize on VOCDataset with RandomColorAdjust op
/// Expectation: The data is processed successfully
TEST_F(MindDataTestDeserialize, TestDeserializeSubsetColorAdjust) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-SubsetColorAdjust.";
  std::string dataset_dir = "./data/dataset/testPK/data";
  std::vector<int64_t> indices = {0, 1};
  std::shared_ptr<SamplerObj> sampler = std::make_shared<SubsetRandomSamplerObj>(indices, 3);
  std::map<std::string, int32_t> class_indexing = {};
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::shared_ptr<DatasetNode> ds = std::make_shared<ImageFolderNode>(dataset_dir, true, sampler, false,
                                                                      std::set<std::string>{}, class_indexing, cache);
  std::vector<float> brightness = {0.5, 0.5};
  std::vector<float> contrast = {1.0, 1.0};
  std::vector<float> hue = {0.0, 0.0};
  std::vector<float> saturation = {1.0, 1.0};
  std::shared_ptr<TensorOperation> operation =
    std::make_shared<vision::RandomColorAdjustOperation>(brightness, contrast, saturation, hue);
  std::vector<std::shared_ptr<TensorOperation>> ops = {operation};
  ds = std::make_shared<MapNode>(ds, ops);
  ds = std::make_shared<SkipNode>(ds, 2);
  compare_dataset(ds);
}

/// Feature: Deserialize
/// Description: Test Deserialize on CLUEDataset with Decode op
/// Expectation: The data is processed successfully
TEST_F(MindDataTestDeserialize, TestDeserializeCLUE) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-CLUE.";
  std::string train_file = "./data/dataset/testCLUE/afqmc/train.json";
  std::string task = "AFQMC";
  std::string usage = "train";
  std::vector<std::string> files = {train_file};
  std::shared_ptr<DatasetCache> cache = nullptr;
  std::shared_ptr<DatasetNode> ds = std::make_shared<CLUENode>(files, task, usage, 1, ShuffleMode::kFalse, 1, 0, cache);
  ds = std::make_shared<RepeatNode>(ds, 1);
  std::shared_ptr<TensorOperation> operation1 = std::make_shared<vision::DecodeOperation>(true);
  std::vector<std::shared_ptr<TensorOperation>> ops = {operation1};
  ds = std::make_shared<MapNode>(ds, ops);
  compare_dataset(ds);
}

/// Feature: Deserialize
/// Description: Test Deserialize with invalid json path or object
/// Expectation: Throw correct error and message
TEST_F(MindDataTestDeserialize, TestDeserializeInvalidJson) {
  std::shared_ptr<DatasetNode> ds;
  // check the invalid json path would return error
  ASSERT_ERROR(Serdes::Deserialize("invalid_dataset.json", &ds));
  // check the invalid json object would return error
  ASSERT_ERROR(Serdes::Deserialize("./data/dataset/testDataset1/datasetTestInvalidJson.json", &ds));
  EXPECT_EQ(ds, nullptr);
}

/// Feature: Deserialize
/// Description: Test Deserialize on Tensor
/// Expectation: The data is processed successfully
TEST_F(MindDataTestDeserialize, TestDeserializeTensor) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-Tensor.";
  std::shared_ptr<Tensor> test_tensor;
  std::vector<float> input = {1.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.2, 0.7, 0.8, 0.9, 1.0, 2.0, 1.3, 3.0, 4.0};
  ASSERT_OK(Tensor::CreateFromVector(input, TensorShape{3, 5}, &test_tensor));
  nlohmann::json json_obj;
  ASSERT_OK(test_tensor->to_json(&json_obj));
  std::shared_ptr<Tensor> test_tensor1;
  ASSERT_OK(Tensor::from_json(json_obj, &test_tensor1));
  nlohmann::json json_obj1;
  ASSERT_OK(test_tensor1->to_json(&json_obj1));
  std::stringstream json_ss;
  json_ss << json_obj;
  std::stringstream json_ss1;
  json_ss1 << json_obj1;
  EXPECT_EQ(json_ss.str(), json_ss1.str());
}

/// Feature: Deserialize
/// Description: Test Deserialize on created Dataset Cache
/// Expectation: The data is processed successfully
TEST_F(MindDataTestDeserialize, DISABLED_TestDeserializeCache) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-Cache.";
  std::string data_dir = "./data/dataset/testPK/data";
  session_id_type env_session;
  ASSERT_OK(GetSessionFromEnv(&env_session));
  std::shared_ptr<DatasetCache> some_cache = CreateDatasetCache(env_session, 0, false, "127.0.0.1", 50052, 1, 1);

  std::shared_ptr<SamplerObj> sampler = std::make_shared<SequentialSamplerObj>(0, 10);
  std::shared_ptr<DatasetNode> ds = std::make_shared<ImageFolderNode>(
    data_dir, false, sampler, false, std::set<std::string>{}, std::map<std::string, int32_t>{}, some_cache);
  compare_dataset(ds);
}

/// Feature: Deserialize
/// Description: Test Deserialize when Python is initialized
/// Expectation: The data is processed successfully
TEST_F(MindDataTestDeserialize, TestDeserializePyFunc) {
  MS_LOG(INFO) << "Doing MindDataTestDeserialize-PyFunc.";
  if (Py_IsInitialized() != 0) {
    std::shared_ptr<DatasetNode> ds1;
    ASSERT_OK(Serdes::Deserialize("./data/dataset/tf_file_dataset/pyvision_dataset_pipeline.json", &ds1));
    EXPECT_NE(ds1, nullptr);
  }
}
