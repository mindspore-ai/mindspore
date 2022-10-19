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

#include "minddata/dataset/include/dataset/datasets.h"

using namespace mindspore::dataset;
using mindspore::dataset::DataType;
using mindspore::dataset::Tensor;
using mindspore::dataset::TensorShape;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: LJSpeechDataset
/// Description: Basic test of LJSpeechDataset
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestLJSpeechDataset) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLJSpeechDataset.";
  std::string folder_path = datasets_root_path_ + "/testLJSpeechData/";
  std::shared_ptr<Dataset> ds = LJSpeech(folder_path, std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds, nullptr);
  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  MS_LOG(INFO) << "iter->GetNextRow(&row) OK";

  EXPECT_NE(row.find("waveform"), row.end());
  EXPECT_NE(row.find("sample_rate"), row.end());
  EXPECT_NE(row.find("transcription"), row.end());
  EXPECT_NE(row.find("normalized_transcription"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto waveform = row["waveform"];
    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: LJSpeechDataset
/// Description: Test LJSpeechDataset in pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestLJSpeechDatasetWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLJSpeechDatasetWithPipeline.";

  // Create two LJSpeech Dataset.
  std::string folder_path = datasets_root_path_ + "/testLJSpeechData/";
  std::shared_ptr<Dataset> ds1 = LJSpeech(folder_path, std::make_shared<RandomSampler>(false, 3));
  std::shared_ptr<Dataset> ds2 = LJSpeech(folder_path, std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds1, nullptr);
  EXPECT_NE(ds2, nullptr);

  // Create two Repeat operation on ds.
  int32_t repeat_num = 1;
  ds1 = ds1->Repeat(repeat_num);
  EXPECT_NE(ds1, nullptr);
  repeat_num = 1;
  ds2 = ds2->Repeat(repeat_num);
  EXPECT_NE(ds2, nullptr);

  // Create two Project operation on ds.
  std::vector<std::string> column_project = {"waveform", "sample_rate", "transcription", "normalized_transcription"};
  ds1 = ds1->Project(column_project);
  EXPECT_NE(ds1, nullptr);
  ds2 = ds2->Project(column_project);
  EXPECT_NE(ds2, nullptr);

  // Create a Concat operation on the ds.
  ds1 = ds1->Concat({ds2});
  EXPECT_NE(ds1, nullptr);

  // Create an iterator over the result of the above dataset.
  // This will trigger the creation of the Execution Tree and launch it.
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row.
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  EXPECT_NE(row.find("waveform"), row.end());
  EXPECT_NE(row.find("sample_rate"), row.end());
  EXPECT_NE(row.find("transcription"), row.end());
  EXPECT_NE(row.find("normalized_transcription"), row.end());

  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    auto waveform = row["waveform"];
    MS_LOG(INFO) << "Tensor waveform shape: " << waveform.Shape();
    ASSERT_OK(iter->GetNextRow(&row));
  }

  EXPECT_EQ(i, 6);

  // Manually terminate the pipeline.
  iter->Stop();
}

/// Feature: LJSpeechDataset
/// Description: Test iterator of LJSpeechDataset with only the waveform column
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestLJSpeechDatasetIteratorOneColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLJSpeechDatasetIteratorOneColumn.";
  // Create a  LJSpeech dataset
  std::string folder_path = datasets_root_path_ + "/testLJSpeechData/";
  std::shared_ptr<Dataset> ds = LJSpeech(folder_path, std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds, nullptr);

  // Create a Batch operation on ds
  int32_t batch_size = 1;
  ds = ds->Batch(batch_size);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // Only select "waveform" column and drop others
  std::vector<std::string> columns = {"waveform"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  // Iterate the dataset and get each row
  std::vector<mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (row.size() != 0) {
    for (auto &v : row) {
      MS_LOG(INFO) << "waveform shape:" << v.Shape();
    }
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }

  EXPECT_EQ(i, 3);

  // Manually terminate the pipeline
  iter->Stop();
}

/// Feature: LJSpeechDataset
/// Description: Test iterator of LJSpeechDataset with wrong column
/// Expectation: Error message is logged, and CreateIterator for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestLJSpeechDatasetIteratorWrongColumn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLJSpeechDatasetIteratorWrongColumn.";
  // Create a LJSpeech Dataset
  std::string folder_path = datasets_root_path_ + "/testLJSpeechData/";
  std::shared_ptr<Dataset> ds = LJSpeech(folder_path, std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds, nullptr);

  // Pass wrong column name
  std::vector<std::string> columns = {"digital"};
  std::shared_ptr<ProjectDataset> project_ds = ds->Project(columns);
  std::shared_ptr<Iterator> iter = project_ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: LJSpeechDataset
/// Description: Test getting size of LJSpeechDataset
/// Expectation: The size is correct
TEST_F(MindDataTestPipeline, TestLJSpeechGetDatasetSize) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLJSpeechGetDatasetSize.";

  // Create a LJSpeech Dataset.
  std::string folder_path = datasets_root_path_ + "/testLJSpeechData/";
  std::shared_ptr<Dataset> ds = LJSpeech(folder_path);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 3);
}

/// Feature: LJSpeechDataset
/// Description: Test LJSpeechDataset Getters method
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestLJSpeechGetters) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLJSpeechMixGetter.";

  // Create a LJSpeech Dataset.
  std::string folder_path = datasets_root_path_ + "/testLJSpeechData/";
  std::shared_ptr<Dataset> ds = LJSpeech(folder_path);
  EXPECT_NE(ds, nullptr);

  EXPECT_EQ(ds->GetDatasetSize(), 3);
  std::vector<DataType> types = ToDETypes(ds->GetOutputTypes());
  std::vector<TensorShape> shapes = ToTensorShapeVec(ds->GetOutputShapes());
  std::vector<std::string> column_names = {"waveform", "sample_rate", "transcription", "normalized_transcription"};
  EXPECT_EQ(types.size(), 4);
  EXPECT_EQ(types[0].ToString(), "float32");
  EXPECT_EQ(types[1].ToString(), "int32");
  EXPECT_EQ(types[2].ToString(), "string");
  EXPECT_EQ(types[3].ToString(), "string");
  EXPECT_EQ(shapes.size(), 4);
  EXPECT_EQ(shapes[1].ToString(), "<>");
  EXPECT_EQ(shapes[2].ToString(), "<>");
  EXPECT_EQ(shapes[3].ToString(), "<>");
  EXPECT_EQ(ds->GetBatchSize(), 1);
  EXPECT_EQ(ds->GetRepeatCount(), 1);

  EXPECT_EQ(ds->GetDatasetSize(), 3);
  EXPECT_EQ(ToDETypes(ds->GetOutputTypes()), types);
  EXPECT_EQ(ToTensorShapeVec(ds->GetOutputShapes()), shapes);

  EXPECT_EQ(ds->GetColumnNames(), column_names);
}

/// Feature: LJSpeechDataset
/// Description: Test LJSpeechDataset with the fail of reading dataset
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestLJSpeechDatasetFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLJSpeechDatasetFail.";

  // Create a LJSpeech Dataset.
  std::shared_ptr<Dataset> ds = LJSpeech("", std::make_shared<RandomSampler>(false, 3));
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid LJSpeech input.
  EXPECT_EQ(iter, nullptr);
}

/// Feature: LJSpeechDataset
/// Description: Test LJSpeechDataset with the null sampler
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestLJSpeechDatasetWithNullSamplerFail) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLJSpeechDatasetWithNullSamplerFail.";

  // Create a LJSpeech Dataset.
  std::string folder_path = datasets_root_path_ + "/testLJSpeechData/";
  std::shared_ptr<Dataset> ds = LJSpeech(folder_path, nullptr);
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure: invalid LJSpeech input, sampler cannot be nullptr.
  EXPECT_EQ(iter, nullptr);
}
