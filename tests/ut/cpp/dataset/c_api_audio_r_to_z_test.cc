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
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/audio.h"

using namespace mindspore::dataset;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: RiaaBiquad op
/// Description: Test RiaaBiquad op basic usage with sample rate 44100
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRiaaBiquadBasicSampleRate44100) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRiaaBiquadBasicSampleRate44100.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto RiaaBiquadOp = audio::RiaaBiquad(44100);

  ds = ds->Map({RiaaBiquadOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by riaabiquad
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {2, 200};

  int i = 0;
  while (!row.empty()) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);

  iter->Stop();
}

/// Feature: RiaaBiquad op
/// Description: Test RiaaBiquad op basic usage with sample rate 48000
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRiaaBiquadBasicSampleRate48000) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRiaaBiquadBasicSampleRate48000.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {30, 40}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto RiaaBiquadOp = audio::RiaaBiquad(48000);

  ds = ds->Map({RiaaBiquadOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by riaabiquad
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {30, 40};

  int i = 0;
  while (!row.empty()) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);

  iter->Stop();
}

/// Feature: RiaaBiquad op
/// Description: Test RiaaBiquad op basic usage with sample rate 88200
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRiaaBiquadBasicSampleRate88200) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRiaaBiquadBasicSampleRate88200.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {5, 4}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto RiaaBiquadOp = audio::RiaaBiquad(88200);

  ds = ds->Map({RiaaBiquadOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by riaabiquad
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {5, 4};

  int i = 0;
  while (!row.empty()) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);

  iter->Stop();
}

/// Feature: RiaaBiquad op
/// Description: Test RiaaBiquad op basic usage with sample rate 96000
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestRiaaBiquadBasicSampleRate96000) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRiaaBiquadBasicSampleRate96000.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 3}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto RiaaBiquadOp = audio::RiaaBiquad(96000);

  ds = ds->Map({RiaaBiquadOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by riaabiquad
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {2, 3};

  int i = 0;
  while (!row.empty()) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);

  iter->Stop();
}

/// Feature: RiaaBiquad op
/// Description: Test RiaaBiquad op basic usage with invalid sample_rate=0
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestRiaaBiquadWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestRiaaBiquadWrongArg.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  EXPECT_NE(ds, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  auto riaa_biquad_op_01 = audio::RiaaBiquad(0);
  ds01 = ds->Map({riaa_biquad_op_01});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);
}

/// Feature: TrebleBiquad op
/// Description: Test TrebleBiquad op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestTrebleBiquadBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTrebleBiquadBasic.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto TrebleBiquadOp = audio::TrebleBiquad(44100, 200.0, 2000, 0.604);

  ds = ds->Map({TrebleBiquadOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by treblebiquad
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {2, 200};

  int i = 0;
  while (!row.empty()) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);

  iter->Stop();
}

/// Feature: TrebleBiquad op
/// Description: Test TrebleBiquad op with invalid sample_rate and Q_
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestTrebleBiquadWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTrebleBiquadWrongArg.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  std::shared_ptr<Dataset> ds02;
  EXPECT_NE(ds, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  auto treble_biquad_op_01 = audio::TrebleBiquad(0, 200);
  ds01 = ds->Map({treble_biquad_op_01});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);

  // Check Q_
  MS_LOG(INFO) << "Q_ is zero.";
  auto treble_biquad_op_02 = audio::TrebleBiquad(44100, 200.0, 3000.0, 0);
  ds02 = ds->Map({treble_biquad_op_02});
  EXPECT_NE(ds02, nullptr);

  std::shared_ptr<Iterator> iter02 = ds02->CreateIterator();
  EXPECT_EQ(iter02, nullptr);
}
