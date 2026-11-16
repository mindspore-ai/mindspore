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

/// Feature: Vad op
/// Description: Test Vad op with pipeline mode
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestVadPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVadPipeline.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 1000}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);
  auto vad = audio::Vad(600);
  ds = ds->Map({vad});
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::vector<int64_t> expected = {2, 660};
  int i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);
  iter->Stop();

  // Original waveform
  std::shared_ptr<SchemaObj> schema2 = Schema();
  ASSERT_OK(schema2->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1000}));
  std::shared_ptr<Dataset> ds2 = RandomData(50, schema2);
  EXPECT_NE(ds2, nullptr);
  ds2 = ds2->SetNumWorkers(4);
  EXPECT_NE(ds2, nullptr);
  auto vad2 = audio::Vad(1600);
  ds2 = ds2->Map({vad2});
  EXPECT_NE(ds2, nullptr);
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  EXPECT_NE(ds2, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row2;
  ASSERT_OK(iter2->GetNextRow(&row2));
  std::vector<int64_t> expected2 = {760};
  i = 0;
  while (row2.size() != 0) {
    auto col = row2["waveform"];
    ASSERT_EQ(col.Shape(), expected2);
    ASSERT_EQ(col.Shape().size(), 1);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter2->GetNextRow(&row2));
    i++;
  }
  EXPECT_EQ(i, 50);
  iter->Stop();
}

/// Feature: Vad op
/// Description: Test Vad op with invalid parameters
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestVadWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVadWrongArgs.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 1000}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  // invalid sample rate
  auto vad_op = audio::Vad(-10);
  ds = ds->Map({vad_op});
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid search_time
  vad_op = audio::Vad(100, 7.0, 0.25, -1.0);
  ds = ds->Map({vad_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid allowed_gap
  vad_op = audio::Vad(100, 7.0, 0.25, 1.0, -10);
  ds = ds->Map({vad_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid pre_trigger_time
  vad_op = audio::Vad(100, 7.0, 0.25, 1.0, 0.25, -10);
  ds = ds->Map({vad_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid boot_time
  vad_op = audio::Vad(100, 7.0, 0.25, 1.0, 0.25, 0.0, -10);
  ds = ds->Map({vad_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid noise_up_time
  vad_op = audio::Vad(100, 7.0, 0.25, 1.0, 0.25, 0.0, 0.35, -10);
  ds = ds->Map({vad_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid noise_down_time
  vad_op = audio::Vad(100, 7.0, 0.25, 1.0, 0.25, 0.0, 0.35, 0.1, -10);
  ds = ds->Map({vad_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid noise_up_time and noise_down_time
  vad_op = audio::Vad(100, 7.0, 0.25, 1.0, 0.25, 0.0, 0.35, 0.1, 0.2);
  ds = ds->Map({vad_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid noise_reduction_amount
  vad_op = audio::Vad(100, 7.0, 0.25, 1.0, 0.25, 0.0, 0.35, 0.1, 0.01, -10);
  ds = ds->Map({vad_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid measure_freq
  vad_op = audio::Vad(100, 7.0, 0.25, 1.0, 0.25, 0.0, 0.35, 0.1, 0.01, 1.35, -10);
  ds = ds->Map({vad_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid measure_duration
  vad_op = audio::Vad(100, 7.0, 0.25, 1.0, 0.25, 0.0, 0.35, 0.1, 0.01, 1.35, 20.0, -10);
  ds = ds->Map({vad_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid measure_smooth_time
  vad_op = audio::Vad(100, 7.0, 0.25, 1.0, 0.25, 0.0, 0.35, 0.1, 0.01, 1.35, 20.0, 0.0, -10);
  ds = ds->Map({vad_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid hp_filter_freq
  vad_op = audio::Vad(100, 7.0, 0.25, 1.0, 0.25, 0.0, 0.35, 0.1, 0.01, 1.35, 20.0, 0.0, 0.4, -10);
  ds = ds->Map({vad_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid lp_filter_freq
  vad_op = audio::Vad(100, 7.0, 0.25, 1.0, 0.25, 0.0, 0.35, 0.1, 0.01, 1.35, 20.0, 0.0, 0.4, 50, -10);
  ds = ds->Map({vad_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid hp_lifter_freq
  vad_op = audio::Vad(100, 7.0, 0.25, 1.0, 0.25, 0.0, 0.35, 0.1, 0.01, 1.35, 20.0, 0.0, 0.4, 50, 6000.0, -10);
  ds = ds->Map({vad_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid lp_lifter_freq
  vad_op = audio::Vad(100, 7.0, 0.25, 1.0, 0.25, 0.0, 0.35, 0.1, 0.01, 1.35, 20.0, 0.0, 0.4, 50, 6000.0, 150.0, -10);
  ds = ds->Map({vad_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Vol op
/// Description: Test Vol op with pipeline mode
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestVolPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVolPipeline.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto vol = audio::Vol(0.3);

  ds = ds->Map({vol});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {2, 200};

  int i = 0;
  while (!row.empty()) {
    auto col = row["inputData"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);

  iter->Stop();
}

/// Feature: Vol op
/// Description: Test Vol op with wrong arguments
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestVolWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestVolWrongArgs.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto vol_op = audio::Vol(-1.5, GainType::kPower);

  ds = ds->Map({vol_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);
}
