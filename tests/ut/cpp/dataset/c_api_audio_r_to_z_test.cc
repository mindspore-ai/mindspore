/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

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
}

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
}

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
}

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
}

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

/// Feature: SlidingWindowCmn
/// Description: test basic function of SlidingWindowCmn
/// Expectation: get correct number of data
TEST_F(MindDataTestPipeline, TestSlidingWindowCmn) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSlidingWindowCmn.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeFloat32, {1, 2, 400}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);
  auto sliding_window_cmn = audio::SlidingWindowCmn(600, 100, false, false);
  auto ds1 = ds->Map({sliding_window_cmn});
  EXPECT_NE(ds1, nullptr);
  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  uint64_t i = 0;
  while (row.size() != 0) {
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }
  EXPECT_EQ(i, 8);
  iter->Stop();
}

/// Feature: SlidingWindowCmn
/// Description: test wrong input args of SlidingWindowCmn
/// Expectation: get nullptr of iterator
TEST_F(MindDataTestPipeline, TestSlidingWindowCmnWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSlidingWindowCmnWrongArgs.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeFloat32, {1, 2, 400}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  // SlidingWindowCmn: cmn_window must be greater than or equal to 0.
  auto sliding_window_cmn_1 = audio::SlidingWindowCmn(-1, 100, false, false);
  auto ds_1 = ds->Map({sliding_window_cmn_1});
  EXPECT_NE(ds_1, nullptr);
  std::shared_ptr<Iterator> iter_1 = ds_1->CreateIterator();
  EXPECT_EQ(iter_1, nullptr);

  // SlidingWindowCmn: min_cmn_window must be greater than or equal to 0.
  auto sliding_window_cmn_2 = audio::SlidingWindowCmn(600, -1, false, false);
  auto ds2 = ds->Map({sliding_window_cmn_2});
  EXPECT_NE(ds2, nullptr);
  std::shared_ptr<Iterator> iter_2 = ds2->CreateIterator();
  EXPECT_EQ(iter_2, nullptr);
}

TEST_F(MindDataTestPipeline, TestTimeMaskingPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTimeMaskingPipeline.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto timemasking = audio::TimeMasking(true, 6);

  ds = ds->Map({timemasking});
  EXPECT_NE(ds, nullptr);

  // mask waveform
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {2, 200};

  int i = 0;
  while (row.size() != 0) {
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

TEST_F(MindDataTestPipeline, TestTimeMaskingWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTimeMaskingWrongArgs.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 20}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto timemasking = audio::TimeMasking(true, -100);

  ds = ds->Map({timemasking});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);
}

TEST_F(MindDataTestPipeline, TestTimeStretchPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTimeStretchPipeline.";
  // op param
  int freq = 1025;
  int hop_length = 512;
  float rate = 1.2;
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, freq, 400, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto TimeStretchOp = audio::TimeStretch(hop_length, freq, rate);

  ds = ds->Map({TimeStretchOp});
  EXPECT_NE(ds, nullptr);

  // apply timestretch
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {2, freq, static_cast<int64_t>(std::ceil(400 / rate)), 2};

  int i = 0;
  while (row.size() != 0) {
    auto col = row["inputData"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);

  iter->Stop();
}

TEST_F(MindDataTestPipeline, TestTimeStretchPipelineWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestTimeStretchPipelineWrongArgs.";
  // op param
  int freq = 1025;
  int hop_length = 512;
  float rate = -2;
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, freq, 400, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto TimeStretchOp = audio::TimeStretch(hop_length, freq, rate);

  ds = ds->Map({TimeStretchOp});
  EXPECT_NE(ds, nullptr);

  // apply timestretch
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);
}

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
}

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
  while (row.size() != 0) {
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
