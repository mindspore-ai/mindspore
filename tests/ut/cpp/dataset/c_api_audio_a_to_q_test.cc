/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/include/dataset/audio.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using namespace std;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

/// Feature: AmplitudeToDB op
/// Description: Test AmplitudeToDB op pipelined
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestAmplitudeToDBPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAmplitudeToDBPipeline.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto amplitude_to_db_op = audio::AmplitudeToDB();

  ds = ds->Map({amplitude_to_db_op});
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

/// Feature: AmplitudeToDB op
/// Description: Test AmplitudeToDB op with wrong arguments
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestAmplitudeToDBWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAmplitudeToDBWrongArgs.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto amplitude_to_db_op = audio::AmplitudeToDB(ScaleType::kPower, 1.0, -1e-10, 80.0);

  ds = ds->Map({amplitude_to_db_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);
}

/// Feature: BandBiquad op
/// Description: Test BandBiquad op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestBandBiquadBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBandBiquadBasic.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto BandBiquadOp = audio::BandBiquad(44100, 200.0);

  ds = ds->Map({BandBiquadOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by bandbiquad
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

/// Feature: BandBiquad op
/// Description: Test BandBiquad op with invalid Q_ and sample_rate
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestBandBiquadParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBandBiquadParamCheck.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  std::shared_ptr<Dataset> ds02;
  EXPECT_NE(ds, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  auto band_biquad_op_01 = audio::BandBiquad(0, 200);
  ds01 = ds->Map({band_biquad_op_01});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);

  // Check Q_
  MS_LOG(INFO) << "Q_ is zero.";
  auto band_biquad_op_02 = audio::BandBiquad(44100, 200, 0);
  ds02 = ds->Map({band_biquad_op_02});
  EXPECT_NE(ds02, nullptr);

  std::shared_ptr<Iterator> iter02 = ds02->CreateIterator();
  EXPECT_EQ(iter02, nullptr);
}

/// Feature: AllpassBiquad op
/// Description: Test AllpassBiquad op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestAllpassBiquadBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAllpassBiquadBasic.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto AllpassBiquadOp = audio::AllpassBiquad(44100, 200.0);

  ds = ds->Map({AllpassBiquadOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by allpassbiquad
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

/// Feature: AllpassBiquad op
/// Description: Test AllpassBiquad op with invalid Q_ and sample_rate
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestAllpassBiquadParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAllpassBiquadParamCheck.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  std::shared_ptr<Dataset> ds02;
  EXPECT_NE(ds, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "Sample_rate_ is zero.";
  auto allpass_biquad_op_01 = audio::AllpassBiquad(0, 200.0, 0.707);
  ds01 = ds->Map({allpass_biquad_op_01});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);

  // Check Q_
  MS_LOG(INFO) << "Q_ is zero.";
  auto allpass_biquad_op_02 = audio::AllpassBiquad(44100, 200, 0);
  ds02 = ds->Map({allpass_biquad_op_02});
  EXPECT_NE(ds02, nullptr);

  std::shared_ptr<Iterator> iter02 = ds02->CreateIterator();
  EXPECT_EQ(iter02, nullptr);
}

/// Feature: BandpassBiquad op
/// Description: Test BandpassBiquad op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestBandpassBiquadBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBandpassBiquadBasic.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto BandpassBiquadOp = audio::BandpassBiquad(44100, 200.0);

  ds = ds->Map({BandpassBiquadOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by bandpassbiquad
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

/// Feature: BandpassBiquad op
/// Description: Test BandpassBiquad op with invalid Q_ and sample_rate
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestBandpassBiquadParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBandpassBiquadParamCheck.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  std::shared_ptr<Dataset> ds02;
  EXPECT_NE(ds, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  auto bandpass_biquad_op_01 = audio::BandpassBiquad(0, 200);
  ds01 = ds->Map({bandpass_biquad_op_01});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);

  // Check Q_
  MS_LOG(INFO) << "Q_ is zero.";
  auto bandpass_biquad_op_02 = audio::BandpassBiquad(44100, 200, 0);
  ds02 = ds->Map({bandpass_biquad_op_02});
  EXPECT_NE(ds02, nullptr);

  std::shared_ptr<Iterator> iter02 = ds02->CreateIterator();
  EXPECT_EQ(iter02, nullptr);
}

/// Feature: BandrejectBiquad op
/// Description: Test BandrejectBiquad op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestBandrejectBiquadBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBandrejectBiquadBasic.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto BandrejectBiquadOp = audio::BandrejectBiquad(44100, 200.0);

  ds = ds->Map({BandrejectBiquadOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by bandrejectbiquad
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

/// Feature: BandrejectBiquad op
/// Description: Test BandrejectBiquad op with invalid Q_ and sample_rate
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestBandrejectBiquadParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBandrejectBiquadParamCheck.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  std::shared_ptr<Dataset> ds02;
  EXPECT_NE(ds, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  auto bandreject_biquad_op_01 = audio::BandrejectBiquad(0, 200);
  ds01 = ds->Map({bandreject_biquad_op_01});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);

  // Check Q_
  MS_LOG(INFO) << "Q_ is zero.";
  auto bandreject_biquad_op_02 = audio::BandrejectBiquad(44100, 200, 0);
  ds02 = ds->Map({bandreject_biquad_op_02});
  EXPECT_NE(ds02, nullptr);

  std::shared_ptr<Iterator> iter02 = ds02->CreateIterator();
  EXPECT_EQ(iter02, nullptr);
}

/// Feature: BassBiquad op
/// Description: Test BassBiquad op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestBassBiquadBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBassBiquadBasic.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto BassBiquadOp = audio::BassBiquad(44100, 50, 200.0);

  ds = ds->Map({BassBiquadOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by bassbiquad
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

/// Feature: BassBiquad op
/// Description: Test BassBiquad op with invalid Q_ and sample_rate
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestBassBiquadParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBassBiquadParamCheck.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  std::shared_ptr<Dataset> ds02;
  EXPECT_NE(ds, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  auto bass_biquad_op_01 = audio::BassBiquad(0, 50, 200.0);
  ds01 = ds->Map({bass_biquad_op_01});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);

  // Check Q_
  MS_LOG(INFO) << "Q_ is zero.";
  auto bass_biquad_op_02 = audio::BassBiquad(44100, 50, 200.0, 0);
  ds02 = ds->Map({bass_biquad_op_02});
  EXPECT_NE(ds02, nullptr);

  std::shared_ptr<Iterator> iter02 = ds02->CreateIterator();
  EXPECT_EQ(iter02, nullptr);
}

/// Feature: Angle op
/// Description: Test Angle op with pipeline mode
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestAnglePipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAnglePipeline.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("complex", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto angle_op = audio::Angle();

  ds = ds->Map({angle_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {2};

  int i = 0;
  while (row.size() != 0) {
    auto col = row["complex"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 1);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);

  iter->Stop();
}

/// Feature: Angle op
/// Description: Test Angle op with pipeline mode with invalid input
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestAnglePipelineError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestAnglePipelineError.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("complex", mindspore::DataType::kNumberTypeFloat32, {3, 2, 1}));
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto angle_op = audio::Angle();

  ds = ds->Map({angle_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  std::unordered_map<std::string, mindspore::MSTensor> row;
  EXPECT_ERROR(iter->GetNextRow(&row));
}

/// Feature: EqualizerBiquad op
/// Description: Test EqualizerBiquad op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestEqualizerBiquadSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestEqualizerBiquadSuccess.";

  // Create an input tensor
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeFloat32, {1, 200}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  // Create a filter object
  auto equalizer_biquad = std::make_shared<audio::EqualizerBiquad>(44100, 3.5, 5.5, 0.707);
  auto ds1 = ds->Map({equalizer_biquad}, {"col1"}, {"audio"});
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

/// Feature: EqualizerBiquad op
/// Description: Test EqualizerBiquad op with invalid sample_rate and Q_
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestEqualizerBiquadWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestEqualizerBiquadWrongArgs.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 10}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  std::shared_ptr<Dataset> ds02;
  EXPECT_NE(ds, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  auto equalizer_biquad_op_01 = audio::EqualizerBiquad(0, 200.0, 5.5, 0.7);
  ds01 = ds->Map({equalizer_biquad_op_01});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);

  // Check Q
  MS_LOG(INFO) << "Q is zero.";
  auto equalizer_biquad_op_02 = audio::EqualizerBiquad(44100, 2000.0, 5.5, 0);
  ds02 = ds->Map({equalizer_biquad_op_02});
  EXPECT_NE(ds02, nullptr);

  std::shared_ptr<Iterator> iter02 = ds02->CreateIterator();
  EXPECT_EQ(iter02, nullptr);
}

/// Feature: LowpassBiquad op
/// Description: Test LowpassBiquad op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestLowpassBiquadSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLowpassBiquadSuccess.";

  // Create an input tensor
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeFloat32, {1, 200}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  // Create a filter object
  auto lowpass_biquad = std::make_shared<audio::LowpassBiquad>(44100, 3000.5, 0.707);
  auto ds1 = ds->Map({lowpass_biquad}, {"col1"}, {"audio"});
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

/// Feature: LowpassBiquad op
/// Description: Test LowpassBiquad op with invalid sample_rate and Q_
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestLowpassBiquadWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLowpassBiquadWrongArgs.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 10}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  std::shared_ptr<Dataset> ds02;
  EXPECT_NE(ds, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  auto lowpass_biquad_op_01 = audio::LowpassBiquad(0, 200.0, 0.7);
  ds01 = ds->Map({lowpass_biquad_op_01});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);

  // Check Q
  MS_LOG(INFO) << "Q is zero.";
  auto lowpass_biquad_op_02 = audio::LowpassBiquad(44100, 2000.0, 0);
  ds02 = ds->Map({lowpass_biquad_op_02});
  EXPECT_NE(ds02, nullptr);

  std::shared_ptr<Iterator> iter02 = ds02->CreateIterator();
  EXPECT_EQ(iter02, nullptr);
}

/// Feature: FrequencyMasking op
/// Description: Test FrequencyMaking op with pipeline mode
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFrequencyMaskingPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFrequencyMaskingPipeline.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {200, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto frequencymasking = audio::FrequencyMasking(true, 6);

  ds = ds->Map({frequencymasking});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {200, 200};

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

/// Feature: FrequencyMasking op
/// Description: Test FrequencyMaking op with invalid arguments
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestFrequencyMaskingWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFrequencyMaskingWrongArgs.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {20, 20}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto frequencymasking = audio::FrequencyMasking(true, -100);

  ds = ds->Map({frequencymasking});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by bandbiquad
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);
}

/// Feature: ComplexNorm op
/// Description: Test ComplexNorm op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestComplexNormBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComplexNormBasic.";

  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeInt64, {3, 2, 4, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto ComplexNormOp = audio::ComplexNorm(3.0);

  ds = ds->Map({ComplexNormOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by ComplexNorm
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {3, 2, 4};

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

/// Feature: ComplexNorm op
/// Description: Test ComplexNorm op with wrong arguments
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestComplexNormWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComplexNormWrongArgs.";

  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeInt64, {3, 2, 4, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto ComplexNormOp = audio::ComplexNorm(-10);

  ds = ds->Map({ComplexNormOp});
  std::shared_ptr<Iterator> iter1 = ds->CreateIterator();
  EXPECT_EQ(iter1, nullptr);
}

/// Feature: Contrast op
/// Description: Test Contrast op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestContrastBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestContrastBasic.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto ContrastOp = audio::Contrast();

  ds = ds->Map({ContrastOp});
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

/// Feature: Contrast op
/// Description: Test Contrast op with invalid enhancement_amount
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestContrastParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestContrastParamCheck.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat64, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  std::shared_ptr<Dataset> ds02;
  EXPECT_NE(ds, nullptr);

  // Check enhancement_amount
  MS_LOG(INFO) << "enhancement_amount is negative.";
  auto contrast_op_01 = audio::Contrast(-10);
  ds01 = ds->Map({contrast_op_01});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);

  MS_LOG(INFO) << "enhancement_amount is out of range.";
  auto contrast_op_02 = audio::Contrast(101);
  ds02 = ds->Map({contrast_op_02});
  EXPECT_NE(ds02, nullptr);

  std::shared_ptr<Iterator> iter02 = ds02->CreateIterator();
  EXPECT_EQ(iter02, nullptr);
}

/// Feature: DeemphBiquad op
/// Description: Test DeemphBiquad op basic usage in pipeline mode
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestDeemphBiquadPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDeemphBiquadPipeline.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto DeemphBiquadOp = audio::DeemphBiquad(44100);

  ds = ds->Map({DeemphBiquadOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by deemphbiquad
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

/// Feature: DeemphBiquad op
/// Description: Test DeemphBiquad op with invalid sample_rate and Q_
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestDeemphBiquadWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDeemphBiquadWrongArgs.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  EXPECT_NE(ds, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "Sample_rate_ is zero.";
  auto deemph_biquad_op_01 = audio::DeemphBiquad(0);
  ds01 = ds->Map({deemph_biquad_op_01});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);
}

/// Feature: Dither op
/// Description: Test basic usage of Dither op in pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestDitherBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDitherBasic.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(2);
  EXPECT_NE(ds, nullptr);

  auto DitherOp = audio::Dither();

  ds = ds->Map({DitherOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by Dither
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
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);
  iter->Stop();
}

/// Feature: HighpassBiquad op
/// Description: Test HighpassBiquad op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestHighpassBiquadSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestHighpassBiquadSuccess.";

  // Create an input tensor
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeFloat32, {1, 200}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  // Create a filter object
  auto highpass_biquad = make_shared<audio::HighpassBiquad>(44100, 3000.5, 0.707);
  auto ds1 = ds->Map({highpass_biquad}, {"col1"}, {"audio"});
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

/// Feature: HighpassBiquad op
/// Description: Test HighpassBiquad op with invalid sample_rate and Q_
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestHighpassBiquadWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestHighpassBiquadWrongArgs.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 10}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  std::shared_ptr<Dataset> ds02;
  EXPECT_NE(ds, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  auto highpass_biquad_op_01 = audio::HighpassBiquad(0, 200.0, 0.7);
  ds01 = ds->Map({highpass_biquad_op_01});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);

  // Check Q
  MS_LOG(INFO) << "Q is zero.";
  auto highpass_biquad_op_02 = audio::HighpassBiquad(44100, 2000.0, 0);
  ds02 = ds->Map({highpass_biquad_op_02});
  EXPECT_NE(ds02, nullptr);

  std::shared_ptr<Iterator> iter02 = ds02->CreateIterator();
  EXPECT_EQ(iter02, nullptr);
}

/// Feature: InverseMelScale op
/// Description: Test basic usage of InverseMelScale op
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestInverseMelScalePipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestInverseMelScalePipeline.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {4, 3, 7}));
  std::shared_ptr<Dataset> ds = RandomData(10, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto inverse_mel_scale_op1 = audio::InverseMelScale(20, 3, 16000, 0, 8000, 10);
  ds = ds->Map({inverse_mel_scale_op1});
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::vector<int64_t> expected = {4, 20, 7};
  int i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 3);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 10);
  iter->Stop();

  std::shared_ptr<SchemaObj> schema2 = Schema();
  ASSERT_OK(schema2->add_column("waveform", mindspore::DataType::kNumberTypeFloat64, {10, 20, 30}));
  ds = RandomData(10, schema2);
  EXPECT_NE(ds, nullptr);
  auto inverse_mel_scale_op2 = audio::InverseMelScale(128, 20, 16000, 0, 8000, 100);
  ds = ds->Map({inverse_mel_scale_op2});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);
  ASSERT_OK(iter->GetNextRow(&row));
  expected = {10, 128, 30};
  i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 3);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat64);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 10);
  iter->Stop();

  std::shared_ptr<SchemaObj> schema3 = Schema();
  ASSERT_OK(schema3->add_column("waveform", mindspore::DataType::kNumberTypeInt16, {3, 4, 5}));
  ds = RandomData(10, schema3);
  EXPECT_NE(ds, nullptr);
  auto inverse_mel_scale_op3 = audio::InverseMelScale(128, 4, 16000, 0, 8000, 100);
  ds = ds->Map({inverse_mel_scale_op3});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);
  ASSERT_OK(iter->GetNextRow(&row));
  expected = {3, 128, 5};
  i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 3);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 10);
  iter->Stop();

  std::shared_ptr<SchemaObj> schema4 = Schema();
  ASSERT_OK(schema4->add_column("waveform", mindspore::DataType::kNumberTypeInt16, {4, 20}));
  ds = RandomData(10, schema4);
  EXPECT_NE(ds, nullptr);
  auto inverse_mel_scale_op4 = audio::InverseMelScale(20, 4, 16000, 0, 8000, 100);
  ds = ds->Map({inverse_mel_scale_op4});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);
  ASSERT_OK(iter->GetNextRow(&row));
  expected = {1, 20, 20};
  i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 3);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 10);
  iter->Stop();
}

/// Feature: InverseMelScale op
/// Description: Test wrong arguments for InverseMelScale op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestInverseMelScaleWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestInverseMelScaleWrongArgs.";
  // MelScale: f_max must be greater than f_min.
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {3, 4, 5}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);
  auto inverse_mel_scale_op = audio::InverseMelScale(128, 4, 1000, -100, -100);
  ds = ds->Map({inverse_mel_scale_op});
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);

  // MelScale: n_mels must be greater than 0.
  inverse_mel_scale_op = audio::InverseMelScale(-128, 16000, 1000, 10, 100);
  ds = ds->Map({inverse_mel_scale_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);

  // MelScale: sample_rate must be greater than f_min.
  inverse_mel_scale_op = audio::InverseMelScale(128, -16000, 1000, 10, 100);
  ds = ds->Map({inverse_mel_scale_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);

  // MelScale: max_iter must be greater than 0.
  inverse_mel_scale_op = audio::InverseMelScale(128, 16000, 1000, 10, 100, -10);
  ds = ds->Map({inverse_mel_scale_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);

  // MelScale: tolerance_loss must be greater than 0.
  inverse_mel_scale_op = audio::InverseMelScale(128, 16000, 1000, 10, 100, 10, -10);
  ds = ds->Map({inverse_mel_scale_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);

  // MelScale: tolerance_change must be greater than 0.
  inverse_mel_scale_op = audio::InverseMelScale(128, 16000, 1000, 10, 100, 10, 10, -10);
  ds = ds->Map({inverse_mel_scale_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: MelscaleFbanks op
/// Description: Test normal operation
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestMelscaleFbanksNormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-MelscaleFbanksNormal.";
  mindspore::MSTensor output;
  NormType norm = NormType::kSlaney;
  MelType mel_type = MelType::kHtk;
  Status s01 = audio::MelscaleFbanks(&output, 1024, 0, 1000, 40, 16000, norm, mel_type);
  EXPECT_TRUE(s01.IsOk());
}

/// Feature: LinearFbanks.
/// Description: Test normal operation.
/// Expectation: As expected.
TEST_F(MindDataTestPipeline, TestLinearFbanksNormal) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-LinearFbanksNormal.";
  mindspore::MSTensor output;
  Status s01 = audio::LinearFbanks(&output, 1024, 0, 1000, 40, 16000);
  EXPECT_TRUE(s01.IsOk());
}

/// Feature: LinearFbanks.
/// Description: Test operation with invalid input.
/// Expectation: Throw exception as expected.
TEST_F(MindDataTestPipeline, TestLinearFbanksWithInvalidInput) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLinearFbanksWithInvalidInput.";
  mindspore::MSTensor output;
  MS_LOG(INFO) << "n_freqs is too low.";
  Status s01 = audio::LinearFbanks(&output, 0, 50, 1000, 20, 16000);
  EXPECT_FALSE(s01.IsOk());
  MS_LOG(INFO) << "f_max is not greater than f_min.";
  Status s02 = audio::LinearFbanks(&output, 1024, 1000, 50, 20, 16000);
  EXPECT_FALSE(s02.IsOk());
  MS_LOG(INFO) << "n_filter is too low.";
  Status s03 = audio::LinearFbanks(&output, 1024, 50, 1000, 0, 16000);
  EXPECT_FALSE(s03.IsOk());
  MS_LOG(INFO) << "sample_rate is too low.";
  Status s04 = audio::LinearFbanks(&output, 1024, 50, 1000, 20, 0);
  EXPECT_FALSE(s04.IsOk());
}

/// Feature: MuLawDecoding op
/// Description: Test MuLawDecoding op basic usage
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestMuLawDecodingBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMuLawDecodingBasic.";

  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeInt32, {1, 100}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto mu_law_decoding_op = audio::MuLawDecoding();

  ds = ds->Map({mu_law_decoding_op});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by MuLawDecoding
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {1, 100};

  int i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);

  iter->Stop();
}

/// Feature: MuLawDecoding op
/// Description: Test MuLawDecoding op with invalid quantization_channels
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMuLawDecodingWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMuLawDecodingWrongArgs.";

  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeInt32, {1, 100}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  // quantization_channels is negative
  auto mu_law_decoding_op1 = audio::MuLawDecoding(-10);

  ds = ds->Map({mu_law_decoding_op1});
  std::shared_ptr<Iterator> iter1 = ds->CreateIterator();
  EXPECT_EQ(iter1, nullptr);

  // quantization_channels is 0
  auto mu_law_decoding_op2 = audio::MuLawDecoding(0);

  ds = ds->Map({mu_law_decoding_op2});
  std::shared_ptr<Iterator> iter2 = ds->CreateIterator();
  EXPECT_EQ(iter1, nullptr);
}

/// Feature: MuLawEncoding op
/// Description: Test MuLawEncoding op in pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestMuLawEncodingBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMuLawEncodingBasic.";

  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 100}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto mu_law_encoding_op = audio::MuLawEncoding();

  ds = ds->Map({mu_law_encoding_op});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by MuLawEncoding
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {1, 100};

  int i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeInt32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);

  iter->Stop();
}

/// Feature: MuLawEncoding op
/// Description: Test invalid parameter of MuLawEncoding op
/// Expectation: Throw exception correctly
TEST_F(MindDataTestPipeline, TestMuLawEncodingWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMuLawEncodingWrongArgs.";

  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 100}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  // quantization_channels is negative
  auto mu_law_encoding_op1 = audio::MuLawEncoding(-10);

  ds = ds->Map({mu_law_encoding_op1});
  std::shared_ptr<Iterator> iter1 = ds->CreateIterator();
  EXPECT_EQ(iter1, nullptr);

  // quantization_channels is 0
  auto mu_law_encoding_op2 = audio::MuLawEncoding(0);

  ds = ds->Map({mu_law_encoding_op2});
  std::shared_ptr<Iterator> iter2 = ds->CreateIterator();
  EXPECT_EQ(iter1, nullptr);
}

/// Feature: Overdrive op
/// Description: Test basic usage of Overdrive op
/// Expectation: Get correct number of data
TEST_F(MindDataTestPipeline, TestOverdriveBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOverdriveBasic.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto OverdriveOp = audio::Overdrive();

  ds = ds->Map({OverdriveOp});
  EXPECT_NE(ds, nullptr);

  // Apply a phasing effect to the audio
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

/// Feature: Overdrive op
/// Description: Test invalid parameter of Overdrive op
/// Expectation: Throw exception correctly
TEST_F(MindDataTestPipeline, TestOverdriveWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestOverdriveWrongArg.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  std::shared_ptr<Dataset> ds02;
  EXPECT_NE(ds, nullptr);

  // Check gain out of range [0,100]
  MS_LOG(INFO) << "gain is less than 0.";
  auto overdrive_op_01 = audio::Overdrive(-0.2, 20.0);
  ds01 = ds->Map({overdrive_op_01});
  EXPECT_NE(ds01, nullptr);
  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);

  // Check color out of range [0,100]
  MS_LOG(INFO) << "color is greater than 100.";
  auto overdrive_op_02 = audio::Overdrive(20.0, 102.3);
  ds02 = ds->Map({overdrive_op_02});
  EXPECT_NE(ds02, nullptr);
  std::shared_ptr<Iterator> iter02 = ds02->CreateIterator();
  EXPECT_EQ(iter02, nullptr);
}

/// Feature: Phaser op
/// Description: Test basic usage of Phaser op
/// Expectation: Get correct number of data
TEST_F(MindDataTestPipeline, TestPhaserBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPhaserBasic";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto PhaserOp = audio::Phaser(44100);

  ds = ds->Map({PhaserOp});
  EXPECT_NE(ds, nullptr);

  // Apply a phasing effect to the audio
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

/// Feature: Phaser op
/// Description: Test invalid parameter of Phaser op
/// Expectation: Throw exception correctly
TEST_F(MindDataTestPipeline, TestPhaserWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPhaserWrongArg.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  std::shared_ptr<Dataset> ds02;
  std::shared_ptr<Dataset> ds03;
  std::shared_ptr<Dataset> ds04;
  std::shared_ptr<Dataset> ds05;
  std::shared_ptr<Dataset> ds06;
  std::shared_ptr<Dataset> ds07;
  std::shared_ptr<Dataset> ds08;
  std::shared_ptr<Dataset> ds09;
  std::shared_ptr<Dataset> ds10;
  EXPECT_NE(ds, nullptr);

  // Check gain_in out of range [0,1]
  MS_LOG(INFO) << "gain_in is less than 0.";
  auto phaser_op_01 = audio::Phaser(44100, -0.2);
  ds01 = ds->Map({phaser_op_01});
  EXPECT_NE(ds01, nullptr);
  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);

  MS_LOG(INFO) << "gain_in is greater than 1.";
  auto phaser_op_02 = audio::Phaser(44100, 1.2);
  ds02 = ds->Map({phaser_op_02});
  EXPECT_NE(ds02, nullptr);
  std::shared_ptr<Iterator> iter02 = ds02->CreateIterator();
  EXPECT_EQ(iter02, nullptr);

  // Check gain_out out of range [0,1e9]
  MS_LOG(INFO) << "gain_out is less than 0.";
  auto phaser_op_03 = audio::Phaser(44100, 0.2, -1.3);
  ds03 = ds->Map({phaser_op_03});
  EXPECT_NE(ds03, nullptr);
  std::shared_ptr<Iterator> iter03 = ds03->CreateIterator();
  EXPECT_EQ(iter03, nullptr);

  MS_LOG(INFO) << "gain_out is greater than 1e9.";
  auto phaser_op_04 = audio::Phaser(44100, 0.3, 1e10);
  ds04 = ds->Map({phaser_op_04});
  EXPECT_NE(ds04, nullptr);
  std::shared_ptr<Iterator> iter04 = ds04->CreateIterator();
  EXPECT_EQ(iter04, nullptr);

  // Check delay_ms out of range [0,5.0]
  MS_LOG(INFO) << "delay_ms is less than 0.";
  auto phaser_op_05 = audio::Phaser(44100, 0.2, 2, -2.0);
  ds05 = ds->Map({phaser_op_05});
  EXPECT_NE(ds05, nullptr);
  std::shared_ptr<Iterator> iter05 = ds05->CreateIterator();
  EXPECT_EQ(iter05, nullptr);

  MS_LOG(INFO) << "delay_ms is greater than 5.0.";
  auto phaser_op_06 = audio::Phaser(44100, 0.3, 2, 6.0);
  ds06 = ds->Map({phaser_op_06});
  EXPECT_NE(ds06, nullptr);
  std::shared_ptr<Iterator> iter06 = ds06->CreateIterator();
  EXPECT_EQ(iter06, nullptr);

  // Check decay out of range [0,0.99]
  MS_LOG(INFO) << "decay is less than 0.";
  auto phaser_op_07 = audio::Phaser(44100, 0.2, 2, 2.0, -1.0);
  ds07 = ds->Map({phaser_op_07});
  EXPECT_NE(ds07, nullptr);
  std::shared_ptr<Iterator> iter07 = ds07->CreateIterator();
  EXPECT_EQ(iter07, nullptr);

  MS_LOG(INFO) << "decay is greater than 0.99.";
  auto phaser_op_08 = audio::Phaser(44100, 0.3, 2, 2.0, 1.2);
  ds08 = ds->Map({phaser_op_08});
  EXPECT_NE(ds08, nullptr);
  std::shared_ptr<Iterator> iter08 = ds08->CreateIterator();
  EXPECT_EQ(iter08, nullptr);

  // Check mod_speed out of range [0.1,10]
  MS_LOG(INFO) << "mod_speed is less than 0.1 .";
  auto phaser_op_09 = audio::Phaser(44100, 0.2, 2, 2.0, 0.5, 0.002);
  ds09 = ds->Map({phaser_op_09});
  EXPECT_NE(ds09, nullptr);
  std::shared_ptr<Iterator> iter09 = ds09->CreateIterator();
  EXPECT_EQ(iter09, nullptr);

  MS_LOG(INFO) << "mod_speed is greater than 10.";
  auto phaser_op_10 = audio::Phaser(44100, 0.3, 2, 2.0, 0.5, 12.0);
  ds10 = ds->Map({phaser_op_10});
  EXPECT_NE(ds10, nullptr);
  std::shared_ptr<Iterator> iter10 = ds10->CreateIterator();
  EXPECT_EQ(iter10, nullptr);
}

/// Feature: LFilter op
/// Description: Test LFilter op with pipeline mode
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestLfilterPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLfilterPipeline.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  std::vector<float> a_coeffs = {0.1, 0.2, 0.3};
  std::vector<float> b_coeffs = {0.1, 0.2, 0.3};
  auto LFilterOp = audio::LFilter(a_coeffs, b_coeffs);

  ds = ds->Map({LFilterOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by lfilter
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

/// Feature: LFilter op
/// Description: Test LFilter op with invalid sample_rate
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestLfilterWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLfilterWrongArgs.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  EXPECT_NE(ds, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "a_coeffs size not equal to b_coeffs";
  std::vector<float> a_coeffs = {0.1, 0.2, 0.3};
  std::vector<float> b_coeffs = {0.1, 0.2};
  auto LFilterOp = audio::LFilter(a_coeffs, b_coeffs);
  ds01 = ds->Map({LFilterOp});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);
}

/// Feature: DCShift op
/// Description: Test DCShift op with pipeline mode
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestDCShiftPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDCShiftPipeline.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 2, 100}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  auto dc_shift_op = audio::DCShift(0.8, 0.02);

  ds = ds->Map({dc_shift_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {1, 2, 100};

  int i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 3);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);

  iter->Stop();
}

/// Feature: DCShift op
/// Description: Test DCShift op with pipeline mode with invalid input
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestDCShiftPipelineError) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDCShiftPipelineError.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {100}));
  std::shared_ptr<Dataset> ds = RandomData(4, schema);
  EXPECT_NE(ds, nullptr);

  auto dc_shift_op = audio::DCShift(3, 0.02);

  ds = ds->Map({dc_shift_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Biquad op
/// Description: Test Biquad op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestBiquadBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBiquadBasic.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto BiquadOp = audio::Biquad(0.01, 0.02, 0.13, 1, 0.12, 0.3);

  ds = ds->Map({BiquadOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by biquad
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

/// Feature: Biquad op
/// Description: Test Biquad op with invalid a0
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestBiquadParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestBiquadParamCheck.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  EXPECT_NE(ds, nullptr);

  // Check a0
  MS_LOG(INFO) << "a0 is zero.";
  auto biquad_op_01 = audio::Biquad(0.01, 0.02, 0.13, 0, 0.12, 0.3);
  ds01 = ds->Map({biquad_op_01});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);
}

/// Feature: Fade op
/// Description: Test Fade op with pipeline mode
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFadeWithPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFadeWithPipeline.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {1, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto fade_op = audio::Fade(20, 30, FadeShape::kExponential);

  ds = ds->Map({fade_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {1, 200};

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

/// Feature: Fade op
/// Description: Test Fade op with FadeShape::kLinear
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFadeWithLinear) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFadeWithLinear.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 10}));
  std::shared_ptr<Dataset> ds = RandomData(10, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto fade_op = audio::Fade(5, 5, FadeShape::kLinear);

  ds = ds->Map({fade_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {2, 10};

  int i = 0;
  while (row.size() != 0) {
    auto col = row["inputData"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 10);

  iter->Stop();
}

/// Feature: Fade op
/// Description: Test Fade op with FadeShape::kLogarithmic
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFadeWithLogarithmic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFadeWithLogarithmic.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat64, {1, 150}));
  std::shared_ptr<Dataset> ds = RandomData(30, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto fade_op = audio::Fade(80, 100, FadeShape::kLogarithmic);

  ds = ds->Map({fade_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {1, 150};

  int i = 0;
  while (row.size() != 0) {
    auto col = row["inputData"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat64);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 30);

  iter->Stop();
}

/// Feature: Fade op
/// Description: Test Fade op with FadeShape::kQuarterSine
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFadeWithQuarterSine) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFadeWithQuarterSine.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeInt32, {20, 20000}));
  std::shared_ptr<Dataset> ds = RandomData(40, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto fade_op = audio::Fade(1000, 1000, FadeShape::kQuarterSine);

  ds = ds->Map({fade_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {20, 20000};

  int i = 0;
  while (row.size() != 0) {
    auto col = row["inputData"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 40);

  iter->Stop();
}

/// Feature: Fade op
/// Description: Test Fade op with FadeShape::kHalfSine
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFadeWithHalfSine) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFadeWithHalfSine.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeInt16, {1, 200}));
  std::shared_ptr<Dataset> ds = RandomData(40, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto fade_op = audio::Fade(100, 100, FadeShape::kHalfSine);

  ds = ds->Map({fade_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {1, 200};

  int i = 0;
  while (row.size() != 0) {
    auto col = row["inputData"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 40);

  iter->Stop();
}

/// Feature: Fade op
/// Description: Test Fade op with invalid arguments
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestFadeWithInvalidArg) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFadeWithInvalidArg.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {1, 200}));
  std::shared_ptr<Dataset> ds_01 = RandomData(50, schema);
  EXPECT_NE(ds_01, nullptr);

  ds_01 = ds_01->SetNumWorkers(4);
  EXPECT_NE(ds_01, nullptr);

  auto fade_op_01 = audio::Fade(-20, 30, FadeShape::kLogarithmic);

  ds_01 = ds_01->Map({fade_op_01});
  EXPECT_NE(ds_01, nullptr);
  // Expect failure, fade in length less than zero
  std::shared_ptr<Iterator> iter_01 = ds_01->CreateIterator();
  EXPECT_EQ(iter_01, nullptr);

  std::shared_ptr<Dataset> ds_02 = RandomData(50, schema);
  EXPECT_NE(ds_02, nullptr);

  ds_02 = ds_02->SetNumWorkers(4);
  EXPECT_NE(ds_02, nullptr);

  auto fade_op_02 = audio::Fade(5, -3, FadeShape::kExponential);

  ds_02 = ds_02->Map({fade_op_02});
  EXPECT_NE(ds_02, nullptr);
  // Expect failure, fade out length less than zero
  std::shared_ptr<Iterator> iter_02 = ds_02->CreateIterator();
  EXPECT_EQ(iter_02, nullptr);
}

/// Feature: Filtfilt
/// Description: Test Filtfilt pipeline usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFiltfiltpipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFiltfiltpipeline.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  std::vector<float> a_coeffs = {0.1, 0.2, 0.3};
  std::vector<float> b_coeffs = {0.1, 0.2, 0.3};
  auto filtfiltop = audio::Filtfilt(a_coeffs, b_coeffs);

  ds = ds->Map({filtfiltop});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by filtfilt
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

/// Feature: Filtfilt
/// Description: Test Filtfilt invalid Args
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFiltfiltWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFiltfiltWrongArgs.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  EXPECT_NE(ds, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "a_coeffs size not equal to b_coeffs";
  std::vector<float> a_coeffs = {0.1, 0.2, 0.3};
  std::vector<float> b_coeffs = {0.1, 0.2};
  auto FiltfiltOp = audio::Filtfilt(a_coeffs, b_coeffs);
  ds01 = ds->Map({FiltfiltOp});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);
}

/// Feature: Magphase op
/// Description: Test Magphase op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestMagphase) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMagphase.";

  float power = 2.0;
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeFloat32, {1, 2}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);
  auto magphase = std::make_shared<audio::Magphase>(power);
  auto ds1 = ds->Map({magphase}, {"col1"}, {"mag", "phase"});
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

/// Feature: Magphase op
/// Description: Test Magphase op with invalid power
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMagphaseWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMagphaseWrongArgs.";

  float power_wrong = -1.0;
  auto magphase = std::make_shared<audio::Magphase>(power_wrong);
  std::unordered_map<std::string, mindspore::MSTensor> row;

  // Magphase: power must be greater than or equal to 0.
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->Map({magphase}, {"col1"}, {"mag", "phase"});
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: DetectPitchFrequency op
/// Description: Test DetectPitchFrequency op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestDetectPitchFrequencyBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDetectPitchFrequencyBasic.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 10000}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto DetectPitchFrequencyOp = audio::DetectPitchFrequency(44100);

  ds = ds->Map({DetectPitchFrequencyOp});
  EXPECT_NE(ds, nullptr);

  // Detect pitch frequency
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {2, 8};

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

/// Feature: DetectPitchFrequency op
/// Description: Test DetectPitchFrequency op with invalid parameters
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestDetectPitchFrequencyParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDetectPitchFrequencyParamCheck.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  std::shared_ptr<Dataset> ds02;
  std::shared_ptr<Dataset> ds03;
  std::shared_ptr<Dataset> ds04;
  std::shared_ptr<Dataset> ds05;
  EXPECT_NE(ds, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  auto detect_pitch_frequency_op_01 = audio::DetectPitchFrequency(0);
  ds01 = ds->Map({detect_pitch_frequency_op_01});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);

  // Check frame_time
  MS_LOG(INFO) << "frame_time is zero.";
  auto detect_pitch_frequency_op_02 = audio::DetectPitchFrequency(30, 0);
  ds02 = ds->Map({detect_pitch_frequency_op_02});
  EXPECT_NE(ds02, nullptr);

  std::shared_ptr<Iterator> iter02 = ds02->CreateIterator();
  EXPECT_EQ(iter02, nullptr);

  // Check win_length
  MS_LOG(INFO) << "win_length is zero.";
  auto detect_pitch_frequency_op_03 = audio::DetectPitchFrequency(30, 0.1, 0);
  ds03 = ds->Map({detect_pitch_frequency_op_03});
  EXPECT_NE(ds03, nullptr);

  std::shared_ptr<Iterator> iter03 = ds03->CreateIterator();
  EXPECT_EQ(iter03, nullptr);

  // Check freq_low
  MS_LOG(INFO) << "freq_low is zero.";
  auto detect_pitch_frequency_op_04 = audio::DetectPitchFrequency(30, 0.1, 3, 0);
  ds04 = ds->Map({detect_pitch_frequency_op_04});
  EXPECT_NE(ds04, nullptr);

  std::shared_ptr<Iterator> iter04 = ds04->CreateIterator();
  EXPECT_EQ(iter04, nullptr);

  // Check freq_high
  MS_LOG(INFO) << "freq_high is zero.";
  auto detect_pitch_frequency_op_05 = audio::DetectPitchFrequency(30, 0.1, 3, 5, 0);
  ds05 = ds->Map({detect_pitch_frequency_op_05});
  EXPECT_NE(ds05, nullptr);

  std::shared_ptr<Iterator> iter05 = ds05->CreateIterator();
  EXPECT_EQ(iter05, nullptr);
}

/// Feature: Flanger op
/// Description: Test Flanger op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestFlangerBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFlangerBasic.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto FlangerOp = audio::Flanger(44100);

  ds = ds->Map({FlangerOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by flanger
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
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);
  iter->Stop();
}

/// Feature: Flanger op
/// Description: Test Flanger op with invalid parameters
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestFlangerParamCheck) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestFlangerParamCheck.";
  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 2}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is zero.";
  auto flanger_op_sample_rate =
    audio::Flanger(0, 0.0, 2.0, 0.0, 71.0, 0.5, 25.0, Modulation::kSinusoidal, Interpolation::kLinear);
  std::shared_ptr<Dataset> dsSample_rate = ds->Map({flanger_op_sample_rate});
  EXPECT_NE(dsSample_rate, nullptr);
  std::shared_ptr<Iterator> iterSample_rate = dsSample_rate->CreateIterator();
  EXPECT_EQ(iterSample_rate, nullptr);

  // Check delay
  MS_LOG(INFO) << "delay is out of range.";
  auto flanger_op_delay =
    audio::Flanger(44100, 50.0, 2.0, 0.0, 71.0, 0.5, 25.0, Modulation::kSinusoidal, Interpolation::kLinear);
  std::shared_ptr<Dataset> dsDelay = ds->Map({flanger_op_delay});
  EXPECT_NE(dsDelay, nullptr);
  std::shared_ptr<Iterator> iterDelay = dsDelay->CreateIterator();
  EXPECT_EQ(iterDelay, nullptr);

  // Check depth
  MS_LOG(INFO) << "depth is out of range.";
  auto flanger_op_depth =
    audio::Flanger(44100, 0.0, 20.0, 0.0, 71.0, 0.5, 25.0, Modulation::kSinusoidal, Interpolation::kLinear);
  std::shared_ptr<Dataset> dsDepth = ds->Map({flanger_op_depth});
  EXPECT_NE(dsDepth, nullptr);
  std::shared_ptr<Iterator> iterDepth = dsDepth->CreateIterator();
  EXPECT_EQ(iterDepth, nullptr);

  // Check regen
  MS_LOG(INFO) << "regen is out of range.";
  auto flanger_op_regen =
    audio::Flanger(44100, 0.0, 2.0, 100.0, 71.0, 0.5, 25.0, Modulation::kSinusoidal, Interpolation::kLinear);
  std::shared_ptr<Dataset> dsRegen = ds->Map({flanger_op_regen});
  EXPECT_NE(dsRegen, nullptr);
  std::shared_ptr<Iterator> iterRegen = dsRegen->CreateIterator();
  EXPECT_EQ(iterRegen, nullptr);

  // Check width
  MS_LOG(INFO) << "width is out of range.";
  auto flanger_op_width =
    audio::Flanger(44100, 0.0, 2.0, 0.0, 200.0, 0.5, 25.0, Modulation::kSinusoidal, Interpolation::kLinear);
  std::shared_ptr<Dataset> dsWidth = ds->Map({flanger_op_width});
  EXPECT_NE(dsWidth, nullptr);
  std::shared_ptr<Iterator> iterWidth = dsWidth->CreateIterator();
  EXPECT_EQ(iterWidth, nullptr);

  // Check speed
  MS_LOG(INFO) << "speed is out of range.";
  auto flanger_op_speed =
    audio::Flanger(44100, 0.0, 2.0, 0.0, 71.0, 20, 25.0, Modulation::kSinusoidal, Interpolation::kLinear);
  std::shared_ptr<Dataset> dsSpeed = ds->Map({flanger_op_speed});
  EXPECT_NE(dsSpeed, nullptr);
  std::shared_ptr<Iterator> iterSpeed = dsSpeed->CreateIterator();
  EXPECT_EQ(iterSpeed, nullptr);

  // Check phase
  MS_LOG(INFO) << "phase is out of range.";
  auto flanger_op_phase =
    audio::Flanger(44100, 0.0, 2.0, 0.0, 71.0, 20, 25.0, Modulation::kSinusoidal, Interpolation::kLinear);
  std::shared_ptr<Dataset> dsPhase = ds->Map({flanger_op_phase});
  EXPECT_NE(dsPhase, nullptr);
  std::shared_ptr<Iterator> iterPhase = dsPhase->CreateIterator();
  EXPECT_EQ(iterPhase, nullptr);
}

/// Feature: CreateDct op
/// Description: Test CreateDct op in eager mode with NormMode::kNone
/// Expectation: The returned result is as expected
TEST_F(MindDataTestPipeline, TestCreateDctNone) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCreateDctNone.";
  mindspore::MSTensor output;
  Status s01 = audio::CreateDct(&output, 200, 400, NormMode::kNone);
  EXPECT_TRUE(s01.IsOk());
}

/// Feature: CreateDct op
/// Description: Test CreateDct op in eager mode with NormMode::kOrtho
/// Expectation: The returned result is as expected
TEST_F(MindDataTestPipeline, TestCreateDctOrtho) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCreateDctOrtho.";
  mindspore::MSTensor output;
  Status s02 = audio::CreateDct(&output, 200, 400, NormMode::kOrtho);
  EXPECT_TRUE(s02.IsOk());
}

/// Feature: CreateDct op
/// Description: Test wrong arguments for CreateDct op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestCreateDctWrongArg) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCreateDctWrongArg.";
  mindspore::MSTensor output;
  // Check n_mfcc
  MS_LOG(INFO) << "n_mfcc is negative.";
  Status s03 = audio::CreateDct(&output, -200, 400, NormMode::kNone);
  EXPECT_FALSE(s03.IsOk());
  // Check n_mels
  MS_LOG(INFO) << "n_mels is negative.";
  Status s04 = audio::CreateDct(&output, 200, -400, NormMode::kOrtho);
  EXPECT_FALSE(s04.IsOk());
}

/// Feature: DBToAmplitude op
/// Description: Test DBToAmplitude op in pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestDBToAmplitudePipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestDBToAmplitudePipeline.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto db_to_amplitude_op = audio::DBToAmplitude(2, 2);

  ds = ds->Map({db_to_amplitude_op});
  EXPECT_NE(ds, nullptr);

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

/// Feature: ComputeDeltas op
/// Description: Test basic function of ComputeDeltas op
/// Expectation: Get correct number of data
TEST_F(MindDataTestPipeline, TestComputeDeltas) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComputeDeltas.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto compute_deltas_op = audio::ComputeDeltas();

  ds = ds->Map({compute_deltas_op});
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

/// Feature: ComputeDeltas op
/// Description: Test wrong input args of ComputeDeltas op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestComputeDeltasWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestComputeDeltasWrongArgs.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("inputData", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);
  auto compute_deltas_op = audio::ComputeDeltas(2, mindspore::dataset::BorderType::kEdge);
  ds = ds->Map({compute_deltas_op});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Gain op
/// Description: Test Gain op in pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestGainPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGainPipeline.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto GainOp = audio::Gain();

  ds = ds->Map({GainOp});
  EXPECT_NE(ds, nullptr);

  // Filtered waveform by Gain
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

/// Feature: MelScale op
/// Description: Test basic usage of MelScale op
/// Expectation: Get correct number of data
TEST_F(MindDataTestPipeline, TestMelScalePipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMelScalePipeline.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 4, 3}));
  std::shared_ptr<Dataset> ds = RandomData(10, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  auto mel_scale_op1 = audio::MelScale(2, 10, 0, 0, 4);
  ds = ds->Map({mel_scale_op1});
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::vector<int64_t> expected = {2, 2, 3};
  int i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 3);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 10);
  iter->Stop();

  std::shared_ptr<SchemaObj> schema2 = Schema();
  ASSERT_OK(schema2->add_column("waveform", mindspore::DataType::kNumberTypeFloat64, {4, 20}));
  ds = RandomData(10, schema2);
  EXPECT_NE(ds, nullptr);
  auto mel_scale_op2 = audio::MelScale(2, 10, -50, 100, 4);
  ds = ds->Map({mel_scale_op2});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);
  ASSERT_OK(iter->GetNextRow(&row));
  expected = {2, 20};
  i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat64);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 10);
  iter->Stop();

  std::shared_ptr<SchemaObj> schema3 = Schema();
  ASSERT_OK(schema3->add_column("waveform", mindspore::DataType::kNumberTypeInt16, {8, 50}));
  ds = RandomData(10, schema3);
  EXPECT_NE(ds, nullptr);
  auto mel_scale_op3 = audio::MelScale(2, 100, 10, 100, 8, NormType::kSlaney, MelType::kHtk);
  ds = ds->Map({mel_scale_op3});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);
  ASSERT_OK(iter->GetNextRow(&row));
  expected = {2, 50};
  i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 10);
  iter->Stop();
}

/// Feature: MelScale op
/// Description: Test wrong arguments for MelScale op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMelScaleWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMelScaleWrongArgs.";

  // MelScale: f_max must be greater than f_min.
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);
  auto mel_scale_op = audio::MelScale(128, 16000, 1000, -100, -100, NormType::kSlaney, MelType::kSlaney);
  ds = ds->Map({mel_scale_op});
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);

  // MelScale: n_mels must be greater than 0.
  mel_scale_op = audio::MelScale(-128, 16000, 1000, 10, 100, NormType::kSlaney, MelType::kSlaney);
  ds = ds->Map({mel_scale_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);

  // MelScale: sample_rate must be greater than f_min.
  mel_scale_op = audio::MelScale(128, -16000, 1000, 10, 100, NormType::kSlaney, MelType::kSlaney);
  ds = ds->Map({mel_scale_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: PhaseVocoder op
/// Description: Test PhaseVocoder op in pipeline mode
/// Expectation: The data is processed successfully
TEST_F(MindDataTestPipeline, TestPhaseVocoderPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPhaseVocoderPipeline.";
  std::shared_ptr<SchemaObj> schema = Schema();

  int freq = 1025;
  int hop_length = 512;
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, freq, 30, 2}));

  std::vector<float> phase_advance;
  float tpnum = 0;
  float step = (1.0 * M_PI * hop_length / (freq - 1));
  for (int i = 0; i < freq; i++) {
    phase_advance.push_back(tpnum);
    tpnum += step;
  }

  std::shared_ptr<Dataset> ds = RandomData(5, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Tensor> phase_advance_tensor;
  Tensor::CreateFromVector(phase_advance, TensorShape({1025, 1}), &phase_advance_tensor);
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(phase_advance_tensor));

  float rate = 1.3;
  auto PhaseVocoder = audio::PhaseVocoder(rate, input_ms);

  ds = ds->Map({PhaseVocoder});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {2, 1025, 24, 2};

  int i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 4);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 5);

  iter->Stop();
}

/// Feature: PhaseVocoder op
/// Description: Test PhaseVocoder op with wrong input
/// Expectation: Throw exception as expected
TEST_F(MindDataTestPipeline, TestPhaseVocoderWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPhaseVocoderWrongArgs.";
  std::shared_ptr<SchemaObj> schema = Schema();

  int freq = 1025;
  int hop_length = 512;
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, freq, 30, 2}));

  std::vector<float> phase_advance;
  float tpnum = 0;
  float step = (1.0 * M_PI * hop_length / (freq - 1));
  for (int i = 0; i < freq; i++) {
    phase_advance.push_back(tpnum);
    tpnum += step;
  }

  std::shared_ptr<Tensor> phase_advance_tensor;
  Tensor::CreateFromVector(phase_advance, TensorShape({1025, 1}), &phase_advance_tensor);
  auto input_ms = mindspore::MSTensor(std::make_shared<mindspore::dataset::DETensor>(phase_advance_tensor));

  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  float rate = -2.0;
  auto PhaseVocoder = audio::PhaseVocoder(rate, input_ms);

  ds = ds->Map({PhaseVocoder});
  EXPECT_NE(ds, nullptr);

  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: MaskAlongAxisIID op
/// Description: Test MaskAlongAxisIID op pipeline
/// Expectation: The returned result is as expected
TEST_F(MindDataTestPipeline, TestMaskAlongAxisIIDPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMaskAlongAxisIIDPipeline.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 1, 200, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  int mask_param = 40;
  float mask_value = 1.0;
  int axis = 1;
  auto MaskAlongAxisIID = audio::MaskAlongAxisIID(mask_param, mask_value, axis);

  ds = ds->Map({MaskAlongAxisIID});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  // Now the parameter check for RandomNode would fail and we would end up with a nullptr iter.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {1, 1, 200, 200};

  int i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 4);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);

  iter->Stop();
}

/// Feature: MaskAlongAxisIID op
/// Description: Test MaskAlongAxisIID op with invalid mask_param
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMaskAlongAxisIIDInvalidMaskParam) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMaskAlongAxisIIDInvalidMaskParam.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 1, 20, 20}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  // The negative mask_param is invalid
  int mask_param = -10;
  float mask_value = 1.0;
  int axis = 2;
  auto MaskAlongAxisIID = audio::MaskAlongAxisIID(mask_param, mask_value, axis);

  ds = ds->Map({MaskAlongAxisIID});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  // Now the parameter check for RandomNode would fail and we would end up with a nullptr iter.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: MaskAlongAxisIID op
/// Description: Test MaskAlongAxisIID op with wrong axis
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMaskAlongAxisIIDInvaildAxis) {
  MS_LOG(INFO) << "MindDataTestPipeline-TestMaskAlongAxisIIDInvaildAxis.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 1, 20, 20}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  // The axis value is invilid
  int mask_param = 10;
  float mask_value = 1.0;
  int axis = 0;
  auto MaskAlongAxisIID = audio::MaskAlongAxisIID(mask_param, mask_value, axis);

  ds = ds->Map({MaskAlongAxisIID});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  // Now the parameter check for RandomNode would fail and we would end up with a nullptr iter.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: MaskAlongAxis op
/// Description: Test MaskAlongAxis op in pipeline mode
/// Expectation: The returned result is as expected
TEST_F(MindDataTestPipeline, TestMaskAlongAxisPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMaskAlongAxisPipeline.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 20, 20}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  int mask_start = 0;
  int mask_width = 10;
  float mask_value = 1.0;
  int axis = 1;
  auto MaskAlongAxis = audio::MaskAlongAxis(mask_start, mask_width, mask_value, axis);

  ds = ds->Map({MaskAlongAxis});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  // Now the parameter check for RandomNode would fail and we would end up with a nullptr iter.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  std::vector<int64_t> expected = {1, 20, 20};

  int i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 3);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);

  iter->Stop();
}

/// Feature: MaskAlongAxis op
/// Description: Test MaskAlongAxis op with invalid mask_param
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMaskAlongAxisWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMaskAlongAxisWrongArgs.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 20, 20}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  // The negative mask_param is invilid
  int mask_start = -10;
  int mask_width = 100;
  float mask_value = 1.0;
  int axis = 1;
  auto MaskAlongAxis = audio::MaskAlongAxis(mask_start, mask_width, mask_value, axis);

  ds = ds->Map({MaskAlongAxis});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  // Now the parameter check for RandomNode would fail and we would end up with a nullptr iter.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: MaskAlongAxis op
/// Description: Test MaskAlongAxis op with wrong mask_width
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMaskAlongAxisNegativeMaskWidth) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMaskAlongAxisNegativeMaskWidth.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 20, 20}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  // The negative mask_width is invalid
  int mask_start = 1;
  int mask_width = -10;
  float mask_value = 1.0;
  int axis = 1;
  auto MaskAlongAxis = audio::MaskAlongAxis(mask_start, mask_width, mask_value, axis);

  ds = ds->Map({MaskAlongAxis});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  // Now the parameter check for RandomNode would fail and we would end up with a nullptr iter.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: MaskAlongAxis op
/// Description: Test MaskAlongAxis op with wrong axis
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMaskAlongAxisInvaildAxis) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMaskAlongAxisInvaildAxis.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 20, 20}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  // The axis value is invilid
  int mask_start = 1;
  int mask_width = -10;
  float mask_value = 1.0;
  int axis = 0;
  auto MaskAlongAxis = audio::MaskAlongAxis(mask_start, mask_width, mask_value, axis);

  ds = ds->Map({MaskAlongAxis});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  // Now the parameter check for RandomNode would fail and we would end up with a nullptr iter.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: MaskAlongAxis op
/// Description: Test MaskAlongAxis op with wrong axis rank
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMaskAlongAxisRank) {
  MS_LOG(INFO) << "Doing TestMaskAlongAxis-TestMaskAlongAxisRank.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 20, 20, 20}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  // The axis value is invilid
  int mask_start = 1;
  int mask_width = 1;
  float mask_value = 1.0;
  int axis = 0;
  auto MaskAlongAxis = audio::MaskAlongAxis(mask_start, mask_width, mask_value, axis);

  ds = ds->Map({MaskAlongAxis});
  EXPECT_NE(ds, nullptr);

  // Create an iterator over the result of the above dataset
  // This will trigger the creation of the Execution Tree and launch it.
  // Now the parameter check for RandomNode would fail and we would end up with a nullptr iter.
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: Resample
/// Description: test Resample in pipeline mode
/// Expectation: the data is processed successfully
TEST_F(MindDataTestPipeline, TestResampleSincInterpolation) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResampleSincInterpolation.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("audio", mindspore::DataType::kNumberTypeFloat32, {3, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);
  auto resample_op = audio::Resample(48000, 16000);
  ds = ds->Map({resample_op});
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::vector<int64_t> expected = {3, 67};
  int i = 0;
  while (row.size() != 0) {
    auto col = row["audio"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);
  iter->Stop();
}

/// Feature: Resample
/// Description: test Resample in pipeline mode
/// Expectation: the data is processed successfully
TEST_F(MindDataTestPipeline, TestResampleKaiserWindow) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResampleKaiserWindow.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("audio", mindspore::DataType::kNumberTypeFloat32, {10, 200}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);
  auto resample_op = audio::Resample(3, 2, ResampleMethod::kKaiserWindow);
  ds = ds->Map({resample_op});
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::vector<int64_t> expected = {10, 134};
  int i = 0;
  while (row.size() != 0) {
    auto col = row["audio"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);
  iter->Stop();
}

/// Feature: Resample
/// Description: test Resample with wrong input
/// Expectation: the data is processed failed
TEST_F(MindDataTestPipeline, TestResampleWithInvalidArg) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestResampleInvalidArgs.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("audio", mindspore::DataType::kNumberTypeFloat32, {1, 200}));
  std::shared_ptr<Dataset> ds_01 = RandomData(50, schema);
  EXPECT_NE(ds_01, nullptr);

  ds_01 = ds_01->SetNumWorkers(4);
  EXPECT_NE(ds_01, nullptr);

  auto resample_01 = audio::Resample(-2, 3, ResampleMethod::kKaiserWindow);

  ds_01 = ds_01->Map({resample_01});
  EXPECT_NE(ds_01, nullptr);

  std::shared_ptr<Iterator> iter_01 = ds_01->CreateIterator();
  // Expect failure, orig_freq can not be negative.
  EXPECT_EQ(iter_01, nullptr);

  std::shared_ptr<Dataset> ds_02 = RandomData(50, schema);
  EXPECT_NE(ds_02, nullptr);

  ds_02 = ds_02->SetNumWorkers(4);
  EXPECT_NE(ds_02, nullptr);

  auto resample_02 = audio::Resample(2, -3, ResampleMethod::kSincInterpolation);

  ds_02 = ds_02->Map({resample_02});
  EXPECT_NE(ds_02, nullptr);

  std::shared_ptr<Iterator> iter_02 = ds_02->CreateIterator();
  // Expect failure, new_freq can not be negative.
  EXPECT_EQ(iter_02, nullptr);
}

/// Feature: LFCC op
/// Description: Test pipeline for LFCC op
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestLFCCPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLFCCPipeline.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 1, 300}));
  std::shared_ptr<Dataset> ds = RandomData(10, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);
  auto lfcc_op1 = audio::LFCC(16000, 128, 40, 0.0, 10000.0, 2, NormMode::kOrtho, true);
  ds = ds->Map({lfcc_op1});
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::vector<int64_t> expected = {1, 1, 40, 2};
  int i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 4);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 10);
  iter->Stop();
}

/// Feature: LFCC op
/// Description: Test wrong arguments for LFCC op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestLFCCWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLFCCWrongArgs.";
  // LFCC: negative sample_rate.
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 1, 300}));
  std::shared_ptr<Dataset> ds = RandomData(10, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);
  auto lfcc_op0 = audio::LFCC(-1);
  ds = ds->Map({lfcc_op0});
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: MFCC op
/// Description: Test pipeline for MFCC op
/// Expectation: Generate expected output after cases were executed
TEST_F(MindDataTestPipeline, TestMFCCPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMFCCPipeline.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 1, 300}));
  std::shared_ptr<Dataset> ds = RandomData(10, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);
  auto mfcc_op1 = audio::MFCC(16000, 40, 2, NormMode::kOrtho, true);
  ds = ds->Map({mfcc_op1});
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::vector<int64_t> expected = {1, 1, 40, 2};
  int i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 4);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 10);
  iter->Stop();
}

/// Feature: MFCC op
/// Description: Test wrong arguments for MFCC op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestMFCCWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestInverseMelScaleWrongArgs.";
  // MFCC: negative sample_rate.
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 1, 300}));
  std::shared_ptr<Dataset> ds = RandomData(10, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);
  auto mfcc_op0 = audio::MFCC(-1);
  ds = ds->Map({mfcc_op0});
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}

/// Feature: MelSpectrogram op
/// Description: Test input MelSpectrogram op
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestMelSpectrogramPipeline) {
  // 3 dimension
  std::shared_ptr<SchemaObj> schema3 = Schema();
  ASSERT_OK(schema3->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 1, 30}));
  std::shared_ptr<Dataset> ds3 = RandomData(10, schema3);
  EXPECT_NE(ds3, nullptr);

  ds3 = ds3->SetNumWorkers(4);
  EXPECT_NE(ds3, nullptr);

  auto mel_spectrogram_op3 = audio::MelSpectrogram(16000, 16, 16, 8, 0.0, 10000.0, 0, 8, WindowType::kHann, 2.0, false,
                                                   true, BorderType::kReflect, true, NormType::kNone, MelType::kHtk);
  ds3 = ds3->Map({mel_spectrogram_op3});
  EXPECT_NE(ds3, nullptr);
  std::shared_ptr<Iterator> iter3 = ds3->CreateIterator();
  EXPECT_NE(ds3, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row3;
  ASSERT_OK(iter3->GetNextRow(&row3));
  std::vector<int64_t> expected3 = {1, 1, 8, 4};
  int i = 0;
  while (row3.size() != 0) {
    auto col = row3["waveform"];
    ASSERT_EQ(col.Shape(), expected3);
    ASSERT_EQ(col.Shape().size(), 4);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter3->GetNextRow(&row3));
    i++;
  }
  EXPECT_EQ(i, 10);
  iter3->Stop();
}

/// Feature: InverseSpectrogram op
/// Description: Test input InverseSpectrogram op
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestInverseSpectrogramPipeline) {
  // 4 dimension
  std::shared_ptr<SchemaObj> schema4 = Schema();
  ASSERT_OK(schema4->add_column("spectrogram", mindspore::DataType::kNumberTypeFloat32, {1, 9, 10, 2}));
  std::shared_ptr<Dataset> ds4 = RandomData(10, schema4);
  EXPECT_NE(ds4, nullptr);

  ds4 = ds4->SetNumWorkers(4);
  EXPECT_NE(ds4, nullptr);

  auto inverse_spectrogram_op4 =
    audio::InverseSpectrogram(1, 16, 16, 8, 0, WindowType::kHann, false, true, BorderType::kReflect, true);
  ds4 = ds4->Map({inverse_spectrogram_op4});
  EXPECT_NE(ds4, nullptr);
  std::shared_ptr<Iterator> iter4 = ds4->CreateIterator();
  EXPECT_NE(ds4, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row4;
  ASSERT_OK(iter4->GetNextRow(&row4));
  std::vector<int64_t> expected4 = {1, 1};
  int i = 0;
  while (row4.size() != 0) {
    auto col = row4["spectrogram"];
    ASSERT_EQ(col.Shape(), expected4);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter4->GetNextRow(&row4));
    i++;
  }
  EXPECT_EQ(i, 10);
  iter4->Stop();
}

/// Feature: PitchShift op
/// Description: Test input 1d of PitchShift op
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestPitchShiftPipeline1D) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPitchShiftPipeline.";
  // 1 dimension
  std::shared_ptr<SchemaObj> schema1 = Schema();
  ASSERT_OK(schema1->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {30}));
  std::shared_ptr<Dataset> ds1 = RandomData(10, schema1);
  EXPECT_NE(ds1, nullptr);

  ds1 = ds1->SetNumWorkers(4);
  EXPECT_NE(ds1, nullptr);

  auto PitchShift_op1 = audio::PitchShift(16000, 4, 12, 16, 16, 4, WindowType::kHann);
  ds1 = ds1->Map({PitchShift_op1});
  EXPECT_NE(ds1, nullptr);
  std::shared_ptr<Iterator> iter1 = ds1->CreateIterator();
  EXPECT_NE(ds1, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row1;
  ASSERT_OK(iter1->GetNextRow(&row1));
  std::vector<int64_t> expected1 = {30};
  int i = 0;
  while (row1.size() != 0) {
    auto col = row1["waveform"];
    ASSERT_EQ(col.Shape(), expected1);
    ASSERT_EQ(col.Shape().size(), 1);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter1->GetNextRow(&row1));
    i++;
  }
  EXPECT_EQ(i, 10);
  iter1->Stop();
}

/// Feature: PitchShift op
/// Description: Test PitchShift op with wrong input
/// Expectation: Throw exception as expected
TEST_F(MindDataTestPipeline, TestPitchShiftWrongArgsHopLength) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestPitchShiftWrongArgs.";
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 1, 30}));
  ;
  std::shared_ptr<Dataset> ds = RandomData(10, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);
  auto PitchShift_op = audio::PitchShift(16000, 4, 12, 16, 16, -4, WindowType::kHann);
  ds = ds->Map({PitchShift_op});
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_EQ(iter, nullptr);
}
