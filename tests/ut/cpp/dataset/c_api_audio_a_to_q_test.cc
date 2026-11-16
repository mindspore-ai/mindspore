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
