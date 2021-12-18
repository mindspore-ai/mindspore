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
#include "include/api/types.h"
#include "utils/log_adapter.h"

#include "minddata/dataset/include/dataset/audio.h"
#include "minddata/dataset/include/dataset/datasets.h"
#include "minddata/dataset/include/dataset/transforms.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;
using namespace std;

class MindDataTestPipeline : public UT::DatasetOpTesting {
 protected:
};

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

TEST_F(MindDataTestPipeline, TestEqualizerBiquadSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestEqualizerBiquadSuccess.";

  // Create an input tensor
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeFloat32, {1, 200}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  // Create a filter object
  std::shared_ptr<TensorTransform> equalizer_biquad(new audio::EqualizerBiquad(44100, 3.5, 5.5, 0.707));
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

TEST_F(MindDataTestPipeline, TestLowpassBiquadSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestLowpassBiquadSuccess.";

  // Create an input tensor
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeFloat32, {1, 200}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  // Create a filter object
  std::shared_ptr<TensorTransform> lowpass_biquad(new audio::LowpassBiquad(44100, 3000.5, 0.707));
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

/// Feature: Dither
/// Description: test basic usage of Dither in pipeline mode
/// Expectation: the data is processed successfully
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

TEST_F(MindDataTestPipeline, TestHighpassBiquadSuccess) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestHighpassBiquadSuccess.";

  // Create an input tensor
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeFloat32, {1, 200}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  // Create a filter object
  std::shared_ptr<TensorTransform> highpass_biquad(new audio::HighpassBiquad(44100, 3000.5, 0.707));
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

/// Feature: MuLawEncoding
/// Description: test MuLawEncoding in pipeline mode
/// Expectation: the data is processed successfully
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

/// Feature: MuLawEncoding
/// Description: test invalid parameter of MuLawEncoding
/// Expectation: throw exception correctly
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

/// Feature: Overdrive
/// Description: test basic usage of Overdrive
/// Expectation: get correct number of data
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

/// Feature: Overdrive
/// Description: test invalid parameter of Overdrive
/// Expectation: throw exception correctly
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

/// Feature: Phaser
/// Description: test basic usage of Phaser
/// Expectation: get correct number of data
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

/// Feature: Phaser
/// Description: test invalid parameter of Phaser
/// Expectation: throw exception correctly
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

TEST_F(MindDataTestPipeline, TestMagphase) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMagphase.";

  float power = 2.0;
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("col1", mindspore::DataType::kNumberTypeFloat32, {1, 2}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<TensorTransform> magphase(new audio::Magphase(power));
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

TEST_F(MindDataTestPipeline, TestMagphaseWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestMagphaseWrongArgs.";

  float power_wrong = -1.0;
  std::shared_ptr<TensorTransform> magphase(new audio::Magphase(power_wrong));
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

/// Feature: CreateDct
/// Description: test CreateDct in eager mode
/// Expectation: the returned result is as expected
TEST_F(MindDataTestPipeline, TestCreateDctNone) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCreateDctNone.";
  mindspore::MSTensor output;
  Status s01 = audio::CreateDct(&output, 200, 400, NormMode::kNone);
  EXPECT_TRUE(s01.IsOk());
}

/// Feature: CreateDct
/// Description: test CreateDct in eager mode
/// Expectation: the returned result is as expected
TEST_F(MindDataTestPipeline, TestCreateDctOrtho) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestCreateDctOrtho.";
  mindspore::MSTensor output;
  Status s02 = audio::CreateDct(&output, 200, 400, NormMode::kOrtho);
  EXPECT_TRUE(s02.IsOk());
}

/// Feature: CreateDct
/// Description: test WrongArg of CreateDct
/// Expectation: return error
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

/// Feature: DBToAmplitude
/// Description: test DBToAmplitude in pipeline mode
/// Expectation: the data is processed successfully
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

/// Feature: ComputeDeltas
/// Description: test basic function of ComputeDeltas
/// Expectation: get correct number of data
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

/// Feature: ComputeDeltas
/// Description: test wrong input args of ComputeDeltas
/// Expectation: get nullptr of iterator
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

/// Feature: Gain
/// Description: test Gain in pipeline mode
/// Expectation: the data is processed successfully
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