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

/// Feature: SlidingWindowCmn op
/// Description: Test basic function of SlidingWindowCmn op
/// Expectation: Get correct number of data
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
  while (!row.empty()) {
    i++;
    ASSERT_OK(iter->GetNextRow(&row));
  }
  EXPECT_EQ(i, 8);
  iter->Stop();
}

/// Feature: SlidingWindowCmn op
/// Description: Test wrong input args of SlidingWindowCmn op
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
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

/// Feature: SpectralCentroid op
/// Description: Test SpectralCentroid op basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSpectralCentroidBasic) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpectralCentroidBasic.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 60}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  auto spectral_centroid = audio::SpectralCentroid(44100, 8, 8, 4, 1, WindowType::kHann);

  auto ds1 = ds->Map({spectral_centroid}, {"waveform"});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (!row.empty()) {
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 8);
  iter->Stop();
}

/// Feature: SpectralCentroid op
/// Description: Test SpectralCentroid op with default inputs
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSpectralCentroidDefault) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpectralCentroidDefault.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 60}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  auto spectral_centroid = audio::SpectralCentroid(44100);

  auto ds1 = ds->Map({spectral_centroid}, {"waveform"});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (!row.empty()) {
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 8);
  iter->Stop();
}

/// Feature: SpectralCentroid op
/// Description: Test SpectralCentroid op with invalid arguments
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestSpectralCentroidWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpectralCentroidWrongArgs.";

  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("col", mindspore::DataType::kNumberTypeFloat32, {1, 50}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  std::shared_ptr<Dataset> ds02;
  std::shared_ptr<Dataset> ds03;
  std::shared_ptr<Dataset> ds04;
  std::shared_ptr<Dataset> ds05;
  EXPECT_NE(ds, nullptr);

  // Check n_fft
  MS_LOG(INFO) << "n_fft is zero.";
  auto spectral_centroid_op_1 = audio::SpectralCentroid(44100, 0, 8, 4, 1, WindowType::kHann);
  ds01 = ds->Map({spectral_centroid_op_1});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);

  // Check win_length
  MS_LOG(INFO) << "win_length is -1.";
  auto spectral_centroid_op_2 = audio::SpectralCentroid(44100, 8, -1, 4, 1, WindowType::kHann);
  ds02 = ds->Map({spectral_centroid_op_2});
  EXPECT_NE(ds02, nullptr);

  std::shared_ptr<Iterator> iter02 = ds02->CreateIterator();
  EXPECT_EQ(iter02, nullptr);

  // Check hop_length
  MS_LOG(INFO) << "hop_length is -1.";
  auto spectral_centroid_op_3 = audio::SpectralCentroid(44100, 8, 8, -1, 1, WindowType::kHann);
  ds03 = ds->Map({spectral_centroid_op_3});
  EXPECT_NE(ds03, nullptr);

  std::shared_ptr<Iterator> iter03 = ds03->CreateIterator();
  EXPECT_EQ(iter03, nullptr);

  // Check pad
  MS_LOG(INFO) << "pad is -1.";
  auto spectral_centroid_op_4 = audio::SpectralCentroid(44100, 8, 8, 4, -1, WindowType::kHann);
  ds04 = ds->Map({spectral_centroid_op_4});
  EXPECT_NE(ds04, nullptr);

  std::shared_ptr<Iterator> iter04 = ds04->CreateIterator();
  EXPECT_EQ(iter04, nullptr);

  // Check sample_rate
  MS_LOG(INFO) << "sample_rate is -1.";
  auto spectral_centroid_op_5 = audio::SpectralCentroid(-1, 8, 8, 4, 8, WindowType::kHann);
  ds05 = ds->Map({spectral_centroid_op_5});
  EXPECT_NE(ds05, nullptr);

  std::shared_ptr<Iterator> iter05 = ds04->CreateIterator();
  EXPECT_EQ(iter05, nullptr);
}

/// Feature: Spectrogram op
/// Description: Test Spectrogram op with default inputs
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSpectrogramDefault) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpectrogramDefault.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 60}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  auto spectrogram =
    audio::Spectrogram(40, 40, 20, 0, WindowType::kHann, 2.0, false, true, BorderType::kReflect, true);

  auto ds1 = ds->Map({spectrogram}, {"waveform"});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (!row.empty()) {
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 8);
  iter->Stop();
}

/// Feature: Spectrogram op
/// Description: Test Spectrogram op with onesided parameter to be false
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSpectrogramOnesidedFalse) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpectrogramOnesidedFalse.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {3, 50}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  auto spectrogram =
    audio::Spectrogram(40, 40, 20, 0, WindowType::kHann, 2.0, false, true, BorderType::kReflect, false);

  auto ds1 = ds->Map({spectrogram}, {"waveform"});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (!row.empty()) {
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 8);
  iter->Stop();
}

/// Feature: Spectrogram op
/// Description: Test Spectrogram op with center parameter to be false
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSpectrogramCenterFalse) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpectrogramCenterFalse.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeInt32, {2, 3, 50}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  auto spectrogram =
    audio::Spectrogram(40, 40, 20, 0, WindowType::kHann, 2.0, false, false, BorderType::kReflect, true);

  auto ds1 = ds->Map({spectrogram}, {"waveform"});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (!row.empty()) {
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 8);
  iter->Stop();
}

/// Feature: Spectrogram op
/// Description: Test Spectrogram op with normalized parameter to be true
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSpectrogramNormalizedTrue) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpectrogramNormalizedTrue.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeInt32, {5, 40}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  auto spectrogram =
    audio::Spectrogram(40, 40, 20, 0, WindowType::kHann, 2.0, true, true, BorderType::kReflect, true);

  auto ds1 = ds->Map({spectrogram}, {"waveform"});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (!row.empty()) {
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 8);
  iter->Stop();
}

/// Feature: Spectrogram op
/// Description: Test Spectrogram op with WindowType::kHamming
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSpectrogramWindowHamming) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpectrogramWindowHamming.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat64, {3, 50}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  auto spectrogram =
    audio::Spectrogram(40, 40, 20, 0, WindowType::kHamming, 2.0, false, true, BorderType::kReflect, true);

  auto ds1 = ds->Map({spectrogram}, {"waveform"});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (!row.empty()) {
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 8);
  iter->Stop();
}

/// Feature: Spectrogram op
/// Description: Test Spectrogram op with BorderType::kEdge
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSpectrogramPadmodeEdge) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpectrogramPadmodeEdge.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeInt32, {3, 4, 50}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  auto spectrogram =
    audio::Spectrogram(40, 40, 20, 0, WindowType::kHamming, 2.0, false, true, BorderType::kEdge, true);

  auto ds1 = ds->Map({spectrogram}, {"waveform"});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (!row.empty()) {
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 8);
  iter->Stop();
}

/// Feature: Spectrogram op
/// Description: Test Spectrogram op with power parameter to be 0
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSpectrogramPower0) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpectrogramPower0.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeInt32, {3, 50}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  auto spectrogram =
    audio::Spectrogram(40, 40, 20, 0, WindowType::kHamming, 0, false, true, BorderType::kReflect, true);

  auto ds1 = ds->Map({spectrogram}, {"waveform"});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (!row.empty()) {
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 8);
  iter->Stop();
}

/// Feature: Spectrogram op
/// Description: Test Spectrogram op with n_fft parameter to be 50
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSpectrogramNfft50) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpectrogramNfft600.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {1, 60}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  auto spectrogram =
    audio::Spectrogram(50, 40, 20, 0, WindowType::kHann, 2.0, false, true, BorderType::kReflect, true);

  auto ds1 = ds->Map({spectrogram}, {"waveform"});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (!row.empty()) {
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 8);
  iter->Stop();
}

/// Feature: Spectrogram op
/// Description: Test Spectrogram op with pad parameter to be 10
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSpectrogramPad10) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpectrogramPad50.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {3, 50}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  auto spectrogram =
    audio::Spectrogram(40, 40, 20, 10, WindowType::kHann, 2.0, false, true, BorderType::kReflect, true);

  auto ds1 = ds->Map({spectrogram}, {"waveform"});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (!row.empty()) {
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 8);
  iter->Stop();
}

/// Feature: Spectrogram op
/// Description: Test Spectrogram op with win_length parameter to be 30
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSpectrogramWinlength30) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpectrogramWinlength300.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 2, 50}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  auto spectrogram =
    audio::Spectrogram(40, 30, 20, 0, WindowType::kHann, 2.0, false, true, BorderType::kReflect, true);

  auto ds1 = ds->Map({spectrogram}, {"waveform"});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (!row.empty()) {
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 8);
  iter->Stop();
}

/// Feature: Spectrogram op
/// Description: Test Spectrogram op with hop_length parameter to be 30
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestSpectrogramHoplength30) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpectrogramHoplength300.";

  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {2, 50}));
  std::shared_ptr<Dataset> ds = RandomData(8, schema);
  EXPECT_NE(ds, nullptr);

  auto spectrogram =
    audio::Spectrogram(40, 40, 30, 0, WindowType::kHann, 2.0, false, true, BorderType::kReflect, true);

  auto ds1 = ds->Map({spectrogram}, {"waveform"});
  EXPECT_NE(ds1, nullptr);

  std::shared_ptr<Iterator> iter = ds1->CreateIterator();
  EXPECT_NE(iter, nullptr);

  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));

  uint64_t i = 0;
  while (!row.empty()) {
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 8);
  iter->Stop();
}

/// Feature: Spectrogram op
/// Description: Test Spectrogram op with invalid parameters
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestSpectrogramWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestSpectrogramWrongArgs.";

  std::shared_ptr<SchemaObj> schema = Schema();
  // Original waveform
  ASSERT_OK(schema->add_column("col", mindspore::DataType::kNumberTypeFloat32, {1, 50}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  std::shared_ptr<Dataset> ds01;
  std::shared_ptr<Dataset> ds02;
  std::shared_ptr<Dataset> ds03;
  std::shared_ptr<Dataset> ds04;
  std::shared_ptr<Dataset> ds05;
  std::shared_ptr<Dataset> ds06;
  EXPECT_NE(ds, nullptr);

  // Check n_fft
  MS_LOG(INFO) << "n_fft is zero.";
  auto spectrogram_op_01 =
    audio::Spectrogram(0, 40, 20, 0, WindowType::kHann, 2.0, false, true, BorderType::kReflect, true);
  ds01 = ds->Map({spectrogram_op_01});
  EXPECT_NE(ds01, nullptr);

  std::shared_ptr<Iterator> iter01 = ds01->CreateIterator();
  EXPECT_EQ(iter01, nullptr);

  // Check win_length
  MS_LOG(INFO) << "win_length is -1.";
  auto spectrogram_op_02 =
    audio::Spectrogram(40, -1, 20, 0, WindowType::kHann, 2.0, false, true, BorderType::kReflect, true);
  ds02 = ds->Map({spectrogram_op_02});
  EXPECT_NE(ds02, nullptr);

  std::shared_ptr<Iterator> iter02 = ds02->CreateIterator();
  EXPECT_EQ(iter02, nullptr);

  // Check hop_length
  MS_LOG(INFO) << "hop_length is -1.";
  auto spectrogram_op_03 =
    audio::Spectrogram(40, 40, -1, 0, WindowType::kHann, 2.0, false, true, BorderType::kReflect, true);
  ds03 = ds->Map({spectrogram_op_03});
  EXPECT_NE(ds03, nullptr);

  std::shared_ptr<Iterator> iter03 = ds03->CreateIterator();
  EXPECT_EQ(iter03, nullptr);

  // Check power
  MS_LOG(INFO) << "power is -1.";
  auto spectrogram_op_04 =
    audio::Spectrogram(40, 40, 20, 0, WindowType::kHann, -1, false, true, BorderType::kReflect, true);
  ds04 = ds->Map({spectrogram_op_04});
  EXPECT_NE(ds04, nullptr);

  std::shared_ptr<Iterator> iter04 = ds04->CreateIterator();
  EXPECT_EQ(iter04, nullptr);

  // Check pad
  MS_LOG(INFO) << "pad is -1.";
  auto spectrogram_op_05 =
    audio::Spectrogram(40, 40, 20, -1, WindowType::kHann, 2.0, false, true, BorderType::kReflect, true);
  ds05 = ds->Map({spectrogram_op_05});
  EXPECT_NE(ds05, nullptr);

  std::shared_ptr<Iterator> iter05 = ds05->CreateIterator();
  EXPECT_EQ(iter05, nullptr);

  // Check n_fft and win)length
  MS_LOG(INFO) << "n_fft is 40, win_length is 50.";
  auto spectrogram_op_06 =
    audio::Spectrogram(40, 50, 20, -1, WindowType::kHann, 2.0, false, true, BorderType::kReflect, true);
  ds06 = ds->Map({spectrogram_op_06});
  EXPECT_NE(ds06, nullptr);

  std::shared_ptr<Iterator> iter06 = ds06->CreateIterator();
  EXPECT_EQ(iter06, nullptr);
}

/// Feature: TimeMasking op
/// Description: Test TimeMasking op with pipeline mode
/// Expectation: Output is equal to the expected output
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

/// Feature: TimeMasking op
/// Description: Test TimeMasking op with wrong arguments
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
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

/// Feature: TimeStretch op
/// Description: Test TimeStretch op with pipeline mode
/// Expectation: Output is equal to the expected output
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
  while (!row.empty()) {
    auto col = row["inputData"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);

  iter->Stop();
}

/// Feature: TimeStretch op
/// Description: Test TimeStretch op with pipeline mode with wrong arguments
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
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

/// Feature: GriffinLim op
/// Description: Test GriffinLim op with pipeline mode
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestPipeline, TestGriffinLimPipeline) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGriffinLimPipeline.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {201, 6}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);
  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);
  auto griffin_lim = audio::GriffinLim(400, 32, 0, 0, WindowType::kHann, 2.0, 0.99, 0, true);
  ds = ds->Map({griffin_lim});
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  EXPECT_NE(ds, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row;
  ASSERT_OK(iter->GetNextRow(&row));
  std::vector<int64_t> expected = {1000};
  int i = 0;
  while (row.size() != 0) {
    auto col = row["waveform"];
    ASSERT_EQ(col.Shape(), expected);
    ASSERT_EQ(col.Shape().size(), 1);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat32);
    ASSERT_OK(iter->GetNextRow(&row));
    i++;
  }
  EXPECT_EQ(i, 50);
  iter->Stop();

  // Original waveform
  std::shared_ptr<SchemaObj> schema2 = Schema();
  ASSERT_OK(schema2->add_column("waveform", mindspore::DataType::kNumberTypeFloat64, {2, 301, 6}));
  std::shared_ptr<Dataset> ds2 = RandomData(50, schema2);
  EXPECT_NE(ds2, nullptr);
  ds2 = ds2->SetNumWorkers(4);
  EXPECT_NE(ds2, nullptr);
  auto griffin_lim2 = audio::GriffinLim(600, 10, 0, 300, WindowType::kHann, 1.0, 0.99, 0, false);
  ds2 = ds2->Map({griffin_lim2});
  EXPECT_NE(ds2, nullptr);
  std::shared_ptr<Iterator> iter2 = ds2->CreateIterator();
  EXPECT_NE(ds2, nullptr);
  std::unordered_map<std::string, mindspore::MSTensor> row2;
  ASSERT_OK(iter2->GetNextRow(&row2));
  std::vector<int64_t> expected2 = {2, 1500};
  i = 0;
  while (row2.size() != 0) {
    auto col = row2["waveform"];
    ASSERT_EQ(col.Shape(), expected2);
    ASSERT_EQ(col.Shape().size(), 2);
    ASSERT_EQ(col.DataType(), mindspore::DataType::kNumberTypeFloat64);
    ASSERT_OK(iter2->GetNextRow(&row2));
    i++;
  }
  EXPECT_EQ(i, 50);
  iter->Stop();
}

/// Feature: GriffinLim op
/// Description: Test GriffinLim op with invalid parameters
/// Expectation: Error message is logged, and CreateIterator() for invalid pipeline returns nullptr
TEST_F(MindDataTestPipeline, TestGriffinLimWrongArgs) {
  MS_LOG(INFO) << "Doing MindDataTestPipeline-TestGriffinLimWrongArgs.";
  // Original waveform
  std::shared_ptr<SchemaObj> schema = Schema();
  ASSERT_OK(schema->add_column("waveform", mindspore::DataType::kNumberTypeFloat32, {201, 6}));
  std::shared_ptr<Dataset> ds = RandomData(50, schema);
  EXPECT_NE(ds, nullptr);

  ds = ds->SetNumWorkers(4);
  EXPECT_NE(ds, nullptr);

  // invalid n_fft
  auto griffin_lim_op = audio::GriffinLim(-10);
  ds = ds->Map({griffin_lim_op});
  EXPECT_NE(ds, nullptr);
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid n_iter
  griffin_lim_op = audio::GriffinLim(400, -10);
  ds = ds->Map({griffin_lim_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid win_length
  griffin_lim_op = audio::GriffinLim(400, 10, -2);
  ds = ds->Map({griffin_lim_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);
  griffin_lim_op = audio::GriffinLim(400, 10, 500);
  ds = ds->Map({griffin_lim_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid hop_length
  griffin_lim_op = audio::GriffinLim(400, 10, 0, -10);
  ds = ds->Map({griffin_lim_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid power
  griffin_lim_op = audio::GriffinLim(400, 10, 0, 0, WindowType::kHann, -2);
  ds = ds->Map({griffin_lim_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid momentum
  griffin_lim_op = audio::GriffinLim(400, 10, 0, 0, WindowType::kHann, 2, -10);
  ds = ds->Map({griffin_lim_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);

  // invalid length
  griffin_lim_op = audio::GriffinLim(400, 10, 0, 0, WindowType::kHann, 2, 0.9, -3);
  ds = ds->Map({griffin_lim_op});
  EXPECT_NE(ds, nullptr);
  iter = ds->CreateIterator();
  // Expect failure
  EXPECT_EQ(iter, nullptr);
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
