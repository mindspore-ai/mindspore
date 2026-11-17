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
