/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/skip_first_epoch_sampler.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestSkipFirstEpochSampler : public UT::Common {
 public:
  class DummyRandomAccessOp : public RandomAccessOp {
   public:
    explicit DummyRandomAccessOp(uint64_t num_rows) {
      // row count is in base class as protected member
      // GetNumRowsInDataset does not need an override, the default from base class is fine.
      num_rows_ = num_rows;
    }
  };
};

/// Feature: MindData SkipFirstEpochSampler Support
/// Description: Test MindData SkipFirstEpochSampler Reset with Replacement
/// Expectation: Results are successfully outputted.
TEST_F(MindDataTestSkipFirstEpochSampler, TestResetReplacement) {
  MS_LOG(INFO) << "Doing MindDataTestSkipFirstEpochSampler-TestResetReplacement.";
  uint64_t total_samples = 1000000;

  // create sampler with replacement = true
  SkipFirstEpochSamplerRT m_sampler(0, 0);
  DummyRandomAccessOp dummyRandomAccessOp(total_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());

  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
    }
  }

  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());
  ASSERT_EQ(row.eoe(), true);

  m_sampler.ResetSampler();
  out.clear();

  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());

  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
    }
  }
  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());
  ASSERT_EQ(row.eoe(), true);
}

/// Feature: MindData SkipFirstEpochSampler Support
/// Description: Test MindData SkipFirstEpochSampler Reset without Replacement
/// Expectation: Results are successfully outputted.
TEST_F(MindDataTestSkipFirstEpochSampler, TestResetNoReplacement) {
  MS_LOG(INFO) << "Doing MindDataTestSkipFirstEpochSampler-TestResetNoReplacement.";
  // num samples to draw.
  uint64_t num_samples = 1000000;
  uint64_t total_samples = 1000000;
  std::vector<uint64_t> freq(total_samples, 0);

  // create sampler without replacement
  SkipFirstEpochSamplerRT m_sampler(0, 0);
  DummyRandomAccessOp dummyRandomAccessOp(total_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());

  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
      freq[*it]++;
    }
  }
  ASSERT_EQ(num_samples, out.size());

  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());
  ASSERT_EQ(row.eoe(), true);

  m_sampler.ResetSampler();
  out.clear();
  freq.clear();
  freq.resize(total_samples, 0);
  MS_LOG(INFO) << "Resetting sampler";

  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());

  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
      freq[*it]++;
    }
  }
  ASSERT_EQ(num_samples, out.size());

  // Without replacement, each sample only drawn once.
  for (int i = 0; i < total_samples; i++) {
    if (freq[i]) {
      ASSERT_EQ(freq[i], 1);
    }
  }

  ASSERT_EQ(m_sampler.GetNextSample(&row), Status::OK());
  ASSERT_EQ(row.eoe(), true);
}
