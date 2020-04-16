/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "gtest/gtest.h"

#include "dataset/core/constants.h"
#include "dataset/core/tensor.h"
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/datasetops/source/sampler/sampler.h"
#include "dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"
#include "utils/log_adapter.h"

#include <vector>
#include <unordered_set>

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestWeightedRandomSampler : public UT::Common {
 public:
  class DummyRandomAccessOp : public RandomAccessOp {
   public:
    DummyRandomAccessOp(uint64_t num_rows) : num_rows_(num_rows) {};
    Status GetNumSamples(int64_t *num) const {
      *num = num_rows_;
      return Status::OK();
    }

    Status GetNumRowsInDataset(int64_t *num) const {
      *num = num_rows_;
      return Status::OK();
    }

   private:
    uint64_t num_rows_;
  };
};

TEST_F(MindDataTestWeightedRandomSampler, TestOneshotReplacement) {
  // num samples to draw.
  uint64_t num_samples = 100;
  uint64_t total_samples = 1000;
  std::vector<double> weights(total_samples, std::rand() % 100);
  std::vector<uint64_t> freq(total_samples, 0);

  // create sampler with replacement = true
  WeightedRandomSampler m_sampler(weights, num_samples, true);
  DummyRandomAccessOp dummyRandomAccessOp(total_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  std::unique_ptr<DataBuffer> db;
  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextBuffer(&db), Status::OK());
  db->PopRow(&row);
  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
      freq[*it]++;
    }
  }

  ASSERT_EQ(num_samples, out.size());

  ASSERT_EQ(m_sampler.GetNextBuffer(&db), Status::OK());
  ASSERT_EQ(db->eoe(), true);
}

TEST_F(MindDataTestWeightedRandomSampler, TestOneshotNoReplacement) {
  // num samples to draw.
  uint64_t num_samples = 100;
  uint64_t total_samples = 1000;
  std::vector<double> weights(total_samples, std::rand() % 100);
  std::vector<uint64_t> freq(total_samples, 0);

  // create sampler with replacement = replacement
  WeightedRandomSampler m_sampler(weights, num_samples, false);
  DummyRandomAccessOp dummyRandomAccessOp(total_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  std::unique_ptr<DataBuffer> db;
  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextBuffer(&db), Status::OK());
  db->PopRow(&row);
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

  ASSERT_EQ(m_sampler.GetNextBuffer(&db), Status::OK());
  ASSERT_EQ(db->eoe(), true);
}

TEST_F(MindDataTestWeightedRandomSampler, TestGetNextBufferReplacement) {
  // num samples to draw.
  uint64_t num_samples = 100;
  uint64_t total_samples = 1000;
  uint64_t samples_per_buffer = 10;
  std::vector<double> weights(total_samples, std::rand() % 100);

  // create sampler with replacement = replacement
  WeightedRandomSampler m_sampler(weights, num_samples, true, samples_per_buffer);
  DummyRandomAccessOp dummyRandomAccessOp(total_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  std::unique_ptr<DataBuffer> db;
  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextBuffer(&db), Status::OK());
  int epoch = 0;
  while (!db->eoe()) {
    epoch++;
    db->PopRow(&row);
    for (const auto &t : row) {
      for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
        out.push_back(*it);
      }
    }
    db.reset();
    ASSERT_EQ(m_sampler.GetNextBuffer(&db), Status::OK());
  }

  ASSERT_EQ(epoch, (num_samples + samples_per_buffer - 1) / samples_per_buffer);
  ASSERT_EQ(num_samples, out.size());
}

TEST_F(MindDataTestWeightedRandomSampler, TestGetNextBufferNoReplacement) {
  // num samples to draw.
  uint64_t num_samples = 100;
  uint64_t total_samples = 100;
  uint64_t samples_per_buffer = 10;
  std::vector<double> weights(total_samples, std::rand() % 100);
  weights[1] = 0;
  weights[2] = 0;
  std::vector<uint64_t> freq(total_samples, 0);

  // create sampler with replacement = replacement
  WeightedRandomSampler m_sampler(weights, num_samples, false, samples_per_buffer);
  DummyRandomAccessOp dummyRandomAccessOp(total_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  std::unique_ptr<DataBuffer> db;
  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextBuffer(&db), Status::OK());
  int epoch = 0;
  while (!db->eoe()) {
    epoch++;
    db->PopRow(&row);
    for (const auto &t : row) {
      for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
        out.push_back(*it);
        freq[*it]++;
      }
    }
    db.reset();
    ASSERT_EQ(m_sampler.GetNextBuffer(&db), Status::OK());
  }

  // Without replacement, each sample only drawn once.
  for (int i = 0; i < total_samples; i++) {
    if (freq[i]) {
      ASSERT_EQ(freq[i], 1);
    }
  }

  ASSERT_EQ(epoch, (num_samples + samples_per_buffer - 1) / samples_per_buffer);
  ASSERT_EQ(num_samples, out.size());
}

TEST_F(MindDataTestWeightedRandomSampler, TestResetReplacement) {
  // num samples to draw.
  uint64_t num_samples = 1000000;
  uint64_t total_samples = 1000000;
  std::vector<double> weights(total_samples, std::rand() % 100);
  std::vector<uint64_t> freq(total_samples, 0);

  // create sampler with replacement = true
  WeightedRandomSampler m_sampler(weights, num_samples, true);
  DummyRandomAccessOp dummyRandomAccessOp(total_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  std::unique_ptr<DataBuffer> db;
  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextBuffer(&db), Status::OK());
  db->PopRow(&row);
  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
      freq[*it]++;
    }
  }
  ASSERT_EQ(num_samples, out.size());

  ASSERT_EQ(m_sampler.GetNextBuffer(&db), Status::OK());
  ASSERT_EQ(db->eoe(), true);

  m_sampler.Reset();
  out.clear();

  ASSERT_EQ(m_sampler.GetNextBuffer(&db), Status::OK());
  db->PopRow(&row);
  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
      freq[*it]++;
    }
  }
  ASSERT_EQ(num_samples, out.size());

  ASSERT_EQ(m_sampler.GetNextBuffer(&db), Status::OK());
  ASSERT_EQ(db->eoe(), true);
}

TEST_F(MindDataTestWeightedRandomSampler, TestResetNoReplacement) {
  // num samples to draw.
  uint64_t num_samples = 1000000;
  uint64_t total_samples = 1000000;
  std::vector<double> weights(total_samples, std::rand() % 100);
  std::vector<uint64_t> freq(total_samples, 0);

  // create sampler with replacement = true
  WeightedRandomSampler m_sampler(weights, num_samples, false);
  DummyRandomAccessOp dummyRandomAccessOp(total_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  std::unique_ptr<DataBuffer> db;
  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextBuffer(&db), Status::OK());
  db->PopRow(&row);
  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
      freq[*it]++;
    }
  }
  ASSERT_EQ(num_samples, out.size());

  ASSERT_EQ(m_sampler.GetNextBuffer(&db), Status::OK());
  ASSERT_EQ(db->eoe(), true);

  m_sampler.Reset();
  out.clear();
  freq.clear();
  freq.resize(total_samples, 0);
  MS_LOG(INFO) << "Resetting sampler";

  ASSERT_EQ(m_sampler.GetNextBuffer(&db), Status::OK());
  db->PopRow(&row);
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

  ASSERT_EQ(m_sampler.GetNextBuffer(&db), Status::OK());
  ASSERT_EQ(db->eoe(), true);
}
