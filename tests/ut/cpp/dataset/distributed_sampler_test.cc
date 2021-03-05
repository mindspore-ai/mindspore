/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/distributed_sampler.h"
#include "utils/log_adapter.h"

#include <vector>
#include <unordered_set>

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestDistributedSampler : public UT::Common {
 public:
  class DummyRandomAccessOp : public RandomAccessOp {
   public:
    DummyRandomAccessOp(uint64_t num_rows) {
      // row count is in base class as protected member
      // GetNumRowsInDataset does not need an override, the default from base class is fine.
      num_rows_ = num_rows;
    }
  };
};

TEST_F(MindDataTestDistributedSampler, TestTwoShardsOne) {
  // num samples to draw.
  uint64_t num_samples = 7;

  // create sampler with replacement = true
  DistributedSamplerRT m_sampler(num_samples, 2, 0, false, 0, -1, false);
  DummyRandomAccessOp dummyRandomAccessOp(num_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  std::unique_ptr<DataBuffer> db;
  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextSample(&db), Status::OK());
  db->PopRow(&row);
  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
    }
  }

  ASSERT_EQ(4, out.size());

  ASSERT_EQ(m_sampler.GetNextSample(&db), Status::OK());
  ASSERT_EQ(db->eoe(), true);
}

TEST_F(MindDataTestDistributedSampler, TestTwoShardsTwo) {
  // num samples to draw.
  uint64_t num_samples = 7;

  // create sampler with replacement = true
  DistributedSamplerRT m_sampler(num_samples, 2, 1, false, 0, -1, false);
  DummyRandomAccessOp dummyRandomAccessOp(num_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  std::unique_ptr<DataBuffer> db;
  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextSample(&db), Status::OK());
  db->PopRow(&row);
  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
    }
  }

  ASSERT_EQ(3, out.size());

  ASSERT_EQ(m_sampler.GetNextSample(&db), Status::OK());
  ASSERT_EQ(db->eoe(), true);
}

TEST_F(MindDataTestDistributedSampler, TestThreeShards) {
  // num samples to draw.
  uint64_t num_samples = 2;

  // create sampler with replacement = true
  DistributedSamplerRT m_sampler(num_samples, 3, 2, false, 0, -1, false);
  DummyRandomAccessOp dummyRandomAccessOp(num_samples);
  m_sampler.HandshakeRandomAccessOp(&dummyRandomAccessOp);

  std::unique_ptr<DataBuffer> db;
  TensorRow row;
  std::vector<uint64_t> out;
  ASSERT_EQ(m_sampler.GetNextSample(&db), Status::OK());
  db->PopRow(&row);
  for (const auto &t : row) {
    for (auto it = t->begin<uint64_t>(); it != t->end<uint64_t>(); it++) {
      out.push_back(*it);
    }
  }

  ASSERT_EQ(0, out.size());

  ASSERT_EQ(m_sampler.GetNextSample(&db), Status::OK());
  ASSERT_EQ(db->eoe(), true);
}

