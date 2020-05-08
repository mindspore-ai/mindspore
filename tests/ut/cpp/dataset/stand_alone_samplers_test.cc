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
#include "dataset/core/client.h"
#include "dataset/core/global_context.h"
#include "dataset/engine/datasetops/source/sampler/distributed_sampler.h"
#include "dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "dataset/engine/datasetops/source/sampler/sampler.h"
#include "dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "dataset/util/de_error.h"
#include "dataset/util/status.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"
#include "securec.h"

using namespace mindspore::dataset;

Status CreateINT64Tensor(std::shared_ptr<Tensor> *sample_ids, int64_t num_elements, unsigned char *data = nullptr) {
  TensorShape shape(std::vector<int64_t>(1, num_elements));
  RETURN_IF_NOT_OK(Tensor::CreateTensor(sample_ids, TensorImpl::kFlexible, shape,
                                        DataType(DataType::DE_INT64), data));
  if (data == nullptr) {
    (*sample_ids)->GetMutableBuffer();  // allocate memory in case user forgets!
  }
  return Status::OK();
}

class MindDataTestStandAloneSampler : public UT::DatasetOpTesting {
 protected:
  class MockStorageOp : public RandomAccessOp {
   public:
    MockStorageOp(int64_t val) : m_val_(val) {}

    Status GetNumSamples(int64_t *ptr) const override {
      (*ptr) = m_val_;
      return Status::OK();
    }

    Status GetNumRowsInDataset(int64_t *ptr) const override {
      (*ptr) = m_val_;
      return Status::OK();
    }

   private:
    int64_t m_val_;
  };
};

TEST_F(MindDataTestStandAloneSampler, TestDistributedSampler) {
  std::vector<std::shared_ptr<Tensor>> row;
  uint64_t res[6][7] = {{0, 3, 6, 9, 12, 15, 18},  {1, 4, 7, 10, 13, 16, 19}, {2, 5, 8, 11, 14, 17, 0},
                        {0, 17, 4, 10, 14, 8, 15}, {13, 9, 16, 3, 2, 19, 12}, {1, 11, 6, 18, 7, 5, 0}};
  for (int i = 0; i < 6; i++) {
    std::shared_ptr<Tensor> t;
    Tensor::CreateTensor(&t, TensorImpl::kFlexible, TensorShape({7}),
                         DataType(DataType::DE_INT64), (unsigned char *)(res[i]));
    row.push_back(t);
  }
  MockStorageOp mock(20);
  std::unique_ptr<DataBuffer> db;
  std::shared_ptr<Tensor> tensor;
  for (int i = 0; i < 6; i++) {
    std::unique_ptr<Sampler> sampler = std::make_unique<DistributedSampler>(3, i % 3, (i < 3 ? false : true));
    sampler->HandshakeRandomAccessOp(&mock);
    sampler->GetNextBuffer(&db);
    db->GetTensor(&tensor, 0, 0);
    MS_LOG(DEBUG) << (*tensor);
    if(i < 3) {  // This is added due to std::shuffle()
      EXPECT_TRUE((*tensor) == (*row[i]));
    }
  }
}

TEST_F(MindDataTestStandAloneSampler, TestStandAoneSequentialSampler) {
  std::vector<std::shared_ptr<Tensor>> row;
  MockStorageOp mock(5);
  uint64_t res[5] = {0, 1, 2, 3, 4};
  std::shared_ptr<Tensor> label1, label2;
  CreateINT64Tensor(&label1, 3, reinterpret_cast<unsigned char *>(res));
  CreateINT64Tensor(&label2, 2, reinterpret_cast<unsigned char *>(res + 3));
  std::shared_ptr<Sampler> sampler = std::make_shared<SequentialSampler>(3);
  std::unique_ptr<DataBuffer> db;
  std::shared_ptr<Tensor> tensor;
  sampler->HandshakeRandomAccessOp(&mock);
  sampler->GetNextBuffer(&db);
  db->GetTensor(&tensor, 0, 0);
  EXPECT_TRUE((*tensor) == (*label1));
  sampler->GetNextBuffer(&db);
  db->GetTensor(&tensor, 0, 0);
  EXPECT_TRUE((*tensor) == (*label2));
  sampler->Reset();
  sampler->GetNextBuffer(&db);
  db->GetTensor(&tensor, 0, 0);
  EXPECT_TRUE((*tensor) == (*label1));
  sampler->GetNextBuffer(&db);
  db->GetTensor(&tensor, 0, 0);
  EXPECT_TRUE((*tensor) == (*label2));
}
