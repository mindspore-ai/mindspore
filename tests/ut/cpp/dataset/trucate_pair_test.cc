/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <memory>
#include <string>
#include "minddata/dataset/core/client.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "securec.h"
#include "minddata/dataset/core/tensor.h"
#include "mindspore/ccsrc/minddata/dataset/text/kernels/truncate_sequence_pair_op.h"

using namespace mindspore::dataset;

namespace py = pybind11;

class MindDataTestTruncatePairOp : public UT::Common {
 public:
  MindDataTestTruncatePairOp() {}

  void SetUp() { GlobalInit(); }
};

/// Feature: TruncateSequencePair op
/// Description: Test TruncateSequencePairOp basic usage
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestTruncatePairOp, Basics) {
  std::shared_ptr<Tensor> t1;
  Tensor::CreateFromVector(std::vector<uint32_t>({1, 2, 3}), &t1);
  std::shared_ptr<Tensor> t2;
  Tensor::CreateFromVector(std::vector<uint32_t>({4, 5}), &t2);
  TensorRow in({t1, t2});
  std::shared_ptr<TruncateSequencePairOp> op = std::make_shared<TruncateSequencePairOp>(4);
  TensorRow out;
  ASSERT_TRUE(op->Compute(in, &out).IsOk());
  std::shared_ptr<Tensor> out1;
  Tensor::CreateFromVector(std::vector<uint32_t>({1, 2}), &out1);
  std::shared_ptr<Tensor> out2;
  Tensor::CreateFromVector(std::vector<uint32_t>({4, 5}), &out2);
  ASSERT_EQ(*out1, *out[0]);
  ASSERT_EQ(*out2, *out[1]);
}
