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
#include "dataset/core/client.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "dataset/core/tensor.h"
#include "dataset/kernels/data/duplicate_op.h"

using namespace mindspore::dataset;

namespace py = pybind11;

class MindDataTestDuplicateOp : public UT::Common {
 public:
  MindDataTestDuplicateOp() {}

  void SetUp() { GlobalInit(); }
};

TEST_F(MindDataTestDuplicateOp, Basics) {
  std::shared_ptr<Tensor> t;
  Tensor::CreateTensor(&t, std::vector<uint32_t>({1, 2, 3, 4, 5, 6}));
  std::shared_ptr<Tensor> v;
  Tensor::CreateTensor(&v, std::vector<uint32_t>({3}), TensorShape::CreateScalar());
  std::shared_ptr<DuplicateOp> op = std::make_shared<DuplicateOp>();
  TensorRow in;
  in.push_back(t);
  TensorRow out;
  ASSERT_TRUE(op->Compute(in, &out).IsOk());

  ASSERT_TRUE(*t == *out[0]);
  ASSERT_TRUE(*t == *out[1]);
  ASSERT_TRUE(t->GetBuffer() == out[0]->GetBuffer());
  ASSERT_TRUE(t->GetBuffer() != out[1]->GetBuffer());
}
