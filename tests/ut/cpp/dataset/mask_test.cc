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
#include <memory>
#include <string>
#include "dataset/core/client.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "securec.h"
#include "dataset/core/tensor.h"
#include "dataset/core/cv_tensor.h"
#include "dataset/core/data_type.h"
#include "dataset/kernels/data/mask_op.h"
#include "dataset/kernels/data/data_utils.h"

using namespace mindspore::dataset;

namespace py = pybind11;

class MindDataTestMaskOp : public UT::Common {
 public:
  MindDataTestMaskOp() {}

  void SetUp() { GlobalInit(); }
};

TEST_F(MindDataTestMaskOp, Basics) {
  std::shared_ptr<Tensor> t;
  Tensor::CreateTensor(&t, std::vector<uint32_t>({1, 2, 3, 4, 5, 6}));
  std::shared_ptr<Tensor> v;
  Tensor::CreateTensor(&v, std::vector<uint32_t>({3}), TensorShape::CreateScalar());
  std::shared_ptr<MaskOp> op = std::make_shared<MaskOp>(RelationalOp::kEqual, v, DataType(DataType::DE_UINT16));
  std::shared_ptr<Tensor> out;
  ASSERT_TRUE(op->Compute(t, &out).IsOk());

  op = std::make_shared<MaskOp>(RelationalOp::kNotEqual, v, DataType(DataType::DE_UINT16));
  ASSERT_TRUE(op->Compute(t, &out).IsOk());

  op = std::make_shared<MaskOp>(RelationalOp::kLessEqual, v, DataType(DataType::DE_UINT16));
  ASSERT_TRUE(op->Compute(t, &out).IsOk());

  op = std::make_shared<MaskOp>(RelationalOp::kLess, v, DataType(DataType::DE_UINT16));
  ASSERT_TRUE(op->Compute(t, &out).IsOk());

  op = std::make_shared<MaskOp>(RelationalOp::kGreaterEqual, v, DataType(DataType::DE_UINT16));
  ASSERT_TRUE(op->Compute(t, &out).IsOk());

  op = std::make_shared<MaskOp>(RelationalOp::kGreater, v, DataType(DataType::DE_UINT16));
  ASSERT_TRUE(op->Compute(t, &out).IsOk());
}
