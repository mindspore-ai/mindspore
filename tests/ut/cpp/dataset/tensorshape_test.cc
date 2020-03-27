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
#include <string>
#include "./securec.h"
#include "dataset/core/client.h"
#include "dataset/core/data_type.h"
#include "dataset/core/tensor_shape.h"
#include "dataset/engine/data_schema.h"
#include "common/common.h"
#include "common/utils.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestTensorShape : public UT::Common {
 public:
    MindDataTestTensorShape() = default;
};


TEST_F(MindDataTestTensorShape, TestBasics) {
  std::vector<dsize_t> vec = {4, 5, 6};
  TensorShape t(vec);
  ASSERT_EQ(t.Rank(), 3);
  ASSERT_EQ(t.Size(), 3);
  ASSERT_EQ(t.known(), true);
  ASSERT_EQ(t.empty(), false);
  ASSERT_EQ(t.NumOfElements(), 120);
  for (dsize_t i = 0; i < t.Rank(); i++) {
    ASSERT_EQ(t[i], vec[i]);
  }
  ASSERT_EQ(vec, t.AsVector());
  ASSERT_EQ(t.IsValidIndex({0, 0, 0}), true);
  ASSERT_EQ(t.IsValidIndex({3, 4, 5}), true);
  ASSERT_EQ(t.IsValidIndex({3, 4, 6}), false);
  ASSERT_EQ(t.IsValidIndex({4, 5, 6}), false);
  ASSERT_EQ(t.IsValidIndex({4, 5, 6}), false);
  ASSERT_EQ(t.IsValidIndex({3, 3}), false);
  ASSERT_EQ(t.IsValidIndex({-3, -3, -1}), false);
  ASSERT_EQ(t.IsValidIndex({-1, 4, 5}), false);
  TensorShape t2({4, 5, 6});
  ASSERT_EQ(t, t2);
  TensorShape t3({0});
  ASSERT_EQ(t3.Size(), 1);
  ASSERT_EQ(t3.NumOfElements(), 0);
  t3 = TensorShape({0, 5, 6});
  ASSERT_EQ(t3.Size(), 3);
  ASSERT_EQ(t3.NumOfElements(), 0);
}

TEST_F(MindDataTestTensorShape, TestScalars) {
  TensorShape t = TensorShape::CreateScalar();
  ASSERT_EQ(t.Rank(), 0);
  ASSERT_EQ(t.AsVector(), std::vector<dsize_t>{});
  ASSERT_EQ(t.known(), true);
  TensorShape t2(std::vector<dsize_t>{});
  ASSERT_EQ(t, t2);
  ASSERT_EQ(t.NumOfElements(), 1);
}

TEST_F(MindDataTestTensorShape, TestDims) {
  TensorShape t = TensorShape::CreateScalar();
  t = t.AppendDim(1);
  t = t.AppendDim(2);
  t = t.AppendDim(3);
  ASSERT_EQ(t, TensorShape({1, 2, 3}));
  TensorShape t2 = TensorShape::CreateScalar();
  t2 = t2.PrependDim(3);
  t2 = t2.PrependDim(2);
  t2 = t2.PrependDim(1);
  ASSERT_EQ(t, t2);
  TensorShape t3({4, 5, 6});
  t3 = t3.InsertDim(0, 1);  // 1, 4, 5, 6
  t3 = t3.InsertDim(2, 2);  // 1, 4, 2, 5, 6
  t3 = t3.InsertDim(4, 3);  // 1, 4, 2, 5, 3, 6
  ASSERT_EQ(t3, TensorShape({1, 4, 2, 5, 3, 6}));
}

TEST_F(MindDataTestTensorShape, TestUnknown) {
  TensorShape t1({-1, 5, 6});
  ASSERT_EQ(t1.AsVector(), std::vector<dsize_t>({-1, 5, 6}));
  ASSERT_EQ(t1.known(), false);
  TensorShape t2({5, 6});
  t2 = t2.PrependDim(-1);
  ASSERT_EQ(t1, t2);
  TensorShape t3 = TensorShape::CreateUnknownRankShape();
  ASSERT_EQ(t3.known(), false);
  ASSERT_EQ(t3.Size(), 0);
  TensorShape t4 = TensorShape::CreateUnknownShapeWithRank(3);
  ASSERT_EQ(t4, TensorShape({-1, -1, -1}));
}

// Test materializing a TensorShape by calling method on a given column descriptor
TEST_F(MindDataTestTensorShape, TestColDescriptor) {
  int32_t rank = 0; // not used
  int32_t num_elements = 0;

  // Has no shape
  ColDescriptor c1("col1", DataType(DataType::DE_INT8), TensorImpl::kFlexible, rank);
  TensorShape generated_shape1 = TensorShape::CreateUnknownRankShape();
  num_elements = 4;
  Status rc = c1.MaterializeTensorShape(num_elements, &generated_shape1);
  ASSERT_TRUE(rc.IsOk());
  MS_LOG(INFO) << "generated_shape1: " << common::SafeCStr(generated_shape1.ToString()) << ".";
  ASSERT_EQ(TensorShape({4}),generated_shape1);

  // Has shape <DIM_UNKNOWN> i.e. <*>
  TensorShape requested_shape2({TensorShape::kDimUnknown});
  ColDescriptor c2("col2", DataType(DataType::DE_INT8), TensorImpl::kFlexible, rank, &requested_shape2);
  TensorShape generated_shape2 = TensorShape::CreateUnknownRankShape();
  num_elements = 5;
  rc = c2.MaterializeTensorShape(num_elements, &generated_shape2);
  ASSERT_TRUE(rc.IsOk());
  MS_LOG(INFO) << "generated_shape2: " << common::SafeCStr(generated_shape2.ToString()) << ".";
  ASSERT_EQ(TensorShape({5}),generated_shape2);

  // Compute unknown dimension <*,4>
  TensorShape requested_shape3({TensorShape::kDimUnknown, 4});
  ColDescriptor c3("col3", DataType(DataType::DE_INT8), TensorImpl::kFlexible, rank, &requested_shape3);
  TensorShape generated_shape3 = TensorShape::CreateUnknownRankShape();
  num_elements = 12;
  rc = c3.MaterializeTensorShape(num_elements, &generated_shape3);
  ASSERT_TRUE(rc.IsOk());
  MS_LOG(INFO) << "generated_shape3: " << common::SafeCStr(generated_shape3.ToString()) << ".";
  ASSERT_EQ(TensorShape({3,4}),generated_shape3);

  // Compute unknown dimension <3,*,4>
  TensorShape requested_shape4({3, TensorShape::kDimUnknown, 4});
  ColDescriptor c4("col4", DataType(DataType::DE_INT8), TensorImpl::kFlexible, rank, &requested_shape4);
  TensorShape generated_shape4 = TensorShape::CreateUnknownRankShape();
  num_elements = 24;
  rc = c4.MaterializeTensorShape(num_elements, &generated_shape4);
  ASSERT_TRUE(rc.IsOk());
  MS_LOG(INFO) << "generated_shape4: " << common::SafeCStr(generated_shape4.ToString()) << ".";
  ASSERT_EQ(TensorShape({3,2,4}),generated_shape4);

  // requested and generated should be the same! <2,3,4>
  TensorShape requested_shape5({2, 3, 4});
  ColDescriptor c5("col5", DataType(DataType::DE_INT8), TensorImpl::kFlexible, rank, &requested_shape5);
  TensorShape generated_shape5 = TensorShape::CreateUnknownRankShape();
  num_elements = 24;
  rc = c5.MaterializeTensorShape(num_elements, &generated_shape5);
  ASSERT_TRUE(rc.IsOk());
  MS_LOG(INFO) << "generated_shape5: " << common::SafeCStr(generated_shape5.ToString()) << ".";
  ASSERT_EQ(requested_shape5,generated_shape5);

  // expect fail due to multiple unknown dimensions
  TensorShape requested_shape6({2, TensorShape::kDimUnknown, TensorShape::kDimUnknown});
  ColDescriptor c6("col6", DataType(DataType::DE_INT8), TensorImpl::kFlexible, rank, &requested_shape6);
  TensorShape generated_shape6 = TensorShape::CreateUnknownRankShape();
  num_elements = 24;
  rc = c6.MaterializeTensorShape(num_elements, &generated_shape6);
  ASSERT_FALSE(rc.IsOk());

  // expect fail because the requested shape element count does not match with num elements
  TensorShape requested_shape7({2, 3, 3});
  ColDescriptor c7("col7", DataType(DataType::DE_INT8), TensorImpl::kFlexible, rank, &requested_shape7);
  TensorShape generated_shape7 = TensorShape::CreateUnknownRankShape();
  num_elements = 24;
  rc = c7.MaterializeTensorShape(num_elements, &generated_shape7);
  ASSERT_FALSE(rc.IsOk());
}

TEST_F(MindDataTestTensorShape, TestInvalid) {
  ASSERT_EQ(TensorShape({2147483648}), TensorShape::CreateUnknownRankShape());
  ASSERT_EQ(TensorShape({kDeMaxDim - 1, kDeMaxDim - 1, kDeMaxDim - 1}), TensorShape::CreateUnknownRankShape());
}
