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
#include <vector>
#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"
#include "frontend/parallel/tensor_layout/shape_util.h"

namespace mindspore {
namespace parallel {

/*
 * shape = [2, 8, 32]
 * shape_accum = [2, 2 * 8, 2 * 8 * 32]
 */
TEST(ShapeUtilTest, ShapeToAccumulateProduct) {
  Shape shape = {2, 8, 32};
  std::vector<int64_t> shape_accum;
  Status status = ShapeToAccumulateProduct(shape, &shape_accum);
  ASSERT_EQ(Status::SUCCESS, status);
  std::vector<int64_t> shape_accum_expect = {2, 2 * 8, 2 * 8 * 32};
  ASSERT_EQ(shape_accum_expect, shape_accum);
}

/*
 * shape = [2, 8, 32]
 * shape_accum = [2 * 8 * 32, 8 * 32, 32]
 */
TEST(ShapeUtilTest, ShapeToAccumulateProductReverse) {
  Shape shape = {2, 8, 32};
  std::vector<int64_t> shape_accum;
  Status status = ShapeToAccumulateProductReverse(shape, &shape_accum);
  ASSERT_EQ(Status::SUCCESS, status);
  std::vector<int64_t> shape_accum_expect = {2 * 8 * 32, 8 * 32, 32};
  ASSERT_EQ(shape_accum_expect, shape_accum);
}

/*
 * shape_accum = [2, 2 * 8, 2 * 8 * 32]
 * shape = [2, 8, 32]
 */
TEST(ShapeUtilTest, AccumulateProductToShape) {
  std::vector<int64_t> shape_accum = {2, 2 * 8, 2 * 8 * 32};
  Shape shape;
  Status status = AccumulateProductToShape(shape_accum, &shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Shape shape_expect = {2, 8, 32};
  ASSERT_EQ(shape_expect, shape);
}

/*
 * shape_accum = [2 * 8 * 32, 8 * 32, 32]
 * shape = [2, 8, 32]
 */
TEST(ShapeUtilTest, AccumulateProductReverseToShape) {
  std::vector<int64_t> shape_accum = {2 * 8 * 32, 8 * 32, 32};
  Shape shape;
  Status status = AccumulateProductReverseToShape(shape_accum, &shape);
  ASSERT_EQ(Status::SUCCESS, status);
  Shape shape_expect = {2, 8, 32};
  ASSERT_EQ(shape_expect, shape);
}

/*
 * shape_accum1 = [2, 8]
 * shape_accum2 = [4, 8]
 * out = [2, 4, 8]
 */
TEST(ShapeUtilTest, UnifyAccumulateProduct) {
  std::vector<int64_t> shape_accum1 = {2, 8};
  std::vector<int64_t> shape_accum2 = {4, 8};
  std::vector<int64_t> out;
  Status status = UnifyAccumulateProduct(shape_accum1, shape_accum2, &out);
  ASSERT_EQ(Status::SUCCESS, status);
  std::vector<int64_t> out_expect = {2, 4, 8};
  ASSERT_EQ(out_expect, out);
}

/*
 * in1 = [2, 4]
 * in2 = [4, 2]
 * out = [2, 2, 2]
 */
TEST(ShapeUtilTest, UnifyShape1) {
  Shape in1 = {2, 4};
  Shape in2 = {4, 2};
  Shape out;
  Status status = UnifyShape(in1, in2, &out);
  ASSERT_EQ(Status::SUCCESS, status);
  Shape out_expect = {2, 2, 2};
  ASSERT_EQ(out_expect, out);
}

/*
 * in1 = [8, 4]
 * in2 = [2, 16]
 * out = [2, 4, 4]
 */
TEST(ShapeUtilTest, UnifyShape2) {
  Shape in1 = {8, 4};
  Shape in2 = {2, 16};
  Shape out;
  Status status = UnifyShape(in1, in2, &out);
  ASSERT_EQ(Status::SUCCESS, status);
  Shape out_expect = {2, 4, 4};
  ASSERT_EQ(out_expect, out);
}

/*
 * in_accum_reverse = [2 * 8 * 32, 8 * 32, 32]
 * expand_pos = [2 * 8 * 32, 8 * 32, 32]
 * out_accum_reverse = [2 * 8 * 32, 8 * 32, 32]
 */
TEST(ShapeUtilTest, ExpandAccumulateProduct1) {
  std::vector<int64_t> in_accum_reverse = {2 * 8 * 32, 8 * 32, 32};
  std::vector<int64_t> expand_pos = {2 * 8 * 32, 8 * 32, 32};
  std::vector<int64_t> out_accum_reverse;
  Status status = ExpandAccumulateProduct(in_accum_reverse, expand_pos, &out_accum_reverse);
  ASSERT_EQ(Status::SUCCESS, status);
  std::vector<int64_t> out_accum_reverse_expect = {2 * 8 * 32, 8 * 32, 32};
  ASSERT_EQ(out_accum_reverse_expect, out_accum_reverse);
}

/*
 * in_accum_reverse = [2 * 8 * 32, 8 * 32, 32]
 * expand_pos = [2 * 8 * 32, 8 * 32, 32, 8]
 * out_accum_reverse = [2 * 8 * 32, 8 * 32, 32, 8]
 */
TEST(ShapeUtilTest, ExpandAccumulateProduct2) {
  std::vector<int64_t> in_accum_reverse = {2 * 8 * 32, 8 * 32, 32};
  std::vector<int64_t> expand_pos = {2 * 8 * 32, 8 * 32, 32, 8};
  std::vector<int64_t> out_accum_reverse;
  Status status = ExpandAccumulateProduct(in_accum_reverse, expand_pos, &out_accum_reverse);
  ASSERT_EQ(Status::SUCCESS, status);
  std::vector<int64_t> out_accum_reverse_expect = {2 * 8 * 32, 8 * 32, 32, 8};
  ASSERT_EQ(out_accum_reverse_expect, out_accum_reverse);
}

/*
 * in_accum_reverse = [2 * 8 * 32, 8 * 32, 32]
 * expand_pos = [2 * 8 * 32, 32, 8]
 * out_accum_reverse = [2 * 8 * 32, 8 * 32, 32, 8]
 */
TEST(ShapeUtilTest, ExpandAccumulateProduct3) {
  std::vector<int64_t> in_accum_reverse = {2 * 8 * 32, 8 * 32, 32};
  std::vector<int64_t> expand_pos = {2 * 8 * 32, 32, 8};
  std::vector<int64_t> out_accum_reverse;
  Status status = ExpandAccumulateProduct(in_accum_reverse, expand_pos, &out_accum_reverse);
  ASSERT_EQ(Status::SUCCESS, status);
  std::vector<int64_t> out_accum_reverse_expect = {2 * 8 * 32, 8 * 32, 32, 8};
  ASSERT_EQ(out_accum_reverse_expect, out_accum_reverse);
}

/*
 * in_accum_reverse = [2 * 8 * 32, 8 * 32, 32]
 * expand_pos = [2 * 8 * 32, 8 * 32, 32, 8]
 * out_accum_reverse = [2 * 8 * 32, 8 * 32, 32, 8]
 */
TEST(ShapeUtilTest, ExpandAccumulateProduct4) {
  std::vector<int64_t> in_accum_reverse = {512 * 1024, 1024};
  std::vector<int64_t> expand_pos = {128 * 4096, 4096};
  std::vector<int64_t> out_accum_reverse;
  Status status = ExpandAccumulateProduct(in_accum_reverse, expand_pos, &out_accum_reverse);
  ASSERT_EQ(Status::SUCCESS, status);
  std::vector<int64_t> out_accum_reverse_expect = {128 * 4 * 1024, 4 * 1024, 1024};
  ASSERT_EQ(out_accum_reverse_expect, out_accum_reverse);
}

/*
 * in = [2, 8, 32]
 * expand = [16, 4, 8]
 * out = [2, 8, 4, 8]
 */
TEST(ShapeUtilTest, ExpandShape1) {
  Shape in = {2, 8, 32};
  Shape expand = {16, 4, 8};
  Shape out;
  Status status = ExpandShape(in, expand, &out);
  ASSERT_EQ(Status::SUCCESS, status);
  Shape out_expect = {2, 8, 4, 8};
  ASSERT_EQ(out_expect, out);
}

/*
 * in = [2, 8, 32]
 * expand = [16, 4, 8]
 * out = [2, 8, 4, 8]
 */
TEST(ShapeUtilTest, ExpandShape2) {
  Shape in = {2, 8, 32};
  Shape expand = {2, 4, 8};
  Shape out;
  Status status = ExpandShape(in, expand, &out);
  ASSERT_EQ(Status::SUCCESS, status);
  Shape out_expect = {2, 4, 2, 4, 8};
  ASSERT_EQ(out_expect, out);
}

}  // namespace parallel
}  // namespace mindspore
