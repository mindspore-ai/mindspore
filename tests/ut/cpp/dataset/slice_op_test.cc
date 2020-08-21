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
#include "minddata/dataset/kernels/data/slice_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestSliceOp : public UT::Common {
 protected:
  MindDataTestSliceOp() {}
};

TEST_F(MindDataTestSliceOp, TestOpBasic) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpBasic.";
  std::vector<uint64_t> labels = {1, 1, 3, 2};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, &input);

  std::shared_ptr<Tensor> output;
  Slice slice = Slice(1, 3);
  std::unique_ptr<SliceOp> op(new SliceOp(SliceOption(slice)));
  Status s = op->Compute(input, &output);

  std::vector<uint64_t> out = {1, 3};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, &expected);

  EXPECT_TRUE(s.IsOk());

  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpNeg) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpNeg.";
  std::vector<uint64_t> labels = {1, 1, 3, 6, 4, 2};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, &input);

  std::shared_ptr<Tensor> output;
  Slice slice = Slice(-1, -5, -1);
  std::unique_ptr<SliceOp> op(new SliceOp(slice));
  Status s = op->Compute(input, &output);

  std::vector<uint64_t> out = {2, 4, 6, 3};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOp2D) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOp2D.";
  std::vector<uint64_t> labels = {1, 1, 3, 2, 3, 2};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({2, 3}), &input);

  std::shared_ptr<Tensor> output;
  Slice slice1_ = Slice(0, 2);
  Slice slice2_ = Slice(0, 1);

  std::vector<SliceOption> slices_ = {SliceOption(slice1_), SliceOption(slice2_)};
  std::unique_ptr<SliceOp> op(new SliceOp(slices_));
  Status s = op->Compute(input, &output);

  std::vector<uint64_t> out = {1, 2};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, TensorShape({2, 1}), &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOp3D) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOp3D.";
  std::vector<uint64_t> labels = {1, 2, 3, 4, 5, 6, 7, 8};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({2, 2, 2}), &input);

  std::shared_ptr<Tensor> output;
  Slice slice1_ = Slice(0, 1);
  Slice slice2_ = Slice(0, 2);
  Slice slice3_ = Slice(0, 2);
  std::vector<SliceOption> slices_ = {SliceOption(slice1_), SliceOption(slice2_), SliceOption(slice3_)};
  std::unique_ptr<SliceOp> op(new SliceOp(slices_));
  Status s = op->Compute(input, &output);

  std::vector<uint64_t> out = {1, 2, 3, 4};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, TensorShape({1, 2, 2}), &expected);

  EXPECT_TRUE(s.IsOk());

  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());
  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpReturnNothing) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpReturnNothing.";
  std::vector<uint64_t> labels = {1, 2, 3, 4, 5, 6, 7, 8};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({2, 2, 2}), &input);

  std::shared_ptr<Tensor> output;
  Slice slice1_ = Slice(0, 1);
  Slice slice2_ = Slice(2, 1);
  Slice slice3_ = Slice(0, 2);
  std::vector<SliceOption> slices_ = {SliceOption(slice1_), SliceOption(slice2_), SliceOption(slice3_)};
  std::unique_ptr<SliceOp> op(new SliceOp(slices_));
  Status s = op->Compute(input, &output);

  std::vector<uint64_t> out = {};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, TensorShape({1, 0, 2}), &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpPartialSlice) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpPartialSlice.";
  std::vector<uint64_t> labels = {1, 2, 3, 4, 5, 6, 7, 8};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({4, 2}), &input);

  std::shared_ptr<Tensor> output;
  Slice slice1_ = Slice(0, 2);
  std::unique_ptr<SliceOp> op(new SliceOp(slice1_));
  Status s = op->Compute(input, &output);

  std::vector<uint64_t> out = {1, 2, 3, 4};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, TensorShape({2, 2}), &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpBool1) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpBool1.";
  std::vector<uint64_t> labels = {1, 2, 3, 4, 5, 6, 7, 8};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({2, 2, 2}), &input);

  std::shared_ptr<Tensor> output;
  std::unique_ptr<SliceOp> op(new SliceOp(SliceOption(true)));
  Status s = op->Compute(input, &output);

  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(labels, TensorShape({2, 2, 2}), &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpBool2) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpBool2.";
  std::vector<uint64_t> labels = {1, 2, 3, 4, 5, 6, 7, 8};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({2, 2, 2}), &input);

  std::shared_ptr<Tensor> output;
  std::unique_ptr<SliceOp> op(new SliceOp(true));
  Status s = op->Compute(input, &output);

  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(labels, TensorShape({2, 2, 2}), &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

// testing passing in just indices
TEST_F(MindDataTestSliceOp, TestOpIndices1) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpIndices1.";
  std::vector<uint64_t> labels = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({3, 3}), &input);

  std::shared_ptr<Tensor> output;
  std::vector<SliceOption> indices;
  std::vector<dsize_t> index1 = {1, 2};
  std::vector<dsize_t> index2 = {0, 1};
  indices.emplace_back(SliceOption(index1));
  indices.emplace_back(SliceOption(index2));
  std::unique_ptr<SliceOp> op(new SliceOp(indices));
  Status s = op->Compute(input, &output);

  std::vector<uint64_t> out = {4, 5, 7, 8};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, TensorShape({2, 2}), &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

// testing passing in just indices
TEST_F(MindDataTestSliceOp, TestOpIndices2) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpIndices2.";
  std::vector<uint64_t> labels = {1, 2, 3, 4, 5, 6, 7, 8};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({2, 2, 2}), &input);

  std::shared_ptr<Tensor> output;
  std::vector<dsize_t> indices = {0};
  std::unique_ptr<SliceOp> op(new SliceOp(indices));
  Status s = op->Compute(input, &output);

  std::vector<uint64_t> out = {1, 2, 3, 4};

  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, TensorShape({1, 2, 2}), &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

// Test Index Object
TEST_F(MindDataTestSliceOp, TestOpSliceAndIndex) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpSliceAndIndex.";
  std::vector<uint64_t> labels = {1, 2, 3, 4, 5, 6, 7, 8};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({2, 2, 2}), &input);

  std::shared_ptr<Tensor> output;
  std::vector<dsize_t> indices = {0};
  Slice slice = Slice(1);
  std::vector<SliceOption> slice_options = {SliceOption(indices), SliceOption(slice)};
  std::unique_ptr<SliceOp> op(new SliceOp(slice_options));
  Status s = op->Compute(input, &output);

  std::vector<uint64_t> out = {1, 2};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, TensorShape({1, 1, 2}), &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpLargerStep) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpLargerStep.";
  std::vector<uint64_t> labels = {1, 2, 3, 4, 5};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({1, 5}), &input);

  std::shared_ptr<Tensor> output;
  Slice slice1_ = Slice(0, 1);
  Slice slice2_ = Slice(0, 4, 2);

  std::vector<SliceOption> slice_options = {SliceOption(slice1_), SliceOption(slice2_)};
  std::unique_ptr<SliceOp> op(new SliceOp(slice_options));
  Status s = op->Compute(input, &output);

  std::vector<uint64_t> out = {1, 3};
  std::shared_ptr<Tensor> expected;

  Tensor::CreateFromVector(out, TensorShape({1, 2}), &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpIndicesError1) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpIndicesError1.";
  std::vector<uint64_t> labels = {1, 2, 3, 4, 5, 6, 7, 8};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({2, 2, 2}), &input);

  std::shared_ptr<Tensor> output;
  std::unique_ptr<SliceOp> op(new SliceOp(Slice()));
  Status s = op->Compute(input, &output);

  EXPECT_FALSE(s.IsOk());
  EXPECT_NE(s.ToString().find("Both indices and slices can not be empty."), std::string::npos);

  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpIndicesError2) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpIndicesError2.";
  std::vector<uint64_t> labels = {1, 2, 3, 4, 5, 6, 7, 8};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({2, 2, 2}), &input);

  std::shared_ptr<Tensor> output;
  SliceOption slice_option = SliceOption(Slice(2));
  std::vector<dsize_t> indices = {0};
  slice_option.indices_ = indices;
  std::unique_ptr<SliceOp> op(new SliceOp(slice_option));
  Status s = op->Compute(input, &output);

  EXPECT_FALSE(s.IsOk());
  EXPECT_NE(s.ToString().find("Both indices and slices can not be given."), std::string::npos);

  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpIndicesError3) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpIndicesError3.";
  std::vector<uint64_t> labels = {1, 2, 3, 4, 5, 6, 7, 8};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({8}), &input);

  std::shared_ptr<Tensor> output;
  std::vector<dsize_t> indices = {8};

  std::unique_ptr<SliceOp> op(new SliceOp(SliceOption(indices)));
  Status s = op->Compute(input, &output);

  EXPECT_FALSE(s.IsOk());
  EXPECT_NE(s.ToString().find("Index 8 is out of bounds."), std::string::npos);

  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpBasicString) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpBasicString.";
  std::vector<std::string> labels = {"1", "1", "3", "2d"};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, &input);

  std::shared_ptr<Tensor> output;
  Slice slice = Slice(1, 3);
  std::unique_ptr<SliceOp> op(new SliceOp(slice));
  Status s = op->Compute(input, &output);

  std::vector<std::string> out = {"1", "3"};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOp2DString) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOp2DString.";
  std::vector<std::string> labels = {"1a", "1b", "3", "2", "3", "2"};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({2, 3}), &input);

  std::shared_ptr<Tensor> output;
  Slice slice1_ = Slice(0, 2);
  Slice slice2_ = Slice(0, 2);

  std::vector<SliceOption> slice_option = {SliceOption(slice1_), SliceOption(slice2_)};
  std::unique_ptr<SliceOp> op(new SliceOp(slice_option));
  Status s = op->Compute(input, &output);

  std::vector<std::string> out = {"1a", "1b", "2", "3"};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, TensorShape({2, 2}), &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpPartialSliceString) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpPartialSliceString.";
  std::vector<std::string> labels = {"1a", "1b", "3", "2", "3", "2", "4", "66"};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({2, 2, 2}), &input);

  std::shared_ptr<Tensor> output;
  Slice slice1 = Slice(0, 2);
  Slice slice2 = Slice(0, 1);

  std::vector<SliceOption> slice_options = {SliceOption(slice1), SliceOption(slice2)};
  std::unique_ptr<SliceOp> op(new SliceOp(slice_options));
  Status s = op->Compute(input, &output);

  std::vector<std::string> out = {"1a", "1b", "3", "2"};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, TensorShape({2, 1, 2}), &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpIndicesString) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpIndicesString.";
  std::vector<std::string> labels = {"1", "2", "3", "4", "5", "6", "7", "8", "9"};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({3, 3}), &input);

  std::shared_ptr<Tensor> output;
  std::vector<dsize_t> index1 = {1, 2};
  std::vector<dsize_t> index2 = {0, 1};
  std::vector<SliceOption> slice_options = {SliceOption(index1), SliceOption(index2)};

  std::unique_ptr<SliceOp> op(new SliceOp(slice_options));
  Status s = op->Compute(input, &output);

  std::vector<std::string> out = {"4", "5", "7", "8"};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, TensorShape({2, 2}), &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpIndicesString2) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpIndicesString2.";
  std::vector<std::string> labels = {"1", "2", "3", "4", "5", "6", "7", "8"};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({2, 2, 2}), &input);

  std::shared_ptr<Tensor> output;
  std::vector<dsize_t> indices = {0};
  std::unique_ptr<SliceOp> op(new SliceOp(indices));
  Status s = op->Compute(input, &output);

  std::vector<std::string> out = {"1", "2", "3", "4"};

  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, TensorShape({1, 2, 2}), &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpSliceAndIndexString) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpSliceAndIndexString.";
  std::vector<std::string> labels = {"1", "2", "3", "4", "5", "6", "7", "8"};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({2, 2, 2}), &input);

  std::shared_ptr<Tensor> output;
  std::vector<dsize_t> indices = {0};
  Slice slice = Slice(1);
  std::vector<SliceOption> slice_options = {SliceOption(indices), SliceOption(slice)};
  std::unique_ptr<SliceOp> op(new SliceOp(slice_options));
  Status s = op->Compute(input, &output);

  std::vector<std::string> out = {"1", "2"};
  std::shared_ptr<Tensor> expected;
  Tensor::CreateFromVector(out, TensorShape({1, 1, 2}), &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpLargerStepString) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpLargerStepString.";
  std::vector<std::string> labels = {"1", "2", "3", "4", "5"};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({1, 5}), &input);

  std::shared_ptr<Tensor> output;
  Slice slice1_ = Slice(0, 1);
  Slice slice2_ = Slice(0, 4, 2);

  std::vector<SliceOption> slice_options = {SliceOption(slice1_), SliceOption(slice2_)};
  std::unique_ptr<SliceOp> op(new SliceOp(slice_options));
  Status s = op->Compute(input, &output);

  std::vector<std::string> out = {"1", "3"};
  std::shared_ptr<Tensor> expected;

  Tensor::CreateFromVector(out, TensorShape({1, 2}), &expected);

  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());

  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);
  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpIndicesErrorString1) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpIndicesErrorString1.";
  std::vector<std::string> labels = {"1", "2", "3", "4", "5", "6", "7", "8"};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({2, 2, 2}), &input);

  std::shared_ptr<Tensor> output;
  std::unique_ptr<SliceOp> op(new SliceOp(Slice()));
  Status s = op->Compute(input, &output);

  EXPECT_FALSE(s.IsOk());
  EXPECT_NE(s.ToString().find("Both indices and slices can not be empty."), std::string::npos);

  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpIndicesErrorString2) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpIndicesErrorString2.";
  std::vector<std::string> labels = {"1", "2", "3", "4", "5", "6", "7", "8"};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({2, 2, 2}), &input);

  std::shared_ptr<Tensor> output;
  SliceOption slice_option = SliceOption(Slice(2));
  std::vector<dsize_t> indices = {0};
  slice_option.indices_ = indices;
  std::unique_ptr<SliceOp> op(new SliceOp(slice_option));
  Status s = op->Compute(input, &output);

  EXPECT_FALSE(s.IsOk());
  EXPECT_NE(s.ToString().find("Both indices and slices can not be given."), std::string::npos);

  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}

TEST_F(MindDataTestSliceOp, TestOpIndicesErrorString3) {
  MS_LOG(INFO) << "Doing MindDataTestSliceOp-TestOpIndicesErrorString3.";
  std::vector<uint64_t> labels = {1, 2, 3, 4, 5, 6, 7, 8};
  std::shared_ptr<Tensor> input;
  Tensor::CreateFromVector(labels, TensorShape({2, 4}), &input);

  std::shared_ptr<Tensor> output;
  std::vector<dsize_t> indices = {2};

  std::unique_ptr<SliceOp> op(new SliceOp(SliceOption(indices)));
  Status s = op->Compute(input, &output);

  EXPECT_FALSE(s.IsOk());
  EXPECT_NE(s.ToString().find("Index 2 is out of bounds."), std::string::npos);

  MS_LOG(INFO) << "MindDataTestSliceOp-TestOp end.";
}
