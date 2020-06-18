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
#include "dataset/kernels/data/concatenate_op.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::LogStream;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::MsLogLevel::INFO;

class MindDataTestConcatenateOp : public UT::Common {
 protected:
  MindDataTestConcatenateOp() {}
};

TEST_F(MindDataTestConcatenateOp, TestOp) {
  MS_LOG(INFO) << "Doing MindDataTestConcatenate-TestOp.";
  uint64_t labels[3] = {1, 1, 2};
  TensorShape shape({3});
  std::shared_ptr<Tensor> input =
    std::make_shared<Tensor>(shape, DataType(DataType::DE_UINT64), reinterpret_cast<unsigned char *>(labels));

  uint64_t append_labels[3] = {4, 4, 4};
  std::shared_ptr<Tensor> append =
    std::make_shared<Tensor>(shape, DataType(DataType::DE_UINT64), reinterpret_cast<unsigned char *>(append_labels));

  std::shared_ptr<Tensor> output;
  std::unique_ptr<ConcatenateOp> op(new ConcatenateOp(0, nullptr, append));
  TensorRow in;
  in.push_back(input);
  TensorRow out_row;
  Status s = op->Compute(in, &out_row);
  uint64_t out[6] = {1, 1, 2, 4, 4, 4};

  std::shared_ptr<Tensor> expected =
    std::make_shared<Tensor>(TensorShape{6}, DataType(DataType::DE_UINT64), reinterpret_cast<unsigned char *>(out));
  output = out_row[0];
  EXPECT_TRUE(s.IsOk());
  ASSERT_TRUE(output->shape() == expected->shape());
  ASSERT_TRUE(output->type() == expected->type());
  MS_LOG(DEBUG) << *output << std::endl;
  MS_LOG(DEBUG) << *expected << std::endl;

  ASSERT_TRUE(*output == *expected);

  //  std::vector<TensorShape> inputs = {TensorShape({3})};
  //  std::vector<TensorShape> outputs = {};
  //  s = op->OutputShape(inputs, outputs);
  //  EXPECT_TRUE(s.IsOk());
  //  ASSERT_TRUE(outputs[0] == TensorShape{6});
  //  MS_LOG(INFO) << "MindDataTestConcatenateOp-TestOp end.";
}
