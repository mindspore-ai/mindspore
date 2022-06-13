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

#include <string>
#include <string_view>

#include "common/common.h"
#include "minddata/dataset/text/kernels/jieba_tokenizer_op.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestJiebaTokenizerOp : public UT::DatasetOpTesting {
 public:
  void CheckEqual(const std::shared_ptr<Tensor> &o, const std::vector<dsize_t> &index, const std::string &expect) {
    std::string_view str;
    Status s = o->GetItemAt(&str, index);
    EXPECT_TRUE(s.IsOk());
    EXPECT_EQ(str, expect);
  }
};

/// Feature: JiebaTokenizer op
/// Description: Test JiebaTokenizerOp basic Compute
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestJiebaTokenizerOp, TestJieba_opFuntions) {
  MS_LOG(INFO) << "Doing MindDataTestJiebaTokenizerOp  TestJieba_opFuntions.";
  std::string dataset_path = datasets_root_path_ + "/jiebadict";
  std::string hmm_path = dataset_path + "/hmm_model.utf8";
  std::string mp_path = dataset_path + "/jieba.dict.utf8";
  TensorRow input, output;
  auto op = std::make_unique<JiebaTokenizerOp>(hmm_path, mp_path);

  std::shared_ptr<Tensor> input_tensor;
  Tensor::CreateScalar<std::string>("今天天气太好了我们一起去外面玩吧", &input_tensor);
  input.push_back(input_tensor);
  Status s = op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Rank(), 1);
  EXPECT_EQ(output[0]->Size(), 7);
  CheckEqual(output[0], {0}, "今天天气");
  CheckEqual(output[0], {1}, "太好了");
  CheckEqual(output[0], {2}, "我们");
  CheckEqual(output[0], {3}, "一起");
  CheckEqual(output[0], {4}, "去");
  CheckEqual(output[0], {5}, "外面");
  CheckEqual(output[0], {6}, "玩吧");
}

/// Feature: JiebaTokenizer op
/// Description: Test JiebaTokenizerOp AddWord
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestJiebaTokenizerOp, TestJieba_opAdd) {
  MS_LOG(INFO) << "Doing MindDataTestJiebaTokenizerOp  TestJieba_opAdd.";
  std::string dataset_path = datasets_root_path_ + "/jiebadict";
  std::string hmm_path = dataset_path + "/hmm_model.utf8";
  std::string mp_path = dataset_path + "/jieba.dict.utf8";
  TensorRow input, output;
  auto op = std::make_unique<JiebaTokenizerOp>(hmm_path, mp_path);

  op->AddWord("男默女泪");
  std::shared_ptr<Tensor> input_tensor;
  Tensor::CreateScalar<std::string>("男默女泪", &input_tensor);
  input.push_back(input_tensor);
  Status s = op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Rank(), 1);
  EXPECT_EQ(output[0]->Size(), 1);
  CheckEqual(output[0], {0}, "男默女泪");
}

/// Feature: JiebaTokenizer op
/// Description: Test JiebaTokenizerOp with an empty string input tensor
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestJiebaTokenizerOp, TestJieba_opEmpty) {
  MS_LOG(INFO) << "Doing MindDataTestJiebaTokenizerOp  TestJieba_opEmpty.";
  std::string dataset_path = datasets_root_path_ + "/jiebadict";
  std::string hmm_path = dataset_path + "/hmm_model.utf8";
  std::string mp_path = dataset_path + "/jieba.dict.utf8";
  TensorRow input, output;
  auto op = std::make_unique<JiebaTokenizerOp>(hmm_path, mp_path);

  op->AddWord("男默女泪");
  std::shared_ptr<Tensor> input_tensor;
  Tensor::CreateScalar<std::string>("", &input_tensor);
  input.push_back(input_tensor);
  Status s = op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Rank(), 1);
  EXPECT_EQ(output[0]->Size(), 1);
  CheckEqual(output[0], {0}, "");
}