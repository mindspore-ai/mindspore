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

#include <string>
#include <string_view>

#include "common/common.h"
#include "dataset/kernels/text/jieba_tokenizer_op.h"
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

TEST_F(MindDataTestJiebaTokenizerOp, TestJieba_opFuntions) {
  MS_LOG(INFO) << "Doing MindDataTestJiebaTokenizerOp  TestJieba_opFuntions.";
  std::string dataset_path = datasets_root_path_ + "/jiebadict";
  std::string hmm_path = dataset_path + "/hmm_model.utf8";
  std::string mp_path = dataset_path + "/jieba.dict.utf8";
  std::shared_ptr<Tensor> output_tensor;
  std::unique_ptr<JiebaTokenizerOp> op(new JiebaTokenizerOp(hmm_path, mp_path));

  std::shared_ptr<Tensor> input_tensor = std::make_shared<Tensor>("今天天气太好了我们一起去外面玩吧");
  Status s = op->Compute(input_tensor, &output_tensor);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output_tensor->Rank(), 1);
  EXPECT_EQ(output_tensor->Size(), 7);
  CheckEqual(output_tensor, {0}, "今天天气");
  CheckEqual(output_tensor, {1}, "太好了");
  CheckEqual(output_tensor, {2}, "我们");
  CheckEqual(output_tensor, {3}, "一起");
  CheckEqual(output_tensor, {4}, "去");
  CheckEqual(output_tensor, {5}, "外面");
  CheckEqual(output_tensor, {6}, "玩吧");
}

TEST_F(MindDataTestJiebaTokenizerOp, TestJieba_opAdd) {
  MS_LOG(INFO) << "Doing MindDataTestJiebaTokenizerOp  TestJieba_opAdd.";
  std::string dataset_path = datasets_root_path_ + "/jiebadict";
  std::string hmm_path = dataset_path + "/hmm_model.utf8";
  std::string mp_path = dataset_path + "/jieba.dict.utf8";
  std::shared_ptr<Tensor> output_tensor;
  std::unique_ptr<JiebaTokenizerOp> op(new JiebaTokenizerOp(hmm_path, mp_path));

  op->AddWord("男默女泪");
  std::shared_ptr<Tensor> input_tensor = std::make_shared<Tensor>("男默女泪");
  Status s = op->Compute(input_tensor, &output_tensor);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output_tensor->Rank(), 1);
  EXPECT_EQ(output_tensor->Size(), 1);
  CheckEqual(output_tensor, {0}, "男默女泪");
}

TEST_F(MindDataTestJiebaTokenizerOp, TestJieba_opEmpty) {
  MS_LOG(INFO) << "Doing MindDataTestJiebaTokenizerOp  TestJieba_opEmpty.";
  std::string dataset_path = datasets_root_path_ + "/jiebadict";
  std::string hmm_path = dataset_path + "/hmm_model.utf8";
  std::string mp_path = dataset_path + "/jieba.dict.utf8";
  std::shared_ptr<Tensor> output_tensor;
  std::unique_ptr<JiebaTokenizerOp> op(new JiebaTokenizerOp(hmm_path, mp_path));

  op->AddWord("男默女泪");
  std::shared_ptr<Tensor> input_tensor = std::make_shared<Tensor>("");
  Status s = op->Compute(input_tensor, &output_tensor);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output_tensor->Rank(), 1);
  EXPECT_EQ(output_tensor->Size(), 1);
  CheckEqual(output_tensor, {0}, "");
}