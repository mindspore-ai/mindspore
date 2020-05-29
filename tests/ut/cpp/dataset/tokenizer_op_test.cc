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
#include <string_view>

#include "common/common.h"
#include "dataset/text/kernels/unicode_char_tokenizer_op.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestTokenizerOp : public UT::Common {
 public:
  void CheckEqual(const std::shared_ptr<Tensor> &o,
                  const std::vector<dsize_t> &index,
                  const std::string &expect) {
    std::string_view str;
    Status s = o->GetItemAt(&str, index);
    EXPECT_TRUE(s.IsOk());
    EXPECT_EQ(str, expect);
  }
};

TEST_F(MindDataTestTokenizerOp, TestUnicodeCharTokenizerOp) {
  MS_LOG(INFO) << "Doing TestUnicodeCharTokenizerOp.";
  std::unique_ptr<UnicodeCharTokenizerOp> op(new UnicodeCharTokenizerOp());
  std::shared_ptr<Tensor> input = std::make_shared<Tensor>("Hello World!");
  std::shared_ptr<Tensor> output;
  Status s = op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 12);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor1: " << output->ToString();
  CheckEqual(output, {0}, "H");
  CheckEqual(output, {1}, "e");
  CheckEqual(output, {2}, "l");
  CheckEqual(output, {3}, "l");
  CheckEqual(output, {4}, "o");
  CheckEqual(output, {5}, " ");
  CheckEqual(output, {6}, "W");
  CheckEqual(output, {7}, "o");
  CheckEqual(output, {8}, "r");
  CheckEqual(output, {9}, "l");
  CheckEqual(output, {10}, "d");
  CheckEqual(output, {11}, "!");

  input = std::make_shared<Tensor>("中国 你好!");
  s = op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 6);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor2: " << output->ToString();
  CheckEqual(output, {0}, "中");
  CheckEqual(output, {1}, "国");
  CheckEqual(output, {2}, " ");
  CheckEqual(output, {3}, "你");
  CheckEqual(output, {4}, "好");
  CheckEqual(output, {5}, "!");

  input = std::make_shared<Tensor>("中");
  s = op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 1);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor3: " << output->ToString();
  CheckEqual(output, {0}, "中");

  input = std::make_shared<Tensor>("H");
  s = op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 1);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor4: " << output->ToString();
  CheckEqual(output, {0}, "H");

  input = std::make_shared<Tensor>("  ");
  s = op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 2);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor5: " << output->ToString();
  CheckEqual(output, {0}, " ");
  CheckEqual(output, {1}, " ");

  input = std::make_shared<Tensor>("");
  s = op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 1);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor6: " << output->ToString();
  CheckEqual(output, {0}, "");
}
