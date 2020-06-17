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
#include "dataset/text/kernels/basic_tokenizer_op.h"
#include "dataset/text/kernels/case_fold_op.h"
#include "dataset/text/kernels/normalize_utf8_op.h"
#include "dataset/text/kernels/regex_replace_op.h"
#include "dataset/text/kernels/regex_tokenizer_op.h"
#include "dataset/text/kernels/unicode_char_tokenizer_op.h"
#include "dataset/text/kernels/unicode_script_tokenizer_op.h"
#include "dataset/text/kernels/whitespace_tokenizer_op.h"
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

TEST_F(MindDataTestTokenizerOp, TestWhitespaceTokenizerOp) {
  MS_LOG(INFO) << "Doing TestWhitespaceTokenizerOp.";
  std::unique_ptr<WhitespaceTokenizerOp> op(new WhitespaceTokenizerOp());
  std::shared_ptr<Tensor> input = std::make_shared<Tensor>("Welcome to China.");
  std::shared_ptr<Tensor> output;
  Status s = op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 3);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor1: " << output->ToString();
  CheckEqual(output, {0}, "Welcome");
  CheckEqual(output, {1}, "to");
  CheckEqual(output, {2}, "China.");

  input = std::make_shared<Tensor>("  hello");
  s = op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 1);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor2: " << output->ToString();
  CheckEqual(output, {0}, "hello");

  input = std::make_shared<Tensor>("hello");
  s = op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 1);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor3: " << output->ToString();
  CheckEqual(output, {0}, "hello");

  input = std::make_shared<Tensor>("hello  ");
  s = op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 1);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor4: " << output->ToString();
  CheckEqual(output, {0}, "hello");

  input = std::make_shared<Tensor>("  ");
  s = op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 1);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor5: " << output->ToString();
  CheckEqual(output, {0}, "");
}

TEST_F(MindDataTestTokenizerOp, TestUnicodeScriptTokenizer) {
  MS_LOG(INFO) << "Doing TestUnicodeScriptTokenizer.";
  std::unique_ptr<UnicodeScriptTokenizerOp> keep_whitespace_op(new UnicodeScriptTokenizerOp(true));
  std::unique_ptr<UnicodeScriptTokenizerOp> skip_whitespace_op(new UnicodeScriptTokenizerOp(false));

  std::shared_ptr<Tensor> input = std::make_shared<Tensor>("Welcome to China. \n 中国\t北京");
  std::shared_ptr<Tensor> output;
  Status s = keep_whitespace_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 10);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor1: " << output->ToString();
  CheckEqual(output, {0}, "Welcome");
  CheckEqual(output, {1}, " ");
  CheckEqual(output, {2}, "to");
  CheckEqual(output, {3}, " ");
  CheckEqual(output, {4}, "China");
  CheckEqual(output, {5}, ".");
  CheckEqual(output, {6}, " \n ");
  CheckEqual(output, {7}, "中国");
  CheckEqual(output, {8}, "\t");
  CheckEqual(output, {9}, "北京");
  s = skip_whitespace_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 6);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor2: " << output->ToString();
  CheckEqual(output, {0}, "Welcome");
  CheckEqual(output, {1}, "to");
  CheckEqual(output, {2}, "China");
  CheckEqual(output, {3}, ".");
  CheckEqual(output, {4}, "中国");
  CheckEqual(output, {5}, "北京");

  input = std::make_shared<Tensor>("  Welcome to 中国.  ");
  s = skip_whitespace_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 4);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor3: " << output->ToString();
  CheckEqual(output, {0}, "Welcome");
  CheckEqual(output, {1}, "to");
  CheckEqual(output, {2}, "中国");
  CheckEqual(output, {3}, ".");
  s = keep_whitespace_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 8);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor4: " << output->ToString();
  CheckEqual(output, {0}, "  ");
  CheckEqual(output, {1}, "Welcome");
  CheckEqual(output, {2}, " ");
  CheckEqual(output, {3}, "to");
  CheckEqual(output, {4}, " ");
  CheckEqual(output, {5}, "中国");
  CheckEqual(output, {6}, ".");
  CheckEqual(output, {7}, "  ");

  input = std::make_shared<Tensor>("Hello");
  s = keep_whitespace_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 1);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor5: " << output->ToString();
  CheckEqual(output, {0}, "Hello");

  input = std::make_shared<Tensor>("H");
  s = keep_whitespace_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 1);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor6: " << output->ToString();
  CheckEqual(output, {0}, "H");

  input = std::make_shared<Tensor>("");
  s = keep_whitespace_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 1);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor7: " << output->ToString();
  CheckEqual(output, {0}, "");

  input = std::make_shared<Tensor>("Hello中国Hello世界");
  s = keep_whitespace_op->Compute(input, &output); EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 4);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor8: " << output->ToString();
  CheckEqual(output, {0}, "Hello");
  CheckEqual(output, {1}, "中国");
  CheckEqual(output, {2}, "Hello");
  CheckEqual(output, {3}, "世界");

  input = std::make_shared<Tensor>("   ");
  s = keep_whitespace_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 1);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor10: " << output->ToString();
  CheckEqual(output, {0}, "   ");
  input = std::make_shared<Tensor>("   ");
  s = skip_whitespace_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 1);
  EXPECT_EQ(output->Rank(), 1);
  MS_LOG(INFO) << "Out tensor11: " << output->ToString();
  CheckEqual(output, {0}, "");
}

TEST_F(MindDataTestTokenizerOp, TestCaseFold) {
  MS_LOG(INFO) << "Doing TestCaseFold.";
  std::unique_ptr<CaseFoldOp> case_fold_op(new CaseFoldOp());
  std::shared_ptr<Tensor> input = std::make_shared<Tensor>("Welcome to China. \n 中国\t北京");
  std::shared_ptr<Tensor> output;
  Status s = case_fold_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 1);
  EXPECT_EQ(output->Rank(), 0);
  MS_LOG(INFO) << "Out tensor1: " << output->ToString();
  CheckEqual(output, {}, "welcome to china. \n 中国\t北京");
}

TEST_F(MindDataTestTokenizerOp, TestNormalize) {
  MS_LOG(INFO) << "Doing TestNormalize.";
  std::unique_ptr<NormalizeUTF8Op> nfc_normalize_op(new NormalizeUTF8Op(NormalizeForm::kNfc));
  std::unique_ptr<NormalizeUTF8Op> nfkc_normalize_op(new NormalizeUTF8Op(NormalizeForm::kNfkc));
  std::unique_ptr<NormalizeUTF8Op> nfd_normalize_op(new NormalizeUTF8Op(NormalizeForm::kNfd));
  std::unique_ptr<NormalizeUTF8Op> nfkd_normalize_op(new NormalizeUTF8Op(NormalizeForm::kNfkd));
  std::shared_ptr<Tensor> input = std::make_shared<Tensor>("ṩ");
  std::shared_ptr<Tensor> output;
  Status s = nfc_normalize_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  MS_LOG(INFO) << "NFC str:" << output->ToString();

  nfkc_normalize_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  MS_LOG(INFO) << "NFKC str:" << output->ToString();

  nfd_normalize_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  MS_LOG(INFO) << "NFD str:" << output->ToString();

  nfkd_normalize_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  MS_LOG(INFO) << "NFKD str:" << output->ToString();
}

TEST_F(MindDataTestTokenizerOp, TestRegexReplace) {
  MS_LOG(INFO) << "Doing TestRegexReplace.";
  std::unique_ptr<RegexReplaceOp> regex_replace_op(new RegexReplaceOp("\\s+", "_", true));
  std::shared_ptr<Tensor> input = std::make_shared<Tensor>("Welcome to China. \n 中国\t北京");
  std::shared_ptr<Tensor> output;
  Status s = regex_replace_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 1);
  EXPECT_EQ(output->Rank(), 0);
  MS_LOG(INFO) << "Out tensor1: " << output->ToString();
  CheckEqual(output, {}, "Welcome_to_China._中国_北京");
}

TEST_F(MindDataTestTokenizerOp, TestRegexTokenizer) {
  MS_LOG(INFO) << "Doing TestRegexTokenizerOp.";
  std::unique_ptr<RegexTokenizerOp> regex_tokenizer_op(new RegexTokenizerOp("\\p{Cc}|\\p{Cf}|\\s+", ""));
  std::shared_ptr<Tensor> input = std::make_shared<Tensor>("Welcome to China. \n 中国\t北京");
  std::shared_ptr<Tensor> output;
  Status s = regex_tokenizer_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
}

TEST_F(MindDataTestTokenizerOp, TestBasicTokenizer) {
  MS_LOG(INFO) << "Doing TestBasicTokenizer.";
  //bool lower_case, bool keep_whitespace, 
  // NormalizeForm  normalization_form, bool preserve_unused_token
  std::unique_ptr<BasicTokenizerOp> basic_tokenizer(new BasicTokenizerOp(true, true, NormalizeForm::kNone, false));
  std::shared_ptr<Tensor> input = std::make_shared<Tensor>("Welcome to China. 中国\t北京");
  std::shared_ptr<Tensor> output;
  Status s = basic_tokenizer->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
}