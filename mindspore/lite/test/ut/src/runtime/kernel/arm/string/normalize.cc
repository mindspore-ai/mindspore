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

#include <iostream>
#include "src/runtime/kernel/arm/fp32/skip_gram_fp32.h"
#include "src/runtime/kernel/arm/string/normalize.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/nnacl/skip_gram_parameter.h"
#include "src/common/file_utils.h"
#include "common/common_test.h"
#include "src/common/log_adapter.h"
#include "src/common/string_util.h"

namespace mindspore {
using mindspore::lite::StringPack;
using mindspore::lite::Tensor;

class TestNormalize : public mindspore::CommonTest {
 public:
  TestNormalize() {}
  void NormalizeTestInit();

 public:
  Tensor input_tensor_;
  Tensor output_tensor_;
  std::vector<Tensor *> inputs_{&input_tensor_};
  std::vector<Tensor *> outputs_{&output_tensor_};
  OpParameter parameter_ = {};
  lite::InnerContext ctx_ = lite::InnerContext();
  kernel::KernelKey desc_ = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_CustomNormalize};
  kernel::KernelCreator creator_ = nullptr;
  kernel::LiteKernel *kernel_ = nullptr;
};

void TestNormalize::NormalizeTestInit() {
  input_tensor_.set_data_type(kObjectTypeString);
  input_tensor_.set_format(schema::Format_NHWC);

  std::vector<StringPack> str_pack;
  const char sentence1[] = "  I don't know what happened\n";
  str_pack.push_back({static_cast<int>(strlen(sentence1) + 1), sentence1});
  const char sentence2[] = "She's not here when Alex arrived!!!";
  str_pack.push_back({static_cast<int>(strlen(sentence2) + 1), sentence2});
  mindspore::lite::WriteStringsToTensor(&input_tensor_, str_pack);

  output_tensor_.set_data_type(kObjectTypeString);
  output_tensor_.set_format(schema::Format_NHWC);
}

TEST_F(TestNormalize, TestSentence) {
  NormalizeTestInit();
  ASSERT_EQ(lite::RET_OK, ctx_.Init());
  creator_ = lite::KernelRegistry::GetInstance()->GetCreator(desc_);
  ASSERT_NE(creator_, nullptr);
  kernel_ = creator_(inputs_, outputs_, &parameter_, &ctx_, desc_);
  ASSERT_NE(kernel_, nullptr);
  auto ret = kernel_->Init();
  ASSERT_EQ(ret, 0);
  ret = kernel_->Run();
  ASSERT_EQ(ret, 0);

  std::vector<StringPack> output = mindspore::lite::ParseTensorBuffer(outputs_[0]);
  for (unsigned int i = 0; i < output.size(); i++) {
    for (int j = 0; j < output[i].len; j++) {
      printf("%c", output[i].data[j]);
    }
    printf("\n");
  }

  input_tensor_.set_data(nullptr);
  output_tensor_.set_data(nullptr);
}

}  // namespace mindspore
