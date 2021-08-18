/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "gtest/gtest.h"
#include "common/common_test.h"
#include "include/errorcode.h"
#include "src/common/config_file.h"

namespace mindspore {
class MixDataTypeTest : public mindspore::CommonTest {
 public:
  MixDataTypeTest() {}
};

TEST_F(MixDataTypeTest, Config1) {
  auto ret = system("echo [execution_plan] > MixDataTypeTestConfig");
  ASSERT_EQ(ret, 0);
  ret = system("echo op1=data_type:fp32 >> MixDataTypeTestConfig");
  ASSERT_EQ(ret, 0);
  ret = system("echo \"op2=\\\"data_type:fp16\\\"\" >> MixDataTypeTestConfig");
  ASSERT_EQ(ret, 0);

  std::string filename = "MixDataTypeTestConfig";
  std::string sectionname = "execution_plan";
  auto execution_plan = lite::GetSectionInfoFromConfigFile(filename, sectionname);

  ASSERT_EQ(execution_plan.size(), 2);

  auto info0 = execution_plan.at("op1");
  ASSERT_EQ(info0, "data_type:fp32");

  auto inf01 = execution_plan.at("op2");
  ASSERT_EQ(inf01, "\"data_type:fp16\"");
}

}  // namespace mindspore
