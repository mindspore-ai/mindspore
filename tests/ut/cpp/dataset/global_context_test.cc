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
#include "dataset/core/global_context.h"
#include "common/common.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestGlobalContext : public UT::Common {
 public:
    MindDataTestGlobalContext() {}
};

TEST_F(MindDataTestGlobalContext, TestGCFunction) {
  MS_LOG(INFO) << "Doing Test GlobalContext";
  MS_LOG(INFO) << "Doing instance";
  GlobalContext *global = GlobalContext::Instance();
  ASSERT_NE(global, nullptr);
  (void) global->config_manager();
  global->Print(std::cout);
}
