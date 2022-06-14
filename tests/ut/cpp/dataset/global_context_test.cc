/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/core/global_context.h"
#include "common/common.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestGlobalContext : public UT::Common {
 public:
    MindDataTestGlobalContext() {}
};

/// Feature: GlobalContext
/// Description: Test GlobalContext::Instance()
/// Expectation: Runs successfully
TEST_F(MindDataTestGlobalContext, TestGCFunction) {
  MS_LOG(INFO) << "Doing Test GlobalContext";
  MS_LOG(INFO) << "Doing instance";
  GlobalContext *global = GlobalContext::Instance();
  ASSERT_NE(global, nullptr);
  (void) global->config_manager();
  global->Print(std::cout);
}
