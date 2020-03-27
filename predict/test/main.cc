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

#include <signal.h>
#include <stdlib.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>
#include "test/test_context.h"
#include "common/mslog.h"

int main(int argc, char **argv) {
  // Initialize Google Test.
  testing::InitGoogleTest(&argc, argv);

  for (size_t i = 0; i < argc; i++) {
    std::string arg = std::string(argv[i]);
    if (arg.find("--testRoot") != std::string::npos) {
      auto testContext =
        std::shared_ptr<mindspore::predict::TestContext>(new (std::nothrow) mindspore::predict::TestContext());
      if (testContext == nullptr) {
        MS_LOGE("new testContext failed");
        return 1;
      }
      testContext->SetTestRoot(arg.substr(arg.find("--testRoot=") + 11));
      break;
    }
  }

  int result = RUN_ALL_TESTS();

  return result;
}
