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
#ifndef TESTS_DATASET_UT_CORE_COMMON_DE_UT_COMMON_H_
#define TESTS_DATASET_UT_CORE_COMMON_DE_UT_COMMON_H_

#include "gtest/gtest.h"
#include "utils/log_adapter.h"

namespace UT {
class Common : public testing::Test {
 public:
    // every TEST_F macro will enter one
    virtual void SetUp();

    virtual void TearDown();
};

class DatasetOpTesting : public Common {
 public:
    std::string datasets_root_path_;
    std::string mindrecord_root_path_;
    void SetUp() override;

};
}  // namespace UT
#endif  // TESTS_DATASET_UT_CORE_COMMON_DE_UT_COMMON_H_

