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

#include "common/common_test.h"
#include "fl/worker/fl_worker.h"


namespace mindspore {
namespace fl {
namespace worker {
class TestFlWorker : public UT::Common {
 public:
  TestFlWorker() = default;
  virtual ~TestFlWorker() = default;

  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TestFlWorker, FlWoker) {
  FLWorker::GetInstance().Run();
  FLWorker::GetInstance().Finalize();
  FLWorker::GetInstance().set_fl_iteration_num(1);
  ASSERT_EQ(FLWorker::GetInstance().fl_iteration_num(), 1);
}

}  // namespace worker
}  // namespace fl
}  // namespace mindspore