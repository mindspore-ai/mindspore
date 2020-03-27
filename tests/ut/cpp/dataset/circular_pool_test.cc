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
#include <string>
#include <random>
#include "dataset/util/task_manager.h"
#include "dataset/util/circular_pool.h"
#include "dataset/util/services.h"
#include "common/common.h"
#include "common/utils.h"
#include "utils/log_adapter.h"
#include "./securec.h"

namespace common = mindspore::common;

using namespace mindspore::dataset;
using mindspore::MsLogLevel::INFO;
using mindspore::ExceptionType::NoExceptionType;
using mindspore::LogStream;

class MindDataTestCircularPool : public UT::Common {
 public:
  std::shared_ptr<MemoryPool> mp_;
  TaskGroup vg_;
    MindDataTestCircularPool() {}
    void SetUp() {
      Status rc = CircularPool::CreateCircularPool(&mp_);
      ASSERT_TRUE(rc.IsOk());
    }
};

Status TestMem(MindDataTestCircularPool *tp, int32_t num_iterations) {
  const uint64_t min = 19 * 1024;  // 19k
  const uint64_t max = 20 * 1024 * 1024;  // 20M
  std::mt19937 gen{std::random_device{}()};
  std::uniform_int_distribution<uint64_t> dist(min, max);
  TaskManager::FindMe()->Post();
  for (int i = 0; i < num_iterations; i++) {
    uint64_t old_sz = dist(gen);
    uint64_t new_sz = dist(gen);
    std::string str = "Allocate " + std::to_string(old_sz) +
                      " bytes of memory and then resize to " + std::to_string(new_sz);
    std::cout << str << std::endl;
    std::string id = Services::GetUniqueID();
    void *p;
    RETURN_IF_NOT_OK(tp->mp_->Allocate(old_sz, &p));
    // Copy the id to the start of the memory.
    (void) memcpy_s(p, old_sz, common::SafeCStr(id), UNIQUEID_LEN);
    RETURN_IF_NOT_OK(tp->mp_->Reallocate(&p, old_sz, new_sz));
    int n = memcmp(p, common::SafeCStr(id), UNIQUEID_LEN);
    if (n) {
      RETURN_STATUS_UNEXPECTED("Expect match");
    }
    tp->mp_->Deallocate(p);
  }
  return Status::OK();
}

TEST_F(MindDataTestCircularPool, TestALLFunction) {
  const int32_t iteration = 100;
  Services::CreateInstance();
  auto f = std::bind(TestMem, this, iteration);
  for (int i = 0; i < 3; i++) {
    vg_.CreateAsyncTask("TestMem", f);
  }
  vg_.join_all();
  std::cout << vg_.GetTaskErrorIfAny() << std::endl;
  ASSERT_TRUE(vg_.GetTaskErrorIfAny().IsOk());
  CircularPool *cp = dynamic_cast<CircularPool *>(mp_.get());
  std::cout << *cp << std::endl;
}

