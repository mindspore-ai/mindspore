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
#include "dataset/util/arena.h"
#include "common/common.h"
#include "dataset/util/de_error.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestArena : public UT::Common {
 public:
    MindDataTestArena() {}
};


TEST_F(MindDataTestArena, TestALLFunction) {
  std::shared_ptr<Arena> mp;
  Status rc = Arena::CreateArena(&mp);
  ASSERT_TRUE(rc.IsOk());
  std::shared_ptr<Arena> arena = std::dynamic_pointer_cast<Arena>(mp);
  std::vector<void *> v;

  srand(time(NULL));
  for (int i = 0; i < 1000; i++) {
    uint64_t sz = rand() % 1048576;
    void *ptr = nullptr;
    ASSERT_TRUE(mp->Allocate(sz, &ptr));
    v.push_back(ptr);
  }
  for (int i = 0; i < 1000; i++) {
    mp->Deallocate(v.at(i));
  }
  MS_LOG(DEBUG) << *mp;
}
