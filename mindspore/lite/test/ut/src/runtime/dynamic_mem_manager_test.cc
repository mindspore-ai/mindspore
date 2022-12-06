/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifdef BFC_MEMORY
#include "common/common_test.h"
#include "src/common/utils.h"
#define private public
#include "src/extendrt/dynamic_mem_manager.h"
#include "src/extendrt/numa_adapter.h"
#undef private

namespace mindspore {
namespace {
constexpr size_t kAllocUnitSize = 256 * 1024 * 1024;
static constexpr size_t kMinimumAllocUnitSize = 64 * 1024 * 1024;
constexpr size_t kTestAllocSize = 1024;
constexpr size_t kTestAllocSize2 = 64;
constexpr size_t kTestAllocSize3 = 10240;
constexpr size_t kTestAllocSize4 = 4096;
constexpr size_t kTestAllocSize5 = 32;
static constexpr size_t kMemAlginSize = 64;
// 16G
static constexpr size_t kMinimumSysMemory = 17179869184;
}  // namespace
class DynamicMemManagerTest : public mindspore::CommonTest {
 public:
  DynamicMemManagerTest() = default;
};

TEST_F(DynamicMemManagerTest, test_init) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  ASSERT_EQ(mem->all_datas_.size(), 1);
  ASSERT_EQ(mem->all_datas_.begin()->second, kAllocUnitSize);
}

TEST_F(DynamicMemManagerTest, test_malloc_succ) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  auto data = mem->Malloc(kTestAllocSize);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  static const auto kMaxMallocSize = lite::GetMaxMallocSize();
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kMinimumAllocUnitSize - kTestAllocSize);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kAllocUnitSize - kTestAllocSize);
  }

  ASSERT_EQ(mem->free_blocks_.begin()->second, 1);
  auto index = mem->datas_[data];
  ASSERT_EQ(index, 0);
}

TEST_F(DynamicMemManagerTest, test_malloc_succ2) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  auto data = mem->Malloc(kTestAllocSize5);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  static const auto kMaxMallocSize = lite::GetMaxMallocSize();
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kMinimumAllocUnitSize - kMemAlginSize);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kAllocUnitSize - kMemAlginSize);
  }

  ASSERT_EQ(mem->free_blocks_.begin()->second, 1);
  auto index = mem->datas_[data];
  ASSERT_EQ(index, 0);
}

TEST_F(DynamicMemManagerTest, test_free_succ) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  auto data = mem->Malloc(kTestAllocSize);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(mem->garbage_block_, -1);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  mem->Free(data);
  ASSERT_EQ(mem->garbage_block_, 1);
}

TEST_F(DynamicMemManagerTest, test_double_free) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  auto data = mem->Malloc(kTestAllocSize);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(mem->garbage_block_, -1);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  mem->Free(data);
  ASSERT_EQ(mem->garbage_block_, 1);
  mem->Free(data);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->garbage_block_, 1);
}

TEST_F(DynamicMemManagerTest, test_free_nullptr_succ) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  mem->Free(nullptr);
}

TEST_F(DynamicMemManagerTest, test_free_wrong_addr) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  void *data = malloc(kTestAllocSize);
  mem->Free(data);
  free(data);
}

TEST_F(DynamicMemManagerTest, test_malloc_2_buf) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  auto data = mem->Malloc(kTestAllocSize);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 1);
  static const auto kMaxMallocSize = lite::GetMaxMallocSize();
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kMinimumAllocUnitSize - kTestAllocSize);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kAllocUnitSize - kTestAllocSize);
  }
  auto index = mem->datas_[data];
  ASSERT_EQ(index, 0);
  auto data2 = mem->Malloc(kTestAllocSize);
  ASSERT_NE(data2, nullptr);
  ASSERT_NE(data2, data);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 2);
  index = mem->datas_[data2];
  ASSERT_EQ(index, 1);
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kMinimumAllocUnitSize - kTestAllocSize * 2);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kAllocUnitSize - kTestAllocSize * 2);
  }
}

TEST_F(DynamicMemManagerTest, test_malloc_2_buf_free) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  auto data = mem->Malloc(kTestAllocSize);
  ASSERT_NE(data, nullptr);
  auto index = mem->datas_[data];
  ASSERT_EQ(index, 0);
  auto data2 = mem->Malloc(kTestAllocSize);
  ASSERT_NE(data2, nullptr);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 2);
  index = mem->datas_[data2];
  ASSERT_EQ(index, 1);

  mem->Free(data2);
  ASSERT_EQ(mem->garbage_block_, 2);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 1);
  static const auto kMaxMallocSize = lite::GetMaxMallocSize();
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kMinimumAllocUnitSize - kTestAllocSize);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kAllocUnitSize - kTestAllocSize);
  }
  mem->Free(data);
  ASSERT_EQ(mem->garbage_block_, 1);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 0);
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kMinimumAllocUnitSize);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kAllocUnitSize);
  }
}

TEST_F(DynamicMemManagerTest, test_malloc_2_buf_free2) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  auto data = mem->Malloc(kTestAllocSize);
  ASSERT_NE(data, nullptr);
  auto index = mem->datas_[data];
  ASSERT_EQ(index, 0);
  auto data2 = mem->Malloc(kTestAllocSize);
  ASSERT_NE(data2, nullptr);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 2);
  index = mem->datas_[data2];
  ASSERT_EQ(index, 1);

  mem->Free(data);
  ASSERT_EQ(mem->garbage_block_, -1);
  ASSERT_EQ(mem->free_blocks_.size(), 2);
  ASSERT_EQ(mem->free_blocks_.begin()->first, kTestAllocSize);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 0);
  static const auto kMaxMallocSize = lite::GetMaxMallocSize();
  mem->Free(data2);
  ASSERT_EQ(mem->garbage_block_, 1);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kMinimumAllocUnitSize);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kAllocUnitSize);
  }
  ASSERT_EQ(mem->free_blocks_.begin()->second, 0);
}

TEST_F(DynamicMemManagerTest, test_malloc_3_buf) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  auto data = mem->Malloc(kTestAllocSize);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 1);
  static const auto kMaxMallocSize = lite::GetMaxMallocSize();
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kMinimumAllocUnitSize - kTestAllocSize);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kAllocUnitSize - kTestAllocSize);
  }
  auto index = mem->datas_[data];
  ASSERT_EQ(index, 0);
  auto data2 = mem->Malloc(kTestAllocSize2);
  ASSERT_NE(data2, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 2);
  index = mem->datas_[data2];
  ASSERT_EQ(index, 1);
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kMinimumAllocUnitSize - kTestAllocSize - kTestAllocSize2);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kAllocUnitSize - kTestAllocSize - kTestAllocSize2);
  }
  auto data3 = mem->Malloc(kTestAllocSize3);
  ASSERT_NE(data3, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 3);
  index = mem->datas_[data3];
  ASSERT_EQ(index, 2);
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first,
              kMinimumAllocUnitSize - kTestAllocSize - kTestAllocSize2 - kTestAllocSize3);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kAllocUnitSize - kTestAllocSize - kTestAllocSize2 - kTestAllocSize3);
  }
}

TEST_F(DynamicMemManagerTest, test_malloc_3_buf_free_order) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  auto data = mem->Malloc(kTestAllocSize);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 1);
  auto index = mem->datas_[data];
  ASSERT_EQ(index, 0);
  auto data2 = mem->Malloc(kTestAllocSize2);
  ASSERT_NE(data2, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 2);
  index = mem->datas_[data2];
  ASSERT_EQ(index, 1);
  auto data3 = mem->Malloc(kTestAllocSize3);
  ASSERT_NE(data3, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 3);
  index = mem->datas_[data3];
  ASSERT_EQ(index, 2);

  mem->Free(data3);
  ASSERT_EQ(mem->garbage_block_, 3);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 2);
  static const auto kMaxMallocSize = lite::GetMaxMallocSize();
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kMinimumAllocUnitSize - kTestAllocSize - kTestAllocSize2);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kAllocUnitSize - kTestAllocSize - kTestAllocSize2);
  }
  mem->Free(data2);
  ASSERT_EQ(mem->garbage_block_, 2);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 1);
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kMinimumAllocUnitSize - kTestAllocSize);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kAllocUnitSize - kTestAllocSize);
  }
  mem->Free(data);
  ASSERT_EQ(mem->garbage_block_, 1);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 0);
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kMinimumAllocUnitSize);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kAllocUnitSize);
  }
}

TEST_F(DynamicMemManagerTest, test_malloc_3_buf_free_unorder) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  auto data = mem->Malloc(kTestAllocSize);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 1);
  auto index = mem->datas_[data];
  ASSERT_EQ(index, 0);
  auto data2 = mem->Malloc(kTestAllocSize2);
  ASSERT_NE(data2, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 2);
  index = mem->datas_[data2];
  ASSERT_EQ(index, 1);
  auto data3 = mem->Malloc(kTestAllocSize3);
  ASSERT_NE(data3, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 3);
  index = mem->datas_[data3];
  ASSERT_EQ(index, 2);

  mem->Free(data2);
  ASSERT_EQ(mem->garbage_block_, -1);
  ASSERT_EQ(mem->free_blocks_.size(), 2);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 1);
  static const auto kMaxMallocSize = lite::GetMaxMallocSize();
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kTestAllocSize2);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kTestAllocSize2);
  }
  mem->Free(data);
  ASSERT_EQ(mem->garbage_block_, 1);
  ASSERT_EQ(mem->free_blocks_.size(), 2);
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kTestAllocSize + kTestAllocSize2);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kTestAllocSize + kTestAllocSize2);
  }
  ASSERT_EQ(mem->free_blocks_.begin()->second, 0);
  mem->Free(data3);
  ASSERT_EQ(mem->garbage_block_, 2);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kMinimumAllocUnitSize);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kAllocUnitSize);
  }
  ASSERT_EQ(mem->free_blocks_.begin()->second, 0);
}

TEST_F(DynamicMemManagerTest, test_malloc_3_buf_free_order2) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  auto data = mem->Malloc(kTestAllocSize);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 1);
  auto index = mem->datas_[data];
  ASSERT_EQ(index, 0);
  auto data2 = mem->Malloc(kTestAllocSize2);
  ASSERT_NE(data2, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 2);
  index = mem->datas_[data2];
  ASSERT_EQ(index, 1);
  auto data3 = mem->Malloc(kTestAllocSize3);
  ASSERT_NE(data3, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 3);
  index = mem->datas_[data3];
  ASSERT_EQ(index, 2);

  mem->Free(data);
  ASSERT_EQ(mem->garbage_block_, -1);
  ASSERT_EQ(mem->free_blocks_.size(), 2);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 0);
  static const auto kMaxMallocSize = lite::GetMaxMallocSize();
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kTestAllocSize);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kTestAllocSize);
  }
  mem->Free(data2);
  ASSERT_EQ(mem->garbage_block_, 1);
  ASSERT_EQ(mem->free_blocks_.size(), 2);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 0);
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kTestAllocSize + kTestAllocSize2);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kTestAllocSize + kTestAllocSize2);
  }
  mem->Free(data3);
  ASSERT_EQ(mem->garbage_block_, 2);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 0);
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kMinimumAllocUnitSize);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kAllocUnitSize);
  }
}

TEST_F(DynamicMemManagerTest, test_malloc_4_buf_free_unorder) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  auto data = mem->Malloc(kTestAllocSize);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 1);
  auto index = mem->datas_[data];
  ASSERT_EQ(index, 0);
  auto data2 = mem->Malloc(kTestAllocSize2);
  ASSERT_NE(data2, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 2);
  index = mem->datas_[data2];
  ASSERT_EQ(index, 1);
  auto data3 = mem->Malloc(kTestAllocSize3);
  ASSERT_NE(data3, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 3);
  index = mem->datas_[data3];
  ASSERT_EQ(index, 2);

  mem->Free(data);
  ASSERT_EQ(mem->garbage_block_, -1);
  ASSERT_EQ(mem->free_blocks_.size(), 2);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 0);
  static const auto kMaxMallocSize = lite::GetMaxMallocSize();
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kTestAllocSize);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kTestAllocSize);
  }

  auto data4 = mem->Malloc(kTestAllocSize2);
  ASSERT_NE(data4, nullptr);
  ASSERT_EQ(mem->garbage_block_, -1);
  ASSERT_EQ(mem->free_blocks_.size(), 2);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 4);
  ASSERT_EQ(mem->free_blocks_.begin()->first, kTestAllocSize - kTestAllocSize2);
}

TEST_F(DynamicMemManagerTest, test_malloc_4_buf_free_unorder2) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  auto data = mem->Malloc(kTestAllocSize4);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 1);
  auto index = mem->datas_[data];
  ASSERT_EQ(index, 0);
  auto data2 = mem->Malloc(kTestAllocSize2);
  ASSERT_NE(data2, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 2);
  index = mem->datas_[data2];
  ASSERT_EQ(index, 1);
  auto data3 = mem->Malloc(kTestAllocSize3);
  ASSERT_NE(data3, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 3);
  index = mem->datas_[data3];
  ASSERT_EQ(index, 2);

  mem->Free(data);
  ASSERT_EQ(mem->garbage_block_, -1);
  ASSERT_EQ(mem->free_blocks_.size(), 2);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 0);
  static const auto kMaxMallocSize = lite::GetMaxMallocSize();
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kTestAllocSize4);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kTestAllocSize4);
  }
  mem->Free(data2);
  ASSERT_EQ(mem->garbage_block_, 1);
  ASSERT_EQ(mem->free_blocks_.size(), 2);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 0);
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kTestAllocSize4 + kTestAllocSize2);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kTestAllocSize4 + kTestAllocSize2);
  }

  auto data4 = mem->Malloc(kTestAllocSize2 + kTestAllocSize4);
  ASSERT_NE(data4, nullptr);
  ASSERT_EQ(mem->garbage_block_, 1);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  ASSERT_EQ(mem->free_blocks_.begin()->second, 3);
  if (kMaxMallocSize < kMinimumSysMemory) {
    ASSERT_EQ(mem->free_blocks_.begin()->first,
              kMinimumAllocUnitSize - kTestAllocSize2 - kTestAllocSize3 - kTestAllocSize4);
  } else {
    ASSERT_EQ(mem->free_blocks_.begin()->first, kAllocUnitSize - kTestAllocSize2 - kTestAllocSize3 - kTestAllocSize4);
  }
}

TEST_F(DynamicMemManagerTest, test_set_ref_count) {
  DynamicMemManager mem_manager;
  auto mem = mem_manager.GetMemOperator(-1);
  ASSERT_NE(mem, nullptr);
  auto data = mem->Malloc(kTestAllocSize);
  ASSERT_NE(data, nullptr);
  ASSERT_EQ(mem->free_blocks_.size(), 1);
  mem->SetRefCount(data, 1);
  auto ref_count = mem->RefCount(data);
  ASSERT_EQ(ref_count, 1);
  ref_count = mem->IncRefCount(data, 1);
  ASSERT_EQ(ref_count, 2);
  ref_count = mem->IncRefCount(data, 1);
  ASSERT_EQ(ref_count, 3);
  ref_count = mem->DecRefCount(data, 1);
  ASSERT_EQ(ref_count, 2);
}
}  // namespace mindspore

#endif
