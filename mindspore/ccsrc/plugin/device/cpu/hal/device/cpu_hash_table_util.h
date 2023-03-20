/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_HAL_DEVICE_CPU_HASH_TABLE_UTIL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_HAL_DEVICE_CPU_HASH_TABLE_UTIL_H_

#include <map>
#include <tuple>
#include <utility>
#include <memory>
#include <functional>
#include "include/backend/device_address.h"
#include "plugin/device/cpu/hal/device/cpu_hash_table.h"

namespace mindspore {
namespace device {
namespace cpu {
using CreateHashTableFunc = std::function<void(const UserDataPtr &)>;
using ImportHashTableFunc = std::function<bool(const UserDataPtr &, const void *, size_t)>;
using ClearHashTableFunc = std::function<void(const UserDataPtr &)>;

constexpr size_t kCreateFuncIndex = 0;
constexpr size_t kImportFuncIndex = 1;
constexpr size_t kClearFuncIndex = 2;

/**
 * @brief Create CPU hash table and set into `user_data`.
 * @param[in] `user_data`: The input user data which contains meta information to create CPU hash table.
 */
template <typename KeyType, typename ValueType>
void CreateCPUHashTable(const UserDataPtr &user_data) {
  MS_EXCEPTION_IF_NULL(user_data);
  auto shape_vector = user_data->get<ShapeVector>(kHashTableShapeVector);
  MS_EXCEPTION_IF_NULL(shape_vector);

  int32_t value_size = 1;
  for (size_t i = 0; i < (*shape_vector).size(); ++i) {
    value_size *= (*shape_vector)[i];
  }
  if (value_size <= 0) {
    MS_LOG(WARNING) << "Invalid value size:" << value_size;
  }
  user_data->set<CPUHashTable<KeyType, ValueType>>(kUserDataData,
                                                   std::make_shared<CPUHashTable<KeyType, ValueType>>(value_size));
}

/**
 * @brief Import key, value, status tensors to CPU hash table.
 * @param[in] `user_data`: The input user data which contains CPU hash table need to import.
 * @param[in] `tensor_data`: The host pointer of tensor which need to be imported into CPU hash table.
 * @param[in] `size`: The data length in bytes of tensor data which need to be imported into CPU hash table.
 * @return Whether the function was successfully executed.
 */
template <typename KeyType, typename ValueType>
bool ImportCPUHashTable(const UserDataPtr &user_data, const void *tensor_data, size_t size) {
  MS_EXCEPTION_IF_NULL(user_data);
  MS_EXCEPTION_IF_NULL(tensor_data);
  const auto &cpu_hash_table = user_data->get<CPUHashTable<KeyType, ValueType>>(kUserDataData);
  MS_EXCEPTION_IF_NULL(cpu_hash_table);
  if (!cpu_hash_table->Import({const_cast<void *>(tensor_data), size})) {
    MS_LOG(ERROR) << "Import for hash table failed.";
    return false;
  }
  return true;
}

/**
 * @brief Clear all resource in CPU hash table and reset all statistics.
 * @param[in] `user_data`: The input user data which contains CPU hash table need to clear.
 */
template <typename KeyType, typename ValueType>
void ClearCPUHashTable(const UserDataPtr &user_data) {
  MS_EXCEPTION_IF_NULL(user_data);
  const auto &cpu_hash_table = user_data->get<CPUHashTable<KeyType, ValueType>>(kUserDataData);
  MS_EXCEPTION_IF_NULL(cpu_hash_table);
  if (!cpu_hash_table->Clear()) {
    MS_LOG(EXCEPTION) << "Clear user data failed.";
  }
}

static std::map<std::pair<TypeId, TypeId>, std::tuple<CreateHashTableFunc, ImportHashTableFunc, ClearHashTableFunc>>
  cpu_hash_table_funcs = {
    {std::make_pair(TypeId::kNumberTypeInt32, TypeId::kNumberTypeFloat32),
     std::make_tuple(CreateCPUHashTable<int, float>, ImportCPUHashTable<int, float>, ClearCPUHashTable<int, float>)},
    {std::make_pair(TypeId::kNumberTypeInt64, TypeId::kNumberTypeFloat32),
     std::make_tuple(CreateCPUHashTable<int64_t, float>, ImportCPUHashTable<int64_t, float>,
                     ClearCPUHashTable<int64_t, float>)}};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_HAL_DEVICE_CPU_HASH_TABLE_UTIL_H_
