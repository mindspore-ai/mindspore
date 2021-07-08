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

#ifndef MINDSPORE_CCSRC_FL_SERVER_MEMORY_REGISTER_H_
#define MINDSPORE_CCSRC_FL_SERVER_MEMORY_REGISTER_H_

#include <map>
#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <typeinfo>
#include "fl/server/common.h"

namespace mindspore {
namespace fl {
namespace server {
// Memory allocated in server is normally trainable parameters, hyperparameters, gradients, etc.
// MemoryRegister registers the Memory with key-value format where key refers to address's name("grad", "weights",
// etc) and value is AddressPtr.
class MemoryRegister {
 public:
  MemoryRegister() = default;
  ~MemoryRegister() = default;

  std::map<std::string, AddressPtr> &addresses() { return addresses_; }
  void RegisterAddressPtr(const std::string &name, const AddressPtr &address);

  // In some cases, memory is passed by unique_ptr which is allocated by caller. They needs to be stored as well to
  // avoid its data being released.
  template <typename T>
  void RegisterArray(const std::string &name, std::unique_ptr<T[]> *array, size_t size) {
    MS_EXCEPTION_IF_NULL(array);
    void *data = array->get();
    AddressPtr addr = std::make_shared<Address>();
    addr->addr = data;
    addr->size = size;

    if (typeid(T) == typeid(int)) {
      auto int_arr = CastUniquePtr<int, T>(array);
      StoreInt32Array(&int_arr);
    } else if (typeid(T) == typeid(float)) {
      auto float_arr = CastUniquePtr<float, T>(array);
      StoreFloatArray(&float_arr);
    } else if (typeid(T) == typeid(size_t)) {
      auto uint64_arr = CastUniquePtr<size_t, T>(array);
      StoreUint64Array(&uint64_arr);
    } else if (typeid(T) == typeid(char)) {
      auto char_arr = CastUniquePtr<char, T>(array);
      StoreCharArray(&char_arr);
    } else {
      MS_LOG(ERROR) << "MemoryRegister does not support type " << typeid(T).name();
      return;
    }

    RegisterAddressPtr(name, addr);
    return;
  }

 private:
  std::map<std::string, AddressPtr> addresses_;
  std::vector<std::unique_ptr<float[]>> float_arrays_;
  std::vector<std::unique_ptr<int[]>> int32_arrays_;
  std::vector<std::unique_ptr<size_t[]>> uint64_arrays_;
  std::vector<std::unique_ptr<char[]>> char_arrays_;

  void StoreInt32Array(std::unique_ptr<int[]> *array);
  void StoreFloatArray(std::unique_ptr<float[]> *array);
  void StoreUint64Array(std::unique_ptr<size_t[]> *array);
  void StoreCharArray(std::unique_ptr<char[]> *array);

  template <typename T, typename S>
  std::unique_ptr<T[]> CastUniquePtr(std::unique_ptr<S[]> *array) {
    return std::unique_ptr<T[]>{reinterpret_cast<T *>(array->release())};
  }
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_MEMORY_REGISTER_H_
