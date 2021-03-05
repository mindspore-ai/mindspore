/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_GLOBAL_CONTEXT_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_GLOBAL_CONTEXT_H_

#include <memory>
#include <mutex>

#include "include/api/status.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/util/allocator.h"

namespace mindspore {
namespace dataset {
// forward declare
class MemoryPool;
class Tensor;
class CVTensor;
class DeviceTensor;

using TensorAlloc = Allocator<Tensor>;              // An allocator for Tensors
using CVTensorAlloc = Allocator<CVTensor>;          // An allocator CVTensors
using DeviceTensorAlloc = Allocator<DeviceTensor>;  // An allocator for Device_Tensors
using IntAlloc = Allocator<dsize_t>;

class GlobalContext {
  // some consts for pool config
  static constexpr int kArenaSize = 128;
  static constexpr int kMaxSize = -1;
  static constexpr bool kInitArena = true;

 public:
  // Singleton pattern.  This method either:
  // - creates the single version of the GlobalContext for the first time and returns it
  // OR
  // - returns the already existing single instance of the GlobalContext
  // @return the single global context
  static GlobalContext *Instance();

  // Destructor
  ~GlobalContext() = default;

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  void Print(std::ostream &out) const;

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param g_c - reference to the GlobalContext to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const GlobalContext &g_c) {
    g_c.Print(out);
    return out;
  }

  // Getter method
  // @return the client config as raw const pointer
  static std::shared_ptr<ConfigManager> config_manager() { return Instance()->config_manager_; }

  // Getter method
  // @return the mem pool
  std::shared_ptr<MemoryPool> mem_pool() const { return mem_pool_; }

  // Getter method
  // @return the tensor allocator as raw pointer
  const TensorAlloc *tensor_allocator() const { return tensor_allocator_.get(); }

  // Getter method
  // @return the CVTensor allocator as raw pointer
  const CVTensorAlloc *cv_tensor_allocator() const { return cv_tensor_allocator_.get(); }

  // Getter method
  // @return the DeviceTensor allocator as raw pointer
  const DeviceTensorAlloc *device_tensor_allocator() const { return device_tensor_allocator_.get(); }

  // Getter method
  // @return the integer allocator as raw pointer
  const IntAlloc *int_allocator() const { return int_allocator_.get(); }

 private:
  // Constructor.
  // @note Singleton.  Instantiation flows through instance()
  // @return This is a constructor.
  GlobalContext() = default;

  Status Init();

  static std::once_flag init_instance_flag_;
  static std::unique_ptr<GlobalContext> global_context_;        // The instance of the singleton (global)
  std::shared_ptr<MemoryPool> mem_pool_;                        // A global memory pool
  std::shared_ptr<ConfigManager> config_manager_;               // The configs
  std::unique_ptr<TensorAlloc> tensor_allocator_;               // An allocator for Tensors
  std::unique_ptr<CVTensorAlloc> cv_tensor_allocator_;          // An allocator for CV Tensors
  std::unique_ptr<DeviceTensorAlloc> device_tensor_allocator_;  // An allocator for Device Tensors
  std::unique_ptr<IntAlloc> int_allocator_;                     // An allocator for ints
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_CORE_GLOBAL_CONTEXT_H_
