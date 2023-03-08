/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_ALLOCATOR_ALLOCATOR_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_ALLOCATOR_ALLOCATOR_H_
#include <map>
#include <vector>
#include <memory>
#include <utility>
#include <string>
#include "tools/converter/micro/coder/allocator/memory_manager.h"
#include "tools/converter/micro/coder/log.h"
#include "tools/converter/micro/coder/utils/type_cast.h"
#include "src/tensor.h"
#include "src/common/log_adapter.h"
#include "tools/converter/micro/coder/generator/component/component.h"

namespace mindspore::lite::micro {
/*
 * kOfflinePackWeight, pack weight tensor data when opcode Prepare
 * kOnlinePackWeight, pack weight tensor data when WeightInit function before the Inference
 */
enum MallocType { kOfflinePackWeight = 0, kOnlinePackWeight = 1, kWorkspace = 2 };
inline std::string wrap(const std::string &a) { return "(" + a + ")"; }
/*
 *  while using Malloc(size, kOnlinePackWeight), the size is actually not necessary,
 *  it could be any value that is a multiple of sizeof(T)
 *  it just need a size, so we set kOnlineSize to avoid magic number.
 */
const int kOnlineSize = 4;

class OperatorCoder;
class MemoryAllocator {
 public:
  static MemoryAllocator *GetInstance() {
    static MemoryAllocator allocator;
    return &allocator;
  }

  MemoryAllocator(const MemoryAllocator &) = delete;
  MemoryAllocator &operator=(const MemoryAllocator &) = delete;

  /*
   * assign model's input, original weights and all tensors memory addr
   */
  int Assign(const std::vector<Tensor *> &inputs, const std::vector<std::unique_ptr<OperatorCoder>> &nodes);

  // allocator holds the space malloced by opcoders, will free before session coder destroy
  void Free();
  /*
   * malloc new weight or bias at Prepare
   * in view of weight, bias and workspace
   */

  void *Malloc(TypeId type_id, size_t size, MallocType type, const std::string &tensor_name = "") {
    if (type != kWorkspace) {
      return MallocWeightTensor(type_id, size, type, tensor_name);
    }
    if (size == 0 || size >= UINT_MAX) {
      return nullptr;
    }
    void *buffer = malloc(size);
    if (buffer == nullptr) {
      MS_LOG(ERROR) << "malloc memory failed";
      return nullptr;
    }
    AssignWorkspaces(buffer, size);
    allocated_.push_back(buffer);
    return buffer;
  }

  Tensor *MallocTensor(TypeId data_type, const std::vector<int> &shape);
  void FreeTensor(Tensor *t);
  /*
   * get the actual runtime address with it's type,
   * including tensor, workspace
   */
  template <typename T>
  std::string GetRuntimeAddr(T t, bool immutable = false) {
    if (!t) {
      return "";
    }
    std::string type_name;
    if (std::type_index(typeid(T)) == std::type_index(typeid(Tensor *))) {
      type_name = GetTensorDataType(reinterpret_cast<Tensor *>(t)->data_type()) + "*";
    } else {
      type_name = GetVariableTypeName<T>();
    }
    std::string type_info = wrap(type_name);
    void *variable = reinterpret_cast<void *>(t);
    auto item = workspaces_addr_.find(variable);
    if (item != workspaces_addr_.end()) {
      return type_info + wrap(item->second);
    }

    auto iter = std::find_if(malloc_weights_addr_.begin(), malloc_weights_addr_.end(),
                             [&variable](const std::pair<Tensor *, std::string> &a) {
                               return variable == (a.first)->data() || variable == a.first;
                             });
    if (iter != malloc_weights_addr_.end()) {
      return iter->second;
    }

    Tensor *tensor = reinterpret_cast<Tensor *>(t);
    auto it = tensors_addr_.find(tensor);
    if (it != tensors_addr_.end()) {
      return type_info + wrap(it->second);
    }

    it = origin_weights_addr_.find(tensor);
    if (it != origin_weights_addr_.end()) {
      saved_weights_addr_.insert(std::make_pair(it->second, tensor));
      if (immutable) {
        malloc_weights_addr_.insert(std::make_pair(tensor, it->second));
      }
      return it->second;
    }
    MS_LOG(ERROR) << "uninitialized memory";
    return "";
  }

  std::map<Tensor *, std::string> tensors_map() const;

  /**
   * @return weight tensor map which
   */
  std::map<std::string, Tensor *> saved_weights() const { return saved_weights_addr_; }
  size_t total_buffer_size() const { return tensors_size_ + workspace_size_; }
  void enable_is_next() { is_next_ = true; }
  void *MallocWeightTensor(TypeId type_id, size_t size, MallocType type, const std::string &tensor_name = "");
  void MarkSharedWeight(const Tensor *src, void *pack_weight);
  void *GetSharedWeightAddr(const Tensor *src);

 private:
  int AssignTensors(const std::vector<std::unique_ptr<OperatorCoder>> &nodes);
  void AssignGraphInputs(const std::vector<Tensor *> &inputs);
  void AssignWorkspaces(void *addr, size_t size);
  void RecordOriginWeightsAddr(const std::vector<std::unique_ptr<OperatorCoder>> &nodes);
  void RecordTensorsAddr(const std::map<Tensor *, size_t> &offsets);

  MemoryAllocator() = default;
  ~MemoryAllocator() = default;

  std::map<void *, std::string> workspaces_addr_;
  size_t workspace_size_{0};
  size_t tensors_size_{0};
  size_t weight_index_{0};

  bool is_next_{false};
  size_t offset_{0};
  std::vector<void *> allocated_;
  std::map<std::string, Tensor *> saved_weights_addr_;
  std::map<Tensor *, std::string> origin_weights_addr_;
  std::map<Tensor *, std::string> malloc_weights_addr_;
  std::map<Tensor *, std::string> tensors_addr_;
  std::map<const Tensor *, void *> shared_pack_weights_;
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_MICRO_CODER_ALLOCATOR_ALLOCATOR_H_
