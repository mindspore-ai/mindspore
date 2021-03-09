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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_PS_PUSH_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_PS_PUSH_KERNEL_H_

#include <vector>
#include <algorithm>
#include "ps/worker.h"
#include "ps/util.h"
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class PushKernel : public CPUKernel {
 public:
  PushKernel() : key_(UINT64_MAX) {}
  ~PushKernel() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) {
    if (outputs.size() != 1) {
      MS_LOG(EXCEPTION) << "Outputs size is " << outputs.size() << ", but PushKernel needs 1.";
    }
    std::vector<size_t> keys;
    std::vector<uintptr_t> addrs;
    std::vector<int64_t> sizes;
    for (auto input : inputs) {
      keys.push_back(key_);
      addrs.push_back(reinterpret_cast<uintptr_t>(input->addr));
      sizes.push_back(SizeToLong(input->size) / sizeof(T));
    }
    mindspore::ps::Worker::GetInstance().Push(keys, addrs, sizes);
    auto ret = memcpy_s(outputs[0]->addr, outputs[0]->size, &key_, sizeof(size_t));
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "Lookup id memcpy failed.";
      return false;
    }
    return true;
  }

  void Init(const CNodePtr &kernel_node) {
    key_ = AnfAlgo::GetNodeAttr<size_t>(kernel_node, kAttrPsKey);
    auto optim_input_shapes =
      AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(kernel_node, "optim_input_shapes");
    auto only_shape_indices = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "only_shape_indices");
    MS_LOG(INFO) << "Key " << key_ << " optimizer input shapes are:" << optim_input_shapes;
    MS_LOG(INFO) << "Only init shape indices are " << only_shape_indices;
    for (size_t i = 0; i < optim_input_shapes.size(); i++) {
      auto shape = optim_input_shapes[i];
      mindspore::ps::Worker::GetInstance().SetOptimInputShapes(key_, shape);
      if (std::count(only_shape_indices.begin(), only_shape_indices.end(), i) == 0) {
        size_t size = sizeof(T);
        for (size_t j = 0; j < shape.size(); j++) {
          size *= LongToSize(shape[j]);
        }
        input_size_list_.push_back(size);
      }
    }

    output_size_list_.push_back(sizeof(size_t));
    return;
  }

  void InitKernel(const CNodePtr &kernel_node) { return; }

 private:
  size_t key_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_PS_PUSH_KERNEL_H_
