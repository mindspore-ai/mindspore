/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <tuple>
#include "ps/worker.h"
#include "ps/util.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class PushKernelMod : public NativeCpuKernelMod {
 public:
  PushKernelMod() : key_(UINT64_MAX) {}
  ~PushKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  void Init(const CNodePtr &kernel_node) override {
    auto kernel_attr = GetKernelAttrFromNode(kernel_node);
    auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
    if (!is_match) {
      MS_LOG(EXCEPTION) << "Push does not support this kernel data type: " << kernel_attr;
    }
    kernel_func_ = std::get<1>(func_list_[index]);
    const size_t kTwoIdx = 2;
    init_func_ = std::get<kTwoIdx>(func_list_[index]);

    init_func_(this, kernel_node);
  }

  void InitKernel(const CNodePtr &kernel_node) override { return; }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  void InitFunc(const CNodePtr &kernel_node) {
    key_ = common::AnfAlgo::GetNodeAttr<size_t>(kernel_node, kAttrPsKey);
    auto optim_input_shapes =
      common::AnfAlgo::GetNodeAttr<std::vector<std::vector<int64_t>>>(kernel_node, "optim_input_shapes");
    auto only_shape_indices = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "only_shape_indices");
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

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                    const std::vector<kernel::AddressPtr> &outputs) {
    if (outputs.size() != 1) {
      MS_LOG(EXCEPTION) << "Outputs size is " << outputs.size() << ", but PushKernelMod needs 1.";
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
    }
    return true;
  }

  using PushFunc =
    std::function<bool(PushKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  using PushInitFunc = std::function<void(PushKernelMod *, const CNodePtr &kernel_node)>;
  static std::vector<std::tuple<KernelAttr, PushFunc, PushInitFunc>> func_list_;
  PushFunc kernel_func_;
  PushInitFunc init_func_;

  size_t key_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_PS_PUSH_KERNEL_H_
