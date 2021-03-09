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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_PS_PULL_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_PS_PULL_KERNEL_H_

#include <vector>
#include <string>
#include "ps/worker.h"
#include "ps/util.h"
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class PullKernel : public CPUKernel {
 public:
  PullKernel() : key_(UINT64_MAX), keys_size_(sizeof(size_t)), var_size_(sizeof(size_t)) {}
  ~PullKernel() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &) {
    if (inputs.size() != 2) {
      MS_LOG(EXCEPTION) << "Inputs size is " << inputs.size() << ", but PullKernel needs 2.";
    }
    bool init_in_server = mindspore::ps::Worker::GetInstance().GetParamInitInServer(param_name_);
    // If init_in_server, forward kernel should run in server too.
    if (!init_in_server) {
      mindspore::ps::Worker::GetInstance().Pull(key_, inputs[1]->addr, inputs[1]->size);
    }
    return true;
  }
  void Init(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but pull needs 2 inputs.";
      return;
    }

    auto key_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < key_shape.size(); i++) {
      keys_size_ *= key_shape[i];
    }
    auto var_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    for (size_t i = 0; i < var_shape.size(); i++) {
      var_size_ *= var_shape[i];
    }
    auto param_node = AnfAlgo::GetInputNode(kernel_node, 1);
    MS_EXCEPTION_IF_NULL(param_node);
    param_name_ = param_node->fullname_with_scope();

    if (mindspore::ps::PSContext::instance()->is_worker()) {
      key_ = AnfAlgo::GetNodeAttr<size_t>(kernel_node, kAttrPsKey);
    }
    InitSizeLists();
    return;
  }
  void InitKernel(const CNodePtr &kernel_node) { return; }

 protected:
  void InitSizeLists() {
    input_size_list_.push_back(keys_size_);
    input_size_list_.push_back(var_size_);
    output_size_list_.push_back(0);
  }

 private:
  size_t key_;
  size_t keys_size_;
  size_t var_size_;
  std::string param_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_PS_PULL_KERNEL_H_
