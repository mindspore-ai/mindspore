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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_DENSE_GRAD_ACCUM_KERNEL_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_DENSE_GRAD_ACCUM_KERNEL_H_

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "fl/server/kernel/aggregation_kernel.h"
#include "fl/server/kernel/aggregation_kernel_factory.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
constexpr size_t kDenseGradAccumKernelInputsNum = 2;
template <typename T>
class DenseGradAccumKernel : public AggregationKernel {
 public:
  DenseGradAccumKernel() = default;
  ~DenseGradAccumKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override {
    MS_EXCEPTION_IF_NULL(kernel_node);
    std::string cnode_name = AnfAlgo::GetCNodeName(kernel_node);
    if (kNameToIdxMap.count(cnode_name) == 0 || kNameToIdxMap.at(cnode_name).count("inputs") == 0 ||
        kNameToIdxMap.at(cnode_name).at("inputs").count("grad") == 0) {
      MS_LOG(EXCEPTION) << "Can't find index info of grad for kernel " << cnode_name;
      return;
    }
    size_t cnode_grad_idx = kNameToIdxMap.at(cnode_name).at("inputs").at("grad");
    std::vector<size_t> grad_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, cnode_grad_idx);
    size_t grad_size = std::accumulate(grad_shape.begin(), grad_shape.end(), sizeof(T), std::multiplies<size_t>());
    input_size_list_.push_back(grad_size);
    size_t new_grad_size = grad_size;
    input_size_list_.push_back(new_grad_size);
    GenerateReuseKernelNodeInfo();
    return;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &) override {
    if (inputs.size() != kDenseGradAccumKernelInputsNum) {
      MS_LOG(ERROR) << "The inputs number of DenseGradAccumKernel should be 2, but got " << inputs.size();
      return false;
    }
    MS_ERROR_IF_NULL_W_RET_VAL(inputs[0], false);
    MS_ERROR_IF_NULL_W_RET_VAL(inputs[1], false);
    MS_ERROR_IF_NULL_W_RET_VAL(inputs[0]->addr, false);
    MS_ERROR_IF_NULL_W_RET_VAL(inputs[1]->addr, false);

    if (accum_count_ == 0) {
      int ret = memset_s(inputs[0]->addr, inputs[0]->size, 0x00, inputs[0]->size);
      if (ret != 0) {
        MS_LOG(ERROR) << "memset_s error, errorno(" << ret << ")";
        return false;
      }
    }

    T *grad_addr = reinterpret_cast<T *>(inputs[0]->addr);
    T *new_grad_addr = reinterpret_cast<T *>(inputs[1]->addr);
    for (size_t i = 0; i < inputs[0]->size / sizeof(T); i++) {
      grad_addr[i] += new_grad_addr[i];
    }

    accum_count_++;
    if (accum_count_ > done_count_) {
      MS_LOG(ERROR) << "accum_count_ should not be greater than done_count_ " << done_count_;
      return false;
    }
    if (accum_count_ == done_count_) {
      for (size_t i = 0; i < inputs[0]->size / sizeof(T); i++) {
        grad_addr[i] /= done_count_;
      }
    }
    return true;
  }

  void Reset() { accum_count_ = 0; }

  bool IsAggregationDone() { return accum_count_ >= done_count_; }

  void GenerateReuseKernelNodeInfo() override { return; }
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_DENSE_GRAD_ACCUM_KERNEL_H_
