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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_HCCL_DYNAMIC_KERNEL_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_HCCL_DYNAMIC_KERNEL_H_

#include <condition_variable>
#include <string>
#include "runtime/device/executor/dynamic_kernel.h"

#include "utils/ms_utils.h"

namespace mindspore {
namespace device {
namespace ascend {
class HcclDynamicKernel : public DynamicKernel {
 public:
  HcclDynamicKernel(const std::string &hccl_type, void *input_ptr, void *output_ptr, uint64_t count, int32_t data_type,
                    int32_t op_type, int32_t root, void *stream, const CNodePtr &cnode_ptr)
      : DynamicKernel(stream, cnode_ptr),
        hccl_type_(hccl_type),
        input_ptr_(input_ptr),
        output_ptr_(output_ptr),
        count_(count),
        data_type_(data_type),
        op_type_(op_type),
        root_(root) {}
  ~HcclDynamicKernel() override = default;
  void UpdateArgs() override;
  void Execute() override;
  void PostExecute() override;

 private:
  std::string hccl_type_;
  void *input_ptr_;
  void *output_ptr_;
  uint64_t count_{0};
  int32_t data_type_{0};
  int32_t op_type_{0};
  int32_t root_{0};
  std::mutex hccl_mutex_;
  std::condition_variable cond_;

  void StaticShapeExecute();
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_HCCL_DYNAMIC_KERNEL_H_
