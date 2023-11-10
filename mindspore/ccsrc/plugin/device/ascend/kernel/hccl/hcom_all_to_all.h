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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCOM_ALL_TO_ALL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCOM_ALL_TO_ALL_H_

#include <memory>
#include <vector>
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/kernel/hccl/hccl_kernel.h"

namespace mindspore::kernel {
class HcomAllToAllKernel : public HcclKernel {
 public:
  HcomAllToAllKernel() = default;
  ~HcomAllToAllKernel() override = default;
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

 protected:
  HcclDataType GetHcclDataType() const override { return data_type_; }
  void CalLoopSize() override {}

 private:
  HcclDataType data_type_ = {};
  bool need_drop_input_ = false;
  hccl::HcclAllToAllVParams params_ = {};
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_HCCL_HCOM_ALL_TO_ALL_H_
