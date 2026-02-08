/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DEVICE_ASCEND_ATB_ADAPTER_H_
#define MINDSPORE_CCSRC_DEVICE_ASCEND_ATB_ADAPTER_H_

#include <string>
#include <vector>
#include <utility>
#include "kernel/ascend/acl_ir/op_api_cache.h"
#include "atb/types.h"
#include "atb/operation.h"
#include "atb/utils.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore::device::ascend {
atb::Context *GetAtbContext(const aclrtStream &stream);
uint64_t GetWorkSpaceSize(atb::Operation *operation, atb::VariantPack variant_pack, aclrtStream stream);
void Launch(atb::Operation *operation, atb::VariantPack variant_pack, void *workspace_ptr,
            std::vector<size_t> workpsace_size_list, aclrtStream stream);
void Launch(atb::Operation *op, atb::VariantPack variant_pack, void *workspace_ptr, size_t workpsace_size,
            aclrtStream stream);

template <typename... Args>
uint64_t AtbHash(const Args &... args) {
  g_hash_offset = 0;
  GatherHash(args...);
  return calc_hash_id();
}

class ParamSetter {
 public:
  ParamSetter &Input(mindspore::kernel::KernelTensor *kernel_tensor);
  ParamSetter &Input(std::optional<mindspore::kernel::KernelTensor *> kernel_tensor);
  ParamSetter &Input(const tensor::TensorPtr &base_tensor);
  ParamSetter &Input(const std::optional<tensor::TensorPtr> &base_tensor);
  ParamSetter &Output(mindspore::kernel::KernelTensor *kernel_tensor);
  ParamSetter &Output(std::optional<mindspore::kernel::KernelTensor *> kernel_tensor);
  ParamSetter &Output(const tensor::TensorPtr &base_tensor);
  ParamSetter &Output(const std::optional<tensor::TensorPtr> &base_tensor);

  void Clear() {
    variant_pack.inTensors.clear();
    variant_pack.outTensors.clear();
  }

  ParamSetter &SetIndex(const std::vector<size_t> inputs, const std::vector<size_t> outputs) {
    input_ids = std::move(inputs);
    output_ids = std::move(outputs);
    variant_pack.inTensors.clear();
    variant_pack.outTensors.clear();
    return *this;
  }
  void Update(const std::vector<mindspore::kernel::KernelTensor *> &inputs,
              const std::vector<mindspore::kernel::KernelTensor *> &outputs);
  std::vector<size_t> input_ids;
  std::vector<size_t> output_ids;
  atb::VariantPack variant_pack;
  aclrtStream stream{nullptr};
};

class AtbContextManager {
 public:
  static AtbContextManager &GetInstance();
  atb::Context *GetContext(const aclrtStream &stream);
  ~AtbContextManager();

 private:
  AtbContextManager() = default;
  mindspore::HashMap<aclrtStream, atb::Context *> context_map_{};
};

}  // namespace mindspore::device::ascend

#endif  // MINDSPORE_CCSRC_DEVICE_ASCEND_ATB_ADAPTER_H_
