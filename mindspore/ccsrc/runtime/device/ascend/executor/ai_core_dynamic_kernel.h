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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_AI_CORE_DYNAMIC_KERNEL_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_AI_CORE_DYNAMIC_KERNEL_H_

#include <vector>
#include <map>
#include <string>
#include <memory>
#include "nlohmann/json.hpp"
#include "ir/tensor.h"
#include "runtime/device/device_address.h"
#include "register/op_tiling.h"
#include "mindspore/ccsrc/runtime/device/executor/dynamic_kernel.h"

namespace mindspore {
namespace device {
namespace ascend {
class AiCoreDynamicKernel : public DynamicKernel {
 public:
  AiCoreDynamicKernel(const void *stub_fubc, uint32_t block_dim, void *tiling_data_ptr, uint32_t op_para_size,
                      void *stream, const CNodePtr &cnode_ptr, const std::vector<void *> &runtime_args)
      : DynamicKernel(stream, cnode_ptr),
        stub_func_(stub_fubc),
        block_dim_(block_dim),
        tiling_data_ptr_(tiling_data_ptr),
        op_para_size_(op_para_size),
        runtime_args_(runtime_args) {}
  AiCoreDynamicKernel(void *handle, uint32_t block_dim, void *tiling_data_ptr, uint32_t op_para_size, void *stream,
                      const CNodePtr &cnode_ptr, const std::vector<void *> &runtime_args, const std::string &ori_key)
      : DynamicKernel(stream, cnode_ptr),
        handle_(handle),
        block_dim_(block_dim),
        tiling_data_ptr_(tiling_data_ptr),
        op_para_size_(op_para_size),
        runtime_args_(runtime_args),
        origin_key_(ori_key) {}
  ~AiCoreDynamicKernel() override;

  void Execute() override;
  void UpdateArgs() override;
  void Initialize() override;
  void PostExecute() override;

 protected:
  void AllocateWorkspace();
  void ParseCompileJson();

 private:
  const void *stub_func_{nullptr};
  void *handle_{nullptr};
  uint32_t block_dim_;
  void *tiling_data_ptr_;  // device ptr
  uint32_t op_para_size_;  // size of tiling_data_ptr_
  std::vector<void *> runtime_args_;
  std::string tiling_data_;
  std::vector<int64_t> workspaces_size_;
  std::vector<DeviceAddressPtr> workspace_addr_;
  std::shared_ptr<nlohmann::json> compile_info_json_;
  optiling::OpCompileInfo op_compile_info_{};
  uint32_t tiling_key_{0};
  const std::string origin_key_{""};

  void ComputeTiling();
  bool CopyTilingToDevice();
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_EXECUTOR_AI_CORE_DYNAMIC_KERNEL_H_
