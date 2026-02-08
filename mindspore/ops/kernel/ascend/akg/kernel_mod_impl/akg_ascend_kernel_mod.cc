/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/akg/kernel_mod_impl/akg_ascend_kernel_mod.h"
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>
#include "utils/log_adapter.h"
#include "acl/acl_rt.h"

namespace mindspore {
namespace kernel {
using std::fstream;
using std::map;
using std::mutex;
using std::string;
constexpr uint32_t DEFAULT_BLOCK_DIM = 1;
constexpr size_t ARGS_REMAP_LEN = 2;
/**
 * @brief infotable contain func_stub\blockdim\kernel file buffer
 */
AkgKernelMod::AkgKernelMod(const KernelPackPtr &kernel_pack, const AnfNodePtr &anf_node_ptr)
    : kernel_pack_(kernel_pack) {
  if (kernel_pack != nullptr) {
    auto kernel_json_info = kernel_pack->kernel_json_info();
    kernel_name_ = kernel_json_info.kernel_name;
    args_remap_ = kernel_json_info.args_remap;
  }
}

AkgKernelManagerPtr AkgKernelMod::kernel_manager_ = std::make_shared<AkgKernelManager>();

bool AkgKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (stream_ptr == nullptr) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr. Kernel name: " << kernel_name_;
    return false;
  }

  if (kernel_pack_ == nullptr) {
    MS_LOG(ERROR) << "kernel pack should not be nullptr. Kernel name: " << kernel_name_;
    return false;
  }
  uint32_t block_dim = DEFAULT_BLOCK_DIM;  // default blockdim equal to 1.
  auto func_stub = kernel_manager_->GenFuncStub(*kernel_pack_, false, &block_dim, nullptr);
  if (!func_stub) {
    MS_LOG(ERROR) << "GenFuncStub failed. Kernel name: " << kernel_name_;
    return false;
  }

  // pack all addresses into a vector.
  std::vector<void *> runtime_args;
  if (args_remap_.size() == ARGS_REMAP_LEN) {
    for (const auto &idx : args_remap_[0]) {
      if (idx >= inputs.size()) {
        MS_LOG(ERROR) << "Input index must be in range [0, " << inputs.size() << "), but got " << idx;
        return false;
      }
      runtime_args.push_back(inputs[idx]->device_ptr());
    }
    auto io_size = inputs.size() + outputs.size();
    for (const auto &idx : args_remap_[1]) {
      if (idx >= io_size) {
        MS_LOG(ERROR) << "Output index must be in range [0, " << io_size << "), but got " << idx;
        return false;
      }
      if (idx < inputs.size()) {
        runtime_args.push_back(inputs[idx]->device_ptr());
      } else {
        runtime_args.push_back(outputs[idx - inputs.size()]->device_ptr());
      }
    }
  } else {
    (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(runtime_args),
                         [](const KernelTensor *input) { return input->device_ptr(); });
    (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(runtime_args),
                         [](const KernelTensor *output) { return output->device_ptr(); });
  }
  if (!workspace.empty()) {
    (void)std::transform(std::begin(workspace), std::end(workspace), std::back_inserter(runtime_args),
                         [](const KernelTensor *addr) { return addr->device_ptr(); });
  }

  auto stream = static_cast<aclrtStream *>(stream_ptr);
  typedef void (*CallFunc)(uint32_t, void *, void *, void **);
  auto func_ptr = reinterpret_cast<CallFunc>(func_stub);
  func_ptr(block_dim, nullptr, stream, runtime_args.data());

  return true;
}

std::vector<size_t> AkgKernelMod::GenParameters() {
  auto kernel_json_info = kernel_pack_->kernel_json_info();
  return kernel_json_info.parameters;
}
}  // namespace kernel
}  // namespace mindspore
