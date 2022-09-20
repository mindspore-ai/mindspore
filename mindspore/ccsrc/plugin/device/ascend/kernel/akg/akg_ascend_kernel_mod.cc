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

#include "plugin/device/ascend/kernel/akg/akg_ascend_kernel_mod.h"
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>
#include "runtime/rt.h"
#include "utils/log_adapter.h"
#include "include/common/utils/convert_utils.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"

namespace mindspore {
namespace kernel {
using std::fstream;
using std::map;
using std::mutex;
using std::string;
using TbeTaskInfoPtr = std::shared_ptr<mindspore::ge::model_runner::TbeTaskInfo>;
using tbe::KernelManager;
constexpr uint32_t DEFAULT_BLOCK_DIM = 1;
/**
 * @brief infotable contain func_stub\blockdim\kernel file buffer
 */
AkgKernelMod::AkgKernelMod(const KernelPackPtr &kernel_pack) : kernel_pack_(kernel_pack) {
  if (kernel_pack != nullptr) {
    auto kernel_json_info = kernel_pack->kernel_json_info();
    kernel_name_ = kernel_json_info.kernel_name;
  }
}

bool AkgKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                          const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (stream_ptr == nullptr) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr. Kernel name: " << kernel_name_;
    return false;
  }

  if (kernel_pack_ == nullptr) {
    MS_LOG(ERROR) << "kernel pack should not be nullptr. Kernel name: " << kernel_name_;
    return false;
  }
  uint32_t block_dim = DEFAULT_BLOCK_DIM;  // default blockdim equal to 1.
  auto func_stub = KernelManager::GenFuncStub(*kernel_pack_, false, &block_dim);
  if (func_stub == 0) {
    MS_LOG(ERROR) << "GenFuncStub failed. Kernel name: " << kernel_name_;
    return false;
  }

  // pack all addresses into a vector.
  std::vector<void *> runtime_args;
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(runtime_args),
                       [](const AddressPtr &input) { return input->addr; });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(runtime_args),
                       [](const AddressPtr &output) { return output->addr; });
  if (!workspace.empty()) {
    (void)std::transform(std::begin(workspace), std::end(workspace), std::back_inserter(runtime_args),
                         [](const AddressPtr &addr) { return addr->addr; });
  }

  rtL2Ctrl_t *l2ctrl = nullptr;
  auto stream = static_cast<rtStream_t *>(stream_ptr);
  auto ret = rtKernelLaunch(reinterpret_cast<void *>(func_stub), block_dim, runtime_args.data(),
                            SizeToUint(sizeof(void *) * runtime_args.size()), l2ctrl, stream);
  if (ret != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Call runtime rtKernelLaunch error. Kernel name: " << kernel_name_
                  << ". Error message: " << device::ascend::GetErrorMsg(static_cast<uint32_t>(ret));
    return false;
  }

  return true;
}

std::vector<TaskInfoPtr> AkgKernelMod::GenTask(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &workspace,
                                               const std::vector<AddressPtr> &outputs, uint32_t stream_id) {
  if (kernel_pack_ == nullptr) {
    MS_LOG(EXCEPTION) << "kernel pack should not be nullptr. Kernel name: " << kernel_name_;
  }

  std::vector<uint8_t> args;
  const uint32_t args_size = 0;
  std::vector<uint8_t> sm_desc;
  void *binary = nullptr;
  const uint32_t binary_size = 0;
  std::vector<uint8_t> meta_data;
  std::vector<void *> input_data_addrs;
  std::vector<void *> output_data_addrs;
  std::vector<void *> workspace_addrs;

  // pack all addresses into a vector.
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(input_data_addrs),
                       [](const AddressPtr &input) { return input->addr; });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(output_data_addrs),
                       [](const AddressPtr &output) { return output->addr; });
  if (!workspace.empty()) {
    (void)std::transform(std::begin(workspace), std::end(workspace), std::back_inserter(workspace_addrs),
                         [](const AddressPtr &workspace) { return workspace->addr; });
  }

  uint32_t block_dim = DEFAULT_BLOCK_DIM;  // default blockdim equal to 1.
  auto func_stub = KernelManager::GenFuncStub(*kernel_pack_, false, &block_dim);
  if (func_stub == 0) {
    MS_LOG(ERROR) << "GenFuncStub failed. Kernel name: " << kernel_name_;
  }

  std::string stub_func = KernelManager::GetStubFuncName(kernel_pack_);

  MS_LOG(DEBUG) << "The block_dim is:" << block_dim;

  TbeTaskInfoPtr task_info_ptr = std::make_shared<mindspore::ge::model_runner::TbeTaskInfo>(
    unique_name_, stream_id, stub_func, block_dim, args, args_size, sm_desc, binary, binary_size, meta_data,
    input_data_addrs, output_data_addrs, workspace_addrs, NeedDump());
  return {task_info_ptr};
}

std::vector<size_t> AkgKernelMod::GenParameters() {
  auto kernel_json_info = kernel_pack_->kernel_json_info();
  return kernel_json_info.parameters;
}
}  // namespace kernel
}  // namespace mindspore
