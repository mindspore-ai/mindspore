/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_ASCEND_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_ASCEND_KERNEL_MOD_H_

#include <vector>
#include <memory>
#include <string>
#include "plugin/device/ascend/hal/device/ge_runtime/task_info.h"
#include "kernel/kernel.h"
#include "kernel/common_utils.h"
#ifndef ENABLE_SECURITY
#include "include/backend/debug/data_dump/dump_json_parser.h"
#endif

using TaskInfoPtr = std::shared_ptr<mindspore::ge::model_runner::TaskInfo>;
namespace mindspore {
namespace kernel {
constexpr uint64_t kOverflowAddrSize = 512;
class AscendKernelMod : public KernelMod {
 public:
  AscendKernelMod() = default;
  explicit AscendKernelMod(const AnfNodePtr &anf_node_ptr) : KernelMod(), anf_node_(anf_node_ptr) {}
  virtual std::vector<TaskInfoPtr> GenTask(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                           const std::vector<AddressPtr> &, uint32_t) = 0;
  uint32_t block_dim() const { return block_dim_; }
  uint32_t stream_id() const { return stream_id_; }
  void SetNode(const AnfNodePtr &anf_node_ptr) { anf_node_ = anf_node_ptr; }
  virtual bool NeedDump() {
#ifndef ENABLE_SECURITY
    const auto &dump_json = DumpJsonParser::GetInstance();
    return dump_json.NeedDump(fullname_) && dump_json.async_dump_enabled() && dump_json.op_debug_mode() == 0 &&
           !is_monad_;
#else
    return false;
#endif
  }
  bool IsNeedRetrieveOutputShape() override;
  void SetAtomicCleanNodes(const std::vector<CNodePtr> &atomic_clean_node);
  std::string GetAtomicCompileInfo() const { return atomic_compile_info_; }
  std::vector<KernelAttr> GetOpSupport() override { return {}; }

 protected:
  virtual void UpdateOutputSizeList();

  AnfNodeWeakPtr anf_node_;
  std::vector<CNodeWeakPtr> atomic_clean_nodes_;
  uint32_t block_dim_{1};
  uint32_t stream_id_{0};
  std::string atomic_compile_info_{};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_ASCEND_KERNEL_MOD_H_
