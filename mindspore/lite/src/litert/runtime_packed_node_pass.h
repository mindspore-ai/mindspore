/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_LITERT_RUNTIME_PACKED_NODE_PASS_
#define MINDSPORE_LITE_SRC_LITERT_RUNTIME_PACKED_NODE_PASS_

#include <string>
#include <map>
#include <vector>
#include "src/litert/lite_model.h"
#include "src/tensor.h"
#include "src/litert/kernel_exec.h"

namespace mindspore {
namespace lite {
struct PackInfo {
  bool is_packed_{false};
  int weight_sums_index_{-1};
  int b_batch_;
  int deep_;
  int col_;
  int deep_align_;
  int col_align_;
  bool b_transpose_{false};
  std::string cpu_option_;
};

class PackedNodePass {
 public:
  static PackedNodePass &GetInstance() {
    static PackedNodePass instance;
    return instance;
  }

  PackInfo *GetNodePackInfo(const std::string &node_name) {
    if (this->node_pack_info_map_.find(node_name) == this->node_pack_info_map_.end()) {
      return nullptr;
    }
    return this->node_pack_info_map_[node_name];
  }
  void Run(Model *model, const std::vector<Tensor *> &tensors);
  void CopyWeightBiasSumsTensor(Tensor *tensor);

 protected:
  void AddNodePackInfo(const std::string &node_name, PackInfo *pack_info) {
    if (this->node_pack_info_map_.find(node_name) != this->node_pack_info_map_.end()) {
      MS_LOG(WARNING) << "Key conflict when add weight sums index.";
    }
    this->node_pack_info_map_[node_name] = pack_info;
  }

 private:
  PackedNodePass() = default;
  ~PackedNodePass();

 private:
  std::map<std::string, PackInfo *> node_pack_info_map_;
};

int PackKernelExec(kernel::KernelExec *kernel_exec, const std::vector<Tensor *> &tensors);

// packed weight data -> unpack
int RecoveryPackedWeight(Tensor *weight, const int quant_type, const TypeId data_type, const int node_type,
                         PackInfo *packInfo);
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_LITERT_RUNTIME_PACKED_NODE_PASS_
