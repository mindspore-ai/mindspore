/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_LITE_ADAPTER_EXPANDERS_ACTIVATION_H_
#define MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_LITE_ADAPTER_EXPANDERS_ACTIVATION_H_

#include <memory>
#include <set>

#include "common/graph_kernel/expanders/utils.h"

namespace mindspore::graphkernel::expanders {
class CheckActivationType : public Validator {
 public:
  explicit CheckActivationType(const std::set<int64_t> &s) : activation_types_(s) {}
  explicit CheckActivationType(int64_t type) { activation_types_.insert(type); }
  ~CheckActivationType() = default;
  bool Check(const OpDesc &e) override {
    if (e.Attrs().count("activation_type") == 0) {
      return true;
    }
    auto activation_type = GetValue<int64_t>(e.Attrs()["activation_type"]);
    if (activation_types_.find(activation_type) == activation_types_.end()) {
      MS_LOG(INFO) << "Activation type " << activation_type << " not supported yet!";
      return false;
    }
    return true;
  }

 private:
  std::set<int64_t> activation_types_;
};

NodePtr GetActivationExpander(const inner::LiteGraph::GraphBuilder &gb, const NodePtrList &inputs,
                              int64_t activation_type);
}  // namespace mindspore::graphkernel::expanders
#endif  // MINDSPORE_CCSRC_COMMON_GRAPH_KERNEL_LITE_ADAPTER_EXPANDERS_ACTIVATION_H_
