/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_CONVERT_PAD_V3_PADDINGS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_CONVERT_PAD_V3_PADDINGS_H_

#include <string>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class ConvertBasePaddings : public PatternProcessPass {
 public:
  explicit ConvertBasePaddings(const std::string &pass_name, bool multi_graph = true)
      : PatternProcessPass(pass_name, multi_graph) {}
  ~ConvertBasePaddings() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

  bool HasDynPaddings(const CNodePtr &) const;
  template <typename T, TypeId type_id>
  const AnfNodePtr OptimizePaddingsValue(const FuncGraphPtr &, const AbstractBasePtr &, const size_t &,
                                         bool force_length8) const;
  virtual const AnfNodePtr CreatePaddingsNode(const FuncGraphPtr &, const AbstractBasePtr &, const size_t &,
                                              const TypeId &) const {
    return nullptr;
  }
  virtual bool ExpandInputXDims(const FuncGraphPtr &, const CNodePtr &) const { return false; }
  virtual void ReduceOutputDims(const FuncGraphPtr &, const CNodePtr &) const {}
};

class ConvertPadV3Paddings : public ConvertBasePaddings {
 public:
  explicit ConvertPadV3Paddings(bool multi_graph = true)
      : ConvertBasePaddings("convert_pad_v3_paddings", multi_graph) {}
  ~ConvertPadV3Paddings() override = default;
  const BaseRef DefinePattern() const override;

 private:
  const AnfNodePtr CreatePaddingsNode(const FuncGraphPtr &graph, const AbstractBasePtr &ori_paddings,
                                      const size_t &dst_length, const TypeId &type_id) const override {
    if (type_id == kNumberTypeInt32) {
      return ConvertBasePaddings::OptimizePaddingsValue<int32_t, kNumberTypeInt32>(graph, ori_paddings, dst_length,
                                                                                   false);
    }
    return ConvertBasePaddings::OptimizePaddingsValue<int64_t, kNumberTypeInt64>(graph, ori_paddings, dst_length,
                                                                                 false);
  }
  bool ExpandInputXDims(const FuncGraphPtr &, const CNodePtr &) const override { return false; }
  void ReduceOutputDims(const FuncGraphPtr &, const CNodePtr &) const override {}
};

class ConvertPadV3GradPaddings : public ConvertBasePaddings {
 public:
  explicit ConvertPadV3GradPaddings(bool multi_graph = true)
      : ConvertBasePaddings("convert_pad_v3_grad_paddings", multi_graph) {}
  ~ConvertPadV3GradPaddings() override = default;
  const BaseRef DefinePattern() const override;

 private:
  const AnfNodePtr CreatePaddingsNode(const FuncGraphPtr &graph, const AbstractBasePtr &ori_paddings,
                                      const size_t &dst_length, const TypeId &type_id) const override {
    if (type_id == kNumberTypeInt32) {
      return ConvertBasePaddings::OptimizePaddingsValue<int32_t, kNumberTypeInt32>(graph, ori_paddings, dst_length,
                                                                                   true);
    }
    return ConvertBasePaddings::OptimizePaddingsValue<int64_t, kNumberTypeInt64>(graph, ori_paddings, dst_length, true);
  }
  bool ExpandInputXDims(const FuncGraphPtr &, const CNodePtr &) const override;
  void ReduceOutputDims(const FuncGraphPtr &, const CNodePtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_CONVERT_PAD_V3_PADDINGS_H_
