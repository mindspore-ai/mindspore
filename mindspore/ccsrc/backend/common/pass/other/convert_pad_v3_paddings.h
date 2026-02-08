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

#include <vector>
#include <string>
#include "include/backend/common/pass_manager/optimizer.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
class BACKEND_COMMON_EXPORT ConvertBasePaddings : public PatternProcessPass {
 public:
  explicit ConvertBasePaddings(const std::string &pass_name, bool multi_graph = true)
      : PatternProcessPass(pass_name, multi_graph) {}
  ~ConvertBasePaddings() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

  bool HasDynPaddings(const CNodePtr &) const;
  const CNodePtr CreateReshapeNode(const FuncGraphPtr &, const AnfNodePtr &, const ShapeVector &) const;
  const CNodePtr CreateStridedSliceNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node,
                                        int64_t index) const;
  const CNodePtr CreateConcatNode(const FuncGraphPtr &, const std::vector<AnfNodePtr> &, const AnfNodePtr &) const;
  const CNodePtr ProcessSliceNConcat(const FuncGraphPtr &, const AnfNodePtr &, const AnfNodePtr &, const int64_t &,
                                     const int64_t &) const;

  const AnfNodePtr CreateDynPaddingsPass(const FuncGraphPtr &, const CNodePtr &, const bool &) const;
  virtual const AnfNodePtr CreateDynPaddingsNode(const FuncGraphPtr &, const CNodePtr &) const { return nullptr; }

  template <typename T, TypeId type_id>
  const AnfNodePtr OptimizePaddingsValue(const FuncGraphPtr &, const AbstractBasePtr &, const bool &, const size_t &,
                                         bool force_length8) const;
  virtual const AnfNodePtr CreateConstPaddingsPass(const FuncGraphPtr &, const AbstractBasePtr &, const bool &,
                                                   const size_t &, const TypeId &) const {
    return nullptr;
  }
  const AnfNodePtr CreateConstPaddingsNode(const FuncGraphPtr &, const CNodePtr &) const;

 private:
  virtual bool ExpandInputXDims(const FuncGraphPtr &, const CNodePtr &) const { return false; }
  virtual void ReduceOutputDims(const FuncGraphPtr &, const CNodePtr &) const {}
};

class BACKEND_COMMON_EXPORT ConvertPadV3Paddings : public ConvertBasePaddings {
 public:
  explicit ConvertPadV3Paddings(bool multi_graph = true)
      : ConvertBasePaddings("convert_pad_v3_paddings", multi_graph) {}
  ~ConvertPadV3Paddings() override = default;
  const BaseRef DefinePattern() const override;

 private:
  const AnfNodePtr CreateConstPaddingsPass(const FuncGraphPtr &graph, const AbstractBasePtr &ori_paddings,
                                           const bool &paddings_contiguous, const size_t &dst_length,
                                           const TypeId &type_id) const override {
    if (type_id == kNumberTypeInt32) {
      return ConvertBasePaddings::OptimizePaddingsValue<int32_t, kNumberTypeInt32>(
        graph, ori_paddings, paddings_contiguous, dst_length, false);
    } else if (type_id == kNumberTypeInt64) {
      return ConvertBasePaddings::OptimizePaddingsValue<int64_t, kNumberTypeInt64>(
        graph, ori_paddings, paddings_contiguous, dst_length, false);
    } else {
      MS_LOG_EXCEPTION << "Unsupported data type for PadV3 paddings input.";
    }
  }
  const AnfNodePtr CreateDynPaddingsNode(const FuncGraphPtr &graph, const CNodePtr &pad_node) const override {
    return ConvertBasePaddings::CreateDynPaddingsPass(graph, pad_node, false);
  }
  bool ExpandInputXDims(const FuncGraphPtr &, const CNodePtr &) const override { return false; }
  void ReduceOutputDims(const FuncGraphPtr &, const CNodePtr &) const override {}
  std::vector<std::string> MustExistPrimitiveName() const override;
};

class BACKEND_COMMON_EXPORT ConvertPadV3GradPaddings : public ConvertBasePaddings {
 public:
  explicit ConvertPadV3GradPaddings(bool multi_graph = true)
      : ConvertBasePaddings("convert_pad_v3_grad_paddings", multi_graph) {}
  ~ConvertPadV3GradPaddings() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  const AnfNodePtr CreateConstPaddingsPass(const FuncGraphPtr &graph, const AbstractBasePtr &ori_paddings,
                                           const bool &paddings_contiguous, const size_t &dst_length,
                                           const TypeId &type_id) const override {
    if (type_id == kNumberTypeInt32) {
      return ConvertBasePaddings::OptimizePaddingsValue<int32_t, kNumberTypeInt32>(
        graph, ori_paddings, paddings_contiguous, dst_length, true);
    } else if (type_id == kNumberTypeInt64) {
      return ConvertBasePaddings::OptimizePaddingsValue<int64_t, kNumberTypeInt64>(
        graph, ori_paddings, paddings_contiguous, dst_length, true);
    } else {
      MS_LOG_EXCEPTION << "Unsupported data type for PadV3Grad paddings input.";
    }
  }
  const AnfNodePtr CreateDynPaddingsNode(const FuncGraphPtr &graph, const CNodePtr &pad_node) const override {
    return ConvertBasePaddings::CreateDynPaddingsPass(graph, pad_node, true);
  }
  bool ExpandInputXDims(const FuncGraphPtr &, const CNodePtr &) const override;
  void ReduceOutputDims(const FuncGraphPtr &, const CNodePtr &) const override;
  std::vector<std::string> MustExistPrimitiveName() const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_OPTIMIZER_CONVERT_PAD_V3_PADDINGS_H_
