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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_VMAP_H
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_VMAP_H

#include <memory>
#include <string>
#include <vector>
#include "ir/meta_func_graph.h"
#include "include/frontend/operator/primitive_py.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
using CNodeInpusList = std::vector<std::vector<AnfNodePtr>>;
using InputsAbstractList = std::vector<std::vector<abstract::AbstractBasePtr>>;
constexpr int64_t kValIndex = 0;
constexpr int64_t kDimIndex = 1;
constexpr char kVmapFunctionModelName[] = "mindspore.ops._vmap";
constexpr char kNumpyModelName[] = "mindspore.numpy";
class VmapMatchOutAxis : public MetaFuncGraph {
 public:
  explicit VmapMatchOutAxis(const std::string &name) : MetaFuncGraph(name), fg_(std::make_shared<FuncGraph>()) {}
  ~VmapMatchOutAxis() override = default;
  MS_DECLARE_PARENT(VmapMatchOutAxis, MetaFuncGraph)

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;

 private:
  CNodePtr GenerateFuncGraphInnerBroadcastAxis(const AnfNodePtr &inputs, const AnfNodePtr &out_axis,
                                               const AnfNodePtr &axis_size,
                                               const AbstractBasePtr &inputs_abstract_elements_begin) const;
  CNodePtr GenerateFuncGraphInnerSingleElement(const AnfNodePtr &inputs, const AnfNodePtr &out_axis,
                                               const AnfNodePtr &axis_size,
                                               const AbstractBasePtr &inputs_abstract_elements_end) const;
  CNodePtr GenerateFuncGraphInnerAllTuple(const AnfNodePtr &inputs, const AnfNodePtr &out_axis,
                                          const AnfNodePtr &axis_size,
                                          const AbstractBasePtrList &inputs_abstract_elements,
                                          const AbstractBasePtr &out_axes_abstract) const;
  FuncGraphPtr fg_{nullptr};
};

class VmapGeneralPreprocess : public MetaFuncGraph {
 public:
  explicit VmapGeneralPreprocess(const std::string &name) : MetaFuncGraph(name) {}
  ~VmapGeneralPreprocess() override = default;
  MS_DECLARE_PARENT(VmapGeneralPreprocess, MetaFuncGraph);

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;
};

class VmapGeneralRule : public MetaFuncGraph {
 public:
  explicit VmapGeneralRule(const std::string &name, const PrimitivePtr &prim, int64_t axis_size)
      : MetaFuncGraph(name), prim_(prim), axis_size_(axis_size) {}
  ~VmapGeneralRule() override = default;
  MS_DECLARE_PARENT(VmapGeneralRule, MetaFuncGraph);

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_abs_list) override;

  std::string prim_name() const {
    if (!prim_) {
      return "";
    }
    return prim_->name();
  }

  int64_t axis_size() const { return axis_size_; }

 private:
  CNodeInpusList ConstructMapInput(const InputsAbstractList &unfold_elements_abstract, int64_t args_size,
                                   int64_t tuple_elements_num);
  PrimitivePtr prim_{nullptr};
  int64_t axis_size_ = 0;
  FuncGraphPtr fg_{nullptr};
};
using VmapGeneralRulePtr = std::shared_ptr<VmapGeneralRule>;

class VmapGeneralRulePyAdapter : public VmapGeneralRule {
 public:
  explicit VmapGeneralRulePyAdapter(const std::string &name, const PrimitivePyAdapterPtr &prim, int64_t axis_size)
      : VmapGeneralRule(name, prim->attached_primitive(), axis_size) {}
  ~VmapGeneralRulePyAdapter() override = default;
};
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_VMAP_H
