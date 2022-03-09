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

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
class VmapMatchOutAxis : public MetaFuncGraph {
 public:
  explicit VmapMatchOutAxis(const std::string &name) : MetaFuncGraph(name), fg_(std::make_shared<FuncGraph>()) {}
  ~VmapMatchOutAxis() override = default;
  MS_DECLARE_PARENT(VmapMatchOutAxis, MetaFuncGraph)

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;

 private:
  CNodePtr GenerateFuncGraphInnerBroadcastAxis(const AnfNodePtr &inputs, const AnfNodePtr &out_axis,
                                               const AnfNodePtr &axis_size,
                                               const AbstractBasePtr &abstract_elements_begin) const;
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

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
};
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_VMAP_H
