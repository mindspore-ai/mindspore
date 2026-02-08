/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_PASS_ADD_ATTR_TO_NODE_ADD_ATTR_TO_NODE_REGISTER_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_PASS_ADD_ATTR_TO_NODE_ADD_ATTR_TO_NODE_REGISTER_H_

#include <string>

#include "base/base.h"
#include "ops_utils/op_utils.h"
#include "utils/hash_map.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace opt {
const AnfNodePtr AccumulateNV2FusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr AddDropoutAttrs(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr AddNFusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr BatchMatMulAttrFusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr ConcatOffsetV1FusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr Conv3DBackpropInputPadListFusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr Conv3DBackpropFilterPadListFusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr DynamicRNNFusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr GatherFusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr Im2ColFusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr IOUFusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr LogFusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr MaxPoolWithArgmaxV2FusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr ParallelConcatFusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr RaggedTensorToSparseFusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr ResizeV2FusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr SqueezeAxis(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr SparseConcatFusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr SparseTensorDenseMatMulFusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr SplitFusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr StandardNormalFusionProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr UniformRealDtypeGe(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr TensorShapeAddDtype(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr AddOnlyDependShapeAttr(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr TensorShapeProcess(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr HShrinkModifyLambd(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr ExtractVolumePatchesFormatTranspose(const FuncGraphPtr &, const AnfNodePtr &);
const AnfNodePtr ImagFusionProcess(const FuncGraphPtr &, const AnfNodePtr &);

using AddAttrToNodeImpl = const AnfNodePtr (*)(const FuncGraphPtr &, const AnfNodePtr &);

class AddAttrToNodeImplRegistry {
 public:
  static AddAttrToNodeImplRegistry &GetInstance();
  void Register(const std::string &op_name, const AddAttrToNodeImpl &impl);
  AddAttrToNodeImpl GetImplByOpName(const std::string &op_name) const;

 private:
  AddAttrToNodeImplRegistry();
  ~AddAttrToNodeImplRegistry() = default;
  DISABLE_COPY_AND_ASSIGN(AddAttrToNodeImplRegistry)
  mindspore::HashMap<std::string, AddAttrToNodeImpl> op_add_attr_to_node_map_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_PASS_ADD_ATTR_TO_NODE_ADD_ATTR_TO_NODE_REGISTER_H_
