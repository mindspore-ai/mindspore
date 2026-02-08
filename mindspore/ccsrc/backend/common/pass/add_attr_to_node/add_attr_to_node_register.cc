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
#include "backend/common/pass/add_attr_to_node/add_attr_to_node_register.h"
#include "primitive/array_ops.h"
#include "primitive/conv_pool_ops.h"
#include "primitive/image_ops.h"
#include "primitive/math_ops.h"
#include "primitive/random_ops.h"
#include "primitive/sparse_ops.h"
#include "primitive/nn_ops.h"
#include "utils/log_adapter.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_e.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"
#include "primitive/auto_generate/gen_ops_primitive_h.h"
#include "primitive/auto_generate/gen_ops_primitive_i.h"
#include "primitive/auto_generate/gen_ops_primitive_l.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"
#include "primitive/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace opt {
AddAttrToNodeImplRegistry::AddAttrToNodeImplRegistry() {
  Register(prim::kPrimAddN->name(), AddNFusionProcess);
  Register(prim::kPrimAccumulateNV2->name(), AccumulateNV2FusionProcess);
  Register(prim::kPrimConcatOffsetV1->name(), ConcatOffsetV1FusionProcess);
  Register(prim::kPrimConv3DBackpropInput->name(), Conv3DBackpropInputPadListFusionProcess);
  Register(prim::kPrimConv3DBackpropFilter->name(), Conv3DBackpropFilterPadListFusionProcess);
  Register(prim::kPrimDropout->name(), AddDropoutAttrs);
  Register(prim::kPrimDynamicRNN->name(), DynamicRNNFusionProcess);
  Register(prim::kPrimGather->name(), GatherFusionProcess);
  Register(prim::kPrimIm2Col->name(), Im2ColFusionProcess);
  Register(prim::kPrimIOU->name(), IOUFusionProcess);
  Register(prim::kPrimLog->name(), LogFusionProcess);
  Register(prim::kPrimMaxPoolWithArgmaxV2->name(), MaxPoolWithArgmaxV2FusionProcess);
  Register(prim::kPrimParallelConcat->name(), ParallelConcatFusionProcess);
  Register(prim::kPrimRaggedTensorToSparse->name(), RaggedTensorToSparseFusionProcess);
  Register(prim::kPrimResizeV2->name(), ResizeV2FusionProcess);
  Register(prim::kPrimSqueeze->name(), SqueezeAxis);
  Register(prim::kPrimSparseConcat->name(), SparseConcatFusionProcess);
  Register(prim::kPrimSparseTensorDenseMatmul->name(), SparseTensorDenseMatMulFusionProcess);
  Register(prim::kPrimSplit->name(), SplitFusionProcess);
  Register(prim::kPrimStandardNormal->name(), StandardNormalFusionProcess);
  Register(prim::kPrimUniformReal->name(), UniformRealDtypeGe);
  Register(prim::kPrimShape->name(), TensorShapeProcess);
  Register(prim::kPrimTensorShape->name(), TensorShapeProcess);
  Register(prim::kPrimDynamicShape->name(), TensorShapeProcess);
  Register(prim::kPrimHShrink->name(), HShrinkModifyLambd);
  Register(prim::kPrimHShrinkGrad->name(), HShrinkModifyLambd);
  Register(prim::kPrimExtractVolumePatches->name(), ExtractVolumePatchesFormatTranspose);
  Register(prim::kPrimImag->name(), ImagFusionProcess);
}

AddAttrToNodeImplRegistry &AddAttrToNodeImplRegistry::GetInstance() {
  static AddAttrToNodeImplRegistry instance;
  return instance;
}

void AddAttrToNodeImplRegistry::Register(const std::string &op_name, const AddAttrToNodeImpl &impl) {
  if (op_add_attr_to_node_map_.find(op_name) == op_add_attr_to_node_map_.end()) {
    (void)op_add_attr_to_node_map_.emplace(op_name, impl);
    MS_LOG(DEBUG) << op_name << " addattr2node register successfully!";
  }
}

AddAttrToNodeImpl AddAttrToNodeImplRegistry::GetImplByOpName(const std::string &op_name) const {
  if (op_add_attr_to_node_map_.find(op_name) != op_add_attr_to_node_map_.end()) {
    MS_LOG(DEBUG) << op_name << " addattr2node find in registry.";
    return op_add_attr_to_node_map_.at(op_name);
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
