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

#include "tools/converter/adapter/acl/mapper/matmul_fusion_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "ops/batch_matmul.h"
#include "ops/op_name.h"
#include "ops/op_utils.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
STATUS MatMulFusionMapper::Mapper(const CNodePtr &cnode) {
  auto src_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  auto transpose_a = src_prim->GetAttr(mindspore::ops::kTransposeA);
  auto transpose_b = src_prim->GetAttr(mindspore::ops::kTransposeB);
  std::vector<int64_t> shape_vector;
  if (acl::GetShapeVectorFromCNode(cnode, &shape_vector) != RET_OK) {
    MS_LOG(ERROR) << "Get shape of cnode failed.";
    return RET_ERROR;
  }

  ops::MatMul mat_mul;
  auto dst_prim = mat_mul.GetPrim();
  if (shape_vector.size() != DIMENSION_2D) {
    ops::BatchMatMul batch_mat_mul;
    dst_prim = batch_mat_mul.GetPrim();
  }
  dst_prim->AddAttr("transpose_x1", transpose_a);
  dst_prim->AddAttr("transpose_x2", transpose_b);
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "MatMulFusion mapper failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameMatMulFusion, MatMulFusionMapper)
}  // namespace lite
}  // namespace mindspore
