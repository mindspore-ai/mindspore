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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/where_parameter.h"
#include "nnacl/sparse_to_dense_parameter.h"
#include "nnacl/transpose.h"
#include "nnacl/triu_tril.h"
#include "nnacl/fp32/unique_fp32.h"
#include "nnacl/scatter_nd_parameter.h"
#include "ops/assert.h"
#include "ops/where.h"
#include "ops/unsorted_segment_sum.h"
#include "ops/unique.h"
#include "ops/triu.h"
#include "ops/tril.h"
#include "ops/transpose.h"
#include "ops/sparse_to_dense.h"
#include "ops/sparse_segment_sum.h"
#include "ops/sparse_reshape.h"
#include "ops/sparse_fill_empty_rows.h"
#include "ops/size.h"
#include "ops/shape.h"
#include "ops/select.h"
#include "ops/scatter_nd_update.h"
#include "ops/tensor_scatter_add.h"
#include "ops/scatter_nd.h"
#include "ops/expand_dims.h"

namespace mindspore {
namespace lite {
REG_OP_BASE_POPULATE(Assert)
REG_OP_BASE_POPULATE(UnsortedSegmentSum)
REG_OP_BASE_POPULATE(SparseSegmentSum)
REG_OP_BASE_POPULATE(SparseReshape)
REG_OP_BASE_POPULATE(SparseFillEmptyRows)
REG_OP_BASE_POPULATE(Size)
REG_OP_BASE_POPULATE(Shape)
REG_OP_BASE_POPULATE(Select)
REG_OP_BASE_POPULATE(ExpandDims)

REG_OP_DEFAULT_POPULATE(SparseToDense)
REG_OP_DEFAULT_POPULATE(Transpose)
REG_OP_DEFAULT_POPULATE(Tril)
REG_OP_DEFAULT_POPULATE(Triu)
REG_OP_DEFAULT_POPULATE(Where)
REG_OP_DEFAULT_POPULATE(Unique)
using mindspore::ops::kNameScatterNd;
using mindspore::ops::kNameScatterNdUpdate;
using mindspore::ops::kNameTensorScatterAdd;
using mindspore::schema::PrimitiveType_ScatterNd;
using mindspore::schema::PrimitiveType_ScatterNdUpdate;
using mindspore::schema::PrimitiveType_TensorScatterAdd;
REG_OPERATOR_POPULATE(kNameScatterNd, PrimitiveType_ScatterNd, PopulateOpParameter<ScatterNDParameter>)
REG_OPERATOR_POPULATE(kNameScatterNdUpdate, PrimitiveType_ScatterNdUpdate, PopulateOpParameter<ScatterNDParameter>)
REG_OPERATOR_POPULATE(kNameTensorScatterAdd, PrimitiveType_TensorScatterAdd, PopulateOpParameter<ScatterNDParameter>)
}  // namespace lite
}  // namespace mindspore
