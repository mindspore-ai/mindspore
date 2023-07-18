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

#ifndef MINDSPORE_CORE_BASE_SPARSE_TENSOR_OPS_H_
#define MINDSPORE_CORE_BASE_SPARSE_TENSOR_OPS_H_

#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ops/sparse_op_name.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace prim {
// COOTensor
GVAR_DEF(PrimitivePtr, kPrimMakeCOOTensor, std::make_shared<Primitive>(kMakeCOOTensorOpName));
GVAR_DEF(PrimitivePtr, kPrimCOOTensorGetValues, std::make_shared<Primitive>(kCOOTensorGetValuesOpName));
GVAR_DEF(PrimitivePtr, kPrimCOOTensorGetIndices, std::make_shared<Primitive>(kCOOTensorGetIndicesOpName));
GVAR_DEF(PrimitivePtr, kPrimCOOTensorGetDenseShape, std::make_shared<Primitive>(kCOOTensorGetDenseShapeOpName));
GVAR_DEF(PrimitivePtr, kPrimCOOTensorDenseMatmul, std::make_shared<Primitive>(kCOOTensorDenseMatmulOpName));

// RowTensor
GVAR_DEF(PrimitivePtr, kPrimMakeRowTensor, std::make_shared<Primitive>(kMakeRowTensorOpName));
GVAR_DEF(PrimitivePtr, kPrimRowTensorGetValues, std::make_shared<Primitive>(kRowTensorGetValuesOpName));
GVAR_DEF(PrimitivePtr, kPrimRowTensorGetIndices, std::make_shared<Primitive>(kRowTensorGetIndicesOpName));
GVAR_DEF(PrimitivePtr, kPrimRowTensorGetDenseShape, std::make_shared<Primitive>(kRowTensorGetDenseShapeOpName));
GVAR_DEF(PrimitivePtr, kPrimRowTensorAdd, std::make_shared<Primitive>(kRowTensorAddOpName));

// CSRTensor
GVAR_DEF(PrimitivePtr, kPrimMakeCSRTensor, std::make_shared<Primitive>(kMakeCSRTensorOpName));
GVAR_DEF(PrimitivePtr, kPrimCSRTensorGetValues, std::make_shared<Primitive>(kCSRTensorGetValuesOpName));
GVAR_DEF(PrimitivePtr, kPrimCSRTensorGetIndptr, std::make_shared<Primitive>(kCSRTensorGetIndptrOpName));
GVAR_DEF(PrimitivePtr, kPrimCSRTensorGetIndices, std::make_shared<Primitive>(kCSRTensorGetIndicesOpName));
GVAR_DEF(PrimitivePtr, kPrimCSRTensorGetDenseShape, std::make_shared<Primitive>(kCSRTensorGetDenseShapeOpName));

// MapTensor
GVAR_DEF(PrimitivePtr, kPrimMakeMapParameter, std::make_shared<Primitive>(kMakeMapParameterOpName));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGet, std::make_shared<Primitive>(kMapTensorGetOpName));
GVAR_DEF(PrimitivePtr, kPrimMapTensorPut, std::make_shared<Primitive>(kMapTensorPutOpName));
GVAR_DEF(PrimitivePtr, kPrimMapTensorErase, std::make_shared<Primitive>(kMapTensorEraseOpName));
GVAR_DEF(PrimitivePtr, kPrimMapTensorPutWithStatus, std::make_shared<Primitive>(kMapTensorPutWithStatusOpName));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetDefaultValue, std::make_shared<Primitive>(kMapTensorGetDefaultValueOpName));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetPermitFilterValue,
         std::make_shared<Primitive>(kMapTensorGetPermitFilterValueOpName));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetEvictFilterValue,
         std::make_shared<Primitive>(kMapTensorGetEvictFilterValueOpName));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetKeys, std::make_shared<Primitive>(kMapTensorGetKeysOpName));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetValues, std::make_shared<Primitive>(kMapTensorGetValuesOpName));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetData, std::make_shared<Primitive>(kMapTensorGetDataOpName));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetGrad, std::make_shared<Primitive>(kMapTensorGetGradOpName));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_SPARSE_TENSOR_OPS_H_
