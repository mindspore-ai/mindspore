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

#include <iostream>
#include <memory>
#include <string>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "utils/flags.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace prim {
// COOTensor
constexpr auto kMakeCOOTensor = "MakeCOOTensor";
constexpr auto kCOOTensorGetValues = "COOTensorGetValues";
constexpr auto kCOOTensorGetIndices = "COOTensorGetIndices";
constexpr auto kCOOTensorGetDenseShape = "COOTensorGetDenseShape";
constexpr auto kCOOTensorDenseMatmul = "COOTensorDenseMatmul";

// RowTensor
constexpr auto kMakeRowTensor = "MakeRowTensor";
constexpr auto kRowTensorGetValues = "RowTensorGetValues";
constexpr auto kRowTensorGetIndices = "RowTensorGetIndices";
constexpr auto kRowTensorGetDenseShape = "RowTensorGetDenseShape";
constexpr auto kRowTensorAdd = "RowTensorAdd";

// CSRTensor
constexpr auto kMakeCSRTensor = "MakeCSRTensor";
constexpr auto kCSRTensorGetValues = "CSRTensorGetValues";
constexpr auto kCSRTensorGetIndptr = "CSRTensorGetIndptr";
constexpr auto kCSRTensorGetIndices = "CSRTensorGetIndices";
constexpr auto kCSRTensorGetDenseShape = "CSRTensorGetDenseShape";
constexpr auto kIsCSRFunc = "IsCSRFunc";

// MapTensor
constexpr auto kMakeMapParameter = "MakeMapParameter";
constexpr auto kMapTensorGet = "MapTensorGet";
constexpr auto kMapTensorPut = "MapTensorPut";
constexpr auto kMapTensorErase = "MapTensorErase";
constexpr auto kMapTensorPutWithStatus = "MapTensorPutWithStatus";
constexpr auto kMapTensorGetDefaultValue = "MapTensorGetDefaultValue";
constexpr auto kMapTensorGetPermitFilterValue = "MapTensorGetPermitFilterValue";
constexpr auto kMapTensorGetEvictFilterValue = "MapTensorGetEvictFilterValue";
constexpr auto kMapTensorGetKeys = "MapTensorGetKeys";
constexpr auto kMapTensorGetValues = "MapTensorGetValues";
constexpr auto kMapTensorGetData = "MapTensorGetData";
constexpr auto kMapTensorGetGrad = "MapTensorGetGrad";

// COOTensor
GVAR_DEF(PrimitivePtr, kPrimMakeCOOTensor, std::make_shared<Primitive>(kMakeCOOTensor));
GVAR_DEF(PrimitivePtr, kPrimCOOTensorGetValues, std::make_shared<Primitive>(kCOOTensorGetValues));
GVAR_DEF(PrimitivePtr, kPrimCOOTensorGetIndices, std::make_shared<Primitive>(kCOOTensorGetIndices));
GVAR_DEF(PrimitivePtr, kPrimCOOTensorGetDenseShape, std::make_shared<Primitive>(kCOOTensorGetDenseShape));
GVAR_DEF(PrimitivePtr, kPrimCOOTensorDenseMatmul, std::make_shared<Primitive>(kCOOTensorDenseMatmul));

// RowTensor
GVAR_DEF(PrimitivePtr, kPrimMakeRowTensor, std::make_shared<Primitive>(kMakeRowTensor));
GVAR_DEF(PrimitivePtr, kPrimRowTensorGetValues, std::make_shared<Primitive>(kRowTensorGetValues));
GVAR_DEF(PrimitivePtr, kPrimRowTensorGetIndices, std::make_shared<Primitive>(kRowTensorGetIndices));
GVAR_DEF(PrimitivePtr, kPrimRowTensorGetDenseShape, std::make_shared<Primitive>(kRowTensorGetDenseShape));
GVAR_DEF(PrimitivePtr, kPrimRowTensorAdd, std::make_shared<Primitive>(kRowTensorAdd));

// CSRTensor
GVAR_DEF(PrimitivePtr, kPrimMakeCSRTensor, std::make_shared<Primitive>(kMakeCSRTensor));
GVAR_DEF(PrimitivePtr, kPrimCSRTensorGetValues, std::make_shared<Primitive>(kCSRTensorGetValues));
GVAR_DEF(PrimitivePtr, kPrimCSRTensorGetIndptr, std::make_shared<Primitive>(kCSRTensorGetIndptr));
GVAR_DEF(PrimitivePtr, kPrimCSRTensorGetIndices, std::make_shared<Primitive>(kCSRTensorGetIndices));
GVAR_DEF(PrimitivePtr, kPrimCSRTensorGetDenseShape, std::make_shared<Primitive>(kCSRTensorGetDenseShape));

// MapTensor
GVAR_DEF(PrimitivePtr, kPrimMakeMapParameter, std::make_shared<Primitive>(kMakeMapParameter));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGet, std::make_shared<Primitive>(kMapTensorGet));
GVAR_DEF(PrimitivePtr, kPrimMapTensorPut, std::make_shared<Primitive>(kMapTensorPut));
GVAR_DEF(PrimitivePtr, kPrimMapTensorErase, std::make_shared<Primitive>(kMapTensorErase));
GVAR_DEF(PrimitivePtr, kPrimMapTensorPutWithStatus, std::make_shared<Primitive>(kMapTensorPutWithStatus));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetDefaultValue, std::make_shared<Primitive>(kMapTensorGetDefaultValue));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetPermitFilterValue, std::make_shared<Primitive>(kMapTensorGetPermitFilterValue));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetEvictFilterValue, std::make_shared<Primitive>(kMapTensorGetEvictFilterValue));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetKeys, std::make_shared<Primitive>(kMapTensorGetKeys));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetValues, std::make_shared<Primitive>(kMapTensorGetValues));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetData, std::make_shared<Primitive>(kMapTensorGetData));
GVAR_DEF(PrimitivePtr, kPrimMapTensorGetGrad, std::make_shared<Primitive>(kMapTensorGetGrad));
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CORE_BASE_SPARSE_TENSOR_OPS_H_
