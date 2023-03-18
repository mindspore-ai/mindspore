/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_MINDAPI_BASE_TYPE_ID_H_
#define MINDSPORE_CORE_MINDAPI_BASE_TYPE_ID_H_

namespace mindspore {
/// \brief TypeId defines data type identifiers.
enum TypeId : int {
  kTypeUnknown = 0,
  //
  // Meta types.
  //
  kMetaTypeBegin = kTypeUnknown,
  kMetaTypeType,  // Type
  kMetaTypeAny,
  kMetaTypeObject,
  kMetaTypeTypeType,  // TypeType
  kMetaTypeProblem,
  kMetaTypeExternal,
  kMetaTypeNone,
  kMetaTypeNull,
  kMetaTypeEllipsis,
  kMetaTypeEnd,
  //
  // Object types
  //
  kObjectTypeBegin = kMetaTypeEnd,
  kObjectTypeNumber,
  kObjectTypeString,
  kObjectTypeList,
  kObjectTypeTuple,
  kObjectTypeSlice,
  kObjectTypeKeyword,
  kObjectTypeTensorType,
  kObjectTypeRowTensorType,
  kObjectTypeCOOTensorType,
  kObjectTypeUndeterminedType,
  kObjectTypeClass,
  kObjectTypeDictionary,
  kObjectTypeFunction,
  kObjectTypeJTagged,
  kObjectTypeSymbolicKeyType,
  kObjectTypeEnvType,
  kObjectTypeRefKey,
  kObjectTypeRef,
  kObjectTypeEnd,
  //
  // Number Types
  //
  kNumberTypeBegin = kObjectTypeEnd,
  kNumberTypeBool,
  kNumberTypeInt,
  kNumberTypeInt8,
  kNumberTypeInt16,
  kNumberTypeInt32,
  kNumberTypeInt64,
  kNumberTypeUInt,
  kNumberTypeUInt8,
  kNumberTypeUInt16,
  kNumberTypeUInt32,
  kNumberTypeUInt64,
  kNumberTypeFloat,
  kNumberTypeFloat16,
  kNumberTypeFloat32,
  kNumberTypeFloat64,
  kNumberTypeDouble,
  kNumberTypeComplex,
  kNumberTypeComplex64,
  kNumberTypeComplex128,
  kNumberTypeInt4,
  kNumberTypeGLUInt,
  kNumberTypeEnd,
  //
  // Monad Types
  //
  kMonadTypeBegin = kNumberTypeEnd,
  kObjectTypeMonad,
  kObjectTypeUMonad,
  kObjectTypeIOMonad,
  kMonadTypeEnd,
  //
  // Sparse Types
  //
  kSparseTypeBegin = kMonadTypeEnd,
  kObjectTypeCSRTensorType,
  kObjectTypeSparseTensorType,
  kObjectTypeMapTensorType,
  kSparseTypeEnd,
  // New types should placed at the end of enum,
  // in order to keep fit with the type of existing model on the lite side.
};

/// \brief QuantDataType defines data type for quantization: int1-int16, uint1-uint16.
enum QuantDataType : int {
  kQuantDataTypeInt1 = 0,
  kQuantDataTypeInt2,
  kQuantDataTypeInt3,
  kQuantDataTypeInt4,
  kQuantDataTypeInt5,
  kQuantDataTypeInt6,
  kQuantDataTypeInt7,
  kQuantDataTypeInt8,
  kQuantDataTypeInt9,
  kQuantDataTypeInt10,
  kQuantDataTypeInt11,
  kQuantDataTypeInt12,
  kQuantDataTypeInt13,
  kQuantDataTypeInt14,
  kQuantDataTypeInt15,
  kQuantDataTypeInt16,

  kQuantDataTypeUInt1 = 100,
  kQuantDataTypeUInt2,
  kQuantDataTypeUInt3,
  kQuantDataTypeUInt4,
  kQuantDataTypeUInt5,
  kQuantDataTypeUInt6,
  kQuantDataTypeUInt7,
  kQuantDataTypeUInt8,
  kQuantDataTypeUInt9,
  kQuantDataTypeUInt10,
  kQuantDataTypeUInt11,
  kQuantDataTypeUInt12,
  kQuantDataTypeUInt13,
  kQuantDataTypeUInt14,
  kQuantDataTypeUInt15,
  kQuantDataTypeUInt16,
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_MINDAPI_BASE_TYPE_ID_H_
