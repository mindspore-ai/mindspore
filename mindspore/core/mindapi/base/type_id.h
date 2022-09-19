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
  kMetaTypeAnything,
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
}  // namespace mindspore
#endif  // MINDSPORE_CORE_MINDAPI_BASE_TYPE_ID_H_
