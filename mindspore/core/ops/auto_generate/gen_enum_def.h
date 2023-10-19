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
#ifndef MINDSPORE_CORE_OPS_GEN_ENUM_DEF_
#define MINDSPORE_CORE_OPS_GEN_ENUM_DEF_

#include <cstdint>

namespace mindspore::ops {
enum TypeId : int64_t {
  kTypeUnknown = 0,
  kMetaTypeBegin = 0,
  kMetaTypeType = 1,
  kMetaTypeAny = 2,
  kMetaTypeObject = 3,
  kMetaTypeTypeType = 4,
  kMetaTypeProblem = 5,
  kMetaTypeExternal = 6,
  kMetaTypeNone = 7,
  kMetaTypeNull = 8,
  kMetaTypeEllipsis = 9,
  kMetaTypeEnd = 10,
  kObjectTypeBegin = 10,
  kObjectTypeNumber = 11,
  kObjectTypeString = 12,
  kObjectTypeList = 13,
  kObjectTypeTuple = 14,
  kObjectTypeSlice = 15,
  kObjectTypeKeyword = 16,
  kObjectTypeTensorType = 17,
  kObjectTypeRowTensorType = 18,
  kObjectTypeCOOTensorType = 19,
  kObjectTypeUndeterminedType = 20,
  kObjectTypeClass = 21,
  kObjectTypeDictionary = 22,
  kObjectTypeFunction = 23,
  kObjectTypeJTagged = 24,
  kObjectTypeSymbolicKeyType = 25,
  kObjectTypeEnvType = 26,
  kObjectTypeRefKey = 27,
  kObjectTypeRef = 28,
  kObjectTypeEnd = 29,
  kNumberTypeBegin = 29,
  kNumberTypeBool = 30,
  kNumberTypeInt = 31,
  kNumberTypeInt8 = 32,
  kNumberTypeInt16 = 33,
  kNumberTypeInt32 = 34,
  kNumberTypeInt64 = 35,
  kNumberTypeUInt = 36,
  kNumberTypeUInt8 = 37,
  kNumberTypeUInt16 = 38,
  kNumberTypeUInt32 = 39,
  kNumberTypeUInt64 = 40,
  kNumberTypeFloat = 41,
  kNumberTypeFloat16 = 42,
  kNumberTypeFloat32 = 43,
  kNumberTypeFloat64 = 44,
  kNumberTypeBFloat16 = 45,
  kNumberTypeDouble = 46,
  kNumberTypeComplex = 47,
  kNumberTypeComplex64 = 48,
  kNumberTypeComplex128 = 49,
  kNumberTypeInt4 = 50,
  kNumberTypeGLUInt = 51,
  kNumberTypeEnd = 52,
  kMonadTypeBegin = 52,
  kObjectTypeMonad = 53,
  kObjectTypeUMonad = 54,
  kObjectTypeIOMonad = 55,
  kMonadTypeEnd = 56,
  kSparseTypeBegin = 56,
  kObjectTypeCSRTensorType = 57,
  kObjectTypeSparseTensorType = 58,
  kObjectTypeMapTensorType = 59,
  kSparseTypeEnd = 60,
};

enum OpDtype : int64_t {
  DT_BEGIN = 0,
  DT_BOOL = 1,
  DT_INT = 2,
  DT_FLOAT = 3,
  DT_NUMBER = 4,
  DT_TENSOR = 5,
  DT_STR = 6,
  DT_ANY = 7,
  DT_TUPLE_BOOL = 8,
  DT_TUPLE_INT = 9,
  DT_TUPLE_FLOAT = 10,
  DT_TUPLE_NUMBER = 11,
  DT_TUPLE_TENSOR = 12,
  DT_TUPLE_STR = 13,
  DT_TUPLE_ANY = 14,
  DT_LIST_BOOL = 15,
  DT_LIST_INT = 16,
  DT_LIST_FLOAT = 17,
  DT_LIST_NUMBER = 18,
  DT_LIST_TENSOR = 19,
  DT_LIST_STR = 20,
  DT_LIST_ANY = 21,
  DT_END = 22,
};

enum Format : int64_t {
  NCHW = 0,
  NHWC = 1,
};

enum PadMode : int64_t {
  PAD = 0,
  SAME = 1,
  VALID = 2,
};

enum Reduction : int64_t {
  NONE = 0,
  MEAN = 1,
  SUM = 2,
  ADD = 3,
};

enum Direction : int64_t {
  UNIDIRECTIONAL = 0,
};

enum Activation : int64_t {
  TANH = 0,
};

enum GateOrder : int64_t {
  RZH = 0,
  ZRH = 1,
};

enum CellType : int64_t {
  LSTM = 0,
};

enum Group : int64_t {
  SYNC_BN_GROUP0 = 0,
};

enum InterpolationMode : int64_t {
  BILINEAR = 0,
  NEAREST = 1,
};

enum NormMode : int64_t {
  BACKWARD = 0,
  FORWARD = 1,
  ORTHO = 2,
};

enum GridSamplerPaddingMode : int64_t {
  ZEROS = 0,
  BORDER = 1,
  REFLECTION = 2,
};

enum PadFillingMode : int64_t {
  CONSTANT = 0,
  REFLECT = 1,
  EDGE = 2,
  CIRCULAR = 3,
  SYMMETRIC = 4,
};

enum CoordinateTransformationMode : int64_t {
  ALIGN_CORNERS = 0,
  HALF_PIXEL = 1,
};

}  // namespace mindspore::ops
#endif  // MINDSPORE_CORE_OPS_GEN_ENUM_DEF_
