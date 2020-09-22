/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_INTERNAL_INCLUDE_MS_TENSOR_H_
#define MINDSPORE_LITE_INTERNAL_INCLUDE_MS_TENSOR_H_

#include "internal/include/lite_utils.h"

enum TypeId : int {
  kTypeUnknown = 0,
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
  kObjectTypeSparseTensorType,
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
  kNumberTypeEnd
};

enum Format {
  Format_NCHW = 0,
  Format_NHWC = 1,
  Format_NHWC4 = 2,
  Format_HWKC = 3,
  Format_HWCK = 4,
  Format_KCHW = 5,
  Format_CKHW = 6,
  Format_KHWC = 7,
  Format_CHWK = 8,
  Format_HW = 9,
  Format_HW4 = 10,
  Format_NC = 11,
  Format_NC4 = 12,
  Format_NC4HW4 = 100,
  Format_NUM_OF_FORMAT = 101,
  Format_MIN = Format_NCHW,
  Format_MAX = Format_NUM_OF_FORMAT
};

typedef struct MSTensor {
  enum Category {
    CONST,  // weight tensor
    VAR     // activation tensor
  };
  void *data_ = NULL;
  void *device_data_ = NULL;
  TypeId data_type_;
  Format format_ = Format_NHWC;
  Category category_ = VAR;
  ShapeVector shape_;
  size_t refCount = 0;

  int32_t Batch() const;

  int32_t Channel() const;

  int32_t Height() const;

  int32_t Width() const;

  /// \brief Get size of the dimension of the MindSpore Lite MSTensor index by the parameter index.
  ///
  /// \param[in] index Define index of dimension returned.
  ///
  /// \return Size of dimension of the MindSpore Lite MSTensor.
  int DimensionSize(size_t index) const;

  /// \brief Get number of element in MSTensor.
  ///
  /// \return Number of element in MSTensor.
  int ElementsNum() const;

  int ElementsC4Num() const;

  /// \brief Get byte size of data in MSTensor.
  ///
  /// \return Byte size of data in MSTensor.
  size_t Size() const;

  static void *operator new(size_t sz);

  static void *operator new[](size_t sz);

  static void operator delete(void *ptr, size_t sz);

  static void operator delete[](void *ptr, size_t sz);
} MSTensor;

MSTensor *CreateTensor(TypeId data_type, const ShapeVector &shape);
void DestroyTensor(MSTensor *ptr);
#endif  // MINDSPORE_LITE_INCLUDE_MS_TENSOR_H_
