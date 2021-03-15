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

#ifndef MINDSPORE_LITE_NNACL_COMMON_H_
#define MINDSPORE_LITE_NNACL_COMMON_H_

#include <stddef.h>
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"

#ifdef __cplusplus
extern "C" {
#endif

#define kNCHW_N 0
#define kNCHW_C 1
#define kNCHW_H 2
#define kNCHW_W 3

typedef enum FormatC {
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
  Format_NC4HW4 = 13,
  Format_NUM_OF_FORMAT = 14,
  Format_MIN = Format_NCHW,
  Format_MAX = Format_NUM_OF_FORMAT
} FormatC;

typedef enum TypeIdC {
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
  kNumberTypeComplex64,
  kNumberTypeEnd
} TypeIdC;

enum NNACLLshProjectionType {
  LshProjectionType_UNKNOWN = 0,
  LshProjectionType_SPARSE = 1,
  LshProjectionType_DENSE = 2,
  LshProjectionType_MIN = LshProjectionType_UNKNOWN,
  LshProjectionType_MAX = LshProjectionType_DENSE
};

enum NNACLQuantType {
  QuantType_QUANT_NONE = 0,
  QuantType_AwareTraining = 1,
  QuantType_WeightQuant = 2,
  QuantType_PostTraining = 3,
  QuantType_MIN = QuantType_QUANT_NONE,
  QuantType_MAX = QuantType_PostTraining
};

typedef struct vvector {
  int **shape_;      // value of shapes
  int *shape_size_;  // size of shape
  size_t size_;      // number of shapes
} vvector;

typedef struct TensorListC {
  bool is_ready_;
  int data_type_;
  int format_;

  int tensors_data_type_;  // element_data_type_, keep same as c++
  int max_elements_num_;
  int element_shape_[8];
  size_t element_num_;
  size_t element_shape_size_;
  TensorC *tensors_;
} TensorListC;

typedef struct VectorC {
  int *data_;
  size_t size_;
  size_t max_size_;
  size_t per_malloc_size_;
} VectorC;

int MallocTensorListData(TensorListC *tensor_list, TypeIdC dtype, vvector *tensor_shape);
int TensorListMergeShape(int *element_shape, size_t *element_shape_size, const int *tmp, size_t tmp_size);
bool TensorListIsFullyDefined(int *shape, size_t shape_size);

int GetBatch(const TensorC *tensor);
int GetHeight(const TensorC *tensor);
int GetWidth(const TensorC *tensor);
int GetChannel(const TensorC *tensor);
int GetElementNum(const TensorC *tensor);
int GetDimensionSize(const TensorC *tensor, const size_t index);

int CheckAugmentNull(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter);
int CheckAugmentNullSize(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                         OpParameter *parameter, size_t inputs_size_obj, size_t outputs_size_obj);
int CheckAugmentNullSizeInputTwo(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                 size_t outputs_size, OpParameter *parameter, size_t inputs_size_obj_0,
                                 size_t inputs_size_obj_1, size_t outputs_size_obj);
int CheckAugmentNullInputSize(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                              OpParameter *parameter, size_t inputs_size_obj);
int CheckAugmentNullOutputSize(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                               OpParameter *parameter, size_t outputs_size_obj);
void SetDataTypeFormat(TensorC *dst, const TensorC *src);

int SetShapeTensor(TensorC *dst, const TensorC *src);
int SetShapeArray(TensorC *dst, int *src, size_t src_size);
int ShapeSet(int *dst_shape, size_t *dst_shape_size, const int *src_shape, size_t src_shape_size);
int ShapePush(int *shape, size_t *shape_size, int value);
int ShapeInsert(int *shape, size_t *shape_size, int index, int value);
int ShapeErase(int *shape, size_t *shape_size, int index);
bool ShapeEqual(const int *shape0, size_t shape0_size, const int *shape1, size_t shape1_size);

void iswap(int *a, int *b);

int imin(int a, int b);
int imax(int a, int b);

int CommonInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter);
int FftInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                  OpParameter *parameter);

int VectorCInit(VectorC *vc, size_t per_malloc_size);
void VectorCSet(VectorC *vc, const int *src_shape, size_t src_shape_size);
void VectorCPush(VectorC *vc, int value);
void VectorCInsert(VectorC *vc, int index, int value);
void VectorCErase(VectorC *vc, int index);
bool VectorCEqual(VectorC *vc1, VectorC *vc2);
void VectorCFree(VectorC *vc);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_NNACL_COMMON__H_
