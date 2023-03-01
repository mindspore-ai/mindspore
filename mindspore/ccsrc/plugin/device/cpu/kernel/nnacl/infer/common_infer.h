/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_NNACL_COMMON_H_
#define MINDSPORE_NNACL_COMMON_H_

#include <stddef.h>
#include "nnacl/errorcode.h"
#include "nnacl/op_base.h"
#include "nnacl/tensor_c.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/tensorlist_c_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

#define EPSILON_VALUE 1e-6

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
  QuantType_QUANT_WEIGHT = 4,
  QuantType_QUANT_ALL = 5,
  QuantType_QUANT_DYNAMIC = 6,
  QuantType_MIN = QuantType_QUANT_NONE,
  QuantType_MAX = QuantType_QUANT_DYNAMIC
};

typedef struct VectorC {
  int *data_;
  size_t size_;
  size_t max_size_;
  size_t per_malloc_size_;
} VectorC;

int CheckAugmentNull(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     const OpParameter *parameter);
int CheckAugmentNullSize(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                         const OpParameter *parameter, size_t inputs_size_obj, size_t outputs_size_obj);
int CheckAugmentNullSizeInputTwo(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                 size_t outputs_size, const OpParameter *parameter, size_t inputs_size_obj_0,
                                 size_t inputs_size_obj_1, size_t outputs_size_obj);
int CheckAugmentNullInputSize(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                              const OpParameter *parameter, size_t inputs_size_obj);
int CheckAugmentNullOutputSize(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                               const OpParameter *parameter, size_t outputs_size_obj);
int CheckAugmentWithMinSize(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                            const OpParameter *parameter, size_t inputs_size_obj, size_t outputs_size_obj);
void SetDataTypeFormat(TensorC *dst, const TensorC *src);

void SetShapeTensor(TensorC *dst, const TensorC *src);
void SetShapeArray(TensorC *dst, const int *src, size_t src_size);
void ShapeSet(int *dst_shape, size_t *dst_shape_size, const int *src_shape, size_t src_shape_size);
void ShapePush(int *shape, size_t *shape_size, int value);
int ShapeInsert(int *shape, size_t *shape_size, int index, int value);
int ShapeErase(int *shape, size_t *shape_size, int index);
bool ShapeEqual(const int *shape0, size_t shape0_size, const int *shape1, size_t shape1_size);

void iswap(int *a, int *b);

int imin(int a, int b);
int imax(int a, int b);

int CommonInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter);
int CommonGradInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                         OpParameter *parameter);
int CommonInferShapeWithOneInput(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                 size_t outputs_size, OpParameter *parameter);
int CommonInferShapeWithNHWC(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                             OpParameter *parameter);
int FftInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                  const OpParameter *parameter);
bool InferFlag(const TensorC *const *inputs, size_t inputs_size);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_COMMON__H_
