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

#include "nnacl/arithmetic_common.h"

void TileOneDimension(float *inData, float *outData, int dim, size_t ndim, int *inShape, int *inStrides,
                      int *outStrides, int *multiple) {
  int srcDimSize = inShape[dim];
  if (dim == ndim - 1) {
    for (int i = 0; i < multiple[dim]; i++) {
      memcpy(outData, inData, srcDimSize * sizeof(float));
      outData += srcDimSize;
    }
    return;
  }
  for (size_t i = 0; i < srcDimSize; i++) {
    for (size_t j = 0; j < multiple[dim]; j++) {
      TileOneDimension(inData + inStrides[dim] * i, outData + outStrides[dim] * (i + j * srcDimSize), dim + 1, ndim,
                       inShape, inStrides, outStrides, multiple);
    }
  }
}

void TileOneDimensionUint8(uint8_t *inData, uint8_t *outData, int dim, size_t ndim, int *inShape, int *inStrides,
                           int *outStrides, int *multiple) {
  int srcDimSize = inShape[dim];
  if (dim == ndim - 1) {
    for (int i = 0; i < multiple[dim]; i++) {
      memcpy(outData, inData, srcDimSize * sizeof(uint8_t));
      outData += srcDimSize;
    }
    return;
  }
  for (size_t i = 0; i < srcDimSize; i++) {
    for (size_t j = 0; j < multiple[dim]; j++) {
      TileOneDimensionUint8(inData + inStrides[dim] * i, outData + outStrides[dim] * (i + j * srcDimSize), dim + 1,
                            ndim, inShape, inStrides, outStrides, multiple);
    }
  }
}

void ComputeStrides(int *shape, int *strides, int ndim) {
  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
}

void CalcMultiplesAndStrides(ArithmeticParameter *param) {
  for (auto i = 0; i < param->ndim_; i++) {
    param->multiples0_[i] = param->out_shape_[i] / param->in_shape0_[i];
    param->multiples1_[i] = param->out_shape_[i] / param->in_shape1_[i];
  }
  // cal strides
  ComputeStrides(param->in_shape0_, param->in_strides0_, param->ndim_);
  ComputeStrides(param->in_shape1_, param->in_strides1_, param->ndim_);
  ComputeStrides(param->out_shape_, param->out_strides_, param->ndim_);
}

void TileDimensions(float *data0, float *data1, float *tile_data0, float *tile_data1, ArithmeticParameter *param) {
  CalcMultiplesAndStrides(param);
  TileOneDimension(data0, tile_data0, 0, param->ndim_, param->in_shape0_, param->in_strides0_, param->out_strides_,
                   param->multiples0_);
  TileOneDimension(data1, tile_data1, 0, param->ndim_, param->in_shape1_, param->in_strides1_, param->out_strides_,
                   param->multiples1_);
}

void TileDimensionsUint8(uint8_t *data0, uint8_t *data1, uint8_t *tile_data0, uint8_t *tile_data1,
                         ArithmeticParameter *param) {
  CalcMultiplesAndStrides(param);
  TileOneDimensionUint8(data0, tile_data0, 0, param->ndim_, param->in_shape0_, param->in_strides0_, param->out_strides_,
                        param->multiples0_);
  TileOneDimensionUint8(data1, tile_data1, 0, param->ndim_, param->in_shape1_, param->in_strides1_, param->out_strides_,
                        param->multiples1_);
}

void TileDimensionsInt8(int8_t *data0, int8_t *data1, int8_t *tile_data0, int8_t *tile_data1,
                        ArithmeticParameter *param) {
  CalcMultiplesAndStrides(param);
  TileOneDimensionUint8((uint8_t *)(data0), (uint8_t *)(tile_data0), 0, param->ndim_, param->in_shape0_,
                        param->in_strides0_, param->out_strides_, param->multiples0_);
  TileOneDimensionUint8((uint8_t *)(data1), (uint8_t *)(tile_data1), 0, param->ndim_, param->in_shape1_,
                        param->in_strides1_, param->out_strides_, param->multiples1_);
}
