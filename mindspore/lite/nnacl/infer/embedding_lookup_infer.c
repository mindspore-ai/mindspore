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

#include "nnacl/infer/embedding_lookup_infer.h"
#include "nnacl/infer/infer_register.h"

int EmbeddingLookupInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                              OpParameter *parameter) {
#ifdef Debug
  int check_ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  if (inputs_size < 2 || outputs_size != 1) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
#endif

  const TensorC *params_ = inputs[0];
  const TensorC *ids = inputs[inputs_size - 1];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, params_);
  if (!parameter->infer_flag_) {
    return NNACL_INFER_INVALID;
  }

  int embedding_shape[MAX_SHAPE_SIZE];
  size_t embedding_shape_size = 0;
  ShapeSet(embedding_shape, &embedding_shape_size, params_->shape_, params_->shape_size_);
  ShapeErase(embedding_shape, &embedding_shape_size, 0);
  int output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = 0;
  ShapeSet(output_shape, &output_shape_size, ids->shape_, ids->shape_size_);
  for (size_t i = 0; i < embedding_shape_size; ++i) {
    ShapePush(output_shape, &output_shape_size, embedding_shape[i]);
  }
  for (size_t i = 1; i < inputs_size - 1; ++i) {
    int embedding_shape_t[MAX_SHAPE_SIZE];
    size_t embedding_shape_t_size = 0;
    ShapeSet(embedding_shape_t, &embedding_shape_t_size, inputs[i]->shape_, inputs[i]->shape_size_);
    ShapeErase(embedding_shape_t, &embedding_shape_t_size, 0);
    bool t_equal = ShapeEqual(embedding_shape_t, embedding_shape_t_size, embedding_shape, embedding_shape_size);
    if (!t_equal) {
      return NNACL_INPUT_TENSOR_ERROR;
    }
  }
  SetShapeArray(output, output_shape, output_shape_size);
  return NNACL_OK;
}

REG_INFER(EmbeddingLookup, PrimType_EmbeddingLookupFusion, EmbeddingLookupInferShape)
