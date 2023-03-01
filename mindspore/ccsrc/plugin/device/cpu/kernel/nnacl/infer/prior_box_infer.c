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

#include "nnacl/infer/prior_box_infer.h"
#include <math.h>
#include "nnacl/infer/infer_register.h"

int PriorBoxInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                       OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  output->data_type_ = kNumberTypeFloat32;
  output->format_ = input->format_;
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  float different_aspect_ratios[MAX_SHAPE_SIZE * 2 + 1];  // NOTE: flip double the number
  different_aspect_ratios[0] = 1.0;
  int32_t different_aspect_ratios_size = 1;

  PriorBoxParameter *param = (PriorBoxParameter *)parameter;
  float *aspect_ratios = param->aspect_ratios;
  if (aspect_ratios == NULL) {
    return NNACL_NULL_PTR;
  }
  int32_t aspect_ratios_size = param->aspect_ratios_size;
  MS_CHECK_TRUE_RET(aspect_ratios_size <= MAX_SHAPE_SIZE, NNACL_ERR);
  for (int32_t i = 0; i < aspect_ratios_size; i++) {
    float ratio = aspect_ratios[i];
    if (fabsf(ratio) < EPSILON_VALUE) {
      return NNACL_ERR;
    }

    bool exist = false;
    for (int32_t j = 0; j < different_aspect_ratios_size; j++) {
      if (fabsf(ratio - different_aspect_ratios[j]) < EPSILON_VALUE) {
        exist = true;
        break;
      }
    }
    if (!exist) {
      different_aspect_ratios[different_aspect_ratios_size] = ratio;
      different_aspect_ratios_size++;
      if (param->flip) {
        different_aspect_ratios[different_aspect_ratios_size] = 1.0f / ratio;
        different_aspect_ratios_size++;
      }
    }
  }

  int32_t min_sizes_size = param->min_sizes_size;
  int32_t max_sizes_size = param->max_sizes_size;
  int32_t num_priors_box = min_sizes_size * different_aspect_ratios_size + max_sizes_size;
  const int kPriorBoxPoints = 4;
  const int kPriorBoxN = 1;
  const int kPriorBoxW = 1;
  const int kPriorBoxC = 2;

  int32_t h = GetHeight(input) * GetWidth(input) * num_priors_box * kPriorBoxPoints;
  output->shape_size_ = 4;
  output->shape_[0] = kPriorBoxN;
  output->shape_[1] = h;
  output->shape_[2] = kPriorBoxW;
  output->shape_[3] = kPriorBoxC;
  return NNACL_OK;
}

REG_INFER(PriorBox, PrimType_PriorBox, PriorBoxInferShape)
