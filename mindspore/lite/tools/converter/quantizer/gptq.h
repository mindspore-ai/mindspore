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
#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_GPTQ_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_GPTQ_H_

#include <map>
#include <algorithm>
#include <vector>
#include "tools/converter/quantizer/quantizer.h"
#include "src/tensor.h"
#include "nnacl/matmul_parameter.h"
#include "tools/converter/quantizer/gptq_quantizer.h"

namespace mindspore::lite::quant {
class Gptq {
 public:
  Gptq(lite::Tensor *weight_tensor, WeightInfo *weight_info, float *hessian_data, int hessian_length, int bit_num,
       bool transpose, const MatMulParameter *op_parameter)
      : weight_tensor_(weight_tensor), hessian_data_(hessian_data), hessian_length_(hessian_length), bit_num_(bit_num) {
    weight_data_ = weight_info->weight_data;
    quant_data_ = weight_info->quant_data;
    quant_params_ = &(weight_info->quant_params);
    auto dims = weight_tensor->shape();
    // W * X
    if (!transpose) {
      if (op_parameter->a_transpose_) {
        prefer_dim_ = 1;
        rows_ = dims[1];
        columns_ = dims[0];
        deep_ = dims[0];
      } else {
        prefer_dim_ = 0;
        rows_ = dims[0];
        columns_ = dims[1];
        deep_ = dims[1];
      }
    } else {
      if (op_parameter->b_transpose_) {
        prefer_dim_ = 0;
        rows_ = dims[0];
        columns_ = dims[1];
        deep_ = dims[1];
      } else {
        prefer_dim_ = 1;
        rows_ = dims[1];
        columns_ = dims[0];
        deep_ = dims[0];
      }
    }
  }

  ~Gptq() = default;

  int DoQuantize();

 private:
  int Init();

  int FindQuantParams(const float *weight_data, std::vector<int> dims, int element_count, int prefer_dim,
                      std::vector<schema::QuantParamT> *quant_params);

  int CalculateHessianInv(float *hessian_data, float *hessian_inv_data, int hessian_length, float percdamp);

  int CloneMatrix(float *dest, int dst_rows, int dst_columns, const float *src, int src_rows, int src_columns, int i1,
                  int i2, int j1, int j2);

  int QuantizePerBlock(std::vector<float> *weight_data, std::vector<int> *quant_data, std::vector<float> *error,
                       std::vector<float> *loss, const std::vector<float> *hinv, int count);

  lite::Tensor *weight_tensor_ = nullptr;
  float *weight_data_ = nullptr;
  int8_t *quant_data_ = nullptr;
  float *hessian_data_ = nullptr;

  int hessian_length_{0};
  int deep_{0};
  int block_size_{4};
  int group_size_{-1};
  float percdamp_{0.01};
  int rows_{0};
  int columns_{0};
  int prefer_dim_{0};
  int bit_num_{4};
  bool symmetric_{false};
  // [-128, 127] 8bit
  // [-8, 7] 4bit
  int quant_min_{-8};
  int quant_max_{7};
  bool narrow_range_{false};
  std::vector<schema::QuantParamT> *quant_params_ = nullptr;
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_GPTQ_H_
