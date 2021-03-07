/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PREDICT_BATCHNORM_CONVERT_SCALE_PASS_H
#define MINDSPORE_PREDICT_BATCHNORM_CONVERT_SCALE_PASS_H

#include <unordered_map>
#include <memory>
#include <string>
#include <utility>
#include "tools/common/graph_util.h"
#include "tools/converter/optimizer.h"
#include "tools/converter/converter_flags.h"

using mindspore::schema::TensorT;
namespace mindspore {
namespace lite {
struct BNWeightTensors {
  schema::TensorT *meanTensor = nullptr;
  TensorT *varianceTensor = nullptr;
  TensorT *scaleTensor = nullptr;
  TensorT *biasTensor = nullptr;
};
class BatchNormConvertScalePass : public GraphPass {
 public:
  BatchNormConvertScalePass() = default;

  ~BatchNormConvertScalePass() = default;

  STATUS Run(MetaGraphT *graph) override;

  void SetFmk(converter::FmkType fmk) { this->fmkType = fmk; }

 protected:
  STATUS GetTransParam(MetaGraphT *graph, const std::unique_ptr<CNodeT> &bnNode);

  // Get and check BNNode weight tensor
  STATUS GetBnWeightTensors(MetaGraphT *graph, BNWeightTensors *bnWeightTensors, const std::unique_ptr<CNodeT> &bnNode);

  STATUS GetBnEpsilon(const std::unique_ptr<CNodeT> &bnNode);

  STATUS GenNewScaleTensor(MetaGraphT *graph, const std::unique_ptr<CNodeT> &bnNode);

  STATUS ConvertBNToScale(MetaGraphT *graph, const std::unique_ptr<CNodeT> &bnNode);

  uint32_t bnChannel = 0;
  float eps = 0;
  TensorT *bnMeanTensor = nullptr;
  float *transScale = nullptr;
  float *transBias = nullptr;
  std::unique_ptr<TensorT> newScaleWeightTensor = nullptr;
  std::unique_ptr<TensorT> newScaleBiasTensor = nullptr;
  converter::FmkType fmkType = converter::FmkType_TF;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_PREDICT_BATCHNORM_CONVERT_SCALE_PASS_H
