/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2018-2019. All rights reserved.
 * Description: mslite
 * Author: mslite
 * Create: 2019-12-13
 */

#ifndef MINDSPORE_PREDICT_CONV_SCALE_BIAS_FUSION_PASS_H
#define MINDSPORE_PREDICT_CONV_SCALE_BIAS_FUSION_PASS_H

#include <string>
#include <unordered_map>
#include <memory>
#include "tools/converter/legacy_optimizer/fusion/fusion_pass.h"

namespace mindspore {
namespace lite {
struct BNWeightTensors {
  schema::TensorT *meanTensor = nullptr;
  schema::TensorT *varianceTensor = nullptr;
  schema::TensorT *scaleTensor = nullptr;
  schema::TensorT *biasTensor = nullptr;
};

class ConvScaleBiasFusionPass : public FusionPass {
 public:
  ConvScaleBiasFusionPass() = default;

  ~ConvScaleBiasFusionPass() override;

  STATUS DefinePattern() override = 0;

  // 1. generate biasTensor according to BN weightTensor
  // 2. change attr of conv
  // 3. delete BN node
  STATUS DoFusion(schema::MetaGraphT *graph, const std::string &patternName,
                  std::unordered_map<std::string, std::shared_ptr<Path>> &matchedPath) override;

  STATUS Run(schema::MetaGraphT *graph) override;

 protected:
  // call GetTransParam() and CalConvWeightTensors()
  STATUS GenConvWeightTensors(schema::MetaGraphT *graph, const std::shared_ptr<Path> &convPath,
                              std::shared_ptr<Path> dstPath);

  // fill this->transScale and this->transBias
  virtual STATUS GetTransParam(schema::MetaGraphT *graph, std::shared_ptr<Path> dstPath, int32_t kernelNum) = 0;

  // fill this->newWeightTensor and this->newBiasTensor according to this->transScale and this->transBias
  STATUS CalConvWeightTensors(schema::MetaGraphT *graph, const std::shared_ptr<Path> &convPath, int32_t kernelNum);

  STATUS CalNewWeightTensor(schema::TensorT *oldWeightTensor, int32_t kernelNum, size_t kernelSize);

  STATUS CalNewBiasTensor(schema::TensorT *oldWeightTensor, schema::TensorT *oldBiasTensor, int32_t kernelNum);

 protected:
  float *transScale = nullptr;
  float *transBias = nullptr;
  float *newWeightData = nullptr;
  float *newBiasData = nullptr;
  std::unique_ptr<schema::TensorT> newWeightTensor = nullptr;
  std::unique_ptr<schema::TensorT> newBiasTensor = nullptr;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_PREDICT_CONV_SCALE_BIAS_FUSION_PASS_H
