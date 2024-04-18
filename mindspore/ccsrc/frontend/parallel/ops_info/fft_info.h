/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_FFT_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_FFT_INFO_H_

#include <memory>
#include <string>
#include <vector>

#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/batch_parallel_info.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"
#include "ir/value.h"
#include "utils/hash_map.h"

namespace mindspore {
namespace parallel {
class FFTBase : public OperatorInfo {
 public:
  FFTBase(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<FFTCost>()) {}
  ~FFTBase() override = default;
  void ReComputeBatchSplitFlagList() override;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override { return SetCostUnderStrategyBase(strategy); }

 protected:
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferTensorMap() override;
  Status InferDevMatrixShape() override;
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferMirrorOps() override;
  virtual size_t GetDimIndex() { return kIndex2; }
  virtual size_t GetIndexNum() { return kIndex4; }
  virtual Status GetDimSplit() { return SUCCESS; }

  Shape input_split_;
};

class FFTIntDim : public FFTBase {
 public:
  FFTIntDim(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : FFTBase(name, inputs_shape, outputs_shape, attrs) {}
  ~FFTIntDim() override = default;

 protected:
  Status GetDimSplit() override;
};

class FFTTupleDim : public FFTBase {
 public:
  FFTTupleDim(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : FFTBase(name, inputs_shape, outputs_shape, attrs) {}
  ~FFTTupleDim() override = default;

 protected:
  Status GetDimSplit() override;
};

#define FFT_REG_END }

#define FFT_REG_INDEX_END(dimIndex, indexNum)        \
 protected:                                          \
  size_t GetDimIndex() override { return dimIndex; } \
  size_t GetIndexNum() override { return indexNum; } \
  FFT_REG_END

#define FFT_REG_HEAD(className, bassName)                                                       \
  class className : public bassName {                                                           \
   public:                                                                                      \
    className(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape, \
              const PrimitiveAttrs &attrs)                                                      \
        : bassName(name, inputs_shape, outputs_shape, attrs) {}                                 \
    ~className() override = default;

#define FFT_REG_GET3(_1, _2, _3, ...) _3

#define FFT_REGISTER(className, bassName, ...) \
  FFT_REG_HEAD(className, bassName)            \
  FFT_REG_GET3(__VA_ARGS__, FFT_REG_INDEX_END(__VA_ARGS__), FFT_REG_END, FFT_REG_END)

FFT_REGISTER(FFTShiftInfo, FFTTupleDim, kIndex1, kIndex2);
FFT_REGISTER(IFFTShiftInfo, FFTTupleDim, kIndex1, kIndex2);
FFT_REGISTER(FFTInfo, FFTIntDim);
FFT_REGISTER(IFFTInfo, FFTIntDim);
FFT_REGISTER(FFT2Info, FFTTupleDim);
FFT_REGISTER(IFFT2Info, FFTTupleDim);
FFT_REGISTER(FFTNInfo, FFTTupleDim);
FFT_REGISTER(IFFTNInfo, FFTTupleDim);
FFT_REGISTER(RFFTInfo, FFTIntDim);
FFT_REGISTER(IRFFTInfo, FFTIntDim);
FFT_REGISTER(DCTInfo, FFTIntDim, kIndex3, kIndex7);
FFT_REGISTER(IDCTInfo, FFTIntDim, kIndex3, kIndex7);
FFT_REGISTER(DCTNInfo, FFTTupleDim, kIndex3, kIndex7);
FFT_REGISTER(IDCTNInfo, FFTTupleDim, kIndex3, kIndex7);

#undef FFT_REG_END
#undef FFT_REG_INDEX_END
#undef FFT_REG_HEAD
#undef FFT_REG_GET3
#undef FFT_REGISTER
}  // namespace parallel
}  // namespace mindspore
#endif
