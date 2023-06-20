/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License")
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
#ifndef MINDSPORE_CORE_OPS_EXPORT_INFER_H_
#define MINDSPORE_CORE_OPS_EXPORT_INFER_H_
#include <vector>
#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore::ops {
#define EXPORT_INFER(INFER_FUNC_NAME)                                                                          \
  MIND_API AbstractBasePtr INFER_FUNC_NAME(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive, \
                                           const std::vector<AbstractBasePtr> &input_args);
// export prim infer since backend or frontend infer implement may use them
EXPORT_INFER(ApplyAdadeltaInfer)
EXPORT_INFER(ApplyAdagradInferFunc)
EXPORT_INFER(ApplyAdagradV2Infer)
EXPORT_INFER(ApplyAdaMaxInfer)
EXPORT_INFER(ApplyAddSignInfer)
EXPORT_INFER(BatchNormInferFunc)
EXPORT_INFER(SplitInfer)
EXPORT_INFER(ApplyPowerSignInfer)
EXPORT_INFER(ReduceArithmeticInferFunc)
EXPORT_INFER(SplitVInfer)
EXPORT_INFER(SparseApplyFtrlInfer)
EXPORT_INFER(ApplyAdamInferFunc)
EXPORT_INFER(SparseApplyProximalAdagradInfer)
EXPORT_INFER(FastGeLUInfer)
EXPORT_INFER(FastGeLUGradInfer)
EXPORT_INFER(GeLUInfer)
EXPORT_INFER(HSwishInfer)
EXPORT_INFER(LARSUpdateInfer)
EXPORT_INFER(LogSoftmaxInfer)
EXPORT_INFER(ReLU6GradInferFunc)
EXPORT_INFER(SeLUInfer)
EXPORT_INFER(GeLUGradInfer)
EXPORT_INFER(IouInferFunc)
EXPORT_INFER(ArgminV2Infer)
EXPORT_INFER(CeLUInfer)
EXPORT_INFER(CumSumInfer)
EXPORT_INFER(DropoutDoMaskInfer)
EXPORT_INFER(GatherInfer)
EXPORT_INFER(HSwishGradInfer)
EXPORT_INFER(PReLUInfer)
EXPORT_INFER(ReLUInferFunc)
EXPORT_INFER(ResizeBilinearGradInfer)
EXPORT_INFER(BCEWithLogitsLossInfer)
EXPORT_INFER(SoftmaxInfer)
}  // namespace mindspore::ops
#endif
