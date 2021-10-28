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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_PARAMETER_TUNNER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_PARAMETER_TUNNER_H
#include <utility>
#include <map>
#include <vector>
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/converter/quantizer/weight_quantizer.h"
#include "tools/converter/export_model.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/parser/parser_utils.h"
#include "include/lite_session.h"
#include "include/model.h"
#include "base/base.h"
#include "tools/converter/converter_flags.h"
namespace mindspore::lite::quant {
struct InferenceParam {
  size_t rounds;
  float start_scale;
  float step;
  int thread_num;
};
class ParameterOptimizer {
 public:
  ParameterOptimizer() = default;

  ~ParameterOptimizer() = default;

  int GridSearchForScale(const FuncGraphPtr &func_graph, converter::Flags *flags, double *init_scale);

 private:
  MinMax GetFineTuneRange(std::vector<float> *candidate_scales);

  int CloneFuncGraph(const FuncGraphPtr &func_graph, converter::Flags *flags, FuncGraphPtr *func_graph_bak);

  int WeightQuantModelInference(const FuncGraphPtr &func_graph, converter::Flags *flags,
                                session::LiteSession *origin_session, int origin_model_size,
                                const InferenceParam &param, double *init_scale, std::vector<float> *candidate_scales,
                                bool is_run_all);

  int OriginModelInference(const FuncGraphPtr &func_graph, converter::Flags *flags, SessionModel *sm,
                           int *origin_model_size);
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_PARAMETER_TUNNER_H
