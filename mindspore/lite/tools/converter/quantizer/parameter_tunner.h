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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_PARAMETER_TUNNER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_PARAMETER_TUNNER_H_

#include <utility>
#include <map>
#include <vector>
#include <memory>
#include "tools/converter/quantizer/quantize_util.h"
#include "tools/converter/quantizer/weight_quantizer.h"
#include "tools/converter/export_model.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/parser/parser_utils.h"
#include "include/model.h"
#include "base/base.h"

namespace mindspore::lite::quant {
struct SearchParams {
  int range_start;
  int range_end;
  int step;
};

class ParameterOptimizer {
 public:
  ParameterOptimizer() = default;

  ~ParameterOptimizer() = default;

  int GridSearchForScale(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param,
                         double *init_scale);

 private:
  MinMax GetFineTuneRange(std::vector<float> *candidate_scales);

  int WeightQuantModelInference(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param,
                                const std::shared_ptr<mindspore::Model> &origin_model, size_t origin_model_size,
                                SearchParams *s_param, int *ret_scale, float *best_compress_ratio,
                                bool *found_valid_scale);

  int OriginModelInference(const FuncGraphPtr &func_graph, const std::shared_ptr<ConverterPara> &param,
                           const std::shared_ptr<mindspore::Model> &origin_model, size_t *origin_model_size);

  int CopyDataAndRun(const std::shared_ptr<mindspore::Model> &origin_model,
                     const std::shared_ptr<mindspore::Model> &quant_model);
};
}  // namespace mindspore::lite::quant
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_QUANTIZER_PARAMETER_TUNNER_H_
