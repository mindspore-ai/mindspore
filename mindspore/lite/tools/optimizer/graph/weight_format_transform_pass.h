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

#ifndef MINDSPORE_LITE_SRC_PASS_FUSION_WEIGHT_FORMAT_TRANSFORM_PASS_H_
#define MINDSPORE_LITE_SRC_PASS_FUSION_WEIGHT_FORMAT_TRANSFORM_PASS_H_
#include <string>
#include <vector>
#include "schema/inner/model_generated.h"
#include "tools/converter/converter_flags.h"
#include "backend/optimizer/common/pass.h"

using mindspore::lite::converter::FmkType;
using mindspore::schema::QuantType;
namespace mindspore::opt {
class WeightFormatTransformPass : public Pass {
 public:
  WeightFormatTransformPass() : Pass("weight_format_transform_pass") {}
  ~WeightFormatTransformPass() override = default;
  void SetQuantType(QuantType type);
  void SetFmkType(FmkType fmkType);
  void SetDstFormat(schema::Format format);
  bool Run(const FuncGraphPtr &graph) override;

 private:
  lite::STATUS ConvWeightFormatTrans(const FuncGraphPtr &graph);
  lite::STATUS TransposeInsertForWeightSharing(const FuncGraphPtr &graph, const ParameterPtr &weight_node,
                                               std::vector<int> perm);
  lite::STATUS HandleWeightSharing(const FuncGraphPtr &graph, const ParameterPtr &weight_node,
                                   schema::Format src_format, schema::Format dst_format);

 private:
  QuantType quant_type = schema::QuantType_QUANT_NONE;
  FmkType fmk_type = lite::converter::FmkType_TF;
  schema::Format dst_format = schema::Format::Format_NUM_OF_FORMAT;
};
}  // namespace mindspore::opt
#endif  // MINDSPORE_LITE_SRC_PASS_FUSION_WEIGHT_FORMAT_TRANSFORM_PASS_H_
