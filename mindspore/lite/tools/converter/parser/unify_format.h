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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_UNIFY_FORMAT_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_UNIFY_FORMAT_H_

#include "tools/optimizer/format/to_format_base.h"

using mindspore::lite::converter::FmkType;
namespace mindspore {
namespace lite {
class UnifyFormatToNHWC : public opt::ToFormatBase {
 public:
  explicit UnifyFormatToNHWC(FmkType fmk_type = lite::converter::FmkType_MS, bool train_flag = false,
                             schema::QuantType quant_type = schema::QuantType_QUANT_NONE)
      : ToFormatBase(fmk_type, train_flag), quant_type_(quant_type) {}
  ~UnifyFormatToNHWC() override = default;

 private:
  STATUS GetTransNodeFormatType(const CNodePtr &cnode, opt::TransTypePair *trans_info) override;
  void SetSensitiveOps() override;
  bool DecideWhetherHandleGraphInput(const FuncGraphPtr &func_graph, const ShapeVector &shape) override;
  bool DecideWhetherInferShapeForNewNode() override;
  STATUS DecideConvWeightSrcAndDstFormat(const CNodePtr &cnode, schema::Format *src_format,
                                         schema::Format *dst_format) override;
  schema::QuantType quant_type_{schema::QuantType_QUANT_NONE};
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_UNIFY_FORMAT_H_
