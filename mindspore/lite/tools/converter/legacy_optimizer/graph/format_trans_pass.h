/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PREDICT_FORMAT_TRANS_PASS_H
#define MINDSPORE_PREDICT_FORMAT_TRANS_PASS_H

#include <memory>
#include "tools/converter/optimizer.h"
#include "tools/common/graph_util.h"
#include "tools/converter/converter_flags.h"

namespace mindspore {
namespace lite {
enum FormatTransNodeType { kNCHW2NHWC, kNHWC2NCHW, kNONE };

class FormatTransPass : public GraphPass {
 public:
  FormatTransPass() : id(0) {}

  ~FormatTransPass() override = default;

  STATUS Run(schema::MetaGraphT *graph) override;

  void SetQuantType(QuantType quantType);

  void SetFmk(converter::FmkType fmkType);

 protected:
  NodeIter InsertFormatTransNode(schema::MetaGraphT *graph, NodeIter existNodeIter, InsertPlace place, size_t inoutIdx,
                                 FormatTransNodeType nodeType, STATUS *errorCode);

 private:
  STATUS DoModelInputFormatTrans(schema::MetaGraphT *graph);

  STATUS DoNodeInoutFormatTrans(schema::MetaGraphT *graph);

  int GetFormat(const schema::CNodeT &);

  STATUS GetInsertFormatTrans(const schema::CNodeT &node, FormatTransNodeType *beforeNodeType,
                              FormatTransNodeType *afterNodeType);

 protected:
  size_t id = 0;

 private:
  QuantType quantType = QuantType_QUANT_NONE;
  converter::FmkType fmkType = converter::FmkType_TF;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_PREDICT_FORMAT_TRANS_PASS_H
