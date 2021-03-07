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
  FormatTransPass() : id_(0) {}

  ~FormatTransPass() override = default;

  STATUS Run(schema::MetaGraphT *graph) override;

  void SetQuantType(QuantType quantType) { this->quant_type_ = quantType; }

  void SetFmk(converter::FmkType fmkType) { this->fmk_type_ = fmkType; }

 protected:
  NodeIter InsertFormatTransNode(schema::MetaGraphT *graph, NodeIter existNodeIter, InsertPlace place, size_t inoutIdx,
                                 FormatTransNodeType nodeType, STATUS *errorCode);

  STATUS ChangeOpAxis(schema::MetaGraphT *graph, const std::unique_ptr<schema::CNodeT> &node);

 private:
  STATUS DoModelInputFormatTrans(schema::MetaGraphT *graph);

  STATUS DoNodeInoutFormatTrans(schema::MetaGraphT *graph);

  void TransformAttrByAxes(int *origin_attr, int *axes, int element_size);

  void TransformOpAxisAttr(int *origin_axis, int element_size);

  STATUS ChangeOpSlice(schema::MetaGraphT *graph, const std::unique_ptr<schema::CNodeT> &node);

  STATUS ChangeOpStridedSlice(schema::MetaGraphT *graph, const std::unique_ptr<schema::CNodeT> &node);

  STATUS ChangeOpSliceAndStridedSlice(schema::MetaGraphT *graph, const std::unique_ptr<schema::CNodeT> &node);

  int GetFormat(const schema::CNodeT &);

  STATUS GetInsertFormatTrans(const schema::CNodeT &node, FormatTransNodeType *beforeNodeType,
                              FormatTransNodeType *afterNodeType);

 protected:
  size_t id_ = 0;
  converter::FmkType fmk_type_ = converter::FmkType_TF;

 private:
  QuantType quant_type_ = QuantType_QUANT_NONE;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_PREDICT_FORMAT_TRANS_PASS_H
