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

#ifndef MINDSPORE_PREDICT_ELTWISE_FORMAT_TRANS_PASS_H
#define MINDSPORE_PREDICT_ELTWISE_FORMAT_TRANS_PASS_H

#include <memory>
#include <vector>
#include "tools/common/graph_util.h"
#include "tools/converter/converter_flags.h"
#include "tools/converter/legacy_optimizer/graph/format_trans_pass.h"

namespace mindspore {
namespace lite {
class TransOpInsertPass : public FormatTransPass {
 public:
  TransOpInsertPass() : FormatTransPass() {}

  ~TransOpInsertPass() override = default;

  STATUS Run(schema::MetaGraphT *graph) override;

 private:
  bool CanFusion(schema::MetaGraphT *graph, const std::unique_ptr<CNodeT> &node);

  STATUS FindOutTransType();

 private:
  FormatTransNodeType pre_insert_trans_type_ = kNHWC2NCHW;
  FormatTransNodeType post_insert_trans_type_ = kNHWC2NCHW;
  FormatTransNodeType pre_type_ = kNONE;
  std::vector<int> pre_perm_;
  FormatTransNodeType post_type_ = kNONE;
  std::vector<int> post_perm_;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_PREDICT_ELTWISE_FORMAT_TRANS_PASS_H
