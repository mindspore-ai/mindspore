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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_FORMAT_TO_NHWC_FORMAT_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_FORMAT_TO_NHWC_FORMAT_H_

#include "tools/optimizer/format/to_format_base.h"

namespace mindspore {
namespace opt {
class ToNHWCFormat : public ToFormatBase {
 public:
  explicit ToNHWCFormat(FmkType fmk_type = converter::kFmkTypeMs, bool train_flag = false,
                        ModelType save_type = kMindIR)
      : ToFormatBase(fmk_type, train_flag, save_type, "ToNHWCFormat") {}
  ~ToNHWCFormat() = default;

 protected:
  STATUS GetTransNodeFormatType(const CNodePtr &cnode, opt::TransTypePair *trans_info) override;
  STATUS DecideConvWeightSrcAndDstFormat(const CNodePtr &cnode, schema::Format *src_format,
                                         schema::Format *dst_format) override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_FORMAT_TO_NHWC_FORMAT_H_
