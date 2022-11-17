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

#include "parser/caffe/caffe_reduce_parser.h"
#include <memory>
#include <vector>
#include "include/registry/converter_context.h"
#include "ops/fusion/reduce_fusion.h"
#include "ops/op_name.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeReduceParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::ReduceFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_keep_dims(false);
  prim->set_reduce_to_end(true);

  const caffe::ReductionParameter &reduce_param = proto.reduction_param();
  if (reduce_param.has_operation()) {
    if (reduce_param.operation() == caffe::ReductionParameter_ReductionOp_MEAN) {
      prim->set_mode(mindspore::ReduceMode::Reduce_Mean);
    } else if (reduce_param.operation() == caffe::ReductionParameter_ReductionOp_SUM) {
      prim->set_mode(mindspore::ReduceMode::Reduce_Sum);
    } else if (reduce_param.operation() == caffe::ReductionParameter_ReductionOp_SUMSQ) {
      prim->set_mode(mindspore::ReduceMode::Reduce_Sum_Square);
    } else if (reduce_param.operation() == caffe::ReductionParameter_ReductionOp_ASUM) {
      prim->set_mode(mindspore::ReduceMode::Reduce_ASum);
    } else {
      MS_LOG(ERROR) << "unsupported reduce mode: " << reduce_param.operation();
      return nullptr;
    }
  } else {
    prim->set_mode(mindspore::ReduceMode::Reduce_Sum);
  }

  std::vector<int32_t> axes;
  if (reduce_param.has_axis()) {
    axes = std::vector<int>(1, reduce_param.axis());
  } else {
    axes = std::vector<int>(1, 0);
  }
  (void)prim->AddAttr(ops::kAxes, api::MakeValue(axes));

  if (reduce_param.has_coeff()) {
    (void)prim->AddAttr(ops::kCoeff, api::MakeValue<float>(reduce_param.coeff()));
  }
  int fmk_type = static_cast<int>(converter::FmkType::kFmkTypeCaffe);
  (void)prim->AddAttr(ops::kFmkType, api::MakeValue(static_cast<int64_t>(fmk_type)));
  return prim;
}

CaffeNodeRegistrar g_caffeReduceParser("Reduction", new CaffeReduceParser());
}  // namespace lite
}  // namespace mindspore
