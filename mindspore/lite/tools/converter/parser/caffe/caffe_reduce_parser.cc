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

#include "tools/converter/parser/caffe/caffe_reduce_parser.h"
#include <memory>
#include <vector>
#include "ops/fusion/reduce_fusion.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeReduceParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto primitive_c = new (std::nothrow) ops::ReduceFusion();
  if (primitive_c == nullptr) {
    MS_LOG(ERROR) << "new ReduceFusion failed";
    return nullptr;
  }

  primitive_c->set_keep_dims(false);

  const caffe::ReductionParameter &reduce_param = proto.reduction_param();
  if (reduce_param.has_operation()) {
    if (reduce_param.operation() == caffe::ReductionParameter_ReductionOp_MEAN) {
      primitive_c->set_mode(mindspore::ReduceMode::Reduce_Mean);
    } else if (reduce_param.operation() == caffe::ReductionParameter_ReductionOp_SUM) {
      primitive_c->set_mode(mindspore::ReduceMode::Reduce_Sum);
    } else if (reduce_param.operation() == caffe::ReductionParameter_ReductionOp_SUMSQ) {
      primitive_c->set_mode(mindspore::ReduceMode::Reduce_Sum_Square);
    } else if (reduce_param.operation() == caffe::ReductionParameter_ReductionOp_ASUM) {
      primitive_c->set_mode(mindspore::ReduceMode::Reduce_ASum);
    } else {
      MS_LOG(ERROR) << "nsupported reduce mode: " << reduce_param.operation();
      return nullptr;
    }
  } else {
    primitive_c->set_mode(mindspore::ReduceMode::Reduce_Sum);
  }

  std::vector<int32_t> axes;
  if (reduce_param.has_axis()) {
    axes.push_back(1);
    axes.push_back(reduce_param.axis());
  } else {
    axes.push_back(1);
    axes.push_back(0);
  }
  primitive_c->AddAttr("axes", MakeValue(axes));

  return primitive_c;
}

CaffeNodeRegistrar g_caffeReduceParser("Reduction", new CaffeReduceParser());
}  // namespace lite
}  // namespace mindspore
