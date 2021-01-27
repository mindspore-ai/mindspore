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

namespace mindspore {
namespace lite {
PrimitiveC *CaffeReduceParser::ParseLitePrimitive(const caffe::LayerParameter &proto,
                                                  const caffe::LayerParameter &weight) {
  auto attr = std::make_unique<schema::ReduceT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  attr->keepDims = false;

  const caffe::ReductionParameter &reduce_param = proto.reduction_param();
  if (reduce_param.has_operation()) {
    if (reduce_param.operation() == caffe::ReductionParameter_ReductionOp_MEAN) {
      attr->mode = schema::ReduceMode_ReduceMean;
    } else if (reduce_param.operation() == caffe::ReductionParameter_ReductionOp_SUM) {
      attr->mode = schema::ReduceMode_ReduceSum;
    } else if (reduce_param.operation() == caffe::ReductionParameter_ReductionOp_SUMSQ) {
      attr->mode = schema::ReduceMode_ReduceSumSquare;
    } else if (reduce_param.operation() == caffe::ReductionParameter_ReductionOp_ASUM) {
      attr->mode = schema::ReduceMode_ReduceASum;
    } else {
      MS_LOG(ERROR) << "nsupported reduce mode: " << reduce_param.operation();
      return nullptr;
    }
  } else {
    attr->mode = schema::ReduceMode_ReduceSum;
  }

  std::vector<int32_t> axes;
  if (reduce_param.has_axis()) {
    axes = std::vector<int>(1, reduce_param.axis());
  } else {
    axes = std::vector<int>(1, 0);
  }
  attr->axes = axes;
  attr->reduceToEnd = true;

  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_Reduce;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

CaffeNodeRegistrar g_caffeReduceParser("Reduction", new CaffeReduceParser());
}  // namespace lite
}  // namespace mindspore
