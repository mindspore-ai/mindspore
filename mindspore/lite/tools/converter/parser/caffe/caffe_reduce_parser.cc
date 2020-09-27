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
STATUS CaffeReduceParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight,
                                schema::CNodeT *op, std::vector<schema::TensorT *> *weightVec) {
  MS_LOG(DEBUG) << "parse CaffeReduceParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::ReduceT> attr = std::make_unique<schema::ReduceT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  const caffe::ReductionParameter reduce_param = proto.reduction_param();
  if (reduce_param.has_operation()) {
    switch (reduce_param.operation()) {
      case caffe::ReductionParameter_ReductionOp_MEAN:
        attr->mode = schema::ReduceMode_ReduceMean;
        break;
      case caffe::ReductionParameter_ReductionOp_SUM:
        attr->mode = schema::ReduceMode_ReduceSum;
        break;
      case caffe::ReductionParameter_ReductionOp_SUMSQ:
        attr->mode = schema::ReduceMode_ReduceSumSquare;
        break;
      case caffe::ReductionParameter_ReductionOp_ASUM:
        attr->mode = schema::ReduceMode_ReduceASum;
      default:
        MS_LOG(ERROR) << "reduce parse params fail, unsupported opration: " << reduce_param.operation();
        return RET_ERROR;
    }
  } else {
    attr->mode = schema::ReduceMode_ReduceSum;
  }
  if (reduce_param.has_axis()) {
    attr->axes = std::vector(1, reduce_param.axis());
  } else {
    attr->axes = std::vector(1, 0);
  }
  if (reduce_param.has_coeff()) {
    attr->coeff = reduce_param.coeff();
  } else {
    attr->coeff = 1.0;
  }
  attr->reduceToEnd = true;
  attr->keepDims = false;
  op->name = proto.name();
  op->primitive->value.type = schema::PrimitiveType_Reduce;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

CaffeNodeRegistrar g_caffeReduceParser("Reduction", new CaffeReduceParser());
}  // namespace lite
}  // namespace mindspore
