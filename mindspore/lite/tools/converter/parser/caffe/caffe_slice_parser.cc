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

#include "tools/converter/parser/caffe/caffe_slice_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
STATUS CaffeSliceParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight,
                               schema::CNodeT *op, std::vector<schema::TensorT *> *weightVec) {
  MS_LOG(DEBUG) << "parse CaffeSliceParser";
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is null";
    return RET_NULL_PTR;
  }
  op->primitive = std::make_unique<schema::PrimitiveT>();
  if (op->primitive == nullptr) {
    MS_LOG(ERROR) << "op->primitive is null";
    return RET_NULL_PTR;
  }

  std::unique_ptr<schema::SplitT> attr = std::make_unique<schema::SplitT>();

  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return RET_NULL_PTR;
  }

  const caffe::SliceParameter &slice_param = proto.slice_param();

  if (!slice_param.slice_point().empty()) {
    attr->numberSplit = slice_param.slice_point_size() + 1;
    std::vector<int32_t> size_splits;
    for (int i = 0; i < slice_param.slice_point_size(); ++i) {
      if (i == 0) {
        size_splits.push_back(slice_param.slice_point(i));
      } else {
        size_splits.push_back(slice_param.slice_point(i) - slice_param.slice_point(i - 1));
      }
    }
    size_splits.push_back(-1);
    attr->sizeSplits = size_splits;
  }

  // The axis along which to slice -- may be negative to index from the end (e.g., -1 for the last axis).
  if (slice_param.has_axis()) {
    attr->splitDim = slice_param.axis();
  } else if (slice_param.has_slice_dim()) {
    attr->splitDim = slice_param.slice_dim();
  }
  op->name = proto.name();
  op->primitive->value.type = schema::PrimitiveType_Split;
  op->primitive->value.value = attr.release();
  return RET_OK;
}

CaffeNodeRegistrar g_caffeSliceParser("Slice", new CaffeSliceParser());
}  // namespace lite
}  // namespace mindspore
