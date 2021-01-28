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
#include "ops/split.h"

namespace mindspore {
namespace lite {
ops::PrimitiveC *CaffeSliceParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Split>();

  const caffe::SliceParameter &slice_param = proto.slice_param();
  prim->set_output_num(2);
  if (!slice_param.slice_point().empty()) {
    prim->set_output_num(slice_param.slice_point_size() + 1);
    std::vector<int64_t> size_splits;
    for (int i = 0; i < slice_param.slice_point_size(); ++i) {
      if (i == 0) {
        size_splits.push_back(slice_param.slice_point(i));
      } else {
        size_splits.push_back(slice_param.slice_point(i) - slice_param.slice_point(i - 1));
      }
    }
    size_splits.push_back(-1);
    prim->set_size_splits(size_splits);
  }

  if (slice_param.has_axis()) {
    prim->set_axis(slice_param.axis());
  } else if (slice_param.has_slice_dim()) {
    prim->set_axis(slice_param.slice_dim());
  }

  return prim.release();
}

CaffeNodeRegistrar g_caffeSliceParser("Slice", new CaffeSliceParser());
}  // namespace lite
}  // namespace mindspore
