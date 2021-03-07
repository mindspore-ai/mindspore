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

#include "tools/converter/parser/caffe/caffe_batchnorm_parser.h"
#include <cmath>
#include <memory>
#include "tools/common/tensor_util.h"
#include "ops/batch_norm.h"

namespace mindspore {
namespace lite {
using STATUS = int;
ops::PrimitiveC *CaffeBatchNormParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::BatchNorm>();

  prim->set_is_training(false);
  prim->set_format(mindspore::NCHW);

  const caffe::BatchNormParameter &batchNormParam = proto.batch_norm_param();
  if (proto.bottom_size() != 1) {
    MS_LOG(ERROR) << "Layer " << proto.name().c_str() << "bottom numbers is error, it must be 1, but is "
                  << proto.bottom_size();
    return nullptr;
  }
  if (proto.top_size() != 1) {
    MS_LOG(ERROR) << "Layer " << proto.name().c_str() << "top numbers is error, it must be 1, but is "
                  << proto.top_size();
    return nullptr;
  }

  float epsilon = 1e-5;
  if (batchNormParam.has_eps() && std::fabs(1e-5 - batchNormParam.eps()) >= 1e-9) {
    epsilon = batchNormParam.eps();
  }
  prim->set_epsilon(epsilon);

  return prim.release();
}

CaffeNodeRegistrar g_caffeBatchNormParser("BatchNorm", new CaffeBatchNormParser());
}  // namespace lite
}  // namespace mindspore
