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

#include "parser/caffe/caffe_innerproduct_parser.h"
#include <memory>
#include "common/op_enum.h"
#include "common/op_attr.h"
#include "common/data_transpose_utils.h"
#include "ops/fusion/full_connection.h"

namespace mindspore {
namespace lite {
namespace {
constexpr int kInnerProductAxis = 2;
void TransformShape(caffe::BlobShape *shape) {
  auto origin_row = shape->dim(0);
  auto origin_col = shape->dim(1);
  shape->clear_dim();
  shape->add_dim(origin_col);
  shape->add_dim(origin_row);
}
}  // namespace
BaseOperatorPtr CaffeInnerProductParser::Parse(const caffe::LayerParameter &proto,
                                               const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::FullConnection>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_activation_type(mindspore::ActivationType::NO_ACTIVATION);
  const caffe::InnerProductParameter &innerProductParam = proto.inner_product_param();

  if (innerProductParam.has_transpose() && innerProductParam.transpose()) {
    auto mutable_weight = const_cast<caffe::LayerParameter *>(&weight);
    if (mutable_weight == nullptr) {
      MS_LOG(ERROR) << "weight is nullptr.";
      return nullptr;
    }
    auto blob = mutable_weight->mutable_blobs(0);
    if (blob == nullptr) {
      MS_LOG(ERROR) << "blob is nullptr.";
      return nullptr;
    }
    auto shape = blob->mutable_shape();
    if (shape == nullptr) {
      MS_LOG(ERROR) << "shape is nullptr.";
      return nullptr;
    }
    if (blob->mutable_data() == nullptr) {
      MS_LOG(ERROR) << "blob mutable data is nullptr.";
      return nullptr;
    }
    if (shape->dim_size() < kNums2) {
      MS_LOG(ERROR) << "weight shape size " << shape->dim_size() << " should greater than 1";
      return nullptr;
    }
    if (shape->dim(0) == 0 || shape->dim(1) == 0) {
      MS_LOG(ERROR) << "dim val can't be 0.";
      return nullptr;
    }
    dpico::TransposeMatrix(blob->mutable_data()->mutable_data(), static_cast<int>(shape->dim(0)),
                           static_cast<int>(shape->dim(1)));
    TransformShape(shape);
  }

  if (!innerProductParam.has_num_output()) {
    MS_LOG(ERROR) << "InnerProduct Parse num_output for " << proto.name().c_str() << " failed.";
    return nullptr;
  } else {
    (void)prim->AddAttr(dpico::kNumOutput, api::MakeValue(static_cast<int64_t>(innerProductParam.num_output())));
  }

  if (innerProductParam.axis() == 1 || innerProductParam.axis() == kInnerProductAxis) {
    prim->set_axis(innerProductParam.axis());
    prim->set_use_axis(true);
  } else {
    MS_LOG(ERROR) << "InnerProduct Parse axis only support default 1 OR 2, but actually " << innerProductParam.axis();
    return nullptr;
  }
  if (innerProductParam.bias_term()) {
    prim->set_has_bias(true);
  }

  return prim;
}

CaffeNodeRegistrar g_caffeInnerProductParser("InnerProduct", new CaffeInnerProductParser());
}  // namespace lite
}  // namespace mindspore
