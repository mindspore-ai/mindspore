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

#include <memory>
#include "mindspore/lite/tools/converter/parser/caffe/caffe_pooling_parser.h"
#include "utils/log_adapter.h"

const uint32_t INNERPRODUCT_WINDOW_DEFAULT_VALUE = 0;
const uint32_t INNERPRODUCT_PAD_DEFAULT_VALUE = 0;

namespace mindspore {
namespace lite {
STATUS CaffePoolingParser::Parse(const caffe::LayerParameter &proto,
                                 const caffe::LayerParameter &weight,
                                 schema::CNodeT *op,
                                 std::vector<schema::TensorT *> *weightVec) {
  std::unique_ptr<schema::PoolingT> attr(new schema::PoolingT());
  attr->format = schema::Format_NCHW;

  const caffe::PoolingParameter poolingParam = proto.pooling_param();

  auto status = ParsePads(poolingParam, attr.get());
  if (status != RET_OK) {
    // MS_LOGE("ParsePads for %s failed", proto.name().c_str());
    return RET_ERROR;
  }

  status = ParseStrides(poolingParam, attr.get());
  if (status != RET_OK) {
    // MS_LOGE("ParseStrides for %s failed", proto.name().c_str());
    return RET_ERROR;
  }

  status = ParseWindows(poolingParam, attr.get());
  if (status != RET_OK) {
    // MS_LOGE("ParseWindows for %s failed", proto.name().c_str());
    return RET_ERROR;
  }

  status = ParsePoolingMode(poolingParam, attr.get());
  if (status != RET_OK) {
    // MS_LOGE("ParsePoolingMode for %s failed", proto.name().c_str());
    return RET_ERROR;
  }

  // default roundMode RoundMode_CEIL
  attr->roundMode = schema::RoundMode_CEIL;
  if (poolingParam.has_round_mode()) {
    if (poolingParam.round_mode() == caffe::PoolingParameter_RoundMode_FLOOR) {
      attr->roundMode = schema::RoundMode_FLOOR;
    } else if (poolingParam.round_mode() == caffe::PoolingParameter_RoundMode_CEIL) {
      attr->roundMode = schema::RoundMode_CEIL;
    } else {
      MS_ASSERT(false);
    }
  }

  attr->padMode = schema::PadMode_CAFFE;
  op->primitive = std::make_unique<schema::PrimitiveT>();
  op->primitive->value.value = attr.release();
  op->primitive->value.type = schema::PrimitiveType_Pooling;
  return RET_OK;
}

STATUS CaffePoolingParser::ParsePads(const caffe::PoolingParameter &poolingParam, schema::PoolingT *attr) {
  if (poolingParam.has_pad_h() && poolingParam.has_pad_w()) {
    if (poolingParam.has_pad()) {
      // MS_LOGE("Either pad or pad_h/w should be specified; not both");
      return RET_ERROR;
    }
    attr->padLeft = poolingParam.pad_w();
    attr->padRight = poolingParam.pad_w();
    attr->padUp = poolingParam.pad_h();
    attr->padDown = poolingParam.pad_h();
  } else {
    attr->padLeft = poolingParam.pad();
    attr->padRight = poolingParam.pad();
    attr->padUp = poolingParam.pad();
    attr->padDown = poolingParam.pad();
  }
  return RET_OK;
}

STATUS CaffePoolingParser::ParseStrides(const caffe::PoolingParameter &poolingParam, schema::PoolingT *attr) {
  if (poolingParam.has_stride_h() && poolingParam.has_stride_w()) {
    if (poolingParam.has_stride()) {
      // MS_LOGE("Either stride or stride_h/w should be specified; not both");
      return RET_ERROR;
    }
    attr->strideH = poolingParam.stride_h();
    attr->strideW = poolingParam.stride_w();
  } else {
    attr->strideH = poolingParam.stride();
    attr->strideW = poolingParam.stride();
  }
  return RET_OK;
}

STATUS CaffePoolingParser::ParseWindows(const caffe::PoolingParameter &poolingParam, schema::PoolingT *attr) {
  if (poolingParam.has_global_pooling() && poolingParam.global_pooling()) {
    if (poolingParam.has_kernel_size() || poolingParam.has_kernel_h() || poolingParam.has_kernel_w()) {
      // MS_LOGE("With Global_pooling: true Filter size cannot specified");
      return RET_ERROR;
    }
    attr->windowH = INNERPRODUCT_WINDOW_DEFAULT_VALUE;
    attr->windowW = INNERPRODUCT_WINDOW_DEFAULT_VALUE;
    attr->global = true;
  } else {
    if (poolingParam.has_kernel_size() == (poolingParam.has_kernel_h() || poolingParam.has_kernel_w())) {
      // MS_LOGE("Filter size is kernel_size OR kernel_h and kernel_w; not both");
      return RET_ERROR;
    }
    if (!poolingParam.has_kernel_size() && !(poolingParam.has_kernel_h() && poolingParam.has_kernel_w())) {
      // MS_LOGE("For non-square filters both kernel_h and kernel_w are required.");
      return RET_ERROR;
    }

    if (poolingParam.has_kernel_h() && poolingParam.has_kernel_w()) {
      attr->windowH = poolingParam.kernel_h();
      attr->windowW = poolingParam.kernel_w();
    } else {
      attr->windowH = poolingParam.kernel_size();
      attr->windowW = poolingParam.kernel_size();
    }
  }
  return RET_OK;
}

STATUS CaffePoolingParser::ParsePoolingMode(const caffe::PoolingParameter &poolingParam, schema::PoolingT *attr) {
  if (poolingParam.pool() == caffe::PoolingParameter::MAX) {
    attr->poolingMode = schema::PoolMode_MAX_POOLING;
  } else if (poolingParam.pool() == caffe::PoolingParameter::AVE) {
    attr->poolingMode = schema::PoolMode_MEAN_POOLING;
  } else {
    // MS_LOGE("Pooling param`s PoolingMode is not MAX either AVE. MindSpore support MAX and AVE only.");
    return RET_ERROR;
  }
  return RET_OK;
}

CaffeNodeRegistrar g_caffePoolingParser("Pooling", new CaffePoolingParser());
}  // namespace lite
}  // namespace mindspore

