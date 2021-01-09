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

#include "tools/converter/parser/caffe/caffe_pooling_parser.h"
#include <memory>

namespace mindspore {
namespace lite {
STATUS CaffePoolingParser::ParsePads(const caffe::PoolingParameter &poolingParam, schema::PoolingT *attr) {
  if (poolingParam.has_pad_h() && poolingParam.has_pad_w()) {
    if (poolingParam.has_pad()) {
      MS_LOG(ERROR) << "Either pad or pad_h/w should be specified; not both";
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
      MS_LOG(ERROR) << "Either stride or stride_h/w should be specified; not both";
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
      MS_LOG(ERROR) << "With Global_pooling: true Filter size cannot specified";
      return RET_ERROR;
    }
    attr->windowH = 0;
    attr->windowW = 0;
    attr->global = true;
  } else {
    if (poolingParam.has_kernel_size() == (poolingParam.has_kernel_h() || poolingParam.has_kernel_w())) {
      MS_LOG(ERROR) << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
      return RET_ERROR;
    }
    if (!poolingParam.has_kernel_size() && !(poolingParam.has_kernel_h() && poolingParam.has_kernel_w())) {
      MS_LOG(ERROR) << "For non-square filters both kernel_h and kernel_w are required.";
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
    MS_LOG(ERROR) << "MindSpore support MAX and AVE PoolingMode only.";
    return RET_ERROR;
  }
  return RET_OK;
}
PrimitiveC *CaffePoolingParser::ParseLitePrimitive(const caffe::LayerParameter &proto,
                                                   const caffe::LayerParameter &weight) {
  std::unique_ptr<schema::PoolingT> attr = std::make_unique<schema::PoolingT>();
  if (attr == nullptr) {
    MS_LOG(ERROR) << "new op failed";
    return nullptr;
  }

  attr->format = schema::Format::Format_NCHW;

  const caffe::PoolingParameter &poolingParam = proto.pooling_param();
  auto status = ParsePads(poolingParam, attr.get());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParsePads for " << proto.name().c_str() << " failed";
    return nullptr;
  }

  status = ParseStrides(poolingParam, attr.get());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParseStrides for " << proto.name().c_str() << " failed";
    return nullptr;
  }

  status = ParseWindows(poolingParam, attr.get());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParseWindows for " << proto.name().c_str() << " failed";
    return nullptr;
  }

  status = ParsePoolingMode(poolingParam, attr.get());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "ParsePoolingMode for " << proto.name().c_str() << " failed";
    return nullptr;
  }

  attr->roundMode = schema::RoundMode_CEIL;
  if (poolingParam.has_round_mode()) {
    if (poolingParam.round_mode() == caffe::PoolingParameter_RoundMode_FLOOR) {
      attr->roundMode = schema::RoundMode_FLOOR;
    } else if (poolingParam.round_mode() == caffe::PoolingParameter_RoundMode_CEIL) {
      attr->roundMode = schema::RoundMode_CEIL;
    }
  }
  attr->padMode = schema::PadMode_CAFFE;
  auto primitive = std::make_unique<schema::PrimitiveT>();
  primitive->value.type = schema::PrimitiveType_Pooling;
  primitive->value.value = attr.release();
  return PrimitiveC::Create(primitive.release());
}

CaffeNodeRegistrar g_caffePoolingParser("Pooling", new CaffePoolingParser());
}  // namespace lite
}  // namespace mindspore
