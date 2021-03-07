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
#include "ops/fusion/avg_pool_fusion.h"
#include "ops/fusion/max_pool_fusion.h"

namespace mindspore {
namespace lite {
STATUS CaffePoolingParser::ParsePads(const caffe::PoolingParameter &poolingParam, std::vector<int64_t> *pad) {
  if (poolingParam.has_pad_h() && poolingParam.has_pad_w()) {
    if (poolingParam.has_pad()) {
      MS_LOG(ERROR) << "Either pad or pad_h/w should be specified; not both";
      return RET_ERROR;
    }
    (*pad)[0] = poolingParam.pad_h();
    (*pad)[1] = poolingParam.pad_h();
    (*pad)[2] = poolingParam.pad_w();
    (*pad)[3] = poolingParam.pad_w();
  } else {
    (*pad)[0] = poolingParam.pad();
    (*pad)[1] = poolingParam.pad();
    (*pad)[2] = poolingParam.pad();
    (*pad)[3] = poolingParam.pad();
  }
  return RET_OK;
}

STATUS CaffePoolingParser::ParseStrides(const caffe::PoolingParameter &poolingParam, std::vector<int64_t> *strides) {
  if (poolingParam.has_stride_h() && poolingParam.has_stride_w()) {
    if (poolingParam.has_stride()) {
      MS_LOG(ERROR) << "Either stride or stride_h/w should be specified; not both";
      return RET_ERROR;
    }
    (*strides)[0] = poolingParam.stride_h();
    (*strides)[1] = poolingParam.stride_w();
  } else {
    (*strides)[0] = poolingParam.stride();
    (*strides)[1] = poolingParam.stride();
  }
  return RET_OK;
}

STATUS CaffePoolingParser::ParseWindows(const caffe::PoolingParameter &poolingParam, std::vector<int64_t> *windows) {
  if (poolingParam.has_global_pooling() && poolingParam.global_pooling()) {
    if (poolingParam.has_kernel_size() || poolingParam.has_kernel_h() || poolingParam.has_kernel_w()) {
      MS_LOG(ERROR) << "With Global_pooling: true Filter size cannot specified";
      return RET_ERROR;
    }
    (*windows)[0] = 0;
    (*windows)[1] = 0;
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
      (*windows)[0] = poolingParam.kernel_h();
      (*windows)[1] = poolingParam.kernel_w();
    } else {
      (*windows)[0] = poolingParam.kernel_size();
      (*windows)[1] = poolingParam.kernel_size();
    }
  }
  return RET_OK;
}

mindspore::RoundMode CaffePoolingParser::ParseRoundMode(const caffe::PoolingParameter &poolingParam) {
  mindspore::RoundMode roundMode = mindspore::RoundMode::CEIL;
  if (poolingParam.has_round_mode()) {
    if (poolingParam.round_mode() == caffe::PoolingParameter_RoundMode_FLOOR) {
      roundMode = mindspore::RoundMode::FLOOR;
    } else if (poolingParam.round_mode() == caffe::PoolingParameter_RoundMode_CEIL) {
      roundMode = mindspore::RoundMode::CEIL;
    }
  }
  return roundMode;
}

ops::PrimitiveC *CaffePoolingParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  const caffe::PoolingParameter &poolingParam = proto.pooling_param();

  // parse kernel params
  std::vector<int64_t> windows(2, 0);
  if (ParseWindows(poolingParam, &windows) != RET_OK) {
    MS_LOG(ERROR) << "ParseWindows for " << proto.name().c_str() << " failed";
    return nullptr;
  }

  // parse strides params
  std::vector<int64_t> strides(2, 0);
  if (ParseStrides(poolingParam, &strides) != RET_OK) {
    MS_LOG(ERROR) << "ParseStrides for " << proto.name().c_str() << " failed";
    return nullptr;
  }

  // parse pad params
  std::vector<int64_t> pad(4, 0);
  if (ParsePads(poolingParam, &pad) != RET_OK) {
    MS_LOG(ERROR) << "ParsePads for " << proto.name().c_str() << " failed";
    return nullptr;
  }

  // parse round mode
  auto roundMode = ParseRoundMode(poolingParam);

  if (poolingParam.pool() == caffe::PoolingParameter::MAX) {
    auto prim = std::make_unique<ops::MaxPoolFusion>();
    prim->set_format(mindspore::Format::NCHW);
    prim->set_pad_mode(mindspore::PadMode::PAD);
    prim->set_kernel_size(windows);
    prim->set_strides(strides);
    prim->set_pad(pad);
    prim->set_round_mode(roundMode);
    prim->set_global(poolingParam.global_pooling());
    return prim.release();
  } else if (poolingParam.pool() == caffe::PoolingParameter::AVE) {
    auto prim = std::make_unique<ops::AvgPoolFusion>();
    prim->set_format(mindspore::Format::NCHW);
    prim->set_pad_mode(mindspore::PadMode::PAD);
    prim->set_kernel_size(windows);
    prim->set_strides(strides);
    prim->set_pad(pad);
    prim->set_round_mode(roundMode);
    prim->set_global(poolingParam.global_pooling());
    return prim.release();
  } else {
    MS_LOG(ERROR) << "poolingParam.pool() is not MAX or AVE";
    return nullptr;
  }
}

CaffeNodeRegistrar g_caffePoolingParser("Pooling", new CaffePoolingParser());
}  // namespace lite
}  // namespace mindspore
