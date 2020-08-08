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

#include <algorithm>
#include "mindspore/lite/tools/converter/parser/caffe/caffe_conv_base_parser.h"

const uint32_t PAD_DEFAULT_VALUE = 0;
const uint32_t STRIDE_DEFAULT_VALUE = 1;
const uint32_t DILATION_DEFAULT_VALUE = 1;
const int32_t SPATIAL_DIM_DEFAULT_SIZE = 2;
const uint32_t DEFAULT_CONV_GROUP = 1;
static const int CAFFE_CONV_BIAS_DIM_NUM = 1;

namespace mindspore {
namespace lite {
STATUS CaffeConvBaseParser::ParsePads(const caffe::ConvolutionParameter &convParam, std::vector<int64_t> *pad) {
  /**
   *  padUp = padH;
   *  padDown = padH;
   *  padLeft = padW;
   *  padRight = padW;
   */
  if (convParam.has_pad_h() || convParam.has_pad_w()) {
    if (convParam.pad_size() != 0) {
      // MS_LOGE("Either pad or pad_h/w should be specified; not both");
      return RET_ERROR;
    }

    if (!convParam.has_pad_h()) {
      (*pad)[0] = PAD_DEFAULT_VALUE;
      (*pad)[1] = PAD_DEFAULT_VALUE;
      (*pad)[2] = convParam.pad_w();
      (*pad)[3] = convParam.pad_w();
    } else if (!convParam.has_pad_w()) {
      (*pad)[0] = convParam.pad_h();
      (*pad)[1] = convParam.pad_h();
      (*pad)[2] = PAD_DEFAULT_VALUE;
      (*pad)[3] = PAD_DEFAULT_VALUE;
    } else {
      (*pad)[0] = convParam.pad_h();
      (*pad)[1] = convParam.pad_h();
      (*pad)[2] = convParam.pad_w();
      (*pad)[3] = convParam.pad_w();
    }
  } else {
    // default 2D
    const int num_pad_dims = convParam.pad_size();
    int num_spatial_dims = std::max(num_pad_dims, SPATIAL_DIM_DEFAULT_SIZE);

    std::vector<int64_t> vec;
    for (int i = 0; i < num_spatial_dims; ++i) {
      vec.push_back((num_pad_dims == 0) ? PAD_DEFAULT_VALUE : convParam.pad((num_pad_dims == 1) ? 0 : i));
    }
    // default 2D
    (*pad)[0] = vec[0];
    (*pad)[1] = vec[0];
    (*pad)[2] = vec[1];
    (*pad)[3] = vec[1];
  }
  return RET_OK;
}

STATUS CaffeConvBaseParser::ParseStrides(const caffe::ConvolutionParameter &convParam, std::vector<int64_t> *stride) {
  if (convParam.has_stride_h() || convParam.has_stride_w()) {
    if (convParam.stride_size() != 0) {
      // MS_LOGE("Either stride or stride_h/w should be specified; not both");
      return RET_ERROR;
    }
    if (!convParam.has_stride_h() || !convParam.has_stride_w()) {
      // MS_LOGE("stride_h/w must appear at the same time!");
      return RET_ERROR;
    }
    (*stride)[0] = convParam.stride_h();
    (*stride)[1] = convParam.stride_w();
  } else {
    const int num_stride_dims = convParam.stride_size();
    int num_spatial_dims = std::max(num_stride_dims, SPATIAL_DIM_DEFAULT_SIZE);

    std::vector<int64_t> vec;
    for (int i = 0; i < num_spatial_dims; ++i) {
      vec.push_back((num_stride_dims == 0) ? STRIDE_DEFAULT_VALUE : convParam.stride((num_stride_dims == 1) ? 0 : i));
    }
    // default 2D
    (*stride)[0] = vec[0];
    (*stride)[1] = vec[1];
  }
  return RET_OK;
}

STATUS CaffeConvBaseParser::ParseDilations(const caffe::ConvolutionParameter &convParam,
                                           std::vector<int64_t> *dilation) {
  const int num_dilation_dims = convParam.dilation_size();
  int num_spatial_dims = std::max(num_dilation_dims, SPATIAL_DIM_DEFAULT_SIZE);

  std::vector<int64_t> vec;
  for (int i = 0; i < num_spatial_dims; ++i) {
    vec.push_back((num_dilation_dims == 0) ? DILATION_DEFAULT_VALUE
                                           : convParam.dilation((num_dilation_dims == 1) ? 0 : i));
  }
  // default 2D
  (*dilation)[0] = vec[0];
  (*dilation)[1] = vec[1];

  return RET_OK;
}

STATUS CaffeConvBaseParser::ParseKernels(const caffe::ConvolutionParameter &convParam, std::vector<int64_t> *kernel) {
  if (convParam.has_kernel_h() || convParam.has_kernel_w()) {
    if (convParam.kernel_size_size() != 0) {
      // MS_LOGE("Either kernel_size or kernel_h/w should be specified; not both.")
      return RET_ERROR;
    }
    if (convParam.has_kernel_h() && convParam.has_kernel_w()) {
      (*kernel)[0] = convParam.kernel_h();
      (*kernel)[1] = convParam.kernel_w();
    } else {
      // MS_LOGE("kernel_h/w must appear at the same time!");
      return RET_ERROR;
    }
  } else if (convParam.kernel_size_size() != 0) {
    int kernel_size = convParam.kernel_size_size();
    int num_spatial_dims = std::max(kernel_size, SPATIAL_DIM_DEFAULT_SIZE);
    std::vector<int64_t> vec;
    for (int i = 0; i < num_spatial_dims; i++) {
      vec.push_back(convParam.kernel_size((kernel_size == 1) ? 0 : i));
    }
    // default 2D
    (*kernel)[0] = vec[0];
    (*kernel)[1] = vec[1];
  } else {
    return RET_ERROR;
  }
  return RET_OK;
}

int CaffeConvBaseParser::ParseGroup(const caffe::ConvolutionParameter &convParam, const std::string &layerType) {
  // group default 1
  int group = 0;
  if (convParam.has_group()) {
    group = convParam.group();
  } else {
    layerType == "ConvolutionDepthwise" ? (group = convParam.num_output()) : (group = DEFAULT_CONV_GROUP);
  }
  return group;
}

int CaffeConvBaseParser::ParseChannelIn(const caffe::LayerParameter &proto, const int &group) {
  int res = 0;
  auto &weightBlob = proto.blobs(0);
  if (weightBlob.has_shape()) {
    res = weightBlob.shape().dim(1) * group;
  } else {
    // get shape information from Blob parameters(caffe proto v1)
    if (proto.type() == "Deconvolution") {
      res = weightBlob.num() * group;
    } else {
      res = weightBlob.channels() * group;
    }
  }
  return res;
}

int CaffeConvBaseParser::ParseChannelOut(const caffe::ConvolutionParameter &convParam) {
  if (!convParam.has_num_output()) {
    // MS_LOGE("Parse num_output for failed.");
  }
  return convParam.num_output();
}

STATUS CaffeConvBaseParser::ParseWeight(const caffe::LayerParameter &weight,
                                        std::vector<schema::TensorT *> *weightVec) {
  // Layer must have Filter
  if (weight.blobs_size() == 0) {
    // MS_LOGE("No filter data in layer %s", weight.name().c_str());
    return RET_ERROR;
  }

  auto filter = ConvertWeight(weight.blobs(0));
  if (filter == nullptr) {
    // MS_LOGE("Convert weight for layer %s failed", weight.name().c_str());
    return RET_ERROR;
  }
  weightVec->push_back(filter);

  // parse bias
  const caffe::ConvolutionParameter convParam = weight.convolution_param();
  if (convParam.bias_term() && weight.blobs_size() > 1) {
    auto bias = ConvertWeight(weight.blobs(1));
    if (bias == nullptr) {
      // MS_LOGE("Convert bias for layer %s failed", weight.name().c_str());
      return RET_ERROR;
    }

    std::vector<int32_t> shape = bias->dims;
    if (shape.size() != CAFFE_CONV_BIAS_DIM_NUM) {
      // MS_LOGE("Bias dim-num of layer %s is not supported");
      return RET_ERROR;
    }
    weightVec->push_back(bias);
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore

