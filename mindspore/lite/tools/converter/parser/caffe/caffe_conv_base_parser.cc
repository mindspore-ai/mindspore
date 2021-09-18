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

#include "tools/converter/parser/caffe/caffe_conv_base_parser.h"
#include <algorithm>
#include "nnacl/op_base.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
STATUS CaffeConvBaseParser::ParsePads(const caffe::ConvolutionParameter &convParam, std::vector<int64_t> *pad) {
  /**
   *  padUp = padH;
   *  padDown = padH;
   *  padLeft = padW;
   *  padRight = padW;
   */
  MSLITE_CHECK_PTR(pad);
  if (convParam.has_pad_h() || convParam.has_pad_w()) {
    if (convParam.pad_size() != 0) {
      MS_LOG(ERROR) << "Either pad or pad_h/w should be specified; not both.";
      return RET_ERROR;
    }

    if (!convParam.has_pad_h()) {
      (*pad)[0] = 0;
      (*pad)[1] = 0;
      (*pad)[2] = convParam.pad_w();
      (*pad)[3] = convParam.pad_w();
    } else if (!convParam.has_pad_w()) {
      (*pad)[0] = convParam.pad_h();
      (*pad)[1] = convParam.pad_h();
      (*pad)[2] = 0;
      (*pad)[3] = 0;
    } else {
      (*pad)[0] = convParam.pad_h();
      (*pad)[1] = convParam.pad_h();
      (*pad)[2] = convParam.pad_w();
      (*pad)[3] = convParam.pad_w();
    }
  } else {
    const int num_pad_dims = convParam.pad_size();
    int num_spatial_dims = std::max(num_pad_dims, 2);

    std::vector<int64_t> vec;
    vec.reserve(num_spatial_dims);
    for (int i = 0; i < num_spatial_dims; ++i) {
      vec.push_back((num_pad_dims == 0) ? 0 : convParam.pad((num_pad_dims == 1) ? 0 : i));
    }
    (*pad)[0] = vec[0];
    (*pad)[1] = vec[0];
    (*pad)[2] = vec[1];
    (*pad)[3] = vec[1];
  }
  return RET_OK;
}

STATUS CaffeConvBaseParser::ParseStrides(const caffe::ConvolutionParameter &convParam, std::vector<int64_t> *stride) {
  MSLITE_CHECK_PTR(stride);
  if (convParam.has_stride_h() || convParam.has_stride_w()) {
    if (convParam.stride_size() != 0) {
      MS_LOG(ERROR) << "Either stride or stride_h/w should be specified; not both";
      return RET_ERROR;
    }
    if (!convParam.has_stride_h() || !convParam.has_stride_w()) {
      MS_LOG(ERROR) << "stride_h/w must appear at the same time!";
      return RET_ERROR;
    }
    (*stride)[0] = convParam.stride_h();
    (*stride)[1] = convParam.stride_w();
  } else {
    const int num_stride_dims = convParam.stride_size();
    int num_spatial_dims = std::max(num_stride_dims, 2);

    std::vector<int64_t> vec;
    vec.reserve(num_spatial_dims);
    for (int i = 0; i < num_spatial_dims; ++i) {
      vec.push_back((num_stride_dims == 0) ? 1 : convParam.stride((num_stride_dims == 1) ? 0 : i));
    }
    (*stride)[0] = vec[0];
    (*stride)[1] = vec[1];
  }
  return RET_OK;
}

STATUS CaffeConvBaseParser::ParseDilations(const caffe::ConvolutionParameter &convParam,
                                           std::vector<int64_t> *dilation) {
  MSLITE_CHECK_PTR(dilation);
  const int num_dilation_dims = convParam.dilation_size();
  int num_spatial_dims = std::max(num_dilation_dims, 2);

  std::vector<int64_t> vec;
  vec.reserve(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    vec.push_back((num_dilation_dims == 0) ? 1 : convParam.dilation((num_dilation_dims == 1) ? 0 : i));
  }
  (*dilation)[0] = vec[0];
  (*dilation)[1] = vec[1];
  return RET_OK;
}

STATUS CaffeConvBaseParser::ParseKernels(const caffe::ConvolutionParameter &convParam, std::vector<int64_t> *kernel) {
  MSLITE_CHECK_PTR(kernel);
  if (convParam.has_kernel_h() || convParam.has_kernel_w()) {
    if (convParam.kernel_size_size() != 0) {
      MS_LOG(ERROR) << "Either kernel_size or kernel_h/w should be specified; not both.";
      return RET_ERROR;
    }
    if (convParam.has_kernel_h() && convParam.has_kernel_w()) {
      (*kernel)[0] = convParam.kernel_h();
      (*kernel)[1] = convParam.kernel_w();
    } else {
      MS_LOG(ERROR) << "kernel_h/w must appear at the same time!";
      return RET_ERROR;
    }
  } else if (convParam.kernel_size_size() != 0) {
    const int kernel_size = convParam.kernel_size_size();
    int num_spatial_dims = std::max(kernel_size, 2);

    std::vector<int64_t> vec;
    vec.reserve(num_spatial_dims);
    for (int i = 0; i < num_spatial_dims; i++) {
      vec.push_back(convParam.kernel_size((kernel_size == 1) ? 0 : i));
    }
    // default 2D
    (*kernel)[0] = vec[0];
    (*kernel)[1] = vec[1];
  } else {
    MS_LOG(ERROR) << "conv does not have kernel info.";
    return RET_ERROR;
  }
  return RET_OK;
}

int CaffeConvBaseParser::ParseGroup(const caffe::ConvolutionParameter &convParam, const std::string &layerType) {
  if (convParam.has_group()) {
    return convParam.group();
  } else {
    return layerType == "ConvolutionDepthwise" || layerType == "DepthwiseConv"
             ? static_cast<int>(convParam.num_output())
             : 1;
  }
}

int CaffeConvBaseParser::ParseChannelOut(const caffe::ConvolutionParameter &convParam, int32_t *channelOut) {
  MSLITE_CHECK_PTR(channelOut);
  if (!convParam.has_num_output()) {
    MS_LOG(ERROR) << "Parse num_output for failed.";
    return RET_ERROR;
  }
  *channelOut = convParam.num_output();
  return RET_OK;
}

STATUS CaffeConvBaseParser::ParseWeight(const caffe::LayerParameter &weight,
                                        std::vector<schema::TensorT *> *weightVec) {
  MSLITE_CHECK_PTR(weightVec);

  if (weight.blobs_size() == 0) {
    MS_LOG(ERROR) << "No filter data in layer " << weight.name().c_str();
    return RET_ERROR;
  }

  auto filter = ConvertWeight(weight.blobs(0));
  if (filter == nullptr) {
    MS_LOG(ERROR) << "Convert weight for layer " << weight.name().c_str() << " failed";
    return RET_ERROR;
  }
  weightVec->push_back(filter);

  const caffe::ConvolutionParameter &convParam = weight.convolution_param();
  if (convParam.bias_term() && weight.blobs_size() > 1) {
    auto bias = ConvertWeight(weight.blobs(1));
    if (bias == nullptr) {
      MS_LOG(ERROR) << "Convert bias for layer " << weight.name().c_str() << " failed";
      return RET_ERROR;
    }

    std::vector<int32_t> shape = bias->dims;
    if (shape.size() != 1) {
      MS_LOG(ERROR) << "Bias dim-num of layer " << weight.name().c_str() << " is not supported";
      return RET_ERROR;
    }
    weightVec->push_back(bias);
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
