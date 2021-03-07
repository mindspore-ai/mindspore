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

#include "coder/opcoders/nnacl/fp32/tile_fp32_coder.h"
#include <string>
#include <type_traits>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_TileFusion;

namespace mindspore::lite::micro::nnacl {
void TileFP32Coder::ComputeStrides(const int *shape, int *strides, int ndim) const {
  int stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
}

int TileFP32Coder::Resize() {
  tile_param_ = reinterpret_cast<TileParameter *>(parameter_);
  MS_CHECK_TRUE(tile_param_->in_dim_ < static_cast<int>(std::extent<decltype(tile_param_->in_dim_)>::value),
                "invalid dims count");
  MS_CHECK_TRUE(static_cast<int>(input_tensor_->shape().size()) < tile_param_->in_dim_, "invalid input shape number.");
  MS_CHECK_TRUE(static_cast<int>(output_tensor_->shape().size()) < tile_param_->in_dim_,
                "invalid output shape number.");
  for (int i = 0; i < tile_param_->in_dim_; ++i) {
    tile_param_->in_shape_[i] = input_tensor_->shape().at(i);
    tile_param_->out_shape_[i] = output_tensor_->shape().at(i);
  }
  ComputeStrides(tile_param_->in_shape_, tile_param_->in_strides_, tile_param_->in_dim_);
  ComputeStrides(tile_param_->out_shape_, tile_param_->out_strides_, tile_param_->in_dim_);
  return RET_OK;
}

int TileFP32Coder::Prepare(CoderContext *const context) { return Resize(); }

int TileFP32Coder::DoCode(CoderContext *const context) {
  // generate code .h .c
  Collect(context, {"nnacl/fp32/tile.h"}, {"nnacl/fp32/tile.c"});

  NNaclFp32Serializer code;

  code.CodeStruct("tile_parameter", *tile_param_);
  // call the op function
  code.CodeFunction("Tile", input_tensor_, output_tensor_, "&tile_parameter");

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_TileFusion, CPUOpCoderCreator<TileFP32Coder>)

}  // namespace mindspore::lite::micro::nnacl
