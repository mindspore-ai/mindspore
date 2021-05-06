/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "src/ops/populate/populate_register.h"
#include "nnacl/infer/audio_spectrogram_infer.h"
using mindspore::schema::PrimitiveType_AudioSpectrogram;

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateAudioSpectrogramParameter(const void *prim) {
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_AudioSpectrogram();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<AudioSpectrogramParameter *>(malloc(sizeof(AudioSpectrogramParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc AudioSpectrogramParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(AudioSpectrogramParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->window_size_ = value->window_size();
  param->stride_ = value->stride();
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

REG_POPULATE(PrimitiveType_AudioSpectrogram, PopulateAudioSpectrogramParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
