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

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateAudioSpectrogramParameter(const void *prim) {
  AudioSpectrogramParameter *arg_param =
    reinterpret_cast<AudioSpectrogramParameter *>(malloc(sizeof(AudioSpectrogramParameter)));
  if (arg_param == nullptr) {
    MS_LOG(ERROR) << "malloc AudioSpectrogramParameter failed.";
    return nullptr;
  }
  memset(arg_param, 0, sizeof(AudioSpectrogramParameter));
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  arg_param->op_parameter_.type_ = primitive->value_type();
  auto param = primitive->value_as_AudioSpectrogram();
  arg_param->window_size_ = param->window_size();
  arg_param->stride_ = param->stride();
  return reinterpret_cast<OpParameter *>(arg_param);
}
}  // namespace

Registry g_audioSpectrogramParameterRegistry(schema::PrimitiveType_AudioSpectrogram, PopulateAudioSpectrogramParameter,
                                             SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
