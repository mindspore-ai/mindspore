/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_COMMON_PROTOBUF_UTILS_H
#define MINDSPORE_LITE_TOOLS_COMMON_PROTOBUF_UTILS_H

#include <string>
#include <vector>
#include "google/protobuf/message.h"
#include "proto/caffe.pb.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
namespace lite {
bool ReadProtoFromCodedInputStream(google::protobuf::io::CodedInputStream *coded_stream,
                                   google::protobuf::Message *proto);

STATUS ReadProtoFromText(const char *file, google::protobuf::Message *message);

STATUS ReadProtoFromBinaryFile(const char *file, google::protobuf::Message *message);
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_COMMON_PROTOBUF_UTILS_H
