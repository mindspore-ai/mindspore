/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/infer/primitive_type.h"
#include "nnacl/op_base.h"

namespace mindspore::kernel {
#ifdef ENABLE_CLOUD_INFERENCE
namespace {
class PrimitiveTypeHelper {
 public:
  static PrimitiveTypeHelper &Instance() {
    static PrimitiveTypeHelper helper;
    helper.InitMap();
    return helper;
  }

  schema::PrimitiveType PBType2FBType(const std::string &pb_type) const {
    auto iter = pb2fb_.find(pb_type);
    if (iter == pb2fb_.end()) {
      return schema::PrimitiveType_NONE;
    }
    return iter->second;
  }

  std::string FBType2PBType(const int &fb_type) const {
    if (fb_type < 0 || fb_type > schema::PrimitiveType_MAX) {
      return "";
    }
    return fb2pb_[fb_type];
  }

  std::string FBType2PBType(const schema::PrimitiveType &fb_type) const {
    return FBType2PBType(static_cast<int>(fb_type));
  }

 private:
  PrimitiveTypeHelper() = default;
  ~PrimitiveTypeHelper() = default;
  void InitMap() {
    if (MS_LIKELY(inited_)) {
      return;
    }
    const auto &fbtypes = mindspore::schema::EnumValuesPrimitiveType();
    auto size = mindspore::schema::PrimitiveType_MAX + 1;
    fb2pb_.reserve(size);
    for (int i = 0; i < size; i++) {
      auto fbtype = fbtypes[i];
      auto pbtype = mindspore::schema::EnumNamePrimitiveType(fbtype);
      fb2pb_.emplace_back(pbtype);
      pb2fb_[pbtype] = fbtype;
    }
    inited_ = true;
  }

 private:
  bool inited_ = false;
  std::vector<std::string> fb2pb_;
  std::map<std::string, mindspore::schema::PrimitiveType> pb2fb_;
};
}  // namespace

PrimitiveType::PrimitiveType(std::string primitive_type) : protocolbuffers_type_(std::move(primitive_type)) {
  flatbuffers_type_ = PrimitiveTypeHelper::Instance().PBType2FBType(protocolbuffers_type_);
}

PrimitiveType::PrimitiveType(mindspore::schema::PrimitiveType primitive_type) : flatbuffers_type_(primitive_type) {
  protocolbuffers_type_ = PrimitiveTypeHelper::Instance().FBType2PBType(flatbuffers_type_);
}

PrimitiveType::PrimitiveType(int primitive_type) : flatbuffers_type_(primitive_type) {
  protocolbuffers_type_ = PrimitiveTypeHelper::Instance().FBType2PBType(flatbuffers_type_);
}

bool PrimitiveType::operator==(const std::string &other) const { return protocolbuffers_type_ == other; }
bool PrimitiveType::operator!=(const std::string &other) const { return protocolbuffers_type_ != other; }
bool PrimitiveType::operator==(mindspore::schema::PrimitiveType other) const { return flatbuffers_type_ == other; }
bool PrimitiveType::operator!=(mindspore::schema::PrimitiveType other) const { return flatbuffers_type_ != other; }
bool PrimitiveType::operator==(int other) const { return flatbuffers_type_ == other; }
bool PrimitiveType::operator!=(int other) const { return flatbuffers_type_ != other; }

PrimitiveType &PrimitiveType::operator=(const std::string &other) {
  protocolbuffers_type_ = other;
  flatbuffers_type_ = PrimitiveTypeHelper::Instance().PBType2FBType(protocolbuffers_type_);
  return *this;
}

PrimitiveType &PrimitiveType::operator=(const mindspore::schema::PrimitiveType &other) {
  flatbuffers_type_ = other;
  protocolbuffers_type_ = PrimitiveTypeHelper::Instance().FBType2PBType(flatbuffers_type_);
  return *this;
}

PrimitiveType &PrimitiveType::operator=(int other) {
  flatbuffers_type_ = other;
  protocolbuffers_type_ = PrimitiveTypeHelper::Instance().FBType2PBType(flatbuffers_type_);
  return *this;
}

std::string PrimitiveType::TypeName() const { return this->protocolbuffers_type_; }

schema::PrimitiveType PrimitiveType::SchemaType() const {
  return static_cast<schema::PrimitiveType>(this->flatbuffers_type_);
}
#endif
}  // namespace mindspore::kernel
