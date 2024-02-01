/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

/*!
 * \file op_attr.h
 * \brief
 */

#ifndef CUSTOMIZE_OP_PROTO_UTILS_OP_ATTR_H_
#define CUSTOMIZE_OP_PROTO_UTILS_OP_ATTR_H_

#include <vector>
#include "op_log.h"
#include "graph/operator.h"

namespace ops {
using namespace ge;

// attr base struct
struct AttrBase {
  const int32_t attr_idx;
  const std::string attr_name;
  AttrBase(const int attr_idx, const std::string &attr_name) : attr_idx(attr_idx), attr_name(attr_name) {}
};

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @return bool: flag of success or not
 */
bool GetAttrValue(const Operator &paras, const struct AttrBase &attr_info, int32_t &value) {
  if (!paras.GetAttr(attr_info.attr_name, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. return false", attr_info.attr_name.c_str());
    return false;
  }
  OP_LOGD("GetAttrValue", "Get the attr of %s is %d", attr_info.attr_name.c_str(), value);
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @return bool: flag of success or not
 */
bool GetAttrValue(const Operator &paras, const struct AttrBase &attr_info, int64_t &value) {
  if (!paras.GetAttr(attr_info.attr_name, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. return false", attr_info.attr_name.c_str());
    return false;
  }
  OP_LOGD("GetAttrValue", "Get the attr of %s is %d", attr_info.attr_name.c_str(), value);
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @param [in] default_value: default_value
 * @return bool: flag of success or not
 */
bool GetAttrValue(const Operator &paras, const struct AttrBase &attr_info, int64_t &value,
                  const int64_t default_value) {
  if (!paras.GetAttr(attr_info.attr_name, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. set the default value", attr_info.attr_name.c_str());
    value = default_value;
  }
  OP_LOGD("GetAttrValue", "Get the attr of %s is %d", attr_info.attr_name.c_str(), value);
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @return bool: flag of success or not
 */
template <typename T>
bool GetAttrValue(const Operator &paras, const std::pair<int64_t, std::string> &attr_info, int32_t &value) {
  if (!paras.GetAttr(attr_info.second, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. return false", attr_info.second.c_str());
    return false;
  }
  OP_LOGD("GetAttrValue", "Get the attr of %s is %d", attr_info.second.c_str(), value);
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @param [in] default_value: default_value
 * @return bool: flag of success or not
 */
template <typename T>
bool GetAttrValue(const Operator &paras, const std::pair<int64_t, std::string> &attr_info, int32_t &value,
                  const int32_t default_value) {
  if (!paras.GetAttr(attr_info.second, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. set the default value", attr_info.second.c_str());
    value = default_value;
  }
  OP_LOGD("GetAttrValue", "Get the attr of %s is %d", attr_info.second.c_str(), value);
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @return bool: flag of success or not
 */
template <typename T>
bool GetAttrValue(const Operator &paras, const std::pair<int64_t, std::string> &attr_info, int64_t &value) {
  if (!paras.GetAttr(attr_info.second, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. return false", attr_info.second.c_str());
    return false;
  }
  OP_LOGD("GetAttrValue", "Get the attr of %s is %lld", attr_info.second.c_str(), value);
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @param [in] default_value: default_value
 * @return bool: flag of success or not
 */
template <typename T>
bool GetAttrValue(const Operator &paras, const std::pair<int64_t, std::string> &attr_info, int64_t &value,
                  const int64_t default_value) {
  if (!paras.GetAttr(attr_info.second, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. set the default value", attr_info.second.c_str());
    value = default_value;
  }
  OP_LOGD("GetAttrValue", "Get the attr of %s is %lld", attr_info.second.c_str(), value);
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @param [in] default_value: default_value
 * @return bool: flag of success or not
 */
template <typename T>
bool GetAttrValue(const Operator &paras, const std::pair<int64_t, std::string> &attr_info, uint32_t &value,
                  const uint32_t default_value) {
  if (!paras.GetAttr(attr_info.second, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. set the default value", attr_info.second.c_str());
    value = default_value;
  }
  OP_LOGD("GetAttrValue", "Get the attr of %s is %d", attr_info.second.c_str(), value);
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @param [in] default_value: default_value
 * @return bool: flag of success or not
 */
template <typename T>
bool GetAttrValue(const Operator &paras, const std::pair<int64_t, std::string> &attr_info, bool &value,
                  const bool default_value) {
  if (!paras.GetAttr(attr_info.second, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. set the default value", attr_info.second.c_str());
    value = default_value;
  }
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @return bool: flag of success or not
 */
template <typename T>
bool GetAttrValue(const Operator &paras, const std::pair<int64_t, std::string> &attr_info, bool &value) {
  if (!paras.GetAttr(attr_info.second, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. return false", attr_info.second.c_str());
    return false;
  }
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @return bool: flag of success or not
 */
template <typename T>
bool GetAttrValue(const Operator &paras, const std::pair<int64_t, std::string> &attr_info, vector<int64_t> &value) {
  if (!paras.GetAttr(attr_info.second, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. return false", attr_info.second.c_str());
    return false;
  }
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @return bool: flag of success or not
 */
template <typename T>
bool GetAttrValue(const Operator &paras, const std::pair<int64_t, std::string> &attr_info, vector<int32_t> &value) {
  if (!paras.GetAttr(attr_info.second, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. return false", attr_info.second.c_str());
    return false;
  }
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @return bool: flag of success or not
 */
template <typename T>
bool GetAttrValue(const Operator &paras, const struct AttrBase &attr_info, int32_t &value,
                  const int32_t default_value) {
  if (!paras.GetAttr(attr_info.attr_name, value)) {
    OP_LOGW("GetAttrValue", "Fail to get attr %s automatically. use default value", attr_info.attr_name.c_str());
    value = default_value;
  }
  OP_LOGD("GetAttrValue", "Get the attr of %s is %d", attr_info.attr_name.c_str(), value);
  return true;
}

/*
 * @brief: read constvalue from paras store into values
 * @param [in] paras: ge::Operator
 * @param [in] attr_info: attr info pair(attr_idx, attr_name)
 * @param [out] value: store value.
 * @return bool: flag of success or not
 */
template <typename T>
bool GetAttrValue(const Operator &paras, const struct AttrBase &attr_info, float32_t &value) {
  if (!paras.GetAttr(attr_info.attr_name, value)) {
    OP_LOGW("GetAttrValue", "Get the attr of %s is failed. return false", attr_info.attr_name.c_str());
    return false;
  }
  OP_LOGD("GetAttrValue", "Get the attr of %s is %f", attr_info.attr_name.c_str(), value);
  return true;
}

}  // namespace ops
#endif  // CUSTOMIZE_OP_PROTO_UTILS_OP_ATTR_H_
