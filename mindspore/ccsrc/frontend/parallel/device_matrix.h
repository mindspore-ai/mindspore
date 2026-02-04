/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_DEVICE_MATRIX_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_DEVICE_MATRIX_H_

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <utility>

#include "frontend/parallel/status.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace parallel {
using RankList = std::vector<int64_t>;
using Shape = std::vector<int64_t>;
using Shapes = std::vector<Shape>;

class ShapeBase {
 public:
  explicit ShapeBase(bool is_list) : is_list_(is_list) {}
  virtual ~ShapeBase() = default;
  bool is_list() const { return is_list_; }
  virtual bool empty() const = 0;
  virtual int64_t GetBatchValue() = 0;
  virtual size_t size() = 0;
  virtual std::vector<int64_t> GetValue() = 0;
  virtual std::shared_ptr<ShapeBase> GetElement(int64_t idx) = 0;
  virtual std::vector<std::vector<int64_t>> GetAllElements() = 0;
  virtual void set_shape(const std::shared_ptr<ShapeBase> shape) = 0;
  std::string ToString() {
    std::ostringstream oss;
    ConvertShapeToStr(&oss);
    return oss.str();
  }
  virtual void ConvertShapeToStr(std::ostringstream *oss) { MS_LOG(WARNING) << "Please override this func"; }

 private:
  bool is_list_;
};

using ShapeBasePtr = std::shared_ptr<ShapeBase>;
using NewShapes = std::vector<ShapeBasePtr>;
using NewTensorMaps = std::vector<ShapeBasePtr>;

class ShapeValue : public ShapeBase {
 public:
  explicit ShapeValue(std::vector<int64_t> s) : ShapeBase(false), _s(std::move(s)) {}
  ~ShapeValue() override = default;
  bool empty() const override { return _s.empty(); }
  int64_t GetBatchValue() override { return _s[0]; }
  size_t size() override { return _s.size(); }
  std::vector<int64_t> GetValue() override { return _s; }
  ShapeBasePtr GetElement(int64_t idx) override {
    MS_LOG(WARNING) << "Can not get element from ShapeValue, please use GetValue";
    return std::make_shared<ShapeValue>(_s);
  }
  std::vector<std::vector<int64_t>> GetAllElements() override {
    std::vector<std::vector<int64_t>> all_elements = {_s};
    return all_elements;
  }
  void set_shape(const std::shared_ptr<ShapeBase> shape) override {
    if (!shape->is_list()) {
      _s = shape->GetValue();
    } else {
      MS_LOG(EXCEPTION) << "Can not set list shape to value shape";
    }
  }

 private:
  void ConvertShapeToStr(std::ostringstream *oss) override {
    *oss << "[";
    for (size_t i = 0; i < _s.size(); ++i) {
      *oss << _s[i];
      if (i != _s.size() - 1) {
        *oss << ", ";
      }
    }
    *oss << "]";
  }
  std::vector<int64_t> _s;
};

class ShapeList : public ShapeBase {
 public:
  explicit ShapeList(std::vector<ShapeBasePtr> s_list) : ShapeBase(true), _s_list(std::move(s_list)) {}
  ~ShapeList() override = default;
  bool empty() const override { return _s_list.empty(); }
  int64_t GetBatchValue() override {
    MS_LOG(EXCEPTION) << "Can not get batch value from ShapeList";
    return 0;
  }
  size_t size() override { return _s_list.size(); }
  std::vector<int64_t> GetValue() override {
    MS_LOG(EXCEPTION) << "Can not get value from ShapeList, please use GetElement";
    return {};
  }
  ShapeBasePtr GetElement(int64_t idx) override {
    if (idx < 0 || LongToSize(idx) >= _s_list.size()) {
      MS_LOG(EXCEPTION) << "Index " << idx << " out of range " << _s_list.size();
    }
    return _s_list[LongToSize(idx)];
  }
  std::vector<std::vector<int64_t>> GetAllElements() override {
    std::vector<std::vector<int64_t>> all_elements;
    for (auto &s : _s_list) {
      auto elements = s->GetAllElements();
      all_elements.insert(all_elements.end(), elements.begin(), elements.end());
    }
    return all_elements;
  }
  void set_shape(const std::shared_ptr<ShapeBase> shape) override {
    if (shape->is_list()) {
      std::vector<ShapeBasePtr> new_list;
      for (size_t i = 0; i < shape->size(); ++i) {
        new_list.push_back(shape->GetElement(SizeToLong(i)));
      }
      _s_list = new_list;
    } else {
      MS_LOG(EXCEPTION) << "Can not set value shape to list shape";
    }
  }

 private:
  void ConvertShapeToStr(std::ostringstream *oss) override {
    *oss << "[";
    for (size_t i = 0; i < _s_list.size(); ++i) {
      _s_list[i]->ConvertShapeToStr(oss);
      if (i != _s_list.size() - 1) {
        *oss << ", ";
      }
    }
    *oss << "]";
  }
  std::vector<ShapeBasePtr> _s_list;
};

class DeviceMatrix {
 public:
  DeviceMatrix(int64_t rank, RankList dev_list, Shape dev_shape);
  DeviceMatrix() = default;
  ~DeviceMatrix() = default;
  std::vector<RankList> group_list() const { return group_list_; }
  Status CreateGroupList();
  Status GetDevicesByTensorMap(const Shape &tensor_map, RankList *rank_list);
  Status GetDevicesAlongDim(const uint64_t &dim, RankList *devices) const;
  Status GetDevicesAlongMultiDim(const std::vector<int64_t> &dims, RankList *devices);

 private:
  int64_t rank_ = -1;
  RankList dev_list_;
  // From low dim to high dim. eg: [D0 D1 D2 D3]
  Shape dev_shape_;
  std::vector<RankList> group_list_;
};

std::string ShapeToString(const Shape &shape);
std::string ShapesToString(const Shapes &shapes);
std::string ListToString(const RankList &list);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_DEVICE_MATRIX_H_
