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

#ifndef MINDSPORE_CCSRC_IR_SCALAR_H_
#define MINDSPORE_CCSRC_IR_SCALAR_H_

namespace mindspore {
/* namespace to support inference engine */

#include <type_traits>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <sstream>
#include <utility>
#include <cfloat>
#include "ir/base.h"
#include "ir/dtype.h"

class Scalar : public Value {
 public:
  Scalar() = default;
  explicit Scalar(const TypePtr t) : Value(t) {}
  ~Scalar() override = default;
  MS_DECLARE_PARENT(Scalar, Value)
  virtual bool IsZero() = 0;
  virtual bool IsOne() = 0;
  abstract::AbstractBasePtr ToAbstract() override;

 protected:
  std::size_t hash_ = 0;
};
using ScalarPtr = std::shared_ptr<Scalar>;

class BoolImm : public Scalar {
 public:
  explicit BoolImm(bool b) : Scalar(kBool), v_(b) { hash_ = std::hash<bool>{}(v_); }
  ~BoolImm() override = default;
  MS_DECLARE_PARENT(BoolImm, Scalar)
  std::size_t hash() const override { return hash_; }
  bool value() const { return v_; }
  bool IsZero() override { return v_ == false; }
  bool IsOne() override { return v_ == true; }
  bool operator==(const Value &other) const override;
  bool operator==(const BoolImm &other) const;
  std::string ToString() const override {
    if (v_) {
      return "true";
    } else {
      return "false";
    }
  }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "Bool(" << v_ << ")";
    return oss.str();
  }

 private:
  bool v_;
};
using BoolImmPtr = std::shared_ptr<BoolImm>;
IMM_TRAITS(BoolImmPtr, bool)

class IntergerImm : public Scalar {
 public:
  IntergerImm() = default;
  explicit IntergerImm(const TypePtr &t) : Scalar(t) {}
  ~IntergerImm() override = default;
  MS_DECLARE_PARENT(IntergerImm, Scalar)
};

class Int8Imm : public IntergerImm {
 public:
  Int8Imm() : IntergerImm(kInt8), v_(0) {}
  explicit Int8Imm(int8_t v) : IntergerImm(kInt8), v_(v) { hash_ = std::hash<int>{}(v_); }
  ~Int8Imm() override = default;
  MS_DECLARE_PARENT(Int8Imm, IntergerImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == 0; }
  bool IsOne() override { return v_ == 1; }
  int8_t value() const { return v_; }
  bool operator==(const Value &other) const override;
  bool operator==(const Int8Imm &other) const;
  std::string ToString() const override { return std::to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "I8(" << v_ << ")";
    return oss.str();
  }

 private:
  int8_t v_;
};
using Int8ImmPtr = std::shared_ptr<Int8Imm>;
IMM_TRAITS(Int8ImmPtr, int8_t)

class Int16Imm : public IntergerImm {
 public:
  Int16Imm() : IntergerImm(kInt16), v_(0) {}
  explicit Int16Imm(int16_t v) : IntergerImm(kInt16), v_(v) { hash_ = std::hash<int>{}(v_); }
  ~Int16Imm() override = default;
  MS_DECLARE_PARENT(Int16Imm, IntergerImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == 0; }
  bool IsOne() override { return v_ == 1; }
  int16_t value() const { return v_; }
  bool operator==(const Value &other) const override;
  bool operator==(const Int16Imm &other) const;
  std::string ToString() const override { return std::to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "I16(" << v_ << ")";
    return oss.str();
  }

 private:
  int16_t v_;
};
using Int16ImmPtr = std::shared_ptr<Int16Imm>;
IMM_TRAITS(Int16ImmPtr, int16_t)

class Int32Imm : public IntergerImm {
 public:
  Int32Imm() : IntergerImm(kInt32), v_(0) {}
  explicit Int32Imm(int v) : IntergerImm(kInt32), v_(v) { hash_ = std::hash<int>{}(v_); }
  ~Int32Imm() override = default;
  MS_DECLARE_PARENT(Int32Imm, IntergerImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == 0; }
  bool IsOne() override { return v_ == 1; }
  int32_t value() const { return v_; }
  bool operator==(const Value &other) const override;
  bool operator==(const Int32Imm &other) const;
  std::string ToString() const override { return std::to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "I32(" << v_ << ")";
    return oss.str();
  }

 private:
  int32_t v_;
};
using Int32ImmPtr = std::shared_ptr<Int32Imm>;
IMM_TRAITS(Int32ImmPtr, int32_t)

class Int64Imm : public IntergerImm {
 public:
  Int64Imm() : IntergerImm(kInt64), v_(0) {}
  explicit Int64Imm(int64_t v) : IntergerImm(kInt64), v_(v) { hash_ = std::hash<int64_t>{}(v_); }
  ~Int64Imm() override = default;
  MS_DECLARE_PARENT(Int64Imm, IntergerImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == 0; }
  bool IsOne() override { return v_ == 1; }
  int64_t value() const { return v_; }
  bool operator==(const Value &other) const override;
  bool operator==(const Int64Imm &other) const;
  std::string ToString() const override { return std::to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "I64(" << v_ << ")";
    return oss.str();
  }

 private:
  int64_t v_;
};
using Int64ImmPtr = std::shared_ptr<Int64Imm>;
IMM_TRAITS(Int64ImmPtr, int64_t)

class UInt8Imm : public IntergerImm {
 public:
  UInt8Imm() : IntergerImm(kUInt8), v_(0) {}
  explicit UInt8Imm(uint8_t v) : IntergerImm(kUInt8), v_(v) { hash_ = std::hash<unsigned int>{}(v_); }
  ~UInt8Imm() override = default;
  MS_DECLARE_PARENT(UInt8Imm, IntergerImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == 0; }
  bool IsOne() override { return v_ == 1; }
  uint8_t value() const { return v_; }
  bool operator==(const Value &other) const override;
  bool operator==(const UInt8Imm &other) const;
  std::string ToString() const override { return std::to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "U8(" << v_ << ")";
    return oss.str();
  }

 private:
  uint8_t v_;
};
using UInt8ImmPtr = std::shared_ptr<UInt8Imm>;
IMM_TRAITS(UInt8ImmPtr, uint8_t);

class UInt16Imm : public IntergerImm {
 public:
  UInt16Imm() : IntergerImm(kUInt16), v_(0) {}
  explicit UInt16Imm(uint16_t v) : IntergerImm(kUInt16), v_(v) { hash_ = std::hash<unsigned int>{}(v_); }
  ~UInt16Imm() override = default;
  MS_DECLARE_PARENT(UInt16Imm, IntergerImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == 0; }
  bool IsOne() override { return v_ == 1; }
  uint16_t value() const { return v_; }
  bool operator==(const Value &other) const override;
  bool operator==(const UInt16Imm &other) const;
  std::string ToString() const override { return std::to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "U16(" << v_ << ")";
    return oss.str();
  }

 private:
  uint16_t v_;
};
using UInt16ImmPtr = std::shared_ptr<UInt16Imm>;
IMM_TRAITS(UInt16ImmPtr, uint16_t);

class UInt32Imm : public IntergerImm {
 public:
  UInt32Imm() : IntergerImm(kUInt32), v_(0) {}
  explicit UInt32Imm(uint32_t v) : IntergerImm(kUInt32), v_(v) { hash_ = std::hash<unsigned int>{}(v_); }
  ~UInt32Imm() override = default;
  MS_DECLARE_PARENT(UInt32Imm, IntergerImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == 0; }
  bool IsOne() override { return v_ == 1; }
  uint32_t value() const { return v_; }
  bool operator==(const Value &other) const override;
  bool operator==(const UInt32Imm &other) const;
  std::string ToString() const override { return std::to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "U32(" << v_ << ")";
    return oss.str();
  }

 private:
  uint32_t v_;
};
using UInt32ImmPtr = std::shared_ptr<UInt32Imm>;
IMM_TRAITS(UInt32ImmPtr, uint32_t);

class UInt64Imm : public IntergerImm {
 public:
  UInt64Imm() : IntergerImm(kUInt64), v_(0) {}
  explicit UInt64Imm(uint64_t v) : IntergerImm(kUInt64), v_(v) { hash_ = std::hash<uint64_t>{}(v); }
  ~UInt64Imm() override = default;
  MS_DECLARE_PARENT(UInt64Imm, IntergerImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return v_ == 0; }
  bool IsOne() override { return v_ == 1; }
  uint64_t value() const { return v_; }
  bool operator==(const Value &other) const override;
  bool operator==(const UInt64Imm &other) const;
  std::string ToString() const override { return std::to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "U64(" << v_ << ")";
    return oss.str();
  }

 private:
  uint64_t v_;
};
using UInt64ImmPtr = std::shared_ptr<UInt64Imm>;
IMM_TRAITS(UInt64ImmPtr, uint64_t);

class FloatImm : public Scalar {
 public:
  FloatImm() = default;
  explicit FloatImm(const TypePtr &t) : Scalar(t) {}
  ~FloatImm() override = default;
  MS_DECLARE_PARENT(FloatImm, Scalar)
};
using FloatImmPtr = std::shared_ptr<FloatImm>;

class FP32Imm : public FloatImm {
 public:
  FP32Imm() : FloatImm(kFloat32), v_(0.0) {}
  explicit FP32Imm(float v) : FloatImm(kFloat32), v_(v) { hash_ = std::hash<float>{}(v_); }
  ~FP32Imm() override = default;
  MS_DECLARE_PARENT(FP32Imm, FloatImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return fabs(v_) <= FLT_EPSILON; }
  bool IsOne() override { return fabs(v_ - 1.0) <= FLT_EPSILON; }
  float value() const { return v_; }
  bool operator==(const Value &other) const override;
  bool operator==(const FP32Imm &other) const;
  std::string ToString() const override { return std::to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "F32(" << v_ << ")";
    return oss.str();
  }

 private:
  float v_;
};
using FP32ImmPtr = std::shared_ptr<FP32Imm>;
IMM_TRAITS(FP32ImmPtr, float)

class FP64Imm : public FloatImm {
 public:
  FP64Imm() : FloatImm(kFloat64), v_(0.0) {}
  explicit FP64Imm(double v) : FloatImm(kFloat64), v_(v) { hash_ = std::hash<double>{}(v_); }
  ~FP64Imm() override = default;
  MS_DECLARE_PARENT(FP64Imm, FloatImm)
  std::size_t hash() const override { return hash_; }
  bool IsZero() override { return fabs(v_) <= DBL_EPSILON; }
  bool IsOne() override { return fabs(v_ - 1.0) <= DBL_EPSILON; }
  double value() const { return v_; }
  bool operator==(const Value &other) const override;
  bool operator==(const FP64Imm &other) const;
  std::string ToString() const override { return std::to_string(v_); }

  std::string DumpText() const override {
    std::ostringstream oss;
    oss << "F64(" << v_ << ")";
    return oss.str();
  }

 private:
  double v_;
};
using FP64ImmPtr = std::shared_ptr<FP64Imm>;
IMM_TRAITS(FP64ImmPtr, double)

}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_IR_SCALAR_H_
