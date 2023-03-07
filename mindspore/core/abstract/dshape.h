/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_CORE_ABSTRACT_DSHAPE_H_
#define MINDSPORE_CORE_ABSTRACT_DSHAPE_H_

#include <vector>
#include <string>
#include <sstream>
#include <typeindex>
#include <memory>
#include <algorithm>

#include "utils/hashing.h"
#include "utils/log_adapter.h"
#include "base/base.h"
#include "mindapi/base/shape_vector.h"

namespace mindspore {
namespace abstract {
class BaseShape;
using BaseShapePtr = std::shared_ptr<BaseShape>;
using BaseShapePtrList = std::vector<BaseShapePtr>;

/// \brief BaseShape defines the basic virtual class of NoShape and Shape classes.
class MS_CORE_API BaseShape : public Base {
 public:
  /// \brief Constructor of BaseShape.
  BaseShape() = default;

  /// \brief Destructor of BaseShape.
  ~BaseShape() override = default;

  MS_DECLARE_PARENT(BaseShape, Base)

  /// \brief Check whether 2 objects are equal.
  ///
  /// \param[in] other Another object.
  /// \return True if current object is equal to another, otherwise false.
  virtual bool operator==(const BaseShape &other) const;

  /// \brief Check whether 2 objects are not equal.
  ///
  /// \param[in] other Another object.
  /// \return True if current object is not equal to another, otherwise false.
  bool operator!=(const BaseShape &other) const;

  /// \brief Calculate the hash value of BaseShape.
  ///
  /// \return The hash value of BaseShape.
  std::size_t hash() const override { return tid(); }

  /// \brief Whether the object's dimensions are dynamic.
  ///
  /// \return True if the object's dimensions are dynamic, otherwise false.
  virtual bool IsDynamic() const = 0;

  /// \brief Whether the object's dimension is zero.
  ///
  /// \return True if the object's dimension is zero, otherwise false.
  virtual bool IsDimZero() const = 0;

  /// \brief Whether the object's dimensions are unknown.
  ///
  /// \return True if the object's dimensions are unknown, otherwise false.
  virtual bool IsDimUnknown() const = 0;

  /// \brief Clone a new object by this one.
  ///
  /// \return New cloned object.
  virtual BaseShapePtr Clone() const = 0;

  /// \brief Broaden the shape.
  virtual void Broaden() {}
};

/// \brief NoShape defines an invalid shape.
class MS_CORE_API NoShape final : public BaseShape {
 public:
  MS_DECLARE_PARENT(NoShape, BaseShape)

  BaseShapePtr Clone() const override { return std::make_shared<NoShape>(); }

  /// \brief Get the description string about the NoShape object.
  ///
  /// \return The description string about the NoShape object.
  std::string ToString() const override { return type_name(); }

  bool IsDynamic() const override { return false; }

  bool IsDimZero() const override { return true; };

  bool IsDimUnknown() const override { return false; }
};

GVAR_DEF(std::shared_ptr<NoShape>, kNoShape, std::make_shared<NoShape>());

/// \brief Shape defines dimensions of tensor.
class MS_CORE_API Shape final : public BaseShape {
 public:
  static constexpr ShapeValueDType kShapeDimAny = -1;
  static constexpr ShapeValueDType kShapeRankAny = -2;
  static constexpr ShapeValueDType kShapeError = -3;
  static constexpr size_t kDynamicRankLen = 1;

  /// \brief Constructor of Shape.
  Shape() : shape_() {}

  /// \brief Constructor of Shape.
  ///
  /// \param[in] list Initial shape dimensions.
  Shape(const std::initializer_list<ShapeValueDType> &list) : shape_(list) {}

  /// \brief Constructor of Shape.
  ///
  /// \param[in] list Initial shape dimensions.
  explicit Shape(const ShapeVector &list) : shape_(list) {}

  /// \brief Constructor of Shape.
  ///
  /// \param[in] list Initial shape dimensions.
  /// \param[in] max_shape Maximum shape dimensions of dynamic shape.
  Shape(const ShapeVector &list, const ShapeVector &max_shape) : shape_(list), max_shape_(max_shape) {}

  /// \brief Destructor of Shape.
  ~Shape() override = default;
  MS_DECLARE_PARENT(Shape, BaseShape)

  /// \brief Calculate the hash value for Shape.
  ///
  /// \return The hash value of Shape.
  std::size_t hash() const override {
    auto hash_code = static_cast<std::size_t>(tid());
    for (auto dim : shape_) {
      hash_code = hash_combine(hash_code, static_cast<size_t>(dim));
    }
    return hash_code;
  }

  /// \brief Get the description string about the Shape object.
  ///
  /// \return The description string about the Shape object.
  std::string ToString() const override;

  /// \brief Get the debug information about the Shape object.
  ///
  /// \return The debug information about the Shape object.
  std::string DumpText() const override;

  bool operator==(const BaseShape &other) const override;

  BaseShapePtr Clone() const override { return std::make_shared<Shape>(shape_, max_shape_); }

  void Broaden() override;

  /// \brief Set shape dimensions of Shape object.
  ///
  /// \param[in] shape Dimensions of shape.
  void set_shape(const ShapeVector &shape) { shape_ = shape; }

  /// \brief Get shape dimensions.
  ///
  /// \return Shape dimensions.
  const ShapeVector &shape() const { return shape_; }

  /// \brief Get maximum shape dimensions.
  ///
  /// \return Maximum shape dimensions.
  const ShapeVector &max_shape() const { return max_shape_; }

  bool IsDynamic() const override;

  bool IsDimZero() const override { return shape_.empty(); };

  bool IsDimUnknown() const override {
    return std::any_of(shape_.begin(), shape_.end(), [](ShapeValueDType s) { return s < -1; });
  }

 private:
  ShapeVector shape_;      // use kShapeDimAny to implement the any shape in python
  ShapeVector max_shape_;  // record maximum length for each dynamic dimension
};
using ShapePtr = std::shared_ptr<Shape>;
using ShapePtrList = std::vector<ShapePtr>;

/// \brief DynamicSequenceShape defines shape of dynamic sequence.
class MS_CORE_API DynamicSequenceShape : public BaseShape {
 public:
  /// \brief Constructor of DynamicSequenceShape.
  DynamicSequenceShape() = default;

  /// \brief Constructor of DynamicSequenceShape.
  explicit DynamicSequenceShape(const BaseShapePtr &element_shape) : element_shape_(element_shape) {}

  /// \brief Destructor of DynamicSequenceShape.
  ~DynamicSequenceShape() override = default;
  MS_DECLARE_PARENT(DynamicSequenceShape, BaseShape);

  /// \brief Get the description string about the DynamicSequenceShape object.
  ///
  /// \return The description string about the DynamicSequenceShape object.
  std::string ToString() const override { return type_name(); }

  /// \brief Check whether any element shape of DynamicSequenceShape is dynamic shape or dynamic rank.
  ///
  /// \return True if any element shape of DynamicSequenceShape is dynamic shape or dynamic rank, otherwise false.
  bool IsDynamic() const override;

  /// \brief Check whether all elements shape of DynamicSequenceShape are empty shape.
  ///
  /// \return True if all elements shape of DynamicSequenceShape are empty shape.
  bool IsDimZero() const override;

  /// \brief Check whether any element shape of DynamicSequenceShape is dynamic shape.
  ///
  /// \return True if any element shape of DynamicSequenceShape is dynamic shape.
  bool IsDimUnknown() const override;

  BaseShapePtr Clone() const override { return std::make_shared<DynamicSequenceShape>(element_shape_->Clone()); }

  bool operator==(const BaseShape &other) const override;

  /// \brief Calculate the hash value for DynamicSequenceShape.
  ///
  /// \return The hash value of Shape.
  std::size_t hash() const override;

 private:
  // element's shape
  BaseShapePtr element_shape_{nullptr};
};
using DynamicSequenceShapePtr = std::shared_ptr<DynamicSequenceShape>;
GVAR_DEF(std::shared_ptr<DynamicSequenceShape>, kDynamicSequenceShape, std::make_shared<DynamicSequenceShape>());

/// \brief SequequeShape defines base class of multiple-shape classes.
class MS_CORE_API SequenceShape : public BaseShape {
 public:
  /// \brief Constructor of SequenceShape.
  SequenceShape() : p_shapes_() {}

  /// \brief Constructor of SequenceShape.
  ///
  /// \param[in]  shapes All element-shapes.
  explicit SequenceShape(const BaseShapePtrList &shapes) : p_shapes_(shapes) {}

  /// \brief Destructor of SequenceShape.
  ~SequenceShape() override = default;
  MS_DECLARE_PARENT(SequenceShape, BaseShape)

  /// \brief Get the description string about the SequenceShape object.
  ///
  /// \return The description string about the SequenceShape object.
  std::string ToString() const override;

  /// \brief Clone all element-shapes.
  ///
  /// \return New cloned element-shapes.
  BaseShapePtrList ElementsClone() const;

  /// \brief Check whether SequenceShape object is equal to a BaseShape object.
  ///
  /// \param[in] other Another SequenceShape object.
  /// \return True if current SequenceShape object is equal to another BaseShape object, otherwise false.
  template <typename T>
  bool SequenceEqual(const BaseShape &other) const {
    if (tid() != other.tid()) {
      return false;
    }
    auto other_shapes = static_cast<const T &>(other).p_shapes_;
    if (other_shapes.size() != p_shapes_.size()) {
      return false;
    }
    for (uint64_t i = 0; i < p_shapes_.size(); ++i) {
      MS_EXCEPTION_IF_NULL(p_shapes_[i]);
      MS_EXCEPTION_IF_NULL(other_shapes[i]);
      if (!(*p_shapes_[i] == *other_shapes[i])) {
        return false;
      }
    }
    return true;
  }

  /// \brief Get all element-shapes.
  ///
  /// \return  All element-shapes.
  const BaseShapePtrList &shape() const { return p_shapes_; }

  /// \brief Get the number of element-shapes.
  ///
  /// \return The number of element-shapes.
  size_t size() const { return p_shapes_.size(); }

  /// \brief Get the element-shape by index through operator '[]'.
  ///
  /// \param[in] dim The index of element shape.
  /// \return The element shape got by index.
  const BaseShapePtr &operator[](std::size_t dim) const { return p_shapes_[dim]; }

  /// \brief Check whether any element shape of DynamicSequenceShape is dynamic shape or dynamic rank.
  ///
  /// \return True if any element shape of DynamicSequenceShape is dynamic shape or dynamic rank, otherwise false.
  bool IsDynamic() const override {
    return std::any_of(p_shapes_.begin(), p_shapes_.end(), [](const BaseShapePtr &bs) { return bs->IsDynamic(); });
  }

  /// \brief Check whether all elements shape of DynamicSequenceShape are empty shape.
  ///
  /// \return True if all elements shape of DynamicSequenceShape are empty shape.
  bool IsDimZero() const override {
    return std::all_of(p_shapes_.begin(), p_shapes_.end(), [](const BaseShapePtr &bs) { return bs->IsDimZero(); });
  };

  /// \brief Check whether any element shape of DynamicSequenceShape is dynamic shape.
  ///
  /// \return True if any element shape of DynamicSequenceShape is dynamic shape.
  bool IsDimUnknown() const override {
    return std::any_of(p_shapes_.begin(), p_shapes_.end(), [](const BaseShapePtr &bs) { return bs->IsDimUnknown(); });
  }

 protected:
  BaseShapePtrList p_shapes_;  // shape list of each elements
};
using SequenceShapePtr = std::shared_ptr<SequenceShape>;

/// \brief TupleShape defines shape used by tuple with tensor inside.
class MS_CORE_API TupleShape final : public SequenceShape {
 public:
  /// \brief Constructor of TupleShape.
  TupleShape() : SequenceShape() {}

  /// \brief Constructor of TupleShape.
  ///
  /// \param[in] shapes Element-shapes of TupleShape.
  explicit TupleShape(const BaseShapePtrList &shapes) : SequenceShape(shapes) {}

  /// \brief Destructor of TupleShape.
  ~TupleShape() override = default;
  MS_DECLARE_PARENT(TupleShape, SequenceShape)

  std::string ToString() const override { return type_name() + "(" + SequenceShape::ToString() + ")"; }

  BaseShapePtr Clone() const override { return std::make_shared<TupleShape>(ElementsClone()); }

  bool operator==(const BaseShape &other) const override { return SequenceEqual<TupleShape>(other); }
};
using TupleShapePtr = std::shared_ptr<TupleShape>;

/// \brief ListShape defines shape used by list with tensor inside.
class MS_CORE_API ListShape final : public SequenceShape {
 public:
  /// \brief Constructor of ListShape.
  ListShape() : SequenceShape() {}
  /// \brief Constructor of ListShape.
  ///
  /// \param[in] shapes Element-shapes of ListShape.
  explicit ListShape(const BaseShapePtrList &shapes) : SequenceShape(shapes) {}

  /// \brief Destructor of ListShape.
  ~ListShape() override = default;
  MS_DECLARE_PARENT(ListShape, SequenceShape)

  std::string ToString() const override { return type_name() + "[" + SequenceShape::ToString() + "]"; }

  BaseShapePtr Clone() const override { return std::make_shared<ListShape>(SequenceShape::ElementsClone()); }

  bool operator==(const BaseShape &other) const override { return SequenceEqual<ListShape>(other); }
};
using ListShapePtr = std::shared_ptr<ListShape>;
}  // namespace abstract
}  // namespace mindspore

#endif  // MINDSPORE_CORE_ABSTRACT_DSHAPE_H_
