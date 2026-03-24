---
name: ms-fusion-pass
description: Add a new IR optimization or fusion pass to MindSpore's compiler pipeline. Use when implementing a graph-level optimization, fusing multiple ops into one, eliminating redundant ops, or adding a backend actor fusion. Triggers on phrases like "add fusion pass", "graph optimization", "IR pass", "optimize graph", "eliminate op", "fuse ops", "add pass", "compiler optimization", "AnfVisitor", "OptimizerCaller".
---

# MindSpore Fusion Pass Development

MindSpore has two types of passes depending on when they run:
- **Frontend IR passes** — pattern-match and rewrite the ANF FuncGraph (before backend)
- **Backend actor passes** — optimize the runtime actor graph (after backend compilation)

Most new optimizations are frontend IR passes.

## Frontend IR Pass

### 1. Create the pass class

File: `mindspore/ccsrc/frontend/optimizer/irpass/{your_pass}.h`

```cpp
#pragma once
#include "mindspore/ccsrc/frontend/optimizer/anf_visitor.h"
#include "mindspore/ccsrc/frontend/optimizer/optimizer_caller.h"
#include "ir/primitive.h"
#include "ops/core_ops.h"

namespace mindspore::opt::irpass {

// Pattern: TargetOp(CastOp(x), ...) -> TargetOp(x, ...) if types match
class YourFusionPass : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override;
  void Visit(const CNodePtr &cnode) override;
  void Visit(const AnfNodePtr &node) override;

 private:
  void Reset() {
    x_ = nullptr;
    is_match_ = false;
  }

  AnfNodePtr x_{nullptr};
  bool is_match_{false};
};

}  // namespace mindspore::opt::irpass
```

File: `mindspore/ccsrc/frontend/optimizer/irpass/{your_pass}.cc`

```cpp
#include "frontend/optimizer/irpass/{your_pass}.h"
#include "frontend/optimizer/irpass/symbol_resolver.h"
#include "ir/func_graph.h"
#include "pipeline/jit/ps/pass.h"

namespace mindspore::opt::irpass {

AnfNodePtr YourFusionPass::operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
  Reset();
  // Match pattern: look for the target op
  AnfVisitor::Match(prim::kPrimYourTargetOp, {IsCNode, IsNode})(node);

  if (!is_match_ || x_ == nullptr) return nullptr;  // no match, return nullptr

  auto fg = node->func_graph();
  if (fg == nullptr) return nullptr;

  // Build the rewritten node:
  auto new_node = fg->NewCNode({NewValueNode(prim::kPrimReplacementOp), x_});
  new_node->set_abstract(node->abstract());  // preserve type info
  return new_node;
}

void YourFusionPass::Visit(const CNodePtr &cnode) {
  // Called for each CNode child when matching pattern
  // Check if this is the inner op we want to fuse
  if (!IsPrimitiveCNode(cnode, prim::kPrimCast)) return;
  auto input = cnode->input(1);  // first real input (index 0 = the primitive)
  // ... check types, set x_ if pattern matches
  x_ = input;
  is_match_ = true;
}

void YourFusionPass::Visit(const AnfNodePtr &node) {
  // Called for non-CNode children
}

}  // namespace mindspore::opt::irpass
```

### 2. Register in irpass.h / irpass.cc

File: `mindspore/ccsrc/frontend/optimizer/irpass.h` — Add member declaration:

```cpp
class OptimizeIRPassLib {
 public:
  // ... existing members ...
  SubstitutionPtr your_fusion_pass_;
};
```

File: `mindspore/ccsrc/frontend/optimizer/irpass.cc` — Add in constructor:

```cpp
OptimizeIRPassLib::OptimizeIRPassLib() {
  // ... existing ...
  your_fusion_pass_ = MakeSubstitution(
    std::make_shared<YourFusionPass>(),
    "your_fusion_pass",
    prim::kPrimYourTargetOp  // only run on nodes with this primitive
  );
}
```

The third argument to `MakeSubstitution` is the **pattern predicate** — it filters which nodes this pass is applied to. Options:

```cpp
prim::kPrimCast          // Only CNodes with primitive Cast
{prim::kPrimCast, prim::kPrimAdd}   // Multiple primitives
IsNode                   // All nodes
```

### 3. Add to an optimizer group

Find where to insert in `mindspore/ccsrc/frontend/optimizer/opt.cc` or the pass pipeline configuration. Look for existing `OptPassGroupMap` or `GetOptPassA()` style functions:

```cpp
// In the appropriate pass group:
auto irpass = std::make_shared<irpass::OptimizeIRPassLib>();
OptPassGroupMap map({
  // ...existing passes...
  {"your_fusion_pass", OptPassConfig({irpass->your_fusion_pass_})},
});
```

Alternatively, add to an existing pass group:
```cpp
// Find the relevant opt level (e.g., opt_a, opt_b):
auto opt_pass_group = GetOptPassA(irpass);
// Add your substitution to the appropriate group list
```

### 4. Include the new header

Add `#include "frontend/optimizer/irpass/{your_pass}.h"` in `irpass.cc`.

### 5. Add to CMakeLists.txt

In `mindspore/ccsrc/frontend/optimizer/irpass/CMakeLists.txt`:
```cmake
list(APPEND OBJ_LIST ${CMAKE_CURRENT_SOURCE_DIR}/{your_pass}.cc)
```

## Key Helper Functions

```cpp
// Check if a node is a specific primitive CNode
IsPrimitiveCNode(node, prim::kPrimCast)

// Get primitive from a CNode
auto prim = GetCNodePrimitive(cnode);

// Check node type
node->isa<CNode>()
node->isa<ValueNode>()
node->isa<Parameter>()

// Get abstract type info
auto type = node->abstract()->cast<abstract::AbstractTensorPtr>();
auto dtype = type->element()->BuildType()->type_id();

// Create a new CNode
fg->NewCNode({prim_node, input1, input2})

// Create a ValueNode from primitive
NewValueNode(prim::kPrimAdd)
NewValueNode(MakeValue(1.0f))
```

## Simple Elimination Pass Example

Remove `Cast(x, same_type)` when input and output types are identical:

```cpp
AnfNodePtr CastSameTypeElim::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  // node is Cast(x, dtype)
  if (!IsPrimitiveCNode(node, prim::kPrimCast)) return nullptr;
  auto cnode = node->cast<CNodePtr>();
  auto input = cnode->input(1);   // x
  auto dst_type_node = cnode->input(2);  // dtype

  auto src_type = input->abstract()->GetType();
  auto dst_type = GetValue<TypePtr>(GetValueNode(dst_type_node));
  if (*src_type == *dst_type) return input;  // eliminate cast

  return nullptr;  // keep cast
}
```

## Backend Actor Pass (Advanced)

For runtime-level optimizations, inherit from `ActorPass`:

```cpp
// Header: ccsrc/backend/ms_backend/runtime/graph_scheduler/optimizer/{pass}.h
#include "backend/ms_backend/runtime/graph_scheduler/optimizer/optimizer.h"

class YourActorFusion : public ActorPass {
 public:
  YourActorFusion() : ActorPass("your_actor_fusion") {}
 protected:
  bool MatchPattern(const AbstractActor *actor) const override;
  void Process(ActorSet *const actor_set, AbstractActor *const actor) override;
};
```

Register in the optimizer:
```cpp
// In graph_scheduler/optimizer/optimizer.cc:
optimizer->AddPass(std::make_shared<YourActorFusion>());
```

## Testing a Pass

```python
import mindspore as ms
import mindspore.ops as ops
import os

os.environ["MS_DEV_SAVE_GRAPHS"] = "1"
os.environ["MS_DEV_SAVE_GRAPHS_PATH"] = "/tmp/graphs"
ms.set_context(mode=ms.GRAPH_MODE)

@ms.jit
def test_fn(x):
    # Pattern your pass should optimize
    return ops.cast(ops.cast(x, ms.float16), ms.float32)

x = ms.Tensor([1.0, 2.0], ms.float32)
result = test_fn(x)
# Check /tmp/graphs/ for before/after IR graphs
```

## Checklist

- [ ] Pass class in `irpass/{your_pass}.h/.cc`
- [ ] Added to `irpass.h` (member) and `irpass.cc` (constructor + MakeSubstitution)
- [ ] Added to appropriate pass group in optimizer pipeline
- [ ] Added `.cc` to `CMakeLists.txt`
- [ ] Test: pattern is correctly matched
- [ ] Test: output is semantically equivalent to input
- [ ] Test: pass does NOT fire on non-matching patterns
