# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test schedules for optimizers that read the shared learning-rate Parameters.

These helpers only update ``optimizer.lrs``. They do not update a cached Python
learning rate such as ``mint.optim.SGD``'s ``lr_float``. For that optimizer, a
changed lr Parameter does not establish an effective scheduled parameter update.
"""
from mindspore import ops


def scale_learning_rates(optimizer, factor):
    """Update shared lr Parameters in place; cached scalar learning rates are unchanged."""
    for lr in optimizer.lrs:
        ops.assign(lr, lr * factor)


def apply_step_decay(optimizer, completed_epochs, step_size, gamma):
    """Apply decay after completed epochs, preserving external changes between steps."""
    if completed_epochs % step_size == 0:
        scale_learning_rates(optimizer, gamma)
