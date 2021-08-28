# Copyright 2021 Huawei Technologies Co., Ltd
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
# ===========================================================================
"""DeepLM solver."""
from time import time

from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.composite import GradOperation
from mindspore.nn import Cell
from mindspore.common.api import _pynative_executor

from . import jacobian as jb
from . import listvec as lv

_zeros = P.Zeros()


class Strategy:
    """LM solver strategy"""
    def __init__(self):
        self.decrease_factor = 2
        self.radius = 1e4

    def reject(self):
        self.radius /= self.decrease_factor
        self.decrease_factor *= 2

    def accept(self, quality):
        self.radius /= max(1.0 / 3.0, 1.0 - pow(2.0 * quality - 1.0, 3))
        self.decrease_factor = 2.0


class StepEvaluator:
    """step evaluator"""
    def __init__(self, reference_cost):
        self.reference_cost = reference_cost
        self.minimum_cost = reference_cost
        self.current_cost = reference_cost
        self.candidate_cost = reference_cost

        self.accumulated_reference_model_cost_change = 0
        self.accumulated_candidate_model_cost_change = 0
        self.num_consecutive_nonmonotonic_steps = 0
        self.max_consecutive_nonmonotonic_steps = 0

    def accept(self, cost, model_cost_change):
        """how step evaluator to accept"""
        self.current_cost = cost
        self.accumulated_candidate_model_cost_change += model_cost_change
        self.accumulated_reference_model_cost_change += model_cost_change
        if self.current_cost < self.minimum_cost:
            self.minimum_cost = self.current_cost
            self.num_consecutive_nonmonotonic_steps = 0
            self.candidate_cost = self.current_cost
            self.accumulated_candidate_model_cost_change = 0
        else:
            self.num_consecutive_nonmonotonic_steps += 1
            if self.current_cost > self.candidate_cost:
                self.candidate_cost = self.current_cost
                self.accumulated_candidate_model_cost_change = 0

        if self.num_consecutive_nonmonotonic_steps == \
                self.max_consecutive_nonmonotonic_steps:
            self.reference_cost = self.candidate_cost
            self.accumulated_reference_model_cost_change = \
                self.accumulated_candidate_model_cost_change

    def step_quality(self, cost, model_cost_change):
        relative_decrease = (self.current_cost - cost) / model_cost_change
        historical_relative_decrease = (self.reference_cost - cost) / (
            self.accumulated_reference_model_cost_change + model_cost_change)

        return max(relative_decrease, historical_relative_decrease)


class Summary:
    """summary"""
    def __init__(self):
        self.cost = 0
        self.gradient_norm = 0
        self.gradient_max_norm = 0
        self.cost = None
        self.lm_iteration = 0
        self.num_consecutive_invalid_steps = 0
        self.step_is_valid = True
        self.relative_decrease = 0
        # 0 = no-converge, 1 = success, 2 = fail
        self.linear_termination_type = 0
        self.lm_termination_type = 0


class LMOption:
    """LM solver options"""
    def __init__(self):
        self.min_diagonal = 1e-6
        self.max_diagonal = 1e32
        self.eta = 1e-3
        self.residual_reset_period = 10
        self.max_linear_iterations = 150
        self.max_num_iterations = 16
        self.max_success_iterations = 16
        self.max_invalid_step = 5
        self.min_relative_decrease = 1e-3
        self.parameter_tolerance = 1e-8
        self.function_tolerance = 1e-6
        self.radius = 1e4
        self.span = 4000000


class FunctionBlock:
    def __init__(self, variables, constants, indices, fn=None):
        self.variables = variables
        self.constants = constants
        self.indices = indices
        self.fn = fn


class ResidualFunc(Cell):
    """residual func"""
    def __init__(self, fn, constant_para, res_index=0):
        super(ResidualFunc, self).__init__()
        self.fn = fn
        self.constant_para = constant_para
        self.reduce_sum = F.reduce_sum
        self.mul = F.mul
        self.reshape = F.reshape
        self.res_index = res_index

    def construct(self, *x):
        residuals = self.fn(*x, *self.constant_para)
        residuals = self.reshape(residuals, (residuals.shape[0], -1))
        return self.reduce_sum(residuals[:, self.res_index])


class LMSolver:
    """LM solver"""
    def __init__(self, functions, verbose=True, option=LMOption()):
        self.start_time = time()
        self.verbose = verbose

        self.functions = functions
        self.variable_dict = {}
        self.variables = []
        self.vrange_func = []
        for func in functions:
            for v in func.variables:
                if v in self.variable_dict:
                    continue
                self.variable_dict[v] = len(self.variables)
                self.variables.append(v)
            self.vrange_func.append(range(len(func.variables)))
        self.vranges = range(len(self.variables))

        self.option = option
        self.vids_func = []
        self.res_funcs = []
        for func in functions:
            self.vids_func.append([self.variable_dict[v] for v in func.variables])
            for _, dim in enumerate(range(0, func.indices[0].shape[0], self.option.span)):
                start = dim
                end = start + self.option.span
                if end > func.indices[0].shape[0]:
                    end = func.indices[0].shape[0]
                constants_para = [func.constants[i][start:end] for i in range(len(func.constants))]

                self.res_funcs.append(
                    [ResidualFunc(func.fn, constants_para, index) for index in self.vranges])

        # some mindspore OPs which may be used below
        self.zeros_like = F.zeros_like
        self.reduce_sum = F.reduce_sum
        self.grad_op = GradOperation(get_all=True)

        self.summary = Summary()
        self.strategy = Strategy()
        self.strategy.radius = option.radius
        self.evaluator = StepEvaluator(0)

        self.q_tolerance = 0.0
        self.r_tolerance = 0.0
        self.x_norm = -1
        self.model_cost_change = 0.0
        self.delta = None

        self.q = None
        self.z = None
        self.p = None
        self.r = None
        self.xref = None
        self.bref = None
        self.model_residuals = None
        self.gradients = None
        self.jacobians = None
        self.preconditioner = None

    def memory_tensor(self, b):
        k = 8
        for i in range(len(b.shape)):
            k *= b.shape[i]
        return k

    def memory_list(self, b):
        mem = 0
        for l in b:
            if isinstance(l, list):
                mem += self.memory_list(l)
            elif isinstance(l, Tensor):
                mem += self.memory_tensor(l)
        return mem

    def memory(self):
        mem = 0
        for _, b in self.__dict__.items():
            if isinstance(b, list):
                mem += self.memory_list(b)
            if isinstance(b, Tensor):
                mem += self.memory_tensor(b)
        return mem

    def timing(self):
        _pynative_executor.sync()
        return time()

    def initialize_variables(self):
        """initialize variables to zeros"""
        self.bref = lv.list_zero(self.variables)
        self.xref = lv.list_zero(self.variables)
        self.r = lv.list_zero(self.variables)
        self.z = lv.list_zero(self.variables)
        self.p = lv.list_zero(self.variables)
        self.q = lv.list_zero(self.variables)
        self.gradients = lv.list_zero(self.variables)
        self.jacobian_scale = lv.list_zero(self.variables)
        self.diagonal = lv.list_zero(self.variables)
        self.preconditioner = []
        for v in self.variables:
            l = 1
            for j in range(1, len(v.shape)):
                l *= v.shape[j]

            self.preconditioner.append(_zeros((v.shape[0], l, l), v.dtype))

    def evaluate_cost(self, candidate):
        """evaluate cost"""
        span = self.option.span
        cost = 0

        for func_id in range(len(self.functions)):
            func = self.functions[func_id]
            indices = func.indices
            variables = [candidate[j] for j in self.vids_func[func_id]]
            constants = func.constants
            residual_num = indices[0].shape[0]
            for dim in range(0, residual_num, span):
                start = dim
                end = start + span
                if end > residual_num:
                    end = residual_num

                varIndexed = [variables[i][indices[i][start:end]] \
                              for i in range(len(variables))]
                constants_para = [constants[i][start:end] \
                                  for i in range(len(constants))]

                residuals = func.fn(*varIndexed, *constants_para)
                cost += self.reduce_sum(0.5 * residuals * residuals)

        return cost

    def gradient(self, res_funcs_index, residual_index, varIndexed):
        """grad net"""
        res_func = self.res_funcs[res_funcs_index][residual_index]
        grad_net = self.grad_op(res_func)
        return grad_net(*varIndexed)

    def evaluate(self, is_first_time=False):
        """evaluate process"""
        for i in self.vranges:
            self.variables[i].grad = None
            self.gradients[i] = F.zeros_like(self.gradients[i])
            self.jacobian_scale[i] = F.zeros_like(self.jacobian_scale[i])

        self.cost = 0

        for func_id in range(len(self.functions)):
            indices = self.functions[func_id].indices
            variables = self.variables
            constants = self.functions[func_id].constants
            fn = self.functions[func_id].fn
            vrange = self.vrange_func[func_id]

            residual_num = indices[0].shape[0]

            if is_first_time:
                var_temp = [variables[i][indices[i][:1]] for i in vrange]
                constant_temp = [constants[i][:1] \
                                 for i in range(len(constants))]
                # t0 = self.timing()

                residuals_temp = fn(*var_temp, *constant_temp)
                # t1 = self.timing()
                residuals_temp = F.reshape(residuals_temp, (residuals_temp.shape[0], -1))

                residual_dim = residuals_temp.shape[1]

                self.functions[func_id].jacobians = []
                max_dim = 0
                for i in vrange:
                    v = variables[i]
                    v = F.reshape(v, (v.shape[0], -1))
                    jacobian = _zeros((residual_dim, *indices[i].shape, *v.shape[1:]), v.dtype)

                    if v.shape[1] > max_dim:
                        max_dim = v.shape[1]
                    self.functions[func_id].jacobians.append(jacobian)

                self.functions[func_id].buffer = _zeros(
                    (indices[0].shape[0], max_dim), variables[0].dtype)

                self.functions[func_id].residuals = _zeros(
                    (residual_num, residual_dim), variables[0].dtype)

            span = self.option.span

            for _index, dim in enumerate(range(0, residual_num, span)):
                start = dim
                end = start + span
                if end > residual_num:
                    end = residual_num
                varIndexed = [variables[i][indices[i][start:end]] for i in vrange]

                constants_para = [constants[i][start:end] for i in range(len(constants))]

                residuals = fn(*varIndexed, *constants_para)
                residuals = F.reshape(residuals, (residuals.shape[0], -1))

                cost = self.reduce_sum(0.5 * residuals * residuals)
                # collect gradients and jacobians

                for i in vrange:
                    grads = self.gradient(_index, i, varIndexed)

                    for j in vrange:
                        grad = grads[j]
                        if grad is None:
                            self.functions[func_id].jacobians[j][i, start:end] = self.zeros_like(
                                self.functions[func_id].jacobians[j][i, start:end]
                            )
                            continue

                        grad = F.reshape(grad, (grad.shape[0], grad.shape[1], -1))

                        self.functions[func_id].jacobians[j][i, start:end] = grad
                self.functions[func_id].residuals[start:end] = F.tensor_mul(residuals, 1.0)

                self.cost += cost

            residuals = self.functions[func_id].residuals

            gradients = [self.gradients[i] for i in self.vids_func[func_id]]
            jacobians = self.functions[func_id].jacobians

            jb.jacobi_left_multiply(jacobians, residuals, self.variables, indices, gradients)

            self.jacobian_scale = [self.jacobian_scale[i] \
                                   for i in self.vids_func[func_id]]
            self.jacobian_scale = jb.jacobi_column_square(indices, jacobians, self.jacobian_scale)

        jb.column_inverse_square(self.jacobian_scale)

        for func_id in range(len(self.functions)):
            indices = self.functions[func_id].indices
            jacobians = self.functions[func_id].jacobians
            self.jacobian_scale = jb.jacobi_normalize(jacobians, indices, self.variables)

        self.summary.gradient_norm = lv.list_norm(gradients)
        self.summary.gradient_max_norm = lv.list_max_norm(gradients)

    def preconditioner_initialize(self):
        for i in range(len(self.preconditioner)):
            self.preconditioner[i] = self.zeros_like(self.preconditioner[i])

    def linear_solve(self, lmDiagonal):
        """linear solver"""
        xref = self.xref
        bref = self.bref
        r = self.r
        z = self.z
        p = self.p

        for i in range(len(self.functions)):
            jacobians = self.functions[i].jacobians
            residuals = self.functions[i].residuals

            indices = self.functions[i].indices

            bref = jb.jacobi_left_multiply(jacobians, residuals, self.variables, indices, bref)
        self.preconditioner_initialize()

        for i in range(len(self.functions)):
            jacobians = self.functions[i].jacobians
            indices = self.functions[i].indices

            lmDiagonalTemp = [lmDiagonal[j] for j in self.vids_func[i]]

            jb.jacobi_block_jt(jacobians, lmDiagonalTemp, indices, self.preconditioner)

        lv.list_invert(self.preconditioner)

        self.summary.linear_termination_type = 0
        self.summary.lm_iteration = 0

        norm_b = lv.list_norm(bref)

        for i in self.vranges:
            xref[i] = self.zeros_like(xref[i])
            r[i] = F.mul(bref[i], 1)

        if norm_b == 0:
            self.summary.linear_termination_type = 1
            return xref

        tol_r = self.r_tolerance * norm_b

        rho = 1.0
        q_0 = -0.0

        self.summary.num_iterations = 0

        while True:
            # t1 = self.timing()
            self.summary.num_iterations += 1

            lv.list_right_multiply(self.preconditioner, r, self.z)

            last_rho = rho
            rho = lv.list_dot(r, z)

            if self.summary.num_iterations == 1:
                for i in self.vranges:
                    p[i] = F.mul(z[i], 1)
                # p[i] = Tensor(z[i].asnumpy())

            else:
                beta = rho / last_rho
                for i in self.vranges:
                    p[i] *= beta
                    p[i] += z[i]

            jb.jacobi_jt_jd(jacobians, lmDiagonal, p, self.variables, indices, self.model_residuals,
                            self.q)

            pq = lv.list_dot(p, self.q)
            if pq < 0:
                self.summary.linear_termination_type = 2
                break
            # return xref

            alpha = rho / pq
            for i in self.vranges:
                xref[i] += alpha * p[i]

            # this is to avoid numercial issue: recompute r every reset steps
            if self.summary.num_iterations % \
                    self.option.residual_reset_period == 0:

                jb.jacobi_jt_jd(
                    jacobians, lmDiagonal, xref, self.variables, indices, self.model_residuals,
                    self.q)

                for i in self.vranges:
                    r[i] = F.mul((bref[i] - self.q[i]), 1)
                # r[i] = Tensor((bref[i] - self.q[i]).asnumpy())
                r = [bref[i] - self.q[i] for i in range(len(r))]
            else:
                for i in self.vranges:
                    r[i] -= F.mul(alpha, self.q[i])

            q_1 = -1.0 * lv.list_dot(xref, [bref[i] + r[i] for i in range(len(r))])
            zeta = self.summary.num_iterations * (q_1 - q_0) / q_1

            if zeta < self.q_tolerance:
                self.summary.linear_termination_type = 1
                break

            q_0 = q_1
            norm_r = lv.list_norm(r)

            if norm_r < tol_r:
                self.summary.linear_termination_type = 1
                break

            if self.summary.num_iterations > self.option.max_linear_iterations:
                break
        return xref

    def compute_trust_region_step(self):
        """step of computing the trust region"""
        for i in range(len(self.diagonal)):
            self.diagonal[i] = self.zeros_like(self.diagonal[i])

        for i in range(len(self.functions)):
            indices = self.functions[i].indices
            jacobians = self.functions[i].jacobians
            diagonal = [self.diagonal[j] for j in self.vids_func[i]]
            jb.jacobi_column_square(indices, jacobians, diagonal)

        diagonal = self.diagonal
        lv.list_clamp(diagonal, self.option.min_diagonal, self.option.max_diagonal)

        lmDiagonal = []
        for v in diagonal:
            lmDiagonal.append(F.sqrt(v / self.strategy.radius))

        self.q_tolerance = self.option.eta
        self.r_tolerance = -1.0

        step = self.linear_solve(lmDiagonal)

        for i in self.vranges:
            step[i] = -step[i]

        for i in range(len(self.functions)):
            indices = self.functions[i].indices
            jacobians = self.functions[i].jacobians
            self.model_residuals[i] = jb.jacobi_right_multiply(jacobians, step, self.variables,
                                                               indices, self.model_residuals[i])

        self.model_cost_change = 0

        for i in range(len(self.functions)):
            self.model_cost_change += -self.reduce_sum(
                self.model_residuals[i] * (
                    self.functions[i].residuals + self.model_residuals[i] * 0.5))

        self.summary.step_is_valid = (self.model_cost_change > 0.0)

        if self.summary.step_is_valid:
            self.delta = [step[i] * self.jacobian_scale[i] for i in range(len(step))]
            self.summary.num_consecutive_invalid_steps = 0

    def solve(self):
        """LM solver main func"""
        # t00 = self.timing()
        self.initialize_variables()
        # t0 = self.timing()
        self.evaluate(True)

        if self.option.max_success_iterations == 0:
            return
        if self.verbose:
            print('\nInitial cost = {}, Memory = {} G'.format(
                self.cost, self.memory() / 1024.0 / 1024.0 / 1024.0))

        self.model_residuals = [F.mul(
            self.functions[func_id].residuals, 1.0) for func_id in range(len(self.functions))]

        if self.summary.lm_iteration == 0:
            self.evaluator = StepEvaluator(self.cost)

        outerIterations = 0
        success_iterations = 0
        self.debug = False
        while True:
            # t2 = self.timing()
            outerIterations += 1

            if outerIterations == self.option.max_num_iterations:
                break
            self.compute_trust_region_step()

            # t3 = self.timing()
            if not self.summary.step_is_valid:
                self.summary.num_consecutive_invalid_steps += 1
                if self.summary.num_consecutive_invalid_steps \
                        > self.option.max_invalid_step:
                    self.summary.lm_termination_type = 2
                    return
                self.strategy.reject()
                continue

            candidate_x = [self.variables[i] + F.reshape(
                self.delta[i], self.variables[i].shape) for i in self.vranges]

            cost = self.evaluate_cost(candidate_x)

            # parameter tolerance check
            step_size_tolerance = self.option.parameter_tolerance \
                                  * (self.x_norm + self.option.parameter_tolerance)
            step_norm = lv.list_norm(self.delta)

            if step_norm < step_size_tolerance:
                self.summary.lm_termination_type = 1
                return

            # function tolerance check
            cost_change = self.cost - cost
            absolute_function_tolerance = \
                self.option.function_tolerance * self.cost

            if abs(cost_change.asnumpy()) < absolute_function_tolerance:
                self.summary.lm_termination_type = 1
                return

            # evaluate relative decrease
            self.summary.relative_decrease = self.evaluator.step_quality(
                cost, self.model_cost_change)

            if self.summary.relative_decrease \
                    > self.option.min_relative_decrease:
                for i in self.vranges:
                    self.variables[i] += F.reshape(self.delta[i], self.variables[i].shape)

                self.x_norm = lv.list_norm(self.variables)
                self.strategy.accept(self.summary.relative_decrease)
                self.evaluator.accept(cost, self.model_cost_change)

                if self.verbose:
                    current_time = self.timing()
                    print('iter = {}, cost = {}, radius = {}, CGstep = {}, time = {}'.format(
                        outerIterations,
                        cost,
                        self.strategy.radius,
                        self.summary.num_iterations,
                        current_time - self.start_time))

                success_iterations += 1
                if success_iterations >= self.option.max_success_iterations:
                    self.cost = cost
                    break

                self.evaluate(True)

            else:
                self.strategy.reject()
                if self.verbose:
                    print('iter = %d (rejected)' % outerIterations)

            if self.strategy.radius < 1e-32 or \
                    self.summary.gradient_max_norm < 1e-10:
                self.summary.lm_termination_type = 1
                return


def solve(variables, constants, indices, fn, num_iterations=15,
          num_success_iterations=15, max_linear_iterations=150, verbose=True):
    """main"""
    if not indices:
        return None
    for _, i in enumerate(range(len(indices))):
        if indices[i].shape[0] == 0:
            return None

    func = FunctionBlock(variables=variables, constants=constants, indices=indices, fn=fn)

    option = LMOption()
    option.max_linear_iterations = max_linear_iterations
    solver = LMSolver(functions=[func], verbose=verbose, option=option)
    solver.option.max_num_iterations = num_iterations
    solver.option.max_success_iterations = num_success_iterations
    for _, i in enumerate(range(len(indices))):
        index = indices[i]
        if len(index.shape) == 1:
            index = F.reshape(index, (-1, 1))
        indices[i] = index
    solver.solve()

    return solver.cost
