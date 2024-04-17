/*
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "custom_op_proto/cust_nn_training.h"
#include "register/op_impl_registry.h"
#include "utils/util.h"
#include "utils/common_shape_fns.h"

namespace ge {
// -----------------FusedSparseAdam----------------
CUST_IMPLEMT_INFERFUNC(FusedSparseAdam, FusedSparseAdamInfer) {
  auto var = op.GetInputDescByName("var");
  auto m = op.GetInputDescByName("m");
  auto v = op.GetInputDescByName("v");
  RETURN_IF_FAILURE(op.UpdateOutputDesc("var", var));
  RETURN_IF_FAILURE(op.UpdateOutputDesc("m", m));
  RETURN_IF_FAILURE(op.UpdateOutputDesc("v", v));
  return GRAPH_SUCCESS;
}
CUST_INFER_FUNC_REG(FusedSparseAdam, FusedSparseAdamInfer);
// -----------------FusedSparseAdam End----------------

// -----------------FusedSparseFtrl----------------
CUST_IMPLEMT_INFERFUNC(FusedSparseFtrl, FusedSparseFtrlInfer) {
  auto var = op.GetInputDescByName("var");
  auto accum = op.GetInputDescByName("accum");
  auto linear = op.GetInputDescByName("linear");
  RETURN_IF_FAILURE(op.UpdateOutputDesc("var", var));
  RETURN_IF_FAILURE(op.UpdateOutputDesc("accum", accum));
  RETURN_IF_FAILURE(op.UpdateOutputDesc("linear", linear));
  return GRAPH_SUCCESS;
}
CUST_INFER_FUNC_REG(FusedSparseFtrl, FusedSparseFtrlInfer);
// -----------------FusedSparseAdam End----------------

// -----------------FusedSparseLazyAdam----------------
CUST_IMPLEMT_INFERFUNC(FusedSparseLazyAdam, FusedSparseLazyAdamInfer) {
  auto var = op.GetInputDescByName("var");
  auto m = op.GetInputDescByName("m");
  auto v = op.GetInputDescByName("v");
  RETURN_IF_FAILURE(op.UpdateOutputDesc("var", var));
  RETURN_IF_FAILURE(op.UpdateOutputDesc("m", m));
  RETURN_IF_FAILURE(op.UpdateOutputDesc("v", v));
  return GRAPH_SUCCESS;
}
CUST_INFER_FUNC_REG(FusedSparseLazyAdam, FusedSparseLazyAdamInfer);
// -----------------FusedSparseAdam End----------------

// -----------------FusedSparseProximalAdagrad----------------
CUST_IMPLEMT_INFERFUNC(FusedSparseProximalAdagrad, FusedSparseProximalAdagradInfer) {
  auto var = op.GetInputDescByName("var");
  auto accum = op.GetInputDescByName("accum");
  RETURN_IF_FAILURE(op.UpdateOutputDesc("var", var));
  RETURN_IF_FAILURE(op.UpdateOutputDesc("accum", accum));
  return GRAPH_SUCCESS;
}
CUST_INFER_FUNC_REG(FusedSparseProximalAdagrad, FusedSparseProximalAdagradInfer);
// -----------------FusedSparseProximalAdagrad End----------------
}  // namespace ge
