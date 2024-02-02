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
CUST_IMPLEMT_INFERFUNC(FusedSparseProximalAdagrad, FusedSparseProximalAdagradInfer) {
  auto var = op.GetInputDescByName("var");
  auto accum = op.GetInputDescByName("accum");
  RETURN_IF_FAILURE(op.UpdateOutputDesc("var", var));
  RETURN_IF_FAILURE(op.UpdateOutputDesc("accum", accum));
  return GRAPH_SUCCESS;
}
CUST_INFER_FUNC_REG(FusedSparseProximalAdagrad, FusedSparseProximalAdagradInfer);
}  // namespace ge
