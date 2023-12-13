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
"""structure module"""
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from tests.st.mindscience.mindsponge.mindsponge.cell.initializer import lecun_init


class PredictedLDDTHead(nn.Cell):
    """Head to predict the per-residue LDDT to be used as a confidence measure."""

    def __init__(self, config, seq_channel):
        super().__init__()
        self.config = config
        self.input_layer_norm = nn.LayerNorm([seq_channel,], epsilon=1e-5)
        self.act_0 = nn.Dense(seq_channel, self.config.num_channels,
                              weight_init=lecun_init(seq_channel, initializer_name='relu')
                              ).to_float(mstype.float16)
        self.act_1 = nn.Dense(self.config.num_channels, self.config.num_channels,
                              weight_init=lecun_init(self.config.num_channels, initializer_name='relu')
                              ).to_float(mstype.float16)
        self.logits = nn.Dense(self.config.num_channels, self.config.num_bins, weight_init='zeros'
                               ).to_float(mstype.float16)
        self.relu = nn.ReLU()

    def construct(self, rp_structure_module):
        """Builds ExperimentallyResolvedHead module."""
        act = rp_structure_module
        act = self.input_layer_norm(act.astype(mstype.float32))
        act = self.act_0(act)
        act = self.relu(act.astype(mstype.float32))
        act = self.act_1(act)
        act = self.relu(act.astype(mstype.float32))
        logits = self.logits(act)
        return logits
