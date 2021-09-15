#!/bin/bash
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
PY=/usr/bin/python3.7

export PYTHONPATH=${PYTHONPATH}:.

${PY} generate_map_report.py \
--annotations_json=/data/coco2017/annotations/instances_val2017.json \
--det_result_json=./perf/om_infer_output_on_coco_val2017/om_det_result.json \
--output_path_name=./map_output/map.txt \
--anno_type=bbox