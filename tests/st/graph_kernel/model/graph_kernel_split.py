# Copyright 2020 Huawei Technologies Co., Ltd
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
"""graph kernel split"""
import json
import getopt
import sys
import model


def print_usage():
    print('Usage: graph_kernel_split.py [OPTION] <JSON_FILE>')
    print('Options:')
    print('  -s <config/auto>\tsplit graph with config')
    print('  -e \t\testimate graph')
    print('  -i \t\tnaive estimate')
    print('  -o <prefix>\toutput split graphs')
    print('  -v \t\tverbose mode')
    print('  -h \t\tprint this help')


class Option:
    """Options"""

    def __init__(self):
        self.split = None
        self.estimate = False
        self.estimate_naive = False
        self.output = None
        self.verbose = False
        self.help = False

    def parse(self, options):
        """parse options"""
        for name, val in options:
            if name == '-h':
                self.help = True
            elif name == '-v':
                self.verbose = True
            elif name == '-o':
                self.output = val
            elif name == '-e':
                self.estimate = True
            elif name == '-s':
                self.split = val
            elif name == '-i':
                self.estimate_naive = True


opt = Option()


def estimate(graph_in, parts_in, naive):
    """estimate graphs costs"""
    def _print_cost(name, c):
        print("%s\tdma_ratio=%f, saturation=%f, mix_saturation=%f, type=%s" %
              (name, c.dma_ratio(), c.saturation(), c.mix_saturation(), c.cost_type()))
    main_cost, _ = model.estimate(graph_in, naive)
    split_cost, sub_costs = model.estimate(parts_in, naive) if parts_in else (None, None)
    _print_cost("MainGraph:", main_cost)
    if parts_in:
        _print_cost("Subgraphs:", split_cost)
        if opt.verbose:
            for i, sub_cost in enumerate(sub_costs):
                _print_cost(" |_%d:\t" % (i), sub_cost)


def split_graph(graph_in, config):
    """split graph"""
    if config == 'auto':
        return model.split(graph_in)
    subgraphs = []
    all_tensors = []
    subgraph_idx = 0
    config_parts = config.split('|')
    for part in config_parts:
        tensor_names = part.split(',')
        graph_name = "%s_%d" % (graph_in.name, subgraph_idx)
        g = graph_in.extract_subgraph(graph_name, tensor_names)
        assert len(g.ops) == len(tensor_names)
        subgraphs.append(g)
        all_tensors += tensor_names
        subgraph_idx += 1
    if len(all_tensors) < len(graph_in.ops):
        graph_name = "%s_%d" % (graph_in.name, subgraph_idx)
        g = graph_in.extract_subgraph(graph_name, all_tensors, True)
        subgraphs.append(g)
    return subgraphs


def main():
    opts, args = getopt.getopt(sys.argv[1:], 'heivo:s:')
    opt.parse(opts)
    if len(args) != 1 or opt.help:
        print_usage()
        sys.exit(0)
    in_file = args[0]
    with open(in_file, 'r') as f:
        desc = json.loads(f.read())
        comp = model.load_composite(desc)
        graph = comp.graph
        parts = []
        # 1. split sub-graphs
        if opt.split is not None:
            parts = split_graph(graph, opt.split)
        if opt.verbose:
            print('----------- main graph --------------')
            print(graph)
            for i, _ in enumerate(parts):
                print('---------------- sub graph %d ---------------' % (i))
                print(parts[i])
        # 2. estimate cost
        if opt.estimate:
            print('------------- cost --------------')
            estimate(graph, parts, False)
        if opt.estimate_naive:
            print('------------- naive cost --------------')
            estimate(graph, parts, True)
        # 3. output parts
        if opt.output is not None:
            for graph_part in parts:
                desc = comp.dump(graph_part)
                s_desc = json.dumps(desc)
                fname = "%s_%s.json" % (opt.output, graph_part.name)
                with open(fname, 'w', encoding='utf-8') as of:
                    of.write(s_desc)


if __name__ == '__main__':
    main()
