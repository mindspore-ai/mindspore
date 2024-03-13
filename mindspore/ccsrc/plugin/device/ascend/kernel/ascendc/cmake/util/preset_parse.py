import json
import sys
import os


def get_config_opts(file):
    src_dir = os.path.abspath(os.path.dirname(file))
    opts = ''
    with open(file, 'r') as fd:
        config = json.load(fd)
        for conf in config:
            if conf == 'configurePresets':
                for node in config[conf]:
                    macros = node.get('cacheVariables')
                    if macros is not None:
                        for key in macros:
                            opts += '-D{}={} '.format(key, macros[key]['value'])
    opts = opts.replace('${sourceDir}', src_dir)
    print(opts)


if __name__ == "__main__":
    get_config_opts(sys.argv[1])
