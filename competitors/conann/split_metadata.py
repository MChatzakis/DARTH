"""Read and write ConANN's small, human-readable split manifest."""
import configparser
from pathlib import Path

def read_split(path):
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(path):
        raise ValueError(f"Cannot read split metadata: {path}")
    section = parser['split']
    result = dict(section)
    for key in ('calibration_queries', 'tuning_queries', 'testing_queries', 'total_queries'):
        result[key] = section.getint(key)
    result['query_order'] = section['query_order'].split(',')
    result['k_values'] = [int(value) for value in section['k_values'].split(',')]
    return result

def write_split(path, values):
    parser = configparser.ConfigParser(interpolation=None)
    parser['split'] = {key: ','.join(map(str, value)) if isinstance(value, list) else str(value)
                       for key, value in values.items()}
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as output:
        parser.write(output)
    temporary.replace(path)
