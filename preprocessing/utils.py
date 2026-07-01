def create_directory_if_missing(path: str):
    import os
    if not os.path.exists(path):
        os.makedirs(path)

def get_from_args(args: list[str], flag: str, default: str):
    if flag in args:
        i = args.index(flag) + 1
        if i < len(args):
            return args[i]
    return default

def sequence_to_name(sequence: str):
    import hashlib
    return hashlib.sha3_256(sequence.encode()).hexdigest()[0:6]

def get_partition_start_end(partition_num: int, partition_count: int, task_count: int):
    import math
    tasks_per_partition = math.ceil(task_count / partition_count)
    smaller_count = tasks_per_partition * partition_count - task_count
    full_count = partition_count - smaller_count
    if partition_num < full_count:
        start = partition_num * tasks_per_partition
        end = start + tasks_per_partition
    else:
        start = full_count * tasks_per_partition + (partition_num - full_count) * (tasks_per_partition - 1)
        end = start + tasks_per_partition - 1
    return start, end

def get_start_and_end(args: list[str], task_count: int):
    partition_count = int(get_from_args(args, '--partition-count', '0'))
    if partition_count == 0:
        start = int(get_from_args(args, '--start', 0))
        end = int(get_from_args(args, '--end', task_count))
    else:
        partition_num = get_from_args(args, '--partition-num', None)
        assert(partition_num is not None)
        start, end = get_partition_start_end(int(partition_num) - 1, partition_count, task_count)
    return start, end

def save_json(file: str, obj):
    import json
    with open(file, 'w') as f:
        json.dump(obj, f)
