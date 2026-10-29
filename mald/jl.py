"""jsonl io. a truncated last line (from a copy during a write) is skipped."""
import json


def rows(path):
    try:
        f = open(path)
    except FileNotFoundError:
        return []
    out = []
    with f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def repair(path):
    """rewrite path with only the lines that parse."""
    r = rows(path)
    with open(path, "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in r)
    return r
