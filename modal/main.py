import sys

import modal

stub = modal.Stub("hello-world-01")

@stub.function()
def f(i):
    if i % 2 == 0:
        print("hello", i)
    else:
        print("world", i, file=sys.stderr)

    return i * i

@stub.local_entrypoint()
def main():
    print(f.local(1000))

    print(f.remote(1000))

    total = 0
    for ret in f.map(range(1000)):
        total += ret