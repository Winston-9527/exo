import asyncio
import os

import pytest
from _pytest.capture import CaptureFixture
from exo_rs import (
    FromSwarm,
    NetworkingHandle,
    Pidfile,
)


def test_rejects_invalid_bootstrap_endpoint() -> None:
    with pytest.raises(RuntimeError, match="not-a-zenoh-endpoint"):
        NetworkingHandle.new(
            "1",
            "python-binding-test",
            52414,
            52413,
            ["not-a-zenoh-endpoint"],
        )


@pytest.mark.asyncio
async def test_sleep_on_multiple_items() -> None:
    print("PYTHON: starting handle")
    h = NetworkingHandle.new(
        os.urandom(16).hex().lstrip("0"),
        "python-binding-test",
        52414,
        52413,
        [],
    )
    print("PYTHON: handle started")

    rt = asyncio.create_task(_await_recv(h))

    # sleep for 4 ticks
    for i in range(10):
        await asyncio.sleep(1)

        await h.gossipsub_publish("topic", b"somehting or other")


def test_pidfile(capsys: CaptureFixture[str]):
    with capsys.disabled():
        print("\nbefore python")
        scoped_lock_file()
        print("after python")


async def _await_recv(h: NetworkingHandle):
    while True:
        event = await h.recv()
        match event:
            case FromSwarm.Connection() as c:
                print(f"PYTHON: connection update: {c}")
            case FromSwarm.Message() as m:
                print(f"PYTHON: message: {m}")


def scoped_lock_file():
    a = Pidfile("/tmp/lock.pid", 0o0600)


if __name__ == "__main__":
    asyncio.run(test_sleep_on_multiple_items())
