import asyncio
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")


async def run_in_process_pool(
    pool: ProcessPoolExecutor, function: Callable[[Input], Output], argument: Input
) -> Output:
    """Run a function in a process and wait for cleanup-safe cancellation."""
    future = asyncio.get_running_loop().run_in_executor(pool, function, argument)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        await asyncio.gather(future, return_exceptions=True)
        raise
