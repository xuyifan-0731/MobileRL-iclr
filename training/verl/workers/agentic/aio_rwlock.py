import asyncio


class WriteEnforceRWLock:
    def __init__(self):
        # Protects internal state
        self._lock = asyncio.Lock()

        # Condition variable used to wait for state changes
        self._cond = asyncio.Condition(self._lock)

        # Number of readers currently holding the lock
        self._readers = 0

        # Whether a writer is currently holding the lock
        self._writer_active = False

        # How many writers are queued waiting for a turn
        self._waiting_writers = 0

        self._active_reader_tasks = set()

    @property
    def reader_lock(self):
        """
        A context manager for acquiring a shared (reader) lock.

        Example:
            async with rwlock.reader_lock:
                # read-only access
        """
        return _ReaderLock(self)

    @property
    def writer_lock(self):
        """
        A context manager for acquiring an exclusive (writer) lock.

        Example:
            async with rwlock.writer_lock:
                # exclusive access
        """
        return _WriterLock(self)

    async def acquire_reader(self):
        async with self._lock:
            # Wait until there is no active writer or waiting writer
            # to ensure fairness.
            while self._writer_active or self._waiting_writers > 0:
                await self._cond.wait()
            self._readers += 1
            self._active_reader_tasks.add(asyncio.current_task())

    async def release_reader(self):
        async with self._lock:
            self._readers -= 1
            self._active_reader_tasks.discard(asyncio.current_task())
            # If this was the last reader, wake up anyone waiting
            # (potentially a writer or new readers).
            if self._readers == 0:
                self._cond.notify_all()

    async def acquire_writer(self):
        async with self._lock:
            # Increment the count of writers waiting
            self._waiting_writers += 1
            try:
                # cancel all reader
                for t in list(self._active_reader_tasks):
                    if not t.done():
                        t.cancel()
                # Wait while either a writer is active or readers are present
                while self._writer_active or self._readers > 0:
                    await self._cond.wait()
                self._writer_active = True
            finally:
                # Decrement waiting writers only after we've acquired the writer lock
                self._waiting_writers -= 1

    async def release_writer(self):
        async with self._lock:
            self._writer_active = False
            # Wake up anyone waiting (readers or writers)
            self._cond.notify_all()


class _ReaderLock:
    def __init__(self, rwlock: WriteEnforceRWLock):
        self._rwlock = rwlock

    async def __aenter__(self):
        await self._rwlock.acquire_reader()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._rwlock.release_reader()


class _WriterLock:
    def __init__(self, rwlock: WriteEnforceRWLock):
        self._rwlock = rwlock

    async def __aenter__(self):
        await self._rwlock.acquire_writer()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._rwlock.release_writer()


if __name__ == '__main__':
    async def reader_task(rwlock: WriteEnforceRWLock, id: int):
        while True:
            try:
                async with rwlock.reader_lock:
                    print(f"Reader {id} acquired the lock")
                    await asyncio.sleep(10)
                    print(f"Reader {id} released the lock")
                    break
            except asyncio.CancelledError:
                print(f"Reader {id} was cancelled")

    async def writer_task(rwlock: WriteEnforceRWLock, id: int):
        async with rwlock.writer_lock:
            print(f"Writer {id} acquired the lock")
            await asyncio.sleep(3)
            print(f"Writer {id} released the lock")

    async def main():
        rwlock = WriteEnforceRWLock()
        tasks = []

        for i in range(5):
            tasks.append(asyncio.create_task(reader_task(rwlock, i)))
        await asyncio.sleep(1)
        tasks.append(asyncio.create_task(writer_task(rwlock, 5)))

        await asyncio.gather(*tasks)

    asyncio.run(main())