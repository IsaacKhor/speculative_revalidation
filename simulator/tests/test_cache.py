import unittest

from simulator.main import CacheSimulator, FIFOPolicy, LRUPolicy, Request


def make_request(timestamp: int, key: str, size: int, ttl: int) -> Request:
    return Request(
        timestamp=timestamp,
        key=key,
        zone="zone-a",
        size=size,
        ttl=ttl,
        stale_time=0,
        method="GET",
        mime="text/plain",
    )


class CacheSimulatorTest(unittest.TestCase):
    def test_lru_hit_eviction_tracking(self) -> None:
        simulator = CacheSimulator(capacity=100, policy=LRUPolicy(), auto_expire=True)

        requests = [
            make_request(0, "alpha", 40, 100),
            make_request(5, "alpha", 40, 100),
            make_request(10, "bravo", 70, 100),
            make_request(15, "alpha", 40, 100),
        ]

        for request in requests:
            simulator.process_request(request)

        stats = simulator.stats

        self.assertEqual(stats.requests, 4)
        self.assertEqual(stats.hits, 1, msg="expected a single cache hit for alpha")
        self.assertEqual(stats.miss_first, 2, msg="alpha and bravo should each incur a compulsory miss once")
        self.assertEqual(stats.miss_evicted, 1, msg="alpha should be missed after eviction")
        self.assertEqual(stats.miss_expired, 0)
        self.assertEqual(stats.evictions, 2, msg="two evictions to accommodate oversized inserts")

    def test_expiry_accounting(self) -> None:
        simulator = CacheSimulator(capacity=128, policy=LRUPolicy(), auto_expire=True)

        requests = [
            make_request(0, "charlie", 60, 10),
            make_request(5, "charlie", 60, 10),
            make_request(15, "charlie", 60, 10),
        ]

        for request in requests:
            simulator.process_request(request)

        stats = simulator.stats

        self.assertEqual(stats.hits, 1, msg="middle request should be a cache hit before expiry")
        self.assertEqual(stats.miss_expired, 1, msg="final request should miss because the entry expired")
        self.assertGreaterEqual(stats.expirations, 1, msg="expired items should be purged")

    def test_fifo_policy_eviction_order(self) -> None:
        simulator = CacheSimulator(capacity=100, policy=FIFOPolicy(), auto_expire=True)

        requests = [
            make_request(0, "delta", 30, 100),
            make_request(1, "echo", 30, 100),
            make_request(2, "delta", 30, 100),
            make_request(3, "foxtrot", 50, 100),
            make_request(4, "delta", 30, 100),
        ]

        for request in requests[:-1]:
            simulator.process_request(request)

        self.assertIn("echo", simulator.entries, msg="echo should remain in cache after fifo eviction")
        self.assertIn("foxtrot", simulator.entries, msg="foxtrot should be admitted into the cache")
        self.assertNotIn("delta", simulator.entries, msg="delta should be the victim of FIFO eviction")

        simulator.process_request(requests[-1])

        stats = simulator.stats

        self.assertGreaterEqual(stats.miss_evicted, 1, msg="delta reload should be counted as an eviction miss")
        self.assertGreaterEqual(stats.evictions, 1, msg="inserting foxtrot should trigger an eviction")


if __name__ == "__main__":
    unittest.main()
