#include "eviction.hpp"
#include "absl/container/btree_set.h"

class Lru : public Cache
{

  public:
    Lru(u64 max_bytes, exit_cb cb) : Cache(max_bytes, cb) {}

    // Insert a new cache entry
    void admit(u64 key, const CacheEntry entry) override
    {
        while (cur_bytes + entry.size > max_bytes)
            evict();

        lru_list.push_back(key);
        cur_bytes += entry.size;

        CacheEntry new_entry = entry;
        new_entry.list_pos = std::prev(lru_list.end());
        cachemap[key] = new_entry;
    }

    auto evict() -> void
    {
        auto tail_key = lru_list.front();

        // there might be keys in the list that got removed early
        auto it = cachemap.find(tail_key);
        while (it == cachemap.end()) {
            lru_list.pop_front();
            if (lru_list.empty())
                return;
            tail_key = lru_list.front();
            it = cachemap.find(tail_key);
        }

        lru_list.pop_front();
        erase(it);
    }

    auto remove(u64 key) -> void override
    {
        auto it = cachemap.find(key);
        if (it == cachemap.end())
            return;

        lru_list.erase(it->second.list_pos);
        erase(it);
    }

    // Touch an entry (move to most recently used position)
    auto touch(u64 key) -> void override
    {
        auto it = cachemap.find(key);
        if (it == cachemap.end())
            return;

        auto &entry = it->second;
        lru_list.erase(entry.list_pos);
        lru_list.push_back(key);
        entry.list_pos = std::prev(lru_list.end());
    }

  private:
    std::list<u64> lru_list;
};

class Sieve : public Cache
{
  public:
    Sieve(u64 max_bytes, exit_cb cb) : Cache(max_bytes, cb), hand(queue.end())
    {
    }

    void admit(u64 key, const CacheEntry entry) override
    {
        while (cur_bytes + entry.size > max_bytes)
            evict();

        CacheEntry new_entry = entry;
        new_entry.frequency = 1; // visited bit for SIEVE
        queue.push_back(key);
        new_entry.list_pos = std::prev(queue.end());
        cur_bytes += entry.size;
        cachemap[key] = new_entry;
    }

    auto evict() -> void
    {
        if (cachemap.empty() || queue.empty())
            return;

        // Move hand forward to find an entry to evict
        while (true) {
            if (hand == queue.end())
                hand = queue.begin();

            auto key = *hand;
            auto it = cachemap.find(key);

            // If key not in cache, move hand forward and skip
            if (it == cachemap.end()) {
                hand++;
                continue;
            }

            // If entry has been visited, clear visited bit and move hand
            // forward
            if (it->second.frequency) {
                it->second.frequency = 0;
                hand++;
            } else {
                // Entry has not been visited, evict it
                auto to_erase = it;
                hand = queue.erase(hand);
                erase(to_erase);
                return;
            }
        }
    }

    auto remove(u64 key) -> void override
    {
        auto it = cachemap.find(key);
        if (it == cachemap.end())
            return;

        auto &[k, entry] = *it;
        auto pos = entry.list_pos;
        if (hand == pos)
            hand++;
        queue.erase(entry.list_pos);
        erase(it);
    }

    auto touch(u64 key) -> void override
    {
        auto it = cachemap.find(key);
        if (it == cachemap.end())
            return;

        // Mark as visited
        it->second.frequency = 1;
    }

  private:
    std::list<u64> queue;
    std::list<u64>::iterator hand;
};

class Fifo : public Cache
{
  public:
    Fifo(u64 max_bytes, exit_cb cb) : Cache(max_bytes, cb) {}

    void admit(u64 key, const CacheEntry entry) override
    {
        while (cur_bytes + entry.size > max_bytes)
            evict();

        CacheEntry new_entry = entry;
        fifo_queue.push_back(key);
        new_entry.list_pos = std::prev(fifo_queue.end());
        cur_bytes += entry.size;
        cachemap[key] = new_entry;
    }

    auto evict() -> void
    {
        if (fifo_queue.empty())
            return;

        auto tail_key = fifo_queue.front();

        // there might be keys in the list that got removed early
        auto it = cachemap.find(tail_key);
        while (it == cachemap.end()) {
            fifo_queue.pop_front();
            if (fifo_queue.empty())
                return;
            tail_key = fifo_queue.front();
            it = cachemap.find(tail_key);
        }

        fifo_queue.pop_front();
        erase(it);
    }

    auto remove(u64 key) -> void override
    {
        auto it = cachemap.find(key);
        if (it == cachemap.end())
            return;

        fifo_queue.erase(it->second.list_pos);
        erase(it);
    }

    auto touch(u64 key) -> void override
    {
        // FIFO doesn't change order on access
    }

  private:
    std::list<u64> fifo_queue;
};

class Gdsf : public Cache
{
  private:
    static auto cmp_prio(CacheEntry *e1, CacheEntry *e2) -> bool
    {
        if (e1->priority != e2->priority)
            return e1->priority < e2->priority;
        else
            return e1->key < e2->key;
    }

  public:
    Gdsf(u64 max_bytes, exit_cb cb) : Cache(max_bytes, cb) {}

    auto get_prio(CacheEntry &e) -> f64
    {
        return prio_last_evict + (e.frequency * 1.0e6 / e.size);
    }

    void admit(u64 key, const CacheEntry entry) override
    {
        while (cur_bytes + entry.size > max_bytes)
            evict();

        CacheEntry new_entry = entry;
        new_entry.frequency = 1; // initial frequency
        new_entry.priority = get_prio(new_entry);
        cur_bytes += entry.size;
        cachemap[key] = new_entry;

        // insert into priority queue
        auto [_, is_inserted] = pq.insert(&cachemap[key]);
        if (!is_inserted)
            breakpoint();
    }

    auto evict() -> void
    {
        if (cachemap.empty() || pq.empty())
            return;

        // get entry with lowest priority
        auto eventry = *pq.begin();
        auto key = eventry->key;
        auto evicted_priority = eventry->priority;

        // remove from priority queue
        pq.erase(eventry);
        prio_last_evict = evicted_priority;

        // remove from cache
        auto cmap_it = cachemap.find(key);
        erase(cmap_it);
    }

    auto remove(u64 key) -> void override
    {
        auto it = cachemap.find(key);
        if (it == cachemap.end())
            return;

        // remove from priority queue
        pq.erase(&it->second);
        erase(it);
    }

    auto touch(u64 key) -> void override
    {
        if (pq.size() != cachemap.size())
            breakpoint();

        auto it = cachemap.find(key);
        if (it == cachemap.end())
            return;

        auto &entry = it->second;
        pq.erase(&entry);

        entry.frequency += 1; // keep track separately because it's not reset
        entry.priority =
            prio_last_evict + (entry.frequency * 1.0e6 / entry.size);

        // erase old priority node and insert new one
        pq.insert(&entry);
    }

  private:
    u64 total_reqs = 0;
    f64 prio_last_evict = 0;

    // use set for ordering and ability to find
    absl::btree_set<CacheEntry *, decltype(&Gdsf::cmp_prio)> pq{Gdsf::cmp_prio};
};

class Arc : public Cache
{
    // Helper structure for managing FIFO queues with size tracking
    struct FIFOQueue {
        std::list<u64> queue;
        absl::flat_hash_map<u64, std::pair<u64, std::list<u64>::iterator>>
            entries; // key -> (size, iterator)
        u64 total_bytes = 0;

        void push(u64 key, u64 size, std::list<u64>::iterator &pos_out)
        {
            queue.push_back(key);
            pos_out = std::prev(queue.end());
            entries[key] = {size, pos_out};
            total_bytes += size;
        }

        auto pop_front() -> std::pair<u64, bool>
        {
            if (queue.empty())
                return {0, false};
            auto key = queue.front();
            queue.pop_front();
            auto it = entries.find(key);
            if (it != entries.end()) {
                total_bytes -= it->second.first;
                entries.erase(it);
            }
            return {key, true};
        }

        auto remove(u64 key) -> void
        {
            auto it = entries.find(key);
            if (it == entries.end())
                return;
            auto [size, pos] = it->second;
            queue.erase(pos);
            total_bytes -= size;
            entries.erase(it);
        }

        bool contains(u64 key) const
        {
            return entries.find(key) != entries.end();
        }
        bool empty() const { return queue.empty(); }
        u64 bytes() const { return total_bytes; }
    };

    // Ghost list - stores only keys (no size tracking)
    struct GhostList {
        std::list<u64> queue;
        absl::flat_hash_map<u64, std::list<u64>::iterator> entries;
        u64 entry_count = 0;
        static constexpr u64 MAX_GHOST_ENTRIES = 1'000'000;

        void push(u64 key)
        {
            // Evict oldest if at capacity
            while (entry_count >= MAX_GHOST_ENTRIES) {
                pop_front();
            }
            queue.push_back(key);
            entries[key] = std::prev(queue.end());
            entry_count++;
        }

        bool contains(u64 key) const
        {
            return entries.find(key) != entries.end();
        }

        auto remove(u64 key) -> void
        {
            auto it = entries.find(key);
            if (it == entries.end())
                return;
            queue.erase(it->second);
            entries.erase(it);
            entry_count--;
        }

        auto pop_front() -> bool
        {
            if (queue.empty())
                return false;
            auto key = queue.front();
            queue.pop_front();
            auto it = entries.find(key);
            if (it != entries.end()) {
                entries.erase(it);
                entry_count--;
            }
            return true;
        }

        bool empty() const { return queue.empty(); }
        u64 count() const { return entry_count; }
    };

  public:
    Arc(u64 max_bytes, exit_cb cb) : Cache(max_bytes, cb), target_t1_bytes(0) {}

    void admit(u64 key, const CacheEntry entry) override
    {
        // Check if in ghost lists
        if (b1.contains(key)) {
            // Hit in B1 - increase target for T1
            auto delta = std::max(b2.count() / (b1.count() + 1), 1UL);
            target_t1_bytes = std::min(target_t1_bytes + delta, max_bytes);
            b1.remove(key);

            // Replace and insert into T2
            replace(entry.size, true);
            insert_t2(key, entry);
        } else if (b2.contains(key)) {
            // Hit in B2 - decrease target for T1
            auto delta = std::max(b1.count() / (b2.count() + 1), 1UL);
            target_t1_bytes =
                target_t1_bytes >= delta ? target_t1_bytes - delta : 0;
            b2.remove(key);

            // Replace and insert into T2
            replace(entry.size, false);
            insert_t2(key, entry);
        } else {
            // Evict from cache if needed
            while (cur_bytes + entry.size > max_bytes)
                replace(entry.size, false);

            // Insert into T1
            insert_t1(key, entry);
        }
    }

    auto remove(u64 key) -> void override
    {
        auto it = cachemap.find(key);
        if (it == cachemap.end())
            return;

        if (t1.contains(key)) {
            t1.remove(key);
        } else {
            t2.remove(key);
        }

        erase(it);
    }

    auto touch(u64 key) -> void override
    {
        auto it = cachemap.find(key);
        if (it == cachemap.end())
            return;

        auto &entry = it->second;

        if (t1.contains(key)) {
            // Promote from T1 to T2
            t1.remove(key);
            t2.push(key, entry.size, entry.list_pos);
        } else {
            // Move to MRU position in T2
            t2.remove(key);
            t2.push(key, entry.size, entry.list_pos);
        }
    }

  private:
    auto insert_t1(u64 key, const CacheEntry &entry) -> void
    {
        CacheEntry new_entry = entry;
        t1.push(key, entry.size, new_entry.list_pos);
        cachemap[key] = new_entry;
        cur_bytes += entry.size;
    }

    auto insert_t2(u64 key, const CacheEntry &entry) -> void
    {
        CacheEntry new_entry = entry;
        t2.push(key, entry.size, new_entry.list_pos);
        cachemap[key] = new_entry;
        cur_bytes += entry.size;
    }

    auto evict_from_t1() -> void
    {
        if (t1.empty())
            return;

        auto [key, valid] = t1.pop_front();
        if (!valid)
            return;

        auto it = cachemap.find(key);
        while (it == cachemap.end() && !t1.empty()) {
            auto [k, v] = t1.pop_front();
            if (!v)
                return;
            key = k;
            it = cachemap.find(key);
        }

        if (it == cachemap.end())
            return;

        auto size = it->second.size;
        erase(it);

        // Add to ghost B1
        b1.push(key);
    }

    auto evict_from_t2() -> void
    {
        if (t2.empty())
            return;

        auto [key, valid] = t2.pop_front();
        if (!valid)
            return;

        auto it = cachemap.find(key);
        while (it == cachemap.end() && !t2.empty()) {
            auto [k, v] = t2.pop_front();
            if (!v)
                return;
            key = k;
            it = cachemap.find(key);
        }

        if (it == cachemap.end())
            return;

        auto size = it->second.size;
        erase(it);

        // Add to ghost B2
        b2.push(key);
    }

    auto replace(u64 needed_size, bool from_b1) -> void
    {
        while (cur_bytes + needed_size > max_bytes) {
            if (t1.bytes() > 0 &&
                (t1.bytes() > target_t1_bytes ||
                 (from_b1 && t1.bytes() == target_t1_bytes))) {
                evict_from_t1();
            } else if (t2.bytes() > 0) {
                evict_from_t2();
            } else {
                // Should not reach here normally
                break;
            }
        }
    }

    FIFOQueue t1; // Recent entries (seen once)
    FIFOQueue t2; // Frequent entries (seen multiple times)
    GhostList b1; // Ghost entries evicted from T1
    GhostList b2; // Ghost entries evicted from T2

    u64 target_t1_bytes;
};

auto make_cache(str type, u64 max_bytes, exit_cb cb) -> uptr<Cache>
{
    if (type == "lru")
        return std::make_unique<Lru>(max_bytes, cb);
    else if (type == "gdsf")
        return std::make_unique<Gdsf>(max_bytes, cb);
    else if (type == "sieve")
        return std::make_unique<Sieve>(max_bytes, cb);
    else if (type == "fifo")
        return std::make_unique<Fifo>(max_bytes, cb);
    else if (type == "arc")
        return std::make_unique<Arc>(max_bytes, cb);
    else
        FAIL("unknown cache type: " + type);
}
