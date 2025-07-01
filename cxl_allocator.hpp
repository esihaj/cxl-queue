// cxl_allocator.hpp
// ─────────────────────────────────────────────────────────────────────────────
//  Header-only CXL memory helpers
//
//   • interface      : class CxlAllocator
//   • implementations: class DaxAllocator   – slice of /dev/dax?   (pmem / CXL)
//                      class NumaAllocator  – DRAM on a NUMA node
//
//  Key features
//   • bump-pointer allocator with           allocate()          // no alignment
//                                           allocate_aligned()  // 64-B aligned
//   • run-time debug level  { Off | Low | High }
//   • simple 64-B flush + verify test
//
//  Build notes
//   ▸ DaxAllocator uses   mmap/MAP_SYNC (Linux ≥ 4.15).
//   ▸ NumaAllocator needs libnuma → link with -lnuma.
//
//  Example
//   #include "cxl_allocator.hpp"
//
//   cxl::DaxAllocator dax;                      // defaults → /dev/dax1.0 81-82 GiB
//   void* p = dax.allocate_aligned(256);        // 64-B aligned
//   uint8_t* tiny = static_cast<uint8_t*>(dax.allocate(8));   // tightly packed
//   dax.test_memory();                          // sanity-check cache flush
// ─────────────────────────────────────────────────────────────────────────────
#ifndef CXL_ALLOCATOR_HPP_
#define CXL_ALLOCATOR_HPP_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <sys/mman.h>
#include <unistd.h>
#include <numa.h>

namespace cxl {

// ─────────────────────────────────────────────────────────────────────────────
//  debug utilities
// ─────────────────────────────────────────────────────────────────────────────
enum class DebugLevel { off, low, high };

// helper
inline void log(DebugLevel lvl,
                DebugLevel threshold,
                const std::string& msg) noexcept
{
    if (lvl >= threshold)
        std::clog << "[cxl] " << msg << '\n';
}

// ─────────────────────────────────────────────────────────────────────────────
//  Interface
// ─────────────────────────────────────────────────────────────────────────────
class CxlAllocator {
public:
    virtual ~CxlAllocator() = default;

    virtual void* allocate(std::size_t bytes)          = 0;  // ≙ align=1
    virtual void* allocate_aligned(std::size_t bytes,
                                   std::size_t alignment = 64) = 0;

    [[nodiscard]] virtual std::size_t used() const noexcept      = 0;
    [[nodiscard]] virtual std::size_t remaining() const noexcept = 0;
    [[nodiscard]] virtual std::size_t capacity() const noexcept  = 0;

    /// basic write-flush-verify on first 64 B of the mapping
    virtual bool test_memory() = 0;

    /// change debug level at run-time
    virtual void set_debug(DebugLevel lvl) noexcept = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
//  bump-pointer utility (internal)
// ─────────────────────────────────────────────────────────────────────────────
class BumpPtr {
public:
    BumpPtr(void* base_addr, std::size_t len_bytes)
        : base_addr_{reinterpret_cast<std::uintptr_t>(base_addr)},
          end_addr_{base_addr_ + len_bytes},
          cur_addr_{base_addr_}
    {
        base_64b_aligned_ = (base_addr_ & 63) == 0;
    }

    void* alloc(std::size_t bytes, std::size_t align)
    {
        std::uintptr_t aligned = (cur_addr_ + align - 1) & ~(align - 1);
        if (aligned + bytes > end_addr_)
            throw std::bad_alloc{};

        cur_addr_ = aligned + bytes;
        return reinterpret_cast<void*>(aligned);
    }

    [[nodiscard]] std::size_t used() const noexcept       { return cur_addr_ - base_addr_; }
    [[nodiscard]] std::size_t capacity() const noexcept   { return end_addr_ - base_addr_; }
    [[nodiscard]] std::size_t remaining() const noexcept  { return end_addr_ - cur_addr_; }
    [[nodiscard]] bool        base_aligned() const noexcept { return base_64b_aligned_; }

private:
    std::uintptr_t base_addr_;
    std::uintptr_t end_addr_;
    std::uintptr_t cur_addr_;
    bool           base_64b_aligned_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  DaxAllocator  –  /dev/dax? slice
// ─────────────────────────────────────────────────────────────────────────────
class DaxAllocator : public CxlAllocator {
public:
    static constexpr std::string_view default_path   = "/dev/dax1.0";
    static constexpr std::size_t      default_offset = 81ULL * 1024 * 1024 * 1024; // 81 GiB
    static constexpr std::size_t      default_length =  1ULL * 1024 * 1024 * 1024; // 1 GiB

    explicit DaxAllocator(std::string_view path   = default_path,
                          std::size_t      offset = default_offset,
                          std::size_t      length = default_length,
                          DebugLevel       dbg    = DebugLevel::low)
        : path_{path}, offset_{offset}, length_{length}, debug_level_{dbg}
    {
        fd_ = ::open(path.data(), O_RDWR | O_SYNC);
        if (fd_ == -1)
            throw std::system_error(errno, std::generic_category(),
                                    "open(" + std::string(path) + ")");

        const long page_sz = ::getpagesize();
        if (offset % page_sz)
            throw std::invalid_argument("offset must be page-aligned");

        base_addr_ = ::mmap(nullptr,
                            length,
                            PROT_READ | PROT_WRITE,
                            MAP_SHARED_VALIDATE | MAP_SYNC,
                            fd_,
                            static_cast<off_t>(offset));
        if (base_addr_ == MAP_FAILED) {
            ::close(fd_);
            throw std::system_error(errno, std::generic_category(),
               "mmap(" + std::string(path) +
               ", offset=" + std::to_string(offset) +
               ", length=" + std::to_string(length) + ")");
        }

        bump_ptr_ = std::make_unique<BumpPtr>(base_addr_, length);

        std::ostringstream msg;
        msg << "DAX mmap ok: path=" << path_
            << " offset=" << offset_
            << " length=" << length_
            << " addr=0x" << std::hex << reinterpret_cast<std::uintptr_t>(base_addr_);
        log(debug_level_, DebugLevel::low, msg.str());

    }

    ~DaxAllocator() override
    {
        if (base_addr_) ::munmap(base_addr_, length_);
        if (fd_ != -1) ::close(fd_);
    }

    // ─── allocator API ────────────────────────────────────────────────────
    void* allocate(std::size_t bytes) override
    {
        void* p = bump_ptr_->alloc(bytes, 1);
        log(debug_level_, DebugLevel::high,
            "allocate(" + std::to_string(bytes) + ") → " +
            std::to_string(reinterpret_cast<std::uintptr_t>(p)));
        return p;
    }

    void* allocate_aligned(std::size_t bytes, std::size_t alignment = 64) override
    {
        void* p = bump_ptr_->alloc(bytes, alignment);
        log(debug_level_, DebugLevel::high,
            "allocate_aligned(" + std::to_string(bytes) + ", align=" +
            std::to_string(alignment) + ") → " +
            std::to_string(reinterpret_cast<std::uintptr_t>(p)));
        return p;
    }

    [[nodiscard]] std::size_t used() const noexcept override      { return bump_ptr_->used(); }
    [[nodiscard]] std::size_t remaining() const noexcept override { return bump_ptr_->remaining(); }
    [[nodiscard]] std::size_t capacity() const noexcept override  { return bump_ptr_->capacity(); }

    bool test_memory() override
    {
        alignas(64) uint8_t pattern[64];
        for (int i = 0; i < 64; ++i) pattern[i] = static_cast<uint8_t>(i);

        std::memcpy(base_addr_, pattern, 64);
        _mm_clflush(base_addr_);
        _mm_mfence();

        uint8_t verify[64];
        std::memcpy(verify, base_addr_, 64);
        bool ok = std::memcmp(pattern, verify, 64) == 0;

        log(debug_level_, DebugLevel::low,
            std::string("test_memory ") + (ok ? "✓" : "✗"));
        return ok;
    }

    void set_debug(DebugLevel lvl) noexcept override { debug_level_ = lvl; }

private:
    std::string           path_;
    std::size_t           offset_;
    std::size_t           length_;
    DebugLevel            debug_level_;
    int                   fd_{-1};
    void*                 base_addr_{nullptr};
    std::unique_ptr<BumpPtr> bump_ptr_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  NumaAllocator  –  DRAM slice on a NUMA node
// ─────────────────────────────────────────────────────────────────────────────

class NumaAllocator : public CxlAllocator {
public:
    explicit NumaAllocator(int          node,
                           std::size_t  length = DaxAllocator::default_length,
                           DebugLevel   dbg    = DebugLevel::low)
        : node_{node}, length_{length}, debug_level_{dbg}
    {
        if (numa_available() == -1)
            throw std::runtime_error("NUMA unavailable");

        base_addr_ = numa_alloc_onnode(length, node);
        if (!base_addr_)
            throw std::runtime_error("numa_alloc_onnode failed");

        bump_ptr_ = std::make_unique<BumpPtr>(base_addr_, length);

        std::ostringstream msg;
        msg << "NUMA alloc ok: node=" << node
            << " length=" << length
            << " addr=0x" << std::hex << reinterpret_cast<std::uintptr_t>(base_addr_);
        log(debug_level_, DebugLevel::low, msg.str());
    }

    ~NumaAllocator() override
    {
        if (base_addr_) numa_free(base_addr_, length_);
    }

    // ─── allocator API ────────────────────────────────────────────────────
    void* allocate(std::size_t bytes) override
    {
        void* p = bump_ptr_->alloc(bytes, 1);
        log(debug_level_, DebugLevel::high,
            "allocate(" + std::to_string(bytes) + ") → " +
            std::to_string(reinterpret_cast<std::uintptr_t>(p)));
        return p;
    }

    void* allocate_aligned(std::size_t bytes, std::size_t alignment = 64) override
    {
        void* p = bump_ptr_->alloc(bytes, alignment);
        log(debug_level_, DebugLevel::high,
            "allocate_aligned(" + std::to_string(bytes) + ", align=" +
            std::to_string(alignment) + ") → " +
            std::to_string(reinterpret_cast<std::uintptr_t>(p)));
        return p;
    }

    [[nodiscard]] std::size_t used() const noexcept override      { return bump_ptr_->used(); }
    [[nodiscard]] std::size_t remaining() const noexcept override { return bump_ptr_->remaining(); }
    [[nodiscard]] std::size_t capacity() const noexcept override  { return bump_ptr_->capacity(); }

    bool test_memory() override
    {
        alignas(64) uint8_t pattern[64];
        for (int i = 0; i < 64; ++i) pattern[i] = static_cast<uint8_t>(i + 17);

        std::memcpy(base_addr_, pattern, 64);
        _mm_clflush(base_addr_);
        _mm_mfence();

        uint8_t verify[64];
        std::memcpy(verify, base_addr_, 64);
        bool ok = std::memcmp(pattern, verify, 64) == 0;

        log(debug_level_, DebugLevel::low,
            std::string("test_memory ") + (ok ? "✓" : "✗"));
        return ok;
    }

    void set_debug(DebugLevel lvl) noexcept override { debug_level_ = lvl; }

private:
    int                  node_;
    std::size_t          length_;
    DebugLevel           debug_level_;
    void*                base_addr_{nullptr};
    std::unique_ptr<BumpPtr> bump_ptr_;
};

} // namespace cxl
#endif // CXL_ALLOCATOR_HPP_
