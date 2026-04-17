#pragma once

// Transport layer for TurboQuant distributed inference. Defines the on-the-wire
// tensor header, dtype codes, and payload sizing helpers shared by TCP channels,
// heartbeats, and the cluster runtime. All formats are little-endian because
// every supported deployment target (Apple Silicon, Linux x86_64) is little-endian.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace turboquant {

class TcpChannel;

/// Maximum dimensions for a tensor in the wire protocol.
static constexpr int kWireMaxDims = 8;

/// Maximum encoded header size: 4 (ndim) + 8*4 (shape) + 4 (dtype) + 4 (seq_id) = 44 bytes.
static constexpr size_t kWireHeaderMaxBytes = 44;

/// Dtype codes for wire serialization. Values are stable across versions and
/// map 1:1 to the corresponding MLX dtype so receivers can allocate storage
/// without additional negotiation.
enum class WireDtype : uint32_t {
    Float32  = 0,
    Float16  = 1,
    BFloat16 = 2,
    Int32    = 3,
    UInt32   = 4,
    UInt8    = 5,
};

/// Fixed-size header prepended to every tensor transfer on the data channel.
/// Encodes tensor metadata so the receiver can allocate and interpret the
/// payload without any out-of-band schema negotiation.
struct WireHeader {
    uint32_t ndim = 0;
    uint32_t shape[kWireMaxDims] = {};
    WireDtype dtype = WireDtype::Float32;
    uint32_t sequence_id = 0;
};

/// Encode a WireHeader into a byte buffer. Returns the number of bytes written.
/// The buffer must be at least kWireHeaderMaxBytes.
inline size_t wire_header_encode(const WireHeader& hdr, uint8_t* buf) {
    size_t pos = 0;
    std::memcpy(buf + pos, &hdr.ndim, 4); pos += 4;
    for (uint32_t i = 0; i < hdr.ndim && i < kWireMaxDims; ++i) {
        std::memcpy(buf + pos, &hdr.shape[i], 4); pos += 4;
    }
    uint32_t dtype_val = static_cast<uint32_t>(hdr.dtype);
    std::memcpy(buf + pos, &dtype_val, 4); pos += 4;
    std::memcpy(buf + pos, &hdr.sequence_id, 4); pos += 4;
    return pos;
}

/// Decode a WireHeader from a byte buffer. Returns the number of bytes consumed,
/// or 0 if the buffer is too small or the encoded ndim exceeds kWireMaxDims.
inline size_t wire_header_decode(const uint8_t* buf, size_t buf_len, WireHeader& hdr) {
    size_t pos = 0;
    if (buf_len < 4) return 0;
    std::memcpy(&hdr.ndim, buf + pos, 4); pos += 4;
    if (hdr.ndim > kWireMaxDims) hdr.ndim = kWireMaxDims;
    for (uint32_t i = 0; i < hdr.ndim; ++i) {
        if (pos + 4 > buf_len) return 0;
        std::memcpy(&hdr.shape[i], buf + pos, 4); pos += 4;
    }
    if (pos + 8 > buf_len) return 0;
    uint32_t dtype_val;
    std::memcpy(&dtype_val, buf + pos, 4); pos += 4;
    hdr.dtype = static_cast<WireDtype>(dtype_val);
    std::memcpy(&hdr.sequence_id, buf + pos, 4); pos += 4;
    return pos;
}

/// Compute the byte size of the tensor payload described by a WireHeader.
inline size_t wire_payload_bytes(const WireHeader& hdr) {
    if (hdr.ndim == 0) return 0;
    size_t elements = 1;
    for (uint32_t i = 0; i < hdr.ndim; ++i) {
        elements *= hdr.shape[i];
    }
    size_t bytes_per_element = 4;
    switch (hdr.dtype) {
        case WireDtype::Float32:  bytes_per_element = 4; break;
        case WireDtype::Float16:  bytes_per_element = 2; break;
        case WireDtype::BFloat16: bytes_per_element = 2; break;
        case WireDtype::Int32:    bytes_per_element = 4; break;
        case WireDtype::UInt32:   bytes_per_element = 4; break;
        case WireDtype::UInt8:    bytes_per_element = 1; break;
    }
    return elements * bytes_per_element;
}

/// Lightweight TCP listener for accepting incoming connections.
/// Wraps POSIX socket bind/listen/accept for single-use server sockets.
class TcpListener {
public:
    TcpListener();
    ~TcpListener();

    TcpListener(const TcpListener&) = delete;
    TcpListener& operator=(const TcpListener&) = delete;

    /// Bind to the given address on any available port. Returns the bound port,
    /// or -1 on failure.
    int bind_any(const char* addr);

    /// Start listening with the given backlog.
    void listen(int backlog);

    /// Accept a single incoming connection. Blocks until a client connects.
    TcpChannel accept();

    /// Close the listener socket.
    void close();

private:
    int fd_ = -1;
};

/// Bidirectional TCP channel for sending and receiving tensors.
/// Wraps a connected POSIX socket with the TurboQuant wire protocol.
class TcpChannel {
public:
    TcpChannel();
    explicit TcpChannel(int fd);
    ~TcpChannel();

    TcpChannel(TcpChannel&& other) noexcept;
    TcpChannel& operator=(TcpChannel&& other) noexcept;
    TcpChannel(const TcpChannel&) = delete;
    TcpChannel& operator=(const TcpChannel&) = delete;

    /// Connect to a remote address and port. Returns true on success.
    bool connect(const char* addr, int port);

    /// Send a tensor: wire header followed by raw payload bytes.
    bool send_tensor(const WireHeader& hdr, const uint8_t* payload);

    /// Receive a tensor. The output vector is resized to hold the payload.
    bool recv_tensor(WireHeader& hdr, std::vector<uint8_t>& recv_data);

    /// Send a 1-byte ACK used for flow control.
    bool send_ack();

    /// Receive a 1-byte ACK. Blocks until received.
    bool recv_ack();

    /// Check if the channel holds an open socket.
    bool is_connected() const;

    /// Close the channel.
    void close();

private:
    int fd_ = -1;

    bool send_all(const void* data, size_t len);
    bool recv_all(void* data, size_t len);
};

} // namespace turboquant
