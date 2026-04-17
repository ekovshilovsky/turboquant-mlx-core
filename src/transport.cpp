// POSIX TCP socket implementation of the TurboQuant transport layer.
// TCP_NODELAY is set on every connected channel to minimize per-token latency;
// the small activation payloads common in pipeline-parallel inference suffer
// badly from Nagle's algorithm.

#include "turboquant/transport.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

namespace turboquant {

// ---------------------------------------------------------------------------
// TcpListener
// ---------------------------------------------------------------------------

TcpListener::TcpListener() {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ >= 0) {
        int opt = 1;
        ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    }
}

TcpListener::~TcpListener() { close(); }

int TcpListener::bind_any(const char* addr) {
    if (fd_ < 0) return -1;
    struct sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port = 0; // Kernel assigns a free port
    ::inet_pton(AF_INET, addr, &sa.sin_addr);
    if (::bind(fd_, reinterpret_cast<struct sockaddr*>(&sa), sizeof(sa)) < 0) {
        return -1;
    }
    socklen_t len = sizeof(sa);
    if (::getsockname(fd_, reinterpret_cast<struct sockaddr*>(&sa), &len) < 0) {
        return -1;
    }
    return ntohs(sa.sin_port);
}

void TcpListener::listen(int backlog) {
    if (fd_ >= 0) ::listen(fd_, backlog);
}

TcpChannel TcpListener::accept() {
    if (fd_ < 0) return TcpChannel();
    struct sockaddr_in client_addr{};
    socklen_t len = sizeof(client_addr);
    int client_fd = ::accept(fd_, reinterpret_cast<struct sockaddr*>(&client_addr), &len);
    if (client_fd < 0) return TcpChannel();
    int opt = 1;
    ::setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    return TcpChannel(client_fd);
}

void TcpListener::close() {
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

// ---------------------------------------------------------------------------
// TcpChannel
// ---------------------------------------------------------------------------

TcpChannel::TcpChannel() = default;
TcpChannel::TcpChannel(int fd) : fd_(fd) {}
TcpChannel::~TcpChannel() { close(); }

TcpChannel::TcpChannel(TcpChannel&& other) noexcept : fd_(other.fd_) {
    other.fd_ = -1;
}

TcpChannel& TcpChannel::operator=(TcpChannel&& other) noexcept {
    if (this != &other) {
        close();
        fd_ = other.fd_;
        other.fd_ = -1;
    }
    return *this;
}

bool TcpChannel::connect(const char* addr, int port) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) return false;
    struct sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, addr, &sa.sin_addr);
    if (::connect(fd_, reinterpret_cast<struct sockaddr*>(&sa), sizeof(sa)) < 0) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    int opt = 1;
    ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    return true;
}

bool TcpChannel::send_tensor(const WireHeader& hdr, const uint8_t* payload) {
    uint8_t header_buf[kWireHeaderMaxBytes];
    size_t header_size = wire_header_encode(hdr, header_buf);
    if (!send_all(header_buf, header_size)) return false;
    size_t payload_size = wire_payload_bytes(hdr);
    if (payload_size > 0 && !send_all(payload, payload_size)) return false;
    return true;
}

bool TcpChannel::recv_tensor(WireHeader& hdr, std::vector<uint8_t>& recv_data) {
    // Read ndim first so we know how much of the variable-length header follows.
    uint32_t ndim;
    if (!recv_all(&ndim, 4)) return false;
    if (ndim > kWireMaxDims) return false;

    // Remainder: shape[ndim] + dtype + sequence_id
    size_t remaining = ndim * 4 + 4 + 4;
    std::vector<uint8_t> rest(remaining);
    if (remaining > 0 && !recv_all(rest.data(), remaining)) return false;

    uint8_t full_buf[kWireHeaderMaxBytes];
    std::memcpy(full_buf, &ndim, 4);
    if (remaining > 0) {
        std::memcpy(full_buf + 4, rest.data(), remaining);
    }
    wire_header_decode(full_buf, 4 + remaining, hdr);

    size_t payload_size = wire_payload_bytes(hdr);
    recv_data.resize(payload_size);
    if (payload_size > 0 && !recv_all(recv_data.data(), payload_size)) return false;
    return true;
}

bool TcpChannel::send_ack() {
    uint8_t ack = 0x06; // ASCII ACK
    return send_all(&ack, 1);
}

bool TcpChannel::recv_ack() {
    uint8_t ack;
    return recv_all(&ack, 1) && ack == 0x06;
}

bool TcpChannel::is_connected() const { return fd_ >= 0; }

void TcpChannel::close() {
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

bool TcpChannel::send_all(const void* data, size_t len) {
    if (fd_ < 0) return false;
    const uint8_t* ptr = static_cast<const uint8_t*>(data);
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::send(fd_, ptr + sent, len - sent, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool TcpChannel::recv_all(void* data, size_t len) {
    if (fd_ < 0) return false;
    uint8_t* ptr = static_cast<uint8_t*>(data);
    size_t received = 0;
    while (received < len) {
        ssize_t n = ::recv(fd_, ptr + received, len - received, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        received += static_cast<size_t>(n);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Heartbeat
// ---------------------------------------------------------------------------

size_t heartbeat_encode(const Heartbeat& hb, uint8_t* buf) {
    size_t pos = 0;
    std::memcpy(buf + pos, &hb.rank, 4); pos += 4;
    buf[pos++] = static_cast<uint8_t>(hb.state);
    buf[pos++] = hb.low_memory ? 1 : 0;
    std::memcpy(buf + pos, &hb.available_memory_gb, 4); pos += 4;
    std::memcpy(buf + pos, &hb.layer_start, 4); pos += 4;
    std::memcpy(buf + pos, &hb.layer_end, 4); pos += 4;
    std::memcpy(buf + pos, &hb.tokens_processed, 8); pos += 8;
    std::memcpy(buf + pos, &hb.avg_layer_ms, 4); pos += 4;
    std::memcpy(buf + pos, &hb.syncing_percent, 4); pos += 4;
    return pos;
}

size_t heartbeat_decode(const uint8_t* buf, size_t len, Heartbeat& hb) {
    if (len < kHeartbeatBytes) return 0;
    size_t pos = 0;
    std::memcpy(&hb.rank, buf + pos, 4); pos += 4;
    hb.state = static_cast<NodeStateCode>(buf[pos++]);
    hb.low_memory = buf[pos++] != 0;
    std::memcpy(&hb.available_memory_gb, buf + pos, 4); pos += 4;
    std::memcpy(&hb.layer_start, buf + pos, 4); pos += 4;
    std::memcpy(&hb.layer_end, buf + pos, 4); pos += 4;
    std::memcpy(&hb.tokens_processed, buf + pos, 8); pos += 8;
    std::memcpy(&hb.avg_layer_ms, buf + pos, 4); pos += 4;
    std::memcpy(&hb.syncing_percent, buf + pos, 4); pos += 4;
    return pos;
}

bool TcpChannel::send_heartbeat(const Heartbeat& hb) {
    uint8_t buf[kHeartbeatBytes];
    heartbeat_encode(hb, buf);
    return send_all(buf, kHeartbeatBytes);
}

bool TcpChannel::recv_heartbeat(Heartbeat& hb) {
    uint8_t buf[kHeartbeatBytes];
    if (!recv_all(buf, kHeartbeatBytes)) return false;
    heartbeat_decode(buf, kHeartbeatBytes, hb);
    return true;
}

} // namespace turboquant
