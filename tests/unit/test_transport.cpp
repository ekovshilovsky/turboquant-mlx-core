// Transport layer unit tests. Validates the wire protocol header encode/decode,
// TCP channel send/recv, and heartbeat framing used by the distributed runtime.

#include "turboquant/transport.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

using namespace turboquant;

static void test_wire_header_roundtrip() {
    WireHeader hdr;
    hdr.ndim = 2;
    hdr.shape[0] = 64;
    hdr.shape[1] = 5120;
    hdr.dtype = WireDtype::BFloat16;
    hdr.sequence_id = 42;

    uint8_t buf[kWireHeaderMaxBytes];
    size_t encoded_size = wire_header_encode(hdr, buf);

    WireHeader decoded;
    size_t decoded_size = wire_header_decode(buf, encoded_size, decoded);

    assert(decoded_size == encoded_size);
    assert(decoded.ndim == 2);
    assert(decoded.shape[0] == 64);
    assert(decoded.shape[1] == 5120);
    assert(decoded.dtype == WireDtype::BFloat16);
    assert(decoded.sequence_id == 42);
    printf("  PASS: wire header roundtrip\n");
}

static void test_wire_header_payload_size() {
    WireHeader hdr;
    hdr.ndim = 2;
    hdr.shape[0] = 1;
    hdr.shape[1] = 5120;
    hdr.dtype = WireDtype::BFloat16;
    hdr.sequence_id = 0;

    // BFloat16 = 2 bytes per element, 1 * 5120 = 5120 elements
    assert(wire_payload_bytes(hdr) == 10240);
    printf("  PASS: wire header payload size\n");
}

static void test_wire_header_1d_tensor() {
    WireHeader hdr;
    hdr.ndim = 1;
    hdr.shape[0] = 256;
    hdr.dtype = WireDtype::Float32;
    hdr.sequence_id = 99;

    uint8_t buf[kWireHeaderMaxBytes];
    size_t sz = wire_header_encode(hdr, buf);

    WireHeader decoded;
    wire_header_decode(buf, sz, decoded);

    assert(decoded.ndim == 1);
    assert(decoded.shape[0] == 256);
    assert(decoded.dtype == WireDtype::Float32);
    assert(wire_payload_bytes(decoded) == 1024);
    printf("  PASS: wire header 1d tensor\n");
}

static void test_tcp_loopback_tensor() {
    TcpListener listener;
    int port = listener.bind_any("127.0.0.1");
    assert(port > 0);
    listener.listen(1);

    std::thread client_thread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        TcpChannel client;
        bool ok = client.connect("127.0.0.1", port);
        assert(ok);

        WireHeader hdr;
        hdr.ndim = 2;
        hdr.shape[0] = 2;
        hdr.shape[1] = 3;
        hdr.dtype = WireDtype::Float32;
        hdr.sequence_id = 7;

        float data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        bool sent = client.send_tensor(hdr, reinterpret_cast<const uint8_t*>(data));
        assert(sent);
    });

    TcpChannel server = listener.accept();

    WireHeader recv_hdr;
    std::vector<uint8_t> recv_data;
    bool received = server.recv_tensor(recv_hdr, recv_data);
    assert(received);

    assert(recv_hdr.ndim == 2);
    assert(recv_hdr.shape[0] == 2);
    assert(recv_hdr.shape[1] == 3);
    assert(recv_hdr.dtype == WireDtype::Float32);
    assert(recv_hdr.sequence_id == 7);
    assert(recv_data.size() == 24); // 6 floats * 4 bytes

    const float* recv_floats = reinterpret_cast<const float*>(recv_data.data());
    assert(recv_floats[0] == 1.0f);
    assert(recv_floats[5] == 6.0f);

    client_thread.join();
    printf("  PASS: tcp loopback tensor\n");
}

static void test_tcp_ack() {
    TcpListener listener;
    int port = listener.bind_any("127.0.0.1");
    listener.listen(1);

    std::thread client_thread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        TcpChannel client;
        client.connect("127.0.0.1", port);
        bool got_ack = client.recv_ack();
        assert(got_ack);
    });

    TcpChannel server = listener.accept();
    bool sent_ack = server.send_ack();
    assert(sent_ack);

    client_thread.join();
    printf("  PASS: tcp ack\n");
}

int main() {
    printf("test_transport:\n");
    test_wire_header_roundtrip();
    test_wire_header_payload_size();
    test_wire_header_1d_tensor();
    test_tcp_loopback_tensor();
    test_tcp_ack();
    printf("All transport tests passed.\n");
    return 0;
}
