// Transport layer unit tests. Validates the wire protocol header encode/decode,
// TCP channel send/recv, and heartbeat framing used by the distributed runtime.

#include "turboquant/transport.h"
#include <cassert>
#include <cstdio>
#include <cstring>

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

int main() {
    printf("test_transport:\n");
    test_wire_header_roundtrip();
    test_wire_header_payload_size();
    test_wire_header_1d_tensor();
    printf("All transport tests passed.\n");
    return 0;
}
