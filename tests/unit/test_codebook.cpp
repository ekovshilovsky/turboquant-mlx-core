#include "turboquant/codebook.h"
#include <cassert>
#include <cmath>
#include <cstdio>

using namespace turboquant;

static void test_1bit_centroids_match_paper() {
    // Theorem 3.1 (Zandieh et al.): 1-bit Lloyd-Max centroids for N(0,1) are +/- sqrt(2/pi)
    auto cb = generate_codebook(1);
    assert(cb.bits == 1);
    assert(cb.centroids.size() == 2);
    float expected = std::sqrt(2.0f / M_PI);
    assert(std::abs(cb.centroids[0] - (-expected)) < 1e-5f);
    assert(std::abs(cb.centroids[1] - expected) < 1e-5f);
    printf("  PASS: 1-bit centroids match paper (+/- sqrt(2/pi) = %.7f)\n", expected);
}

static void test_2bit_centroids_match_paper() {
    // Table 1 (Zandieh et al.): 2-bit Lloyd-Max centroids for N(0,1)
    auto cb = generate_codebook(2);
    assert(cb.bits == 2);
    assert(cb.centroids.size() == 4);

    float expected[4] = {-1.510232f, -0.4527800f, 0.4527800f, 1.510232f};
    for (int i = 0; i < 4; i++) {
        assert(std::abs(cb.centroids[i] - expected[i]) < 1e-5f);
    }
    printf("  PASS: 2-bit centroids match paper {-1.510, -0.453, 0.453, 1.510}\n");
}

static void test_4bit_centroids_sorted_symmetric() {
    auto cb = generate_codebook(4);
    assert(cb.bits == 4);
    assert(cb.centroids.size() == 16);

    // All centroids must be strictly increasing
    for (size_t i = 1; i < cb.centroids.size(); i++) {
        assert(cb.centroids[i] > cb.centroids[i-1]);
    }
    // N(0,1) symmetry: centroid[i] == -centroid[N-1-i]
    for (size_t i = 0; i < cb.centroids.size() / 2; i++) {
        assert(std::abs(cb.centroids[i] + cb.centroids[cb.centroids.size()-1-i]) < 1e-5f);
    }
    printf("  PASS: 4-bit centroids sorted and symmetric\n");
}

static void test_boundaries_are_midpoints() {
    // Boundaries must be exactly midway between adjacent centroids (Lloyd-Max optimality)
    auto cb = generate_codebook(3);
    assert(cb.bits == 3);
    assert(cb.centroids.size() == 8);
    assert(cb.boundaries.size() == 7);

    for (size_t i = 0; i < cb.boundaries.size(); i++) {
        float expected = (cb.centroids[i] + cb.centroids[i+1]) / 2.0f;
        assert(std::abs(cb.boundaries[i] - expected) < 1e-5f);
    }
    printf("  PASS: boundaries are midpoints between centroids\n");
}

static void test_quantize_roundtrip_at_centroids() {
    // Values exactly at centroids must round-trip without error
    auto cb = generate_codebook(4);
    assert(cb.centroids.size() == 16);

    auto input_values = mlx::core::array(cb.centroids.data(),
                                         {static_cast<int>(cb.centroids.size())});
    mlx::core::eval(input_values);

    auto indices = quantize(input_values, cb);
    mlx::core::eval(indices);

    auto recovered = dequantize(indices, cb);
    mlx::core::eval(recovered);

    const float* out = recovered.data<float>();
    for (size_t i = 0; i < cb.centroids.size(); i++) {
        assert(std::abs(out[i] - cb.centroids[i]) < 1e-5f);
    }
    printf("  PASS: quantize-dequantize round-trip at centroid values is exact\n");
}

static void test_validate_codebook_accepts_valid() {
    auto cb = generate_codebook(4);
    assert(validate_codebook(cb) == true);
    printf("  PASS: validate_codebook accepts valid 4-bit codebook\n");
}

static void test_validate_codebook_rejects_duplicate_centroids() {
    auto cb = generate_codebook(4);
    assert(cb.centroids.size() == 16);

    auto bad = cb;
    bad.centroids[0] = bad.centroids[1]; // violates strict ordering
    assert(validate_codebook(bad) == false);
    printf("  PASS: validate_codebook rejects codebook with duplicate centroids\n");
}

int main() {
    printf("test_codebook:\n");
    test_1bit_centroids_match_paper();
    test_2bit_centroids_match_paper();
    test_4bit_centroids_sorted_symmetric();
    test_boundaries_are_midpoints();
    test_quantize_roundtrip_at_centroids();
    test_validate_codebook_accepts_valid();
    test_validate_codebook_rejects_duplicate_centroids();
    printf("All codebook tests passed.\n");
    return 0;
}
