"""
Tests for spectral alignment module.
"""

import pytest
import numpy as np
from scipy import sparse

from temporal_spectral_flow.alignment import (
    SpectralMatcher,
    SignConvention,
    SpectralAligner,
    AlignedSpectralPair,
)


class TestSpectralMatcher:
    """Tests for eigenvalue matching."""

    def test_identity_matching(self):
        """Identical eigenvalues should match to identity permutation."""
        matcher = SpectralMatcher()

        lambda_ = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
        perm, cost = matcher.match_with_permutation(lambda_, lambda_)

        assert np.array_equal(perm, np.arange(5))
        assert cost < 1e-10

    def test_permuted_matching(self):
        """Should recover permutation of eigenvalues."""
        matcher = SpectralMatcher()

        lambda_source = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
        true_perm = np.array([2, 0, 4, 1, 3])
        lambda_target = lambda_source[true_perm]

        recovered_perm, cost = matcher.match_with_permutation(
            lambda_source, lambda_target
        )

        # recovered_perm[i] = j means lambda_target[j] matches lambda_source[i]
        # So lambda_target[recovered_perm] should equal lambda_source
        assert np.allclose(lambda_target[recovered_perm], lambda_source)

    def test_crossing_detection(self):
        """Should detect eigenvalue crossings."""
        matcher = SpectralMatcher()

        # Sequence where eigenvalues 1 and 2 cross
        lambda_seq = [
            np.array([0.1, 0.4, 0.6, 1.0]),  # t=0
            np.array([0.1, 0.48, 0.52, 1.0]),  # t=1: approaching
            np.array([0.1, 0.52, 0.48, 1.0]),  # t=2: crossed!
            np.array([0.1, 0.6, 0.4, 1.0]),  # t=3: separated
        ]

        crossings = matcher.detect_crossings(lambda_seq)

        # Should detect crossing between t=1 and t=2
        assert len(crossings) > 0
        # At least one crossing involving modes 1 and 2
        crossing_times = [c[0] for c in crossings]
        assert 1 in crossing_times

    def test_degeneracy_detection(self):
        """Should detect near-degenerate eigenvalues."""
        matcher = SpectralMatcher(degeneracy_threshold=0.01)

        lambda_ = np.array([0.1, 0.5, 0.505, 1.0])  # modes 1,2 are degenerate

        degeneracies = matcher.detect_degeneracies(lambda_)

        assert (1, 2) in degeneracies

    def test_cost_types(self):
        """Test different cost types compute without error."""
        lambda_source = np.array([0.1, 0.5, 1.0])
        lambda_target = np.array([0.15, 0.45, 1.1])

        for cost_type in ["absolute", "relative", "log"]:
            matcher = SpectralMatcher(cost_type=cost_type)
            perm, cost = matcher.match_with_permutation(lambda_source, lambda_target)
            assert len(perm) == 3
            assert cost >= 0

    def test_crossing_penalty(self):
        """Crossing penalty should discourage non-identity permutations."""
        # Setup where eigenvalues could swap
        lambda_source = np.array([0.5, 0.51])
        lambda_target = np.array([0.51, 0.5])  # Swapped

        # Without penalty, should swap
        matcher_no_penalty = SpectralMatcher(crossing_penalty=0.0)
        perm1, _ = matcher_no_penalty.match_with_permutation(lambda_source, lambda_target)

        # With large penalty, should prefer identity
        matcher_with_penalty = SpectralMatcher(crossing_penalty=1.0)
        perm2, _ = matcher_with_penalty.match_with_permutation(lambda_source, lambda_target)

        # The penalty should affect the matching decision
        assert np.array_equal(perm1, np.array([1, 0]))  # Swap
        assert np.array_equal(perm2, np.array([0, 1]))  # Identity preferred

    def test_dimension_mismatch_error(self):
        """Should raise error for mismatched dimensions in match_with_permutation."""
        matcher = SpectralMatcher()

        lambda_source = np.array([0.1, 0.5, 1.0])
        lambda_target = np.array([0.15, 0.45])

        with pytest.raises(ValueError, match="Dimension mismatch"):
            matcher.match_with_permutation(lambda_source, lambda_target)


class TestSignConvention:
    """Tests for sign conventions."""

    def test_max_entry_deterministic(self):
        """max_entry should give deterministic signs."""
        convention = SignConvention(method="max_entry")

        phi = np.array([0.1, -0.8, 0.3, -0.2])
        sign1 = convention.compute_sign(phi)
        sign2 = convention.compute_sign(phi)

        assert sign1 == sign2
        assert sign1 == -1  # max abs is -0.8, so sign is -1

    def test_canonicalize_idempotent(self):
        """Canonicalizing twice should give same result."""
        convention = SignConvention(method="max_entry")

        Phi = np.random.randn(50, 5)
        Phi, _ = np.linalg.qr(Phi)

        Phi_can1, signs1 = convention.canonicalize(Phi)
        Phi_can2, signs2 = convention.canonicalize(Phi_can1)

        assert np.allclose(Phi_can1, Phi_can2)
        assert np.allclose(signs2, np.ones(5))  # Already canonical

    def test_sign_flip_invariance(self):
        """Flipped eigenvector should canonicalize to same result."""
        convention = SignConvention(method="max_entry")

        phi = np.array([0.1, 0.8, 0.3, 0.2])
        phi_flipped = -phi

        Phi = phi.reshape(-1, 1)
        Phi_flipped = phi_flipped.reshape(-1, 1)

        Phi_can, _ = convention.canonicalize(Phi)
        Phi_flipped_can, _ = convention.canonicalize(Phi_flipped)

        assert np.allclose(Phi_can, Phi_flipped_can)

    def test_all_methods(self):
        """All sign convention methods should work."""
        phi = np.array([0.1, 0.8, -0.3, 0.2])

        for method in ["max_entry", "sum_positive", "first_entry", "moment"]:
            convention = SignConvention(method=method)
            sign = convention.compute_sign(phi)
            assert sign in [1.0, -1.0]

    def test_unknown_method_error(self):
        """Should raise error for unknown method."""
        convention = SignConvention(method="unknown")

        phi = np.array([0.1, 0.8, 0.3])
        with pytest.raises(ValueError, match="Unknown method"):
            convention.compute_sign(phi)

    def test_align_to_reference(self):
        """align_to_reference should produce consistent signs."""
        convention = SignConvention(method="max_entry")

        Phi_ref = np.random.randn(50, 5)
        Phi_ref, _ = np.linalg.qr(Phi_ref)

        # Flip some signs
        signs = np.array([1, -1, 1, -1, 1])
        Phi_target = Phi_ref * signs

        Phi_aligned, _ = convention.align_to_reference(Phi_ref, Phi_target)

        # After alignment, canonical forms should match
        ref_can, _ = convention.canonicalize(Phi_ref)
        aligned_can, _ = convention.canonicalize(Phi_aligned)
        assert np.allclose(ref_can, aligned_can)


class TestSpectralAligner:
    """Tests for full alignment pipeline."""

    def test_align_identical(self):
        """Aligning identical frames should give identity transform."""
        aligner = SpectralAligner()

        # Create random orthonormal matrix
        Phi = np.random.randn(50, 5)
        Phi, _ = np.linalg.qr(Phi)
        lambda_ = np.sort(np.random.rand(5))

        result = aligner.align(Phi, lambda_, Phi, lambda_)

        assert np.array_equal(result.eigenvalue_permutation, np.arange(5))
        assert np.allclose(result.eigenvalue_gaps, 0)

    def test_align_permuted(self):
        """Should correctly align permuted eigenvalues."""
        aligner = SpectralAligner()

        Phi = np.random.randn(50, 5)
        Phi, _ = np.linalg.qr(Phi)
        lambda_source = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Permute target
        perm = np.array([3, 1, 4, 0, 2])
        Phi_target = Phi[:, perm]
        lambda_target = lambda_source[perm]

        result = aligner.align(Phi, lambda_source, Phi_target, lambda_target)

        # After alignment, eigenvalues should match
        assert np.allclose(result.lambda_target_aligned, lambda_source)

    def test_align_sign_flipped(self):
        """Should handle sign flips correctly."""
        aligner = SpectralAligner()

        Phi = np.random.randn(50, 5)
        Phi, _ = np.linalg.qr(Phi)
        lambda_ = np.sort(np.random.rand(5))

        # Flip some signs
        signs = np.array([1, -1, 1, -1, 1])
        Phi_flipped = Phi * signs

        result = aligner.align(Phi, lambda_, Phi_flipped, lambda_)

        # After alignment, should match original (or be consistently signed)
        # Check that aligned has same canonical form as source
        convention = SignConvention()
        source_can, _ = convention.canonicalize(Phi)
        target_can, _ = convention.canonicalize(result.Phi_target_aligned)

        assert np.allclose(source_can, target_can)

    def test_sequence_alignment(self):
        """Should align a sequence of frames."""
        aligner = SpectralAligner()

        # Create sequence with gradual drift
        T = 5
        k = 4
        N = 50

        Phi_seq = []
        lambda_seq = []

        # Base frame
        np.random.seed(42)
        Phi_base = np.random.randn(N, k)
        Phi_base, _ = np.linalg.qr(Phi_base)
        lambda_base = np.sort(np.random.rand(k))

        for t in range(T):
            # Add small perturbation
            noise = 0.1 * np.random.randn(N, k)
            Phi_t = Phi_base + noise
            Phi_t, _ = np.linalg.qr(Phi_t)

            lambda_t = lambda_base + 0.05 * np.random.randn(k)
            lambda_t = np.sort(lambda_t)

            Phi_seq.append(Phi_t)
            lambda_seq.append(lambda_t)

        alignments = aligner.align_sequence(Phi_seq, lambda_seq)

        assert len(alignments) == T - 1

        # Check all alignments have reasonable costs
        for alignment in alignments:
            assert alignment.matching_cost < 1.0  # Should be small for gradual drift

    def test_eigenvalue_mismatch_error(self):
        """Should raise error for mismatched eigenvalue counts."""
        aligner = SpectralAligner()

        Phi_source = np.random.randn(50, 5)
        Phi_source, _ = np.linalg.qr(Phi_source)
        lambda_source = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        Phi_target = np.random.randn(50, 4)
        Phi_target, _ = np.linalg.qr(Phi_target)
        lambda_target = np.array([0.1, 0.3, 0.5, 0.7])

        with pytest.raises(ValueError, match="Eigenvalue count mismatch"):
            aligner.align(Phi_source, lambda_source, Phi_target, lambda_target)

    def test_sequence_length_mismatch_error(self):
        """Should raise error for mismatched sequence lengths."""
        aligner = SpectralAligner()

        Phi_seq = [np.random.randn(50, 5) for _ in range(3)]
        lambda_seq = [np.random.rand(5) for _ in range(4)]

        with pytest.raises(ValueError, match="same length"):
            aligner.align_sequence(Phi_seq, lambda_seq)

    def test_custom_components(self):
        """Should work with custom matcher and sign convention."""
        matcher = SpectralMatcher(cost_type="relative", degeneracy_threshold=1e-4)
        sign_convention = SignConvention(method="sum_positive")
        aligner = SpectralAligner(matcher=matcher, sign_convention=sign_convention)

        Phi = np.random.randn(50, 5)
        Phi, _ = np.linalg.qr(Phi)
        lambda_ = np.sort(np.random.rand(5))

        result = aligner.align(Phi, lambda_, Phi, lambda_)

        assert isinstance(result, AlignedSpectralPair)

    def test_compute_spectral_velocity(self):
        """Should compute spectral velocity for same-size frames."""
        aligner = SpectralAligner()

        np.random.seed(42)
        N, k = 50, 5

        Phi_source = np.random.randn(N, k)
        Phi_source, _ = np.linalg.qr(Phi_source)
        lambda_source = np.sort(np.random.rand(k))

        # Create slightly perturbed target
        noise = 0.1 * np.random.randn(N, k)
        Phi_target = Phi_source + noise
        Phi_target, _ = np.linalg.qr(Phi_target)
        lambda_target = lambda_source + 0.05 * np.random.randn(k)

        alignment = aligner.align(Phi_source, lambda_source, Phi_target, lambda_target)
        v_lambda, v_Phi = aligner.compute_spectral_velocity(alignment)

        assert v_lambda.shape == (k,)
        assert v_Phi is not None
        assert v_Phi.shape == (N, k)


class TestAlignedSpectralPair:
    """Tests for AlignedSpectralPair dataclass."""

    def test_dataclass_creation(self):
        """Should create AlignedSpectralPair correctly."""
        N, k = 50, 5
        Phi_source = np.random.randn(N, k)
        Phi_target = np.random.randn(N, k)
        lambda_source = np.random.rand(k)
        lambda_target = np.random.rand(k)
        perm = np.arange(k)
        signs = np.ones(k)
        gaps = np.abs(lambda_source - lambda_target)

        result = AlignedSpectralPair(
            Phi_source=Phi_source,
            Phi_target_aligned=Phi_target,
            lambda_source=lambda_source,
            lambda_target_aligned=lambda_target,
            eigenvalue_permutation=perm,
            sign_flips=signs,
            matching_cost=0.1,
            eigenvalue_gaps=gaps,
        )

        assert result.Phi_source.shape == (N, k)
        assert result.Phi_target_aligned.shape == (N, k)
        assert result.lambda_source.shape == (k,)
        assert result.matching_cost == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
