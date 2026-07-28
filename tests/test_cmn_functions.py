"""Pure analysis functions of the CMN receptive-field pipeline.

These are the numerical core of the analysis (local direction binning, spatial
cluster tracing, preferred-direction vectors) and are tested without a database.
"""
import numpy as np
import pytest

torch = pytest.importorskip('torch', reason='the cmn analysis module imports torch at module level')

from entarchy_vxpy_suite2p.analysis.cmn import functions


def bin_edges(bin_num=16):
    return np.linspace(-np.pi, np.pi, bin_num + 1)


def vectors_at_angle(angle, patch_num=1, frame_num=1, magnitude=1.0):
    """Motion vectors (frames x patches x 2) all pointing at `angle`."""
    vectors = np.zeros((frame_num, patch_num, 2))
    vectors[:, :, 0] = magnitude * np.cos(angle)
    vectors[:, :, 1] = magnitude * np.sin(angle)
    return vectors


class TestCrossProduct:

    def test_matches_numpy(self, rng):
        a, b = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
        assert np.array_equal(functions.crossproduct(a, b), np.cross(a, b))

    def test_right_handed(self):
        result = functions.crossproduct(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
        assert np.allclose(result, [0, 0, 1])


class TestProjectToLocal2dVectors:

    def test_output_shape(self, rng):
        # Normals away from the poles, where the local frame is well defined
        normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.6, 0.0, 0.8]])
        vectors = rng.normal(size=(7, 3, 3))
        result = functions.project_to_local_2d_vectors(normals, vectors)
        assert result.shape == (7, 3, 2)
        assert np.isfinite(result).all()

    def test_tangential_magnitude_is_preserved(self):
        """A vector tangent to the sphere at a patch keeps its length in the local frame."""
        normals = np.array([[1.0, 0.0, 0.0]])
        tangent = np.array([[[0.0, 2.0, 0.0]]])  # perpendicular to the normal

        result = functions.project_to_local_2d_vectors(normals, tangent)
        assert np.linalg.norm(result[0, 0]) == pytest.approx(2.0)

    def test_radial_component_is_discarded(self):
        """A vector along the patch normal has no local 2d component."""
        normals = np.array([[1.0, 0.0, 0.0]])
        radial = np.array([[[3.0, 0.0, 0.0]]])

        result = functions.project_to_local_2d_vectors(normals, radial)
        assert np.allclose(result[0, 0], 0.0, atol=1e-9)

    def test_up_direction_defines_vertical_axis(self):
        """At a patch on the equator, world 'up' maps to the local +y axis."""
        normals = np.array([[1.0, 0.0, 0.0]])
        upward = np.array([[[0.0, 0.0, 1.0]]])

        result = functions.project_to_local_2d_vectors(normals, upward)
        assert result[0, 0, 0] == pytest.approx(0.0, abs=1e-9)
        assert result[0, 0, 1] == pytest.approx(1.0)

    def test_rotated_up_direction_changes_frame(self):
        normals = np.array([[1.0, 0.0, 0.0]])
        vectors = np.array([[[0.0, 1.0, 0.0]]])

        default = functions.project_to_local_2d_vectors(normals, vectors)
        tilted = functions.project_to_local_2d_vectors(normals, vectors,
                                                       vertical_up_direction=[0.0, 0.3, 0.95])
        assert not np.allclose(default, tilted)

    def test_accepts_list_and_array_up_direction(self):
        normals = np.array([[1.0, 0.0, 0.0]])
        vectors = np.array([[[0.0, 0.0, 1.0]]])

        as_list = functions.project_to_local_2d_vectors(normals, vectors, [0, 0, 1])
        as_array = functions.project_to_local_2d_vectors(normals, vectors, np.array([0, 0, 1]))
        assert np.array_equal(as_list, as_array)

    def test_patch_normal_parallel_to_up_is_degenerate(self):
        """At a patch whose normal is parallel to the up direction the local frame
        is undefined and comes out as NaN (the icosphere has no such vertex, so
        this is only reachable with custom geometry)."""
        normals = np.array([[0.0, 0.0, 1.0]])
        vectors = np.array([[[1.0, 0.0, 0.0]]])

        with np.errstate(invalid='ignore'):
            result = functions.project_to_local_2d_vectors(normals, vectors)
        assert np.isnan(result).all()


class TestLocalFrameOrthonormality:
    """The local 2d frame must be orthonormal, otherwise projected motion vectors
    are distorted in both angle and magnitude."""

    @staticmethod
    def pitched_up(degrees):
        """The up direction as process_recording builds it from ants/init_x_rotation."""
        pitch = np.deg2rad(degrees)
        rotation = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                             [0, 1, 0],
                             [-np.sin(pitch), 0, np.cos(pitch)]])
        return np.array([0, 0, 1]) @ rotation

    @staticmethod
    def tangent_vectors(normals):
        """Unit vectors tangent to the sphere at each patch."""
        arbitrary = np.array([0.0, 1.0, 0.0])
        tangents = arbitrary - normals * np.dot(normals, arbitrary)[:, None]
        norms = np.linalg.norm(tangents, axis=1)
        keep = norms > 1e-6
        return normals[keep], tangents[keep] / norms[keep, None]

    def projected_lengths(self, points, up=None):
        """Lengths of projected unit tangents, excluding patches where the local
        frame is degenerate (normal parallel to the up direction, e.g. the poles)."""
        normals, tangents = self.tangent_vectors(points)
        with np.errstate(invalid='ignore'):
            projected = functions.project_to_local_2d_vectors(normals, tangents[None, :, :], up)[0]

        lengths = np.linalg.norm(projected, axis=1)
        finite = np.isfinite(lengths)
        assert finite.sum() > 0.9 * len(lengths), 'unexpectedly many degenerate patches'
        return lengths[finite]

    def test_default_up_direction_is_orthonormal(self, unit_sphere_points):
        lengths = self.projected_lengths(unit_sphere_points)
        assert np.allclose(lengths, 1.0, atol=1e-6)

    @pytest.mark.parametrize('pitch_degrees', [0, 5, 10, 20, 30])
    def test_pitched_up_direction_stays_orthonormal(self, unit_sphere_points, pitch_degrees):
        """Regression: the tangential component of the up direction used to be
        computed against [0,0,1] rather than the up direction itself, which left
        the frame non-orthonormal whenever ants/init_x_rotation was set."""
        lengths = self.projected_lengths(unit_sphere_points, self.pitched_up(pitch_degrees))
        assert np.allclose(lengths, 1.0, atol=1e-6)

    def test_local_axes_are_orthogonal_to_the_patch_normal(self, unit_sphere_points):
        """A vector along the patch normal must have no local 2d component, whatever
        the up direction."""
        up = self.pitched_up(20)
        normals, _ = self.tangent_vectors(unit_sphere_points)

        with np.errstate(invalid='ignore'):
            projected = functions.project_to_local_2d_vectors(normals, normals[None, :, :], up)[0]

        finite = np.isfinite(projected).all(axis=1)
        assert np.allclose(projected[finite], 0.0, atol=1e-6)


class TestCalculateLocalDirections:

    def test_output_shapes(self, rng):
        vectors = rng.normal(size=(20, 6, 2))
        edges = bin_edges(16)

        norms, etas = functions.calculate_local_directions(vectors, edges)
        assert norms.shape == (20, 6, 16)
        assert etas.shape == (6, 16)

    def test_direction_lands_in_expected_bin(self):
        edges = bin_edges(16)
        # Bin centres sit half a bin above each edge; pick the centre of bin 10
        centre = (edges[10] + edges[11]) / 2
        vectors = vectors_at_angle(centre, patch_num=1, frame_num=1, magnitude=2.0)

        _, etas = functions.calculate_local_directions(vectors, edges)
        assert np.argmax(etas[0]) == 10
        assert etas[0, 10] == pytest.approx(2.0)
        assert etas[0].sum() == pytest.approx(2.0)

    def test_eta_is_velocity_weighted_mean_over_frames(self):
        edges = bin_edges(16)
        centre = (edges[3] + edges[4]) / 2
        # Two frames at the same angle, magnitudes 1 and 3 -> mean 2
        vectors = np.concatenate([vectors_at_angle(centre, magnitude=1.0),
                                  vectors_at_angle(centre, magnitude=3.0)], axis=0)

        _, etas = functions.calculate_local_directions(vectors, edges)
        assert etas[0, 3] == pytest.approx(2.0)

    def test_zero_vectors_contribute_nothing(self):
        edges = bin_edges(16)
        vectors = np.zeros((5, 2, 2))
        _, etas = functions.calculate_local_directions(vectors, edges)
        assert np.allclose(etas, 0.0)

    def test_a_single_nan_frame_poisons_the_whole_patch(self):
        """NaN frames are NOT skipped: velocity is NaN and NaN * False == NaN, so one
        NaN frame turns every bin of that patch into NaN, which then silently
        propagates through the significance test as 'not significant'.

        The pipeline avoids this only because signal_selection is ANDed with
        cmn_phase_selection, so unfilled (NaN) frames are never passed in.
        """
        edges = bin_edges(16)
        centre = (edges[5] + edges[6]) / 2
        vectors = vectors_at_angle(centre, frame_num=2, magnitude=2.0)
        vectors[1] = np.nan

        _, etas = functions.calculate_local_directions(vectors, edges)
        assert np.isnan(etas).all()

    def test_frames_within_selection_are_nan_free(self):
        """The guarantee the pipeline actually relies on."""
        edges = bin_edges(16)
        centre = (edges[5] + edges[6]) / 2
        vectors = vectors_at_angle(centre, frame_num=3, magnitude=2.0)
        vectors[2] = np.nan
        selection = np.array([True, True, False])  # excludes the NaN frame

        _, etas = functions.calculate_local_directions(vectors[selection], edges)
        assert etas[0, 5] == pytest.approx(2.0)
        assert not np.isnan(etas).any()

    def test_patches_are_independent(self):
        edges = bin_edges(16)
        vectors = np.zeros((1, 2, 2))
        angle_a = (edges[2] + edges[3]) / 2
        angle_b = (edges[12] + edges[13]) / 2
        vectors[0, 0] = [np.cos(angle_a), np.sin(angle_a)]
        vectors[0, 1] = [np.cos(angle_b), np.sin(angle_b)]

        _, etas = functions.calculate_local_directions(vectors, edges)
        assert np.argmax(etas[0]) == 2
        assert np.argmax(etas[1]) == 12


class TestNumpyTorchEquivalence:
    """The pipeline compares observed ETAs (numpy) against bootstrapped ETAs (torch),
    so the two implementations must agree."""

    def test_random_data_matches(self, rng):
        edges = bin_edges(16)
        vectors = rng.normal(size=(50, 8, 2))

        _, numpy_etas = functions.calculate_local_directions(vectors, edges)
        _, torch_etas = functions.calculate_local_directions_torch(
            torch.tensor(vectors, dtype=torch.float32),
            torch.tensor(edges, dtype=torch.float32))

        assert np.allclose(numpy_etas, torch_etas.numpy(), atol=1e-5)

    def test_boundary_angle_is_handled_differently(self):
        """Known discrepancy: the numpy version uses closed intervals on both ends
        while the torch version uses half-open intervals. A motion vector pointing
        exactly at pi (directly 'left') is therefore counted by numpy but dropped
        by torch. Real data hits this only for exactly-axis-aligned vectors."""
        edges = bin_edges(16)
        vectors = vectors_at_angle(np.pi, magnitude=1.0)  # arctan2(0, -1) == pi exactly

        _, numpy_etas = functions.calculate_local_directions(vectors, edges)
        _, torch_etas = functions.calculate_local_directions_torch(
            torch.tensor(vectors, dtype=torch.float32),
            torch.tensor(edges, dtype=torch.float32))

        assert numpy_etas.sum() == pytest.approx(1.0)
        assert torch_etas.numpy().sum() == pytest.approx(0.0)

    def test_interior_edge_is_double_counted_by_numpy(self):
        """A vector exactly on an interior bin edge falls into both adjacent bins
        in the numpy implementation, but only one in the torch implementation."""
        edges = bin_edges(16)
        vectors = vectors_at_angle(edges[8], magnitude=1.0)  # exactly 0.0 rad

        _, numpy_etas = functions.calculate_local_directions(vectors, edges)
        _, torch_etas = functions.calculate_local_directions_torch(
            torch.tensor(vectors, dtype=torch.float32),
            torch.tensor(edges, dtype=torch.float32))

        assert (numpy_etas[0] > 0).sum() == 2
        assert (torch_etas.numpy()[0] > 0).sum() == 1


class TestPreferredDirections:

    def test_no_significant_bins_gives_zero_vector(self):
        etas = np.array([[1.0, 2.0, 3.0, 4.0]])
        significances = np.zeros((1, 4), dtype=bool)
        centres = np.array([-np.pi / 2, 0.0, np.pi / 2, np.pi])

        vectors = functions._calc_preferred_directions(etas, significances, centres)
        assert np.array_equal(vectors[0], [0, 0])

    def test_single_significant_bin_points_along_it(self):
        etas = np.array([[0.0, 5.0, 0.0, 0.0]])
        significances = np.array([[False, True, False, False]])
        centres = np.array([-np.pi / 2, 0.0, np.pi / 2, np.pi])

        vectors = functions._calc_preferred_directions(etas, significances, centres)
        # Bin centre 0 rad -> +x direction; normalised by the total eta sum
        assert vectors[0, 0] == pytest.approx(1.0)
        assert vectors[0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_opposing_bins_cancel(self):
        etas = np.array([[3.0, 0.0, 3.0, 0.0]])
        significances = np.array([[True, False, True, False]])
        centres = np.array([-np.pi / 2, 0.0, np.pi / 2, np.pi])

        vectors = functions._calc_preferred_directions(etas, significances, centres)
        assert np.allclose(vectors[0], [0, 0], atol=1e-9)

    def test_output_shape_matches_patch_count(self, rng):
        etas = rng.random((7, 16))
        significances = rng.random((7, 16)) > 0.5
        centres = bin_edges(16)[:-1]

        vectors = functions._calc_preferred_directions(etas, significances, centres)
        assert vectors.shape == (7, 2)

    def test_normalisation_uses_total_eta_sum(self):
        """Documents the current normalisation: the population vector is divided by
        the sum over ALL bins, not only the significant ones."""
        etas = np.array([[4.0, 4.0]])
        significances = np.array([[True, False]])
        centres = np.array([0.0, np.pi / 2])

        vectors = functions._calc_preferred_directions(etas, significances, centres)
        assert vectors[0, 0] == pytest.approx(0.5)  # 4 / (4 + 4)


class TestCreateClusters:
    """Spatial clustering of patches that share significant direction bins."""

    @staticmethod
    def ring_neighbours(patch_num):
        """Each patch is adjacent to its two ring neighbours (third slot repeats)."""
        neighbours = np.zeros((patch_num, 3), dtype=np.int64)
        for i in range(patch_num):
            neighbours[i] = [(i - 1) % patch_num, (i + 1) % patch_num, (i + 1) % patch_num]
        return neighbours

    def test_no_significant_bins_gives_no_clusters(self):
        significant = np.zeros((6, 4), dtype=bool)
        maps, indices, unique = functions.create_clusters(significant, self.ring_neighbours(6), 1)

        assert not maps.any()
        assert indices == []
        assert unique == []

    def test_isolated_patch_forms_no_cluster(self):
        """A single significant patch has no neighbour to connect to."""
        significant = np.zeros((6, 4), dtype=bool)
        significant[2, 1] = True

        _, indices, unique = functions.create_clusters(significant, self.ring_neighbours(6), 1)
        assert indices == []
        assert unique == []

    def test_two_adjacent_patches_form_one_cluster(self):
        significant = np.zeros((6, 4), dtype=bool)
        significant[2, 1] = True
        significant[3, 1] = True

        _, indices, unique = functions.create_clusters(significant, self.ring_neighbours(6), 1)
        assert len(unique) == 1
        assert set(unique[0]) == {2, 3}

    def test_neighbours_without_shared_bins_do_not_connect(self):
        significant = np.zeros((6, 4), dtype=bool)
        significant[2, 1] = True
        significant[3, 3] = True  # adjacent patch, different direction

        _, indices, unique = functions.create_clusters(significant, self.ring_neighbours(6), 1)
        assert unique == []

    def test_threshold_requires_enough_shared_bins(self):
        significant = np.zeros((6, 4), dtype=bool)
        significant[2, [0, 1]] = True
        significant[3, [1, 2]] = True  # only bin 1 in common

        _, _, with_threshold_1 = functions.create_clusters(significant, self.ring_neighbours(6), 1)
        _, _, with_threshold_2 = functions.create_clusters(significant, self.ring_neighbours(6), 2)

        assert len(with_threshold_1) == 1
        assert with_threshold_2 == []

    def test_two_separate_clusters(self):
        significant = np.zeros((10, 4), dtype=bool)
        significant[[1, 2], 0] = True
        significant[[6, 7], 2] = True

        _, _, unique = functions.create_clusters(significant, self.ring_neighbours(10), 1)
        assert len(unique) == 2
        assert {frozenset(u) for u in unique} == {frozenset({1, 2}), frozenset({6, 7})}

    def test_chain_is_traced_transitively(self):
        significant = np.zeros((10, 4), dtype=bool)
        significant[[1, 2, 3, 4], 0] = True

        _, _, unique = functions.create_clusters(significant, self.ring_neighbours(10), 1)
        assert len(unique) == 1
        assert set(unique[0]) == {1, 2, 3, 4}

    def test_cluster_map_marks_shared_bins_only(self):
        significant = np.zeros((6, 4), dtype=bool)
        significant[2, [0, 1]] = True
        significant[3, [1]] = True

        maps, indices, _ = functions.create_clusters(significant, self.ring_neighbours(6), 1)
        cluster = indices[0]
        marked_bins = {int(b) for _, b in cluster}
        assert marked_bins == {1}

    def test_full_ring_is_one_cluster(self):
        significant = np.zeros((8, 4), dtype=bool)
        significant[:, 0] = True

        _, _, unique = functions.create_clusters(significant, self.ring_neighbours(8), 1)
        assert len(unique) == 1
        assert set(unique[0]) == set(range(8))

    def test_recursion_depth_on_long_chain(self):
        """Cluster tracing is recursive; a long connected chain must not hit
        Python's recursion limit at realistic patch counts (642 for subdiv 3)."""
        patch_num = 642
        significant = np.zeros((patch_num, 4), dtype=bool)
        significant[:, 0] = True

        _, _, unique = functions.create_clusters(significant, self.ring_neighbours(patch_num), 1)
        assert len(unique) == 1
        assert len(unique[0]) == patch_num
