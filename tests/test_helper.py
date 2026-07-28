"""Geometry helpers: spherical coordinates, rotations, projections, platonic solids."""
import numpy as np
import pytest

from entarchy_vxpy_suite2p.analysis.cmn import helper


class TestSphericalCoordinates:

    def test_roundtrip(self, unit_sphere_points):
        for vec in unit_sphere_points:
            az, el, r = helper.cart2sph(*vec)
            back = helper.sph2cart(az, el, r)
            assert np.allclose(back, vec, atol=1e-6)

    def test_radius(self):
        assert helper.cart2sph(3.0, 4.0, 0.0)[2] == pytest.approx(5.0)
        assert helper.cart2sph(0.0, 0.0, 2.0)[2] == pytest.approx(2.0)

    @pytest.mark.parametrize('name,vector', [
        ('front', (1.0, 0.0, 0.0)),
        ('up', (0.0, 0.0, 1.0)),
        ('right', (0.0, -1.0, 0.0)),
    ])
    def test_universal_directions_match_convention(self, name, vector):
        """The module documents a fish-centric frame: azimuth 0/elevation 0 is
        'front', elevation 90 is 'up', azimuth 90 is 'right'."""
        azimuth, elevation = helper.universal_directions[name]
        cartesian = helper.sph2cart(azimuth, elevation, 1.0)
        assert np.allclose(cartesian, vector, atol=1e-9)

    def test_azimuth_sign_convention(self):
        # Azimuth is negated relative to the mathematical convention
        assert helper.cart2sph(0.0, -1.0, 0.0)[0] == pytest.approx(np.pi / 2)
        assert helper.cart2sph(0.0, 1.0, 0.0)[0] == pytest.approx(-np.pi / 2)

    def test_elevation_range(self, unit_sphere_points):
        elevations = np.array([helper.cart2sph(*v)[1] for v in unit_sphere_points])
        assert np.all(elevations >= -np.pi / 2 - 1e-9)
        assert np.all(elevations <= np.pi / 2 + 1e-9)


class TestRotationMatrix:

    def test_rotates_v1_onto_v2(self, rng):
        for _ in range(20):
            v1 = rng.normal(size=3)
            v2 = rng.normal(size=3)
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)

            matrix = helper.rotmat_from_to(v1, v2)
            assert np.allclose(matrix @ v1, v2, atol=1e-8)

    def test_direction_is_not_inverted(self, rng):
        """Regression: the rotation axis used to be cross(v2, v1), which mapped
        v2 onto v1 instead."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])

        matrix = helper.rotmat_from_to(v1, v2)
        assert np.allclose(matrix @ v1, v2, atol=1e-8)
        assert not np.allclose(matrix @ v2, v1, atol=1e-8)

    def test_is_orthonormal(self, rng):
        v1, v2 = rng.normal(size=3), rng.normal(size=3)
        matrix = helper.rotmat_from_to(v1, v2)

        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-8)
        assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-8)

    def test_normalizes_inputs(self):
        a = helper.rotmat_from_to(np.array([2.0, 0, 0]), np.array([0, 5.0, 0]))
        b = helper.rotmat_from_to(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
        assert np.allclose(a, b)

    def test_parallel_vectors_are_degenerate(self):
        """Known limitation: identical/antiparallel inputs have no unique rotation
        axis, so the cross product is zero and the result is undefined."""
        with np.errstate(invalid='ignore'):
            result = helper.rotmat_from_to(np.array([1.0, 0, 0]), np.array([1.0, 0, 0]))
        assert np.isnan(result).any()


class TestFisherPdf:

    def test_peaks_at_mean_direction(self):
        mu = np.array([0.0, 0.0, 1.0])
        at_mean = helper.fisher_pdf(mu, mu, kappa=4.0)
        away = helper.fisher_pdf(np.array([0.0, 0.0, -1.0]), mu, kappa=4.0)
        orthogonal = helper.fisher_pdf(np.array([1.0, 0.0, 0.0]), mu, kappa=4.0)

        assert at_mean > orthogonal > away
        assert away > 0

    def test_integrates_to_one_over_sphere(self):
        """Normalization check by numeric integration in spherical coordinates."""
        mu = np.array([0.0, 0.0, 1.0])
        kappa = 2.0

        thetas = np.linspace(0, np.pi, 400)
        phis = np.linspace(0, 2 * np.pi, 400)
        d_theta = thetas[1] - thetas[0]
        d_phi = phis[1] - phis[0]

        total = 0.0
        for theta in thetas:
            vec = np.array([np.sin(theta), 0.0, np.cos(theta)])
            total += helper.fisher_pdf(vec, mu, kappa) * np.sin(theta) * d_theta * len(phis) * d_phi

        assert total == pytest.approx(1.0, rel=0.02)

    def test_higher_kappa_concentrates(self):
        mu = np.array([0.0, 0.0, 1.0])
        assert helper.fisher_pdf(mu, mu, 10.0) > helper.fisher_pdf(mu, mu, 1.0)


class TestProjections:

    @pytest.mark.parametrize('projection', [helper.mollweide_projection, helper.eckert_iv_projection])
    def test_central_meridian_maps_to_origin(self, projection):
        x, y = projection(0.0, 0.0)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    @pytest.mark.parametrize('projection', [helper.mollweide_projection, helper.eckert_iv_projection])
    def test_symmetry_about_equator(self, projection):
        x_up, y_up = projection(0.5, 0.4)
        x_down, y_down = projection(0.5, -0.4)
        assert x_up == pytest.approx(x_down)
        assert y_up == pytest.approx(-y_down)

    @pytest.mark.parametrize('projection', [helper.mollweide_projection, helper.eckert_iv_projection])
    def test_scales_with_radius(self, projection):
        x1, y1 = projection(0.5, 0.3, radius=1.0)
        x2, y2 = projection(0.5, 0.3, radius=2.0)
        assert x2 == pytest.approx(2 * x1)
        assert y2 == pytest.approx(2 * y1)

    def test_central_meridian_shifts_longitude(self):
        x_a, _ = helper.mollweide_projection(1.0, 0.0, central_meridian=0.0)
        x_b, _ = helper.mollweide_projection(1.0, 0.0, central_meridian=1.0)
        assert x_b == pytest.approx(0.0)
        assert x_a > x_b


class TestPlatonicSolids:

    @pytest.mark.parametrize('solid_cls,base_vertices,base_faces', [
        (helper.IcosahedronSphere, 12, 20),
        (helper.Tetrahedron, 4, 4),
        (helper.Octahedron, 6, 8),
    ])
    def test_unsubdivided_counts(self, solid_cls, base_vertices, base_faces):
        solid = solid_cls(subdiv_lvl=0)
        assert solid.get_vertices().shape == (base_vertices, 3)
        assert len(solid.get_indices()) == base_faces * 3

    @pytest.mark.parametrize('level,expected_vertices,expected_faces', [
        (0, 12, 20), (1, 42, 80), (2, 162, 320), (3, 642, 1280),
    ])
    def test_icosphere_subdivision_counts(self, level, expected_vertices, expected_faces):
        solid = helper.IcosahedronSphere(subdiv_lvl=level)
        assert solid.get_vertices().shape[0] == expected_vertices
        assert len(solid.get_indices()) == expected_faces * 3

    def test_all_vertices_on_unit_sphere(self):
        vertices = helper.IcosahedronSphere(subdiv_lvl=3).get_vertices()
        norms = np.linalg.norm(vertices, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_vertices_are_unique(self):
        vertices = helper.IcosahedronSphere(subdiv_lvl=2).get_vertices()
        unique = np.unique(np.round(vertices.astype(np.float64), 6), axis=0)
        assert unique.shape[0] == vertices.shape[0]

    def test_face_indices_are_in_range(self):
        solid = helper.IcosahedronSphere(subdiv_lvl=2)
        indices = solid.get_indices()
        assert indices.min() >= 0
        assert indices.max() < solid.get_vertices().shape[0]

    def test_vertex_levels_track_subdivision(self):
        solid = helper.IcosahedronSphere(subdiv_lvl=2)
        levels = solid.get_vertex_levels()

        assert levels.shape[0] == solid.get_vertices().shape[0]
        assert (levels[:12] == 0).all()          # original corners
        assert set(np.unique(levels)) == {0, 1, 2}

    def test_arrays_are_contiguous(self):
        solid = helper.IcosahedronSphere(subdiv_lvl=1)
        assert solid.get_vertices().flags['C_CONTIGUOUS']
        assert solid.get_indices().flags['C_CONTIGUOUS']

    def test_distribution_is_roughly_even(self):
        """Sanity check for use as egomotion axis candidates: no hemisphere bias."""
        vertices = helper.IcosahedronSphere(subdiv_lvl=3).get_vertices()
        assert np.allclose(vertices.mean(axis=0), 0.0, atol=1e-6)

    @pytest.mark.xfail(reason='get_spherical_coordinates indexes rows instead of columns '
                              'and unpacks 3 return values into 2', strict=True)
    def test_spherical_coordinates(self):
        solid = helper.IcosahedronSphere(subdiv_lvl=1)
        azimuth, elevation = solid.get_spherical_coordinates()
        assert azimuth.shape[0] == solid.get_vertices().shape[0]
        assert elevation.shape[0] == solid.get_vertices().shape[0]


class TestDespine:

    def test_hides_requested_spines(self):
        matplotlib = pytest.importorskip('matplotlib')
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        _, axis = plt.subplots()
        helper.despine(axis, spines=['top', 'right'])

        assert not axis.spines['top'].get_visible()
        assert not axis.spines['right'].get_visible()
        assert axis.spines['left'].get_visible()
        plt.close('all')

    def test_hides_all_spines_and_ticks_by_default(self):
        matplotlib = pytest.importorskip('matplotlib')
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        _, axis = plt.subplots()
        helper.despine(axis)

        assert not any(s.get_visible() for s in axis.spines.values())
        assert len(axis.xaxis.get_ticks_position()) >= 0
        assert list(axis.get_xticks()) == []
        plt.close('all')
