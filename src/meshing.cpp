#include "seagullmesh.hpp"

#include <cmath>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <pybind11/functional.h>

#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_tree.h>

#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/fair.h>
#include <CGAL/Polygon_mesh_processing/angle_and_area_smoothing.h>
#include <CGAL/Polygon_mesh_processing/smooth_shape.h>
#include <CGAL/Polygon_mesh_processing/refine.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>
#include <CGAL/Polygon_mesh_processing/tangential_relaxation.h>
#include <CGAL/Polygon_mesh_processing/remesh_planar_patches.h>
#include <CGAL/Polygon_mesh_processing/interpolated_corrected_curvatures.h>
#include <CGAL/Polygon_mesh_processing/Adaptive_sizing_field.h>
#include <CGAL/Polygon_mesh_processing/Uniform_sizing_field.h>
#include <CGAL/Polygon_mesh_processing/refine_mesh_at_isolevel.h>
#include <CGAL/Polygon_mesh_processing/repair_self_intersections.h>
#include <CGAL/Polygon_mesh_processing/surface_Delaunay_remeshing.h>

typedef Mesh3::Property_map<V, Point3>                      VertPoint;
typedef Mesh3::Property_map<V, double>                      VertDouble;
typedef Mesh3::Property_map<V, bool>                        VertBool;
typedef Mesh3::Property_map<E, bool>                        EdgeBool;

typedef Mesh3::Property_map<F, uint32_t>                    FaceIndex;

namespace PMP = CGAL::Polygon_mesh_processing;

typedef PMP::Principal_curvatures_and_directions<Kernel>    PrincipalCurvDir;
typedef Mesh3::Property_map<V, PrincipalCurvDir>            VertPrincipalCurvDir;

template<typename SizingField>
struct SizingFieldWrapper {
    typedef typename SizingField::FT FT;
    SizingField wrapped;
    VertBool& flagged;

    FT at(const V v, const Mesh3& mesh) const {
        return wrapped.at(v, mesh);
    }
    std::optional<FT> is_too_long(const V va, const V vb, const Mesh3& mesh) const {
        return wrapped.is_too_long(va, vb, mesh);
    }
    std::optional<FT> is_too_short(const H h, const Mesh3& mesh) const {
        return wrapped.is_too_short(h, mesh);
    }
    Point3 split_placement(const H h, const Mesh3& mesh) const {
        return wrapped.split_placement(h, mesh);
    }
    void register_split_vertex(const V v, const Mesh3& mesh) {
        wrapped.register_split_vertex(v, mesh);

        bool v_flagged = false;
        for (H h : halfedges_around_source(v, mesh)) {
            if ( flagged[mesh.target(h)] ) {
                v_flagged = true;
                break;
            }
        }
        flagged[v] = v_flagged;
    }
};



struct PerVertexSizingOnMap {
    using FT = double;
    VertDouble& sizes;

    FT at(const V v, const Mesh3&) const { return sizes[v]; }

    std::optional<FT> is_too_long(const V va, const V vb, const Mesh3& mesh) const {
        const double target = 0.5 * (sizes[va] + sizes[vb]);
        const double sq = CGAL::squared_distance(mesh.point(va), mesh.point(vb));
        const double target_sq = target * target;
        if (sq > target_sq) return sq / target_sq;
        return std::nullopt;
    }

    std::optional<FT> is_too_short(const H h, const Mesh3& mesh) const {
        const V s = source(h, mesh);
        const V t = target(h, mesh);
        const double target = 0.5 * (sizes[s] + sizes[t]);
        const double sq = CGAL::squared_distance(mesh.point(s), mesh.point(t));
        const double target_sq = target * target;
        if (sq < target_sq) return sq / target_sq;
        return std::nullopt;
    }

    Point3 split_placement(const H h, const Mesh3& mesh) const {
        return CGAL::midpoint(mesh.point(source(h, mesh)), mesh.point(target(h, mesh)));
    }

    void register_split_vertex(const V v, const Mesh3& mesh) {
        // simple average of 1-ring neighbors
        double sum = 0.0; int n = 0;
        for (H hh : halfedges_around_source(v, mesh)) { sum += sizes[target(hh, mesh)]; ++n; }
        sizes[v] = (n ? sum / n : sizes[v]);
    }
};

struct WeightedMoveVertex {
    // a vertex-point-map-like
    using key_type = V;
    using value_type = Point3;
    using reference = Point3&;
    using category = boost::read_write_property_map_tag;

    VertPoint& points;
    VertDouble& weights;

    WeightedMoveVertex(VertPoint& p, VertDouble& w) : points(p), weights(w) {}

    friend Point3& get (const WeightedMoveVertex& self, V v) { return self.points[v]; }
    friend void put (const WeightedMoveVertex& self, V v, const Point3& p1) {
        const double w1 = self.weights[v];
        if (w1 <= 0.0) {
            return;
        } else if (w1 >= 1.0) {
            self.points[v] = p1;
        } else {
            const Point3& p0 = self.points[v];
            const double w0 = 1.0 - w1;
            self.points[v] = Point3(
                w0 * p0.x() + w1 * p1.x(),
                w0 * p0.y() + w1 * p1.y(),
                w0 * p0.z() + w1 * p1.z()
            );
        }
    }
};

struct SurfaceAabbSizingField {
    using FT = double;

    using Primitive = CGAL::AABB_face_graph_triangle_primitive<Mesh3>;
    using Traits = CGAL::AABB_traits<Kernel, Primitive>;
    using Tree = CGAL::AABB_tree<Traits>;

    const Mesh3* mesh;
    VertDouble sizes;
    std::shared_ptr<Tree> tree;

    SurfaceAabbSizingField(const Mesh3& mesh_, VertDouble sizes_)
        : mesh(&mesh_), sizes(sizes_), tree(std::make_shared<Tree>())
    {
        auto face_range = faces(mesh_);
        tree->insert(face_range.first, face_range.second, mesh_);
        tree->build();
        tree->accelerate_distance_queries();
    }

    template <typename BarePoint, typename Index>
    FT operator()(const BarePoint& p, int /*dim*/, const Index& /*index*/) const {
        return eval(Point3(p.x(), p.y(), p.z()));
    }

    FT eval(const Point3& p) const {
        const auto [closest, f] = tree->closest_point_and_primitive(p);

        const H h0 = halfedge(f, *mesh);
        const V v0 = source(h0, *mesh);
        const V v1 = target(h0, *mesh);
        const V v2 = target(next(h0, *mesh), *mesh);

        const Point3& a = mesh->point(v0);
        const Point3& b = mesh->point(v1);
        const Point3& c = mesh->point(v2);

        const auto v0v = b - a;
        const auto v1v = c - a;
        const auto v2v = closest - a;

        const double d00 = v0v * v0v;
        const double d01 = v0v * v1v;
        const double d11 = v1v * v1v;
        const double d20 = v2v * v0v;
        const double d21 = v2v * v1v;

        const double denom = d00 * d11 - d01 * d01;
        if (std::abs(denom) < 1e-30) {
            const double s0 = sizes[v0];
            const double s1 = sizes[v1];
            const double s2 = sizes[v2];
            return (std::max)(1e-30, (std::min)({s0, s1, s2}));
        }

        const double v = (d11 * d20 - d01 * d21) / denom;
        const double w = (d00 * d21 - d01 * d20) / denom;
        const double u = 1.0 - v - w;

        const double s0 = sizes[v0];
        const double s1 = sizes[v1];
        const double s2 = sizes[v2];

        const double size = u * s0 + v * s1 + w * s2;
        return (std::max)(1e-30, size);
    }
};


void init_meshing(py::module &m) {
    m.def_submodule("meshing")
        .def("uniform_isotropic_remeshing", [](
                Mesh3& mesh,
                const Indices<F>& faces,
                const double target_edge_length,
                unsigned int n_iter,
                bool collapse_constraints,
                bool protect_constraints,
                bool do_project,
                VertBool& vertex_is_constrained_map,
                EdgeBool& edge_is_constrained_map,
                FaceIndex& face_patch_map,
                VertBool& flagged
            ) {
            using SizingField = PMP::Uniform_sizing_field<Mesh3, VertPoint>;
            SizingField sizing_field(target_edge_length, mesh);
            SizingFieldWrapper<SizingField> wrapped_sizing_field{sizing_field, flagged};

            auto params = PMP::parameters::
                number_of_iterations(n_iter)
                .collapse_constraints(protect_constraints)
                .protect_constraints(protect_constraints)
                .do_project(do_project)
                .vertex_is_constrained_map(vertex_is_constrained_map)
                .edge_is_constrained_map(edge_is_constrained_map)
                .face_patch_map(face_patch_map)
            ;
            PMP::isotropic_remeshing(faces.to_vector(), wrapped_sizing_field, mesh, params);
        })
        .def("adaptive_isotropic_remeshing", [](
                Mesh3& mesh,
                const Indices<F>& faces,
                const double tolerance,
                const double ball_radius,
                const std::pair<double, double>& edge_len_min_max,
                unsigned int n_iter,
                bool collapse_constraints,
                bool protect_constraints,
                bool do_project,
                VertBool& vertex_is_constrained_map,
                EdgeBool& edge_is_constrained_map,
                FaceIndex& face_patch_map,
                VertBool& flagged
            ) {

            using SizingField = PMP::Adaptive_sizing_field<Mesh3, VertPoint>;
            // Not sure about this but I think we need to supply the entire face range to the sizing field
            // even if we're only remeshing a subset of faces
            SizingField sizing_field(tolerance, edge_len_min_max, mesh.faces(), mesh);
            SizingFieldWrapper<SizingField> wrapped_sizing_field{sizing_field, flagged};

            auto params = PMP::parameters::
                number_of_iterations(n_iter)
                .collapse_constraints(collapse_constraints)
                .protect_constraints(protect_constraints)
                .do_project(do_project)
                .vertex_is_constrained_map(vertex_is_constrained_map)
                .edge_is_constrained_map(edge_is_constrained_map)
                .face_patch_map(face_patch_map)
            ;
            PMP::isotropic_remeshing(faces.to_vector(), wrapped_sizing_field, mesh, params);
        })
        .def("adaptive_isotropic_remeshing_full", [](
                Mesh3& mesh,
                const Indices<F>& faces,
                VertDouble& vertex_sizing_map,
                unsigned int n_iter,
                bool collapse_constraints,
                bool protect_constraints,
                bool do_project,
                VertBool& vertex_is_constrained_map,
                EdgeBool& edge_is_constrained_map,
                FaceIndex& face_patch_map,
                VertBool& flagged
            ) {

            PerVertexSizingOnMap sizing_field{vertex_sizing_map};
            SizingFieldWrapper<PerVertexSizingOnMap> wrapped_sizing_field{sizing_field, flagged};

            auto params = PMP::parameters::
                number_of_iterations(n_iter)
                .collapse_constraints(collapse_constraints)
                .protect_constraints(protect_constraints)
                .do_project(do_project)
                .vertex_is_constrained_map(vertex_is_constrained_map)
                .edge_is_constrained_map(edge_is_constrained_map)
                .face_patch_map(face_patch_map)
            ;
            PMP::isotropic_remeshing(faces.to_vector(), wrapped_sizing_field, mesh, params);
        })
        .def("remesh_delaunay", [](Mesh3& mesh, EdgeBool& edge_is_constrained_map){
            auto params = PMP::parameters::edge_is_constrained_map(edge_is_constrained_map);
            return PMP::surface_Delaunay_remeshing(mesh, params);

        })
        .def("remesh_delaunay_full", [](
            Mesh3& mesh,
            VertDouble& vertex_sizing_map,
            double facet_angle,
            double facet_distance,
            double features_angle_bound,
            bool protect_constraints,
            EdgeBool& edge_is_constrained_map
        ) {
            SurfaceAabbSizingField sizing_field(mesh, vertex_sizing_map);
            
            auto params = PMP::parameters::
                edge_is_constrained_map(edge_is_constrained_map)
                .features_angle_bound(features_angle_bound)
                .protect_constraints(protect_constraints)
                .mesh_edge_size(sizing_field)
                .mesh_facet_size(sizing_field)
                .mesh_facet_angle(facet_angle)
                .mesh_facet_distance(facet_distance)
            ;
                
            return PMP::surface_Delaunay_remeshing(mesh, params);
        })
        .def("fair", [](Mesh3& mesh, const Indices<V>& verts, const unsigned int fairing_continuity) {
            // A value controling the tangential continuity of the output surface patch.
            // The possible values are 0, 1 and 2, refering to the C0, C1 and C2 continuity.
            auto params = PMP::parameters::fairing_continuity(fairing_continuity);
            bool success = PMP::fair(mesh, verts.to_vector(), params);
            if (!success) {
                throw std::runtime_error("Fairing failed");
            }
        })
        .def("refine", [](Mesh3& mesh, const Indices<F>& faces, double density) {
            std::vector<V> new_verts;
            std::vector<F> new_faces;
            auto params = PMP::parameters::density_control_factor(density);
            PMP::refine(mesh, faces.to_vector(), std::back_inserter(new_faces), std::back_inserter(new_verts), params);
            return std::make_pair(Indices<V>(new_verts), Indices<F>(new_faces));
        })
        .def("smooth_angle_and_area", [](
            Mesh3& mesh,
            const Indices<F>& faces,
            unsigned int n_iter,
            bool use_area_smoothing,
            bool use_angle_smoothing,
            bool use_safety_constraints,
            bool do_project,
            VertBool& vertex_is_constrained_map,
            EdgeBool& edge_is_constrained_map
        ) {
            auto params = PMP::parameters::
                number_of_iterations(n_iter)
                .use_area_smoothing(use_area_smoothing)
                .use_angle_smoothing(use_angle_smoothing)
                .use_safety_constraints(use_safety_constraints)
                .do_project(do_project)
                .vertex_is_constrained_map(vertex_is_constrained_map)
                .edge_is_constrained_map(edge_is_constrained_map)
            ;
            PMP::angle_and_area_smoothing(faces.to_vector(), mesh, params);
        })
        .def("tangential_relaxation", [](
            Mesh3& mesh,
            const Indices<V>& verts,
            unsigned int n_iter,
            bool relax_constraints,
            VertBool& vertex_is_constrained_map,
            EdgeBool& edge_is_constrained_map
        ) {
            auto params = PMP::parameters::
                number_of_iterations(n_iter)
                .relax_constraints(relax_constraints)
                .vertex_is_constrained_map(vertex_is_constrained_map)
                .edge_is_constrained_map(edge_is_constrained_map)
            ;
            PMP::tangential_relaxation(verts.to_vector(), mesh, params);
        })
        .def("smooth_shape", [](
            Mesh3& mesh,
            const Indices<F>& faces,
            const double time,
            unsigned int n_iter,
            VertBool& vertex_is_constrained_map
        ) {
            auto params = PMP::parameters::
                number_of_iterations(n_iter)
                .vertex_is_constrained_map(vertex_is_constrained_map)
            ;
            PMP::smooth_shape(faces.to_vector(), mesh, time, params);
        })
        .def("smooth_shape_weighted", [](
            Mesh3& mesh,
            const Indices<F>& faces,
            const double time,
            unsigned int n_iter,
            VertBool& vertex_is_constrained_map,
            VertDouble& vertex_weights
        ) {
            WeightedMoveVertex vpm(mesh.points(), vertex_weights);

            auto params = PMP::parameters::
                number_of_iterations(n_iter)
                .vertex_is_constrained_map(vertex_is_constrained_map)
                .vertex_point_map(vpm)
            ;
            PMP::smooth_shape(faces.to_vector(), mesh, time, params);
        })
        .def("does_self_intersect", [](const Mesh3& mesh) {
            return PMP::does_self_intersect(mesh);
        })
        .def("do_intersect", [](const Mesh3& mesh, const Mesh3& other) {
            return PMP::do_intersect(mesh, other);
        })
        .def("self_intersections", [](const Mesh3& mesh) {
            std::vector<std::pair<F, F>> pairs;
            PMP::self_intersections(mesh, std::back_inserter(pairs));

            std::vector<F> first, second;
            boost::copy(pairs | boost::adaptors::transformed([](const auto& pair) { return pair.first; }), std::back_inserter(first));
            boost::copy(pairs | boost::adaptors::transformed([](const auto& pair) { return pair.second; }), std::back_inserter(second));

            return std::make_tuple(Indices<F>(first), Indices<F>(second));
        })
        .def("remove_self_intersections", [](Mesh3& mesh) {
            // returns a bool, presumably success
            return PMP::experimental::remove_self_intersections(mesh);
        })
        .def("remesh_planar_patches", [](
                const Mesh3& mesh,
                EdgeBool& edge_is_constrained_map,
                FaceIndex& face_patch_map,
                float cosine_of_maximum_angle
            ) {
            auto params = PMP::parameters::
                edge_is_constrained_map(edge_is_constrained_map)
                .face_patch_map(face_patch_map)
                .cosine_of_maximum_angle(cosine_of_maximum_angle)
            ;

            Mesh3 out;
            PMP::remesh_planar_patches(mesh, out, params);
            return out;
        })
        .def("interpolated_corrected_curvatures", [](
            const Mesh3& mesh,
            VertDouble& mean_curv_map,
            VertDouble& gauss_curv_map,
            VertPrincipalCurvDir& princ_curv_dir_map,
            const double ball_radius
        ) {
            auto params = PMP::parameters::
                vertex_mean_curvature_map(mean_curv_map)
                .vertex_Gaussian_curvature_map(gauss_curv_map)
                .vertex_principal_curvatures_and_directions_map(princ_curv_dir_map)
                .ball_radius(ball_radius)
            ;
            PMP::interpolated_corrected_curvatures(mesh, params);
        })
        .def("refine_mesh_at_isolevel", [](
            Mesh3& mesh,
            VertDouble& value_map,
            double isovalue,
            EdgeBool& edge_is_constrained_map
        ) {
            auto params = PMP::parameters::edge_is_constrained_map(edge_is_constrained_map);
            PMP::refine_mesh_at_isolevel(mesh, value_map, isovalue, params);
        })
        .def("split_long_edges", [](
            Mesh3& mesh,
            const Indices<E>& edges,
            double target_edge_length,
            EdgeBool& edge_is_constrained_map
        ) {
            auto params = PMP::parameters::edge_is_constrained_map(edge_is_constrained_map);
            PMP::split_long_edges(edges.to_vector(), target_edge_length, mesh, params);
        })
    ;
}
