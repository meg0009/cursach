use crate::join::{self, get_mn, InOrOut};
use crate::local::{gamma, gamma_d, join_local, LocalPolygon, Points, U};
use crate::mesh::Mesh;
use crate::triangle::{IndexedPolygon, IndexedTriangle, Polygon};
use crate::vertex::Vertex;
use crate::ZEROS;
use nalgebra as na;

pub struct Cube {
    _x0: na::Vector3<f64>,
    _size: f64,
    _size_edge: usize,
    mesh: Mesh,
}

impl Cube {
    pub fn _with_edge(size_edge: usize) -> Self {
        Self::new(ZEROS.clone(), 1., size_edge)
    }

    pub fn new(pos: na::Vector3<f64>, size: f64, size_edge: usize) -> Self {
        let dl = size / 2.;
        let a = pos + na::Vector3::new(dl, -dl, -dl);
        let b = pos + na::Vector3::new(-dl, -dl, -dl);
        let c = pos + na::Vector3::new(-dl, dl, -dl);
        let _d = pos + na::Vector3::new(dl, dl, -dl);
        let _a2 = pos + na::Vector3::new(dl, -dl, dl);
        let b2 = pos + na::Vector3::new(-dl, -dl, dl);
        let _c2 = pos + na::Vector3::new(-dl, dl, dl);
        let _d2 = pos + na::Vector3::new(dl, dl, dl);

        let capacity = size_edge * size_edge * 6 * 2;
        let mut vertexes = Vec::with_capacity(capacity);
        let mut triangles = Vec::with_capacity(capacity);
        let dl = size / size_edge as f64;
        for i in 0..size_edge {
            for j in 0..size_edge {
                // abcd
                let x0 = b + na::Vector3::new(dl * i as f64, dl * j as f64, 0.);
                vertexes.push(x0);
                vertexes.push(x0 + na::Vector3::new(dl, 0., 0.));
                vertexes.push(x0 + na::Vector3::new(dl, dl, 0.));
                triangles.push(IndexedTriangle::new(
                    vertexes.len() - 3,
                    vertexes.len() - 2,
                    vertexes.len() - 1,
                ));
                vertexes.push(x0 + na::Vector3::new(0., dl, 0.));
                triangles.push(IndexedTriangle::new(
                    vertexes.len() - 4,
                    vertexes.len() - 1,
                    vertexes.len() - 2,
                ));

                // a2b2c2d2
                let x0 = b2 + na::Vector3::new(dl * i as f64, dl * j as f64, 0.);
                vertexes.push(x0);
                vertexes.push(x0 + na::Vector3::new(dl, 0., 0.));
                vertexes.push(x0 + na::Vector3::new(dl, dl, 0.));
                triangles.push(IndexedTriangle::new(
                    vertexes.len() - 3,
                    vertexes.len() - 2,
                    vertexes.len() - 1,
                ));
                vertexes.push(x0 + na::Vector3::new(0., dl, 0.));
                triangles.push(IndexedTriangle::new(
                    vertexes.len() - 4,
                    vertexes.len() - 1,
                    vertexes.len() - 2,
                ));

                // aa2d2d
                let x0 = a + na::Vector3::new(0., dl * i as f64, dl * j as f64);
                vertexes.push(x0);
                vertexes.push(x0 + na::Vector3::new(0., dl, 0.));
                vertexes.push(x0 + na::Vector3::new(0., dl, dl));
                triangles.push(IndexedTriangle::new(
                    vertexes.len() - 3,
                    vertexes.len() - 2,
                    vertexes.len() - 1,
                ));
                vertexes.push(x0 + na::Vector3::new(0., 0., dl));
                triangles.push(IndexedTriangle::new(
                    vertexes.len() - 4,
                    vertexes.len() - 1,
                    vertexes.len() - 2,
                ));

                // bb2c2c
                let x0 = b + na::Vector3::new(0., dl * i as f64, dl * j as f64);
                vertexes.push(x0);
                vertexes.push(x0 + na::Vector3::new(0., dl, 0.));
                vertexes.push(x0 + na::Vector3::new(0., dl, dl));
                triangles.push(IndexedTriangle::new(
                    vertexes.len() - 3,
                    vertexes.len() - 2,
                    vertexes.len() - 1,
                ));
                vertexes.push(x0 + na::Vector3::new(0., 0., dl));
                triangles.push(IndexedTriangle::new(
                    vertexes.len() - 4,
                    vertexes.len() - 1,
                    vertexes.len() - 2,
                ));

                // aa2b2b
                let x0 = b + na::Vector3::new(dl * i as f64, 0., dl * j as f64);
                vertexes.push(x0);
                vertexes.push(x0 + na::Vector3::new(dl, 0., 0.));
                vertexes.push(x0 + na::Vector3::new(dl, 0., dl));
                triangles.push(IndexedTriangle::new(
                    vertexes.len() - 3,
                    vertexes.len() - 2,
                    vertexes.len() - 1,
                ));
                vertexes.push(x0 + na::Vector3::new(0., 0., dl));
                triangles.push(IndexedTriangle::new(
                    vertexes.len() - 4,
                    vertexes.len() - 1,
                    vertexes.len() - 2,
                ));

                // dd2c2c
                let x0 = c + na::Vector3::new(dl * i as f64, 0., dl * j as f64);
                vertexes.push(x0);
                vertexes.push(x0 + na::Vector3::new(dl, 0., 0.));
                vertexes.push(x0 + na::Vector3::new(dl, 0., dl));
                triangles.push(IndexedTriangle::new(
                    vertexes.len() - 3,
                    vertexes.len() - 2,
                    vertexes.len() - 1,
                ));
                vertexes.push(x0 + na::Vector3::new(0., 0., dl));
                triangles.push(IndexedTriangle::new(
                    vertexes.len() - 4,
                    vertexes.len() - 1,
                    vertexes.len() - 2,
                ));
            }
        }
        Cube {
            _x0: pos.clone(),
            _size: size,
            _size_edge: size_edge,
            mesh: Mesh::new(
                &vertexes.iter().map(|v| Vertex::new(v)).collect(),
                &triangles,
            ),
        }
    }

    pub fn _mesh(&self) -> &Mesh {
        &self.mesh
    }

    pub fn run(&self, mesh: &Mesh) -> Mesh {
        let mut vertexes = Vec::with_capacity(self.mesh.vertexes().len());
        for v in self.mesh.vertexes().iter() {
            let (x, old_poly) = Self::found_triangle_projection(&v.pos(), mesh).unwrap();
            let local_poly =
                LocalPolygon::from(&Polygon::from_indexed_polygon(&old_poly, mesh.vertexes()));
            match InOrOut::from((&x, &local_poly)) {
                InOrOut::Out(p1, p2, a) => {
                    let (i1, i2) = {
                        if p1 == join::Points::X {
                            if p2 == join::Points::Y {
                                (
                                    old_poly.triangle_indexes().x_index(),
                                    old_poly.triangle_indexes().y_index(),
                                )
                            } else {
                                (
                                    old_poly.triangle_indexes().x_index(),
                                    old_poly.triangle_indexes().z_index(),
                                )
                            }
                        } else {
                            (
                                old_poly.triangle_indexes().y_index(),
                                old_poly.triangle_indexes().z_index(),
                            )
                        }
                    };
                    let another_poly = mesh.find_neighbor(&old_poly, i1, i2).unwrap();
                    let b = crate::join::get_inside_point(
                        &old_poly,
                        mesh,
                        i1,
                        i2,
                        &local_poly.to_local(&x),
                    ); // b - в глобальной системе координат
                       // a - в локальной системе координат local_poly
                       // x - в глобальной системе координат
                    let (m, n) = get_mn(&x, &b, &old_poly, &another_poly, mesh, i1, i2);
                    let points = Points::from(&local_poly);
                    let u = U::from_local(&local_poly, &old_poly, mesh);
                    let local_m = local_poly.to_local(&m);
                    let new_local_m = na::Vector3::new(
                        local_m[0],
                        local_m[1],
                        gamma(local_m[0], local_m[1], &points, &u),
                    );
                    let local_m_d = gamma_d(
                        local_m[0],
                        local_m[1],
                        &points,
                        &u,
                        &local_poly.to_local(&(m - b)),
                    );
                    let new_m = local_poly.from_local(&new_local_m);

                    let o_z = na::Vector3::new(0., 0., 1.);
                    let local_z = local_poly.get_e3();
                    let local_angle = o_z.angle(&local_z);
                    
                    let another_local_poly = LocalPolygon::from(&Polygon::from_indexed_polygon(
                        &another_poly,
                        mesh.vertexes(),
                    ));
                    let points = Points::from(&another_local_poly);
                    let u = U::from_local(&another_local_poly, &another_poly, mesh);
                    let local_n = another_local_poly.to_local(&n);
                    let new_local_n = na::Vector3::new(
                        local_n[0],
                        local_n[1],
                        gamma(local_n[0], local_n[1], &points, &u),
                    );
                    let local_n_d = gamma_d(
                        local_n[0],
                        local_n[1],
                        &points,
                        &u,
                        &another_local_poly.to_local(&(m - b)),
                    );

                    /*let another_local_z = another_local_poly.get_e3();
                    let another_local_angle = o_z.angle(&another_local_z);*/

                    let new_n = another_local_poly.from_local(&new_local_n);
                    let abs_local = join_local::JoinLocalPolygon::from((
                        mesh.vertexes()[i1].pos(),
                        mesh.vertexes()[i2].pos(),
                        &old_poly.norm(),
                        &another_poly.norm(),
                    ));
                    let local_m = abs_local.to_local(&new_m);
                    let local_n = abs_local.to_local(&new_n);
                    /*let new_local_z = abs_local.get_e3();

                    let new_local_angle = local_angle - o_z.angle(&new_local_z);
                    let new_m_d = (local_m_d - new_local_angle.tan())
                        / (1. + local_m_d * new_local_angle.tan());
                    let new_local_angle = another_local_angle - o_z.angle(&new_local_z);
                    let new_n_d = (local_n_d - new_local_angle.tan())
                        / (1. + local_n_d * new_local_angle.tan());

                    let local_zx = na::Vector3::new(local_z[0], local_z[1], 0.);
                    let another_local_zx = na::Vector3::new(another_local_z[0], another_local_z[1], 0.);
                    let new_local_zx = na::Vector3::new(new_local_z[0], new_local_z[1], 0.);

                    let new_local_angle = local_zx.angle(&new_local_zx);
                    let new_m_d = (new_m_d - new_local_angle.tan())
                        / (1. + new_m_d * new_local_angle.tan());
                    let new_local_angle = another_local_zx.angle(&new_local_zx);
                    let new_n_d = (new_n_d - new_local_angle.tan())
                        / (1. + new_n_d * new_local_angle.tan());*/

                    let bm_abs_local = abs_local.to_local(&(b - m));
                    let bn_abs_local = abs_local.to_local(&(b - n));

                    //let new_m_d = local_m_d.0 * bm_abs_local[0] + local_m_d.1 * bm_abs_local[1];
                    //let new_n_d = local_n_d.0 * bn_abs_local[0] + local_n_d.1 * bn_abs_local[1];
                    //let new_m_d = local_m_d;
                    //let new_n_d = local_n_d;
                    let n_old = old_poly.norm();
                    let n_another = another_poly.norm();
                    let nn = abs_local.get_e3();
                    let angle_old = nn.angle(&n_old);
                    let angle_another = nn.angle(&n_another);
                    let new_m_d = (local_m_d - angle_old.tan()) / (1. + local_m_d * angle_old.tan());
                    let new_n_d = (local_n_d - angle_another.tan()) / (1. + local_n_d * angle_another.tan());

                    let local_x = local_poly.to_local(&x);
                    let points = Points::from(&local_poly);
                    let u = U::from_local(&local_poly, &old_poly, mesh);
                    let new_local_x = na::Vector3::new(
                        local_x[0],
                        local_x[1],
                        gamma(local_x[0], local_x[1], &points, &u),
                    );

                    let local_x = abs_local.to_local(&local_poly.from_local(&new_local_x));
                    //let local_x = abs_local.to_local(&x);

                    // исправления - gamma ниже считает всё в NaN а также возможно выше не правильно пересчитываются в новую локальную систему координат производные m_d и n_d
                    let new_local_x = na::Vector3::new(
                        local_x[0],
                        local_x[1],
                        join_local::gamma(
                            //local_x[0], local_x[1], &local_m, &local_n, local_m[2], local_n[2],
                            &local_x,
                            &abs_local.to_local(&m),
                            &abs_local.to_local(&n),
                            local_m[2],
                            local_n[2],
                            /*
                            -local_x[2] + join_local::gamma(
                            local_x[0], local_x[1], &abs_local.to_local(&m), &abs_local.to_local(&n), local_m[2], local_n[2],
                            new_m_d, new_n_d,
                            ),
                             */
                            new_m_d,
                            new_n_d,
                        ),
                    );
                    let new_x = abs_local.from_local(&new_local_x);
                    vertexes.push(Vertex::new(&new_x));
                    //vertexes.push(Vertex::new(&b));
                    //vertexes.push(Vertex::new(&m));
                    //let new_m_d =
                }
                InOrOut::In => {
                    let points = Points::from(&local_poly);
                    let u = U::from_local(&local_poly, &old_poly, mesh);
                    let local_x = local_poly.to_local(&x);
                    let new_local_x = na::Vector3::new(
                        local_x[0],
                        local_x[1],
                        gamma(local_x[0], local_x[1], &points, &u),
                    );
                    let new_x = local_poly.from_local(&new_local_x);
                    vertexes.push(Vertex::new(&new_x));
                }
            }
        }
        Mesh::new(
            &vertexes,
            &self
                .mesh
                .faces()
                .iter()
                .map(|p| p.triangle_indexes().clone())
                .collect(),
        )
    }

    fn found_triangle_projection(
        x: &na::Vector3<f64>,
        mesh: &Mesh,
    ) -> Option<(na::Vector3<f64>, IndexedPolygon)> {
        for (i, p) in mesh
            .faces()
            .iter()
            .map(|p| Polygon::from_indexed_polygon(p, mesh.vertexes()))
            .enumerate()
        {
            let normal = p.norm();
            let d = -normal.dot(&p.triangle().get_center());
            let lambda_x = -d / (normal[0] * x[0] + normal[1] * x[1] + normal[2] * x[2]);
            let x0 = na::Vector3::new(x[0] * lambda_x, x[1] * lambda_x, x[2] * lambda_x);
            let local = LocalPolygon::from(&p);
            if (x.clone() + normal.clone()).magnitude() > (x.clone() - normal.clone()).magnitude()
                && local.inside_polygon(&x0)
            {
                return Some((x0.clone(), mesh.faces()[i].clone()));
            }
        }
        None
    }
}
