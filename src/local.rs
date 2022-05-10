pub mod join_local;

use crate::mesh::Mesh;
use crate::triangle::{IndexedPolygon, Polygon};
use nalgebra as na;

pub struct LocalPolygon {
    old_poly: Polygon,
    poly: Polygon,
    e1: na::Vector3<f64>,
    _e2: na::Vector3<f64>,
    e3: na::Vector3<f64>,
    e_1: na::Vector3<f64>,
    e_2: na::Vector3<f64>,
    e_3: na::Vector3<f64>,
    x0: na::Vector3<f64>,
    matrix_a: na::Matrix3<f64>,
    matrix_a_1: na::Matrix3<f64>,
}

impl LocalPolygon {
    pub fn _get_matrix_a(&self) -> na::Matrix3<f64> {
        self.matrix_a.clone()
    }

    pub fn to_local(&self, pos: &na::Vector3<f64>) -> na::Vector3<f64> {
        self.matrix_a_1 * pos + self.x0
    }

    pub fn from_local(&self, pos: &na::Vector3<f64>) -> na::Vector3<f64> {
        self.matrix_a * pos + self.old_poly.triangle().x()
    }

    pub fn inside_polygon(&self, vec: &na::Vector3<f64>) -> bool {
        self.inside_triangle(
            self.poly.triangle().x().clone(),
            self.poly.triangle().y().clone(),
            self.poly.triangle().z().clone(),
            vec,
        )
    }

    pub fn inside_triangle(
        &self,
        x: na::Vector3<f64>,
        y: na::Vector3<f64>,
        z: na::Vector3<f64>,
        vec: &na::Vector3<f64>,
    ) -> bool {
        static EPS: f64 = 0.0000001;
        let local = self.to_local(vec);
        let y1 = Self::calculate_projection(&local, &z, &x);
        let z1 = Self::calculate_projection(&local, &z, &y);
        let mut rez = 0;
        if local[0] <= y1 && (y1 - local[0]).abs() > EPS {
            rez += 1;
        }
        if local[0] <= z1 || (z1 - local[0]).abs() < EPS {
            rez += 1;
        }

        rez % 2 != 0 && local[1].signum() == z[1].signum() && local[1].abs() <= z[1].abs()
            || local[1].abs() <= EPS && local[0] <= y[0] && local[0] > 0.
    }

    pub fn get_poly(&self) -> &Polygon {
        &self.poly
    }

    fn calculate_projection(
        local: &na::Vector3<f64>,
        x: &na::Vector3<f64>,
        y: &na::Vector3<f64>,
    ) -> f64 {
        let y21 = x[1] - y[1];
        let x21 = x[0] - y[0];
        (local[1] - y[1]) / y21 * x21 + y[0]
    }
}

impl From<&Polygon> for LocalPolygon {
    fn from(poly: &Polygon) -> Self {
        let triangle = poly.triangle();
        let e_1 = triangle.y() - triangle.x();
        let e_2 = triangle.z() - triangle.x();
        let e_3 = triangle.z() - triangle.y();
        let e1 = e_1.normalize();
        let e2 = poly.norm().cross(&e1);
        let e3 = poly.norm();
        let a = na::Matrix3::from_columns(&[e1, e2, e3]);
        let at = a.try_inverse().unwrap();
        let x0 = -at * triangle.x();

        LocalPolygon {
            old_poly: poly.clone(),
            poly: Polygon::new(
                &(at * triangle.x() + x0),
                &(at * triangle.y() + x0),
                &(at * triangle.z() + x0),
            ),
            e1: e1.clone(),
            _e2: e2.clone(),
            e3: e3.clone(),
            e_1: e_1.clone(),
            e_2: e_2.clone(),
            e_3: e_3.clone(),
            x0: x0.clone(),
            matrix_a: a.clone(),
            matrix_a_1: at.clone(),
        }
    }
}

pub struct Points {
    p1: na::Vector3<f64>,
    p2: na::Vector3<f64>,
    p3: na::Vector3<f64>,
    p4: na::Vector3<f64>,
    p5: na::Vector3<f64>,
    p6: na::Vector3<f64>,
    p7: na::Vector3<f64>,
    p8: na::Vector3<f64>,
    p9: na::Vector3<f64>,
    p10: na::Vector3<f64>,
}

impl From<&LocalPolygon> for Points {
    fn from(local: &LocalPolygon) -> Self {
        let triangle = local.poly.triangle();
        let x = triangle.x();
        let y = triangle.y();
        let z = triangle.z();
        Points {
            p1: x.clone(),
            p2: y.clone(),
            p3: z.clone(),
            p4: (2. * x + y) / 3.,
            p5: (x + 2. * y) / 3.,
            p6: (2. * y + z) / 3.,
            p7: (y + 2. * z) / 3.,
            p8: (x + 2. * z) / 3.,
            p9: (2. * x + z) / 3.,
            p10: (x + y + z) / 3.,
        }
    }
}

impl Points {
    fn get_l(&self, a: f64, b: f64, i: i32, j: i32) -> f64 {
        let pi = match i {
            1 => &self.p1,
            2 => &self.p2,
            3 => &self.p3,
            4 => &self.p4,
            5 => &self.p5,
            6 => &self.p6,
            7 => &self.p7,
            8 => &self.p8,
            9 => &self.p9,
            10 => &self.p10,
            _ => unreachable!(),
        };
        let pj = match j {
            1 => &self.p1,
            2 => &self.p2,
            3 => &self.p3,
            4 => &self.p4,
            5 => &self.p5,
            6 => &self.p6,
            7 => &self.p7,
            8 => &self.p8,
            9 => &self.p9,
            10 => &self.p10,
            _ => unreachable!(),
        };
        (a - pi[0]) * (pj[1] - pi[1]) - (b - pi[1]) * (pj[0] - pi[0])
    }

    pub fn get_fi(
        &self,
        a: f64,
        b: f64,
        first: (i32, i32),
        second: (i32, i32),
        third: (i32, i32),
        i: i32,
    ) -> f64 {
        let pi = match i {
            1 => &self.p1,
            2 => &self.p2,
            3 => &self.p3,
            4 => &self.p4,
            5 => &self.p5,
            6 => &self.p6,
            7 => &self.p7,
            8 => &self.p8,
            9 => &self.p9,
            10 => &self.p10,
            _ => unreachable!(),
        };
        let (pa, pb) = (pi[0], pi[1]);
        self.get_l(a, b, first.0, first.1)
            * self.get_l(a, b, second.0, second.1)
            * self.get_l(a, b, third.0, third.1)
            / (self.get_l(pa, pb, first.0, first.1)
                * self.get_l(pa, pb, second.0, second.1)
                * self.get_l(pa, pb, third.0, third.1))
    }
}

pub struct U {
    u1: f64,
    u2: f64,
    u3: f64,
    u4: f64,
    u5: f64,
    u6: f64,
    u7: f64,
    u8: f64,
    u9: f64,
    u10: f64,
}

impl U {
    pub fn _get(&self, i: i32) -> f64 {
        match i {
            1 => self.u1,
            2 => self.u2,
            3 => self.u3,
            4 => self.u4,
            5 => self.u5,
            6 => self.u6,
            7 => self.u7,
            8 => self.u8,
            9 => self.u9,
            10 => self.u10,
            _ => unreachable!(),
        }
    }

    pub fn from_local(local: &LocalPolygon, poly: &IndexedPolygon, mesh: &Mesh) -> Self {
        let vertexes = mesh.vertexes();
        let triangle = poly.triangle_indexes();
        let nx = vertexes[triangle.x_index()].norm();
        let ny = vertexes[triangle.y_index()].norm();
        let nz = vertexes[triangle.z_index()].norm();

        let q11 = -nx.dot(&local.e1) / (nx.dot(&local.e3));
        let q12 = -ny.dot(&local.e1) / (ny.dot(&local.e3));
        let q21 = -nx.dot(&local.e_2) / (local.e_2.norm() * nx.dot(&local.e3));
        let q23 = -nz.dot(&local.e_2) / (local.e_2.norm() * nz.dot(&local.e3));
        let q32 = -ny.dot(&local.e_3) / (local.e_3.norm() * ny.dot(&local.e3));
        let q33 = -nz.dot(&local.e_3) / (local.e_3.norm() * nz.dot(&local.e3));
        let u_4 = 2. * local.e_1.norm() / 27. * (2. * q11 - q12);
        let u_5 = 2. * local.e_1.norm() / 27. * (q11 - 2. * q12);
        let u_6 = 2. * local.e_3.norm() / 27. * (2. * q32 - q33);
        let u_7 = 2. * local.e_3.norm() / 27. * (q32 - 2. * q33);
        let u_8 = 2. * local.e_2.norm() / 27. * (q21 - 2. * q23);
        let u_9 = 2. * local.e_2.norm() / 27. * (2. * q21 - q23);
        U {
            u1: 0.,
            u2: 0.,
            u3: 0.,
            u4: u_4,
            u5: u_5,
            u6: u_6,
            u7: u_7,
            u8: u_8,
            u9: u_9,
            u10: 0.25 * (u_4 + u_5 + u_6 + u_7 + u_8 + u_9),
        }
    }
}

pub fn gamma(a: f64, b: f64, p: &Points, u: &U) -> f64 {
    let fi1 = p.get_fi(a, b, (2, 3), (5, 8), (4, 9), 1);
    let fi2 = p.get_fi(a, b, (1, 3), (4, 7), (5, 6), 2);
    let fi3 = p.get_fi(a, b, (1, 2), (6, 9), (7, 8), 3);
    let fi4 = p.get_fi(a, b, (1, 3), (2, 3), (5, 8), 4);
    let fi5 = p.get_fi(a, b, (1, 3), (2, 3), (4, 7), 5);
    let fi6 = p.get_fi(a, b, (1, 2), (1, 3), (4, 7), 6);
    let fi7 = p.get_fi(a, b, (1, 2), (1, 3), (6, 9), 7);
    let fi8 = p.get_fi(a, b, (1, 2), (2, 3), (6, 9), 8);
    let fi9 = p.get_fi(a, b, (1, 2), (2, 3), (5, 8), 9);
    let fi10 = p.get_fi(a, b, (1, 2), (2, 3), (1, 3), 10);
    u.u1 * fi1
        + u.u2 * fi2
        + u.u3 * fi3
        + u.u4 * fi4
        + u.u5 * fi5
        + u.u6 * fi6
        + u.u7 * fi7
        + u.u8 * fi8
        + u.u9 * fi9
        + u.u10 * fi10
}
