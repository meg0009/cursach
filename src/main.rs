use nalgebra as na;

use std::convert::From;
use std::fs::File;

const ZEROS: na::Vector3<f64> = na::vector![0., 0., 0.];

#[derive(Clone, Copy, Debug)]
struct Triangle {
    x: na::Vector3<f64>,
    y: na::Vector3<f64>,
    z: na::Vector3<f64>,
}

impl Triangle {
    fn new(x: na::Vector3<f64>, y: na::Vector3<f64>, z: na::Vector3<f64>) -> Self {
        Triangle { x: x, y: y, z: z }
    }

    fn get_center(&self) -> na::Vector3<f64> {
        (self.x + self.y + self.z) / 3.
    }
}

#[derive(Clone)]
struct Polygon {
    normal: na::Vector3<f64>,
    triangle: Triangle,
    nx: na::Vector3<f64>,
    ny: na::Vector3<f64>,
    nz: na::Vector3<f64>,
}

impl Polygon {
    fn new(_normal: &na::Vector3<f64>, triangle: Triangle) -> Self {
        /*let mut nx = (triangle.y[0] - triangle.y[1]) * (triangle.z[0] - triangle.z[1]);
        nx += (triangle.y[1] - triangle.y[2]) * (triangle.z[1] - triangle.z[2]);
        nx += (triangle.y[2] - triangle.y[0]) * (triangle.z[2] - triangle.z[0]);
        println!("nx = {:?}", nx);
        let mut ny = (triangle.z[0] - triangle.z[1]) * (triangle.x[0] - triangle.x[1]);
        ny += (triangle.z[1] - triangle.z[2]) * (triangle.x[1] - triangle.x[2]);
        ny += (triangle.z[2] - triangle.z[0]) * (triangle.x[2] - triangle.x[0]);
        println!("ny = {:?}", ny);
        let mut nz = (triangle.x[0] - triangle.x[1]) * (triangle.y[0] - triangle.y[1]);
        nz += (triangle.x[1] - triangle.x[2]) * (triangle.y[1] - triangle.y[2]);
        nz += (triangle.x[2] - triangle.x[0]) * (triangle.y[2] - triangle.y[0]);
        println!("nz = {:?}", nz);
        let _t = triangle.get_center();
        let a = triangle.x - triangle.y;
        let b = triangle.x - triangle.z;
        let _tt = a.cross(&b);
        let d = (-(nx * triangle.x[0] + ny * triangle.x[1] + nz * triangle.x[2])).signum();
        let l = (nx.powi(2) + ny.powi(2) + nz.powi(2)).sqrt();
        println!("l = {:?}", l);
        nx /= l;
        ny /= l;
        nz /= l;
        nx *= d;
        ny *= d;
        nz *= d;*/
        let a = triangle.y - triangle.x;
        let b = triangle.z - triangle.x;
        let mut n = a.cross(&b);
        n /= n.norm();
        let c = triangle.get_center();
        if n[0].signum() != c[0].signum() {
            n *= -1.;
        }
        /*let mut n = (triangle.x + triangle.y + triangle.z) / 3.;
        n /= n.norm();*/
        /*let a = ((triangle.x + triangle.y) / 2. + triangle.z / 3.) * 2. / 3.;
        let b = ((triangle.y + triangle.z) / 2. + triangle.x / 3.) * 2. / 3.;
        let c = ((triangle.x + triangle.z) / 2. + triangle.y / 3.) * 2. / 3.;
        let aa = a - b;
        let bb = a - c;
        let mut n = aa.cross(&bb);*/
        n /= n.norm();
        //println!("{:?}", n);
        /*Polygon {
            normal: na::vector![nx, ny, nz],
            triangle: triangle,
            nx: na::vector![nx, ny, nz],
            ny: na::vector![nx, ny, nz],
            nz: na::vector![nx, ny, nz],
        }*/
        /*Polygon {
            normal: _normal.clone(),
            triangle: triangle,
            nx: _normal.clone(),
            ny: _normal.clone(),
            nz: _normal.clone(),
        }*/
        Polygon {
            normal: n.clone(),
            triangle: triangle,
            nx: n.clone(),
            ny: n.clone(),
            nz: n.clone(),
        }

        /*let mut nx = 0.;
        let mut ny = 0.;
        let mut nz = 0.;

        let vert = [triangle.x, triangle.y, triangle.z];
        for i in 0..3 {
            let v_curr = vert[i];
            let v_next = vert[(i + 1) % 3];
            nx += (v_next[1] - v_curr[1]) * (v_curr[2] + v_next[2]);
            ny += (v_next[2] - v_curr[2]) * (v_curr[0] + v_next[0]);
            nz += (v_next[0] - v_curr[0]) * (v_curr[1] + v_next[1]);
        }

        //метод ньюэла newell
        /*let mut nx = (triangle.y[1] - triangle.x[1]) * (triangle.x[2] + triangle.y[2]);
        nx += (triangle.z[1] - triangle.x[1]) * (triangle.y[2] + triangle.z[2]);
        nx += (triangle.x[1] - triangle.z[1]) * (triangle.z[2] + triangle.x[2]);
        let mut ny = (triangle.y[2] - triangle.x[2]) * (triangle.x[0] + triangle.y[0]);
        nx += (triangle.z[2] - triangle.y[2]) * (triangle.y[0] + triangle.z[0]);
        nx += (triangle.x[2] - triangle.z[2]) * (triangle.z[0] + triangle.x[0]);
        let mut nz = (triangle.y[0] - triangle.x[0]) * (triangle.x[1] + triangle.y[1]);
        nx += (triangle.z[0] - triangle.y[0]) * (triangle.y[1] + triangle.z[1]);
        nx += (triangle.x[0] - triangle.z[0]) * (triangle.z[1] + triangle.x[1]);*/
        let l = na::vector![nx, ny, nz].norm();
        nx /= -l;
        println!("nx = {:?}", nx);
        ny /= -l;
        println!("ny = {:?}", ny);
        nz /= -l;
        println!("nz = {:?}", nz);
        Polygon {
            normal: na::vector![nx, ny, nz],
            triangle: triangle,
            nx: na::vector![nx, ny, nz],
            ny: na::vector![nx, ny, nz],
            nz: na::vector![nx, ny, nz],
        }*/
    }

    fn new_all(
        normal: na::Vector3<f64>,
        triangle: Triangle,
        nx: na::Vector3<f64>,
        ny: na::Vector3<f64>,
        nz: na::Vector3<f64>,
    ) -> Self {
        Polygon {
            normal: normal,
            triangle: triangle,
            nx: nx,
            ny: ny,
            nz: nz,
        }
    }

    fn set_normal_vertex(&self, mesh: &Vec<Polygon>) -> Self {
        let nx = Self::found_vertex_normal(self.triangle.x, mesh);
        let ny = Self::found_vertex_normal(self.triangle.y, mesh);
        let nz = Self::found_vertex_normal(self.triangle.z, mesh);
        /*println!("nx = {}", nx);
        println!("ny = {}", ny);
        println!("nz = {}", nz);*/
        Polygon {
            normal: self.normal,
            triangle: self.triangle,
            nx: nx,
            ny: ny,
            nz: nz,
        }
    }

    fn found_vertex_normal(vertex: na::Vector3<f64>, mesh: &Vec<Polygon>) -> na::Vector3<f64> {
        static EPS: na::Vector3<f64> = na::vector![0.00000001, 0.00000001, 0.00000001];
        let mut n = ZEROS;
        //println!("vertex = {}", vertex);
        for pol in mesh
            .iter()
            //.filter(|x| vertex == x.triangle.x || vertex == x.triangle.y || vertex == x.triangle.z)
            .filter(|x| {
                (vertex - x.triangle.x).abs() < EPS
                    || (vertex - x.triangle.y).abs() < EPS
                    || (vertex - x.triangle.z).abs() < EPS
            })
        {
            n += pol.normal / vertex.metric_distance(&pol.triangle.get_center());
            //println!("{:?}", pol.triangle);
            //println!("{}", pol.normal);
        }
        n / n.norm()
    }

    fn transition(&self) -> LocalPolygon {
        self.into()
    }
}

impl From<LocalPolygon> for Polygon {
    fn from(poly: LocalPolygon) -> Self {
        poly.old_poly
    }
}

struct Mesh {
    mesh: Vec<Polygon>,
}

impl Mesh {
    fn new(data: Vec<Polygon>) -> Self {
        Mesh { mesh: data }
    }
    fn set_normal_vertex(&self) -> Self {
        Mesh {
            mesh: self
                .mesh
                .iter()
                .map(|x| x.set_normal_vertex(&self.mesh))
                .collect(),
        }
    }
}

impl From<&stl_io::IndexedMesh> for Mesh {
    fn from(mesh: &stl_io::IndexedMesh) -> Self {
        Mesh::new(
            mesh.faces
                .iter()
                .map(|x| {
                    Polygon::new(
                        &na::vector![x.normal[0].into(), x.normal[1].into(), x.normal[2].into()],
                        Triangle::new(
                            na::vector![
                                mesh.vertices[x.vertices[0]][0].into(),
                                mesh.vertices[x.vertices[0]][1].into(),
                                mesh.vertices[x.vertices[0]][2].into()
                            ],
                            na::vector![
                                mesh.vertices[x.vertices[1]][0].into(),
                                mesh.vertices[x.vertices[1]][1].into(),
                                mesh.vertices[x.vertices[1]][2].into()
                            ],
                            na::vector![
                                mesh.vertices[x.vertices[2]][0].into(),
                                mesh.vertices[x.vertices[2]][1].into(),
                                mesh.vertices[x.vertices[2]][2].into()
                            ],
                        ),
                    )
                })
                .collect(),
        )
        .set_normal_vertex()
    }
}

struct LocalPolygon {
    old_poly: Polygon,
    poly: Polygon,
    e1: na::Vector3<f64>,
    e2: na::Vector3<f64>,
    e3: na::Vector3<f64>,
    matrix_a: na::Matrix3<f64>,
    e_1: na::Vector3<f64>,
    e_2: na::Vector3<f64>,
    e_3: na::Vector3<f64>,
    x0: na::Vector3<f64>,
    old_x0: na::Vector3<f64>,
    matrix_a_1: na::Matrix3<f64>,
}

impl LocalPolygon {
    fn get_matrix_a(&self) -> na::Matrix3<f64> {
        self.matrix_a.clone()
    }

    fn get_points(&self) -> Points {
        Points::from(self)
    }

    fn get_local_point(&self, x: na::Vector3<f64>) -> na::Vector3<f64> {
        //println!("{:?}", self.poly.triangle);
        let res = self.matrix_a_1 * x + self.x0;
        /*let poly = &self.poly;
        let e1 = self.matrix_a_1 * self.e_1;
        let e2 = self.matrix_a_1 * self.e_2;
        let e3 = self.matrix_a_1 * self.e_3;
        println!("{}, {}, {}", e1, e2, e3);
        let angel_x = e1.angle(&e2);
        let angel_y = e3.angle(&e1);
        let angel_z = e2.angle(&e3);
        let angel_x_2 = angel_x / 2.;
        let angel_y_2 = angel_y / 2.;
        let angel_z_2 = angel_z / 2.;
        let e_xy = na::Matrix4::new_rotation(na::vector![0., 0., 0.5 * angel_x_2]).transform_vector(&e1);
        let e_yx = na::Matrix4::new_rotation(na::vector![0., 0., 1.5 * angel_y_2]).transform_vector(&e3);
        let e_yz = na::Matrix4::new_rotation(na::vector![0., 0., 0.5 * angel_y_2]).transform_vector(&e3);
        let e_zy = na::Matrix4::new_rotation(na::vector![0., 0., 1.5 * angel_z_2]).transform_vector(&e2);
        let e_zx = na::Matrix4::new_rotation(na::vector![0., 0., 0.5 * angel_z_2]).transform_vector(&e2);
        let e_xz = na::Matrix4::new_rotation(na::vector![0., 0., 1.5 * angel_x_2]).transform_vector(&e1);
        println!("{}, {}, {}, {}, {}, {}", e_xy, e_yx, e_yz, e_zy, e_zx, e_xz);
        let xy = e_xy / 2.;
        let yz = e_yz / 2.;
        let xz = e_zx / 2.;
        let tx = poly.triangle.x;
        let ty = poly.triangle.y;
        let tz = poly.triangle.z;
        let x = res;

        let a1 = (tx[0] - x[0]) * (ty[1] - tx[1]) - (ty[0] - tx[0]) * (tx[1] - x[1]);
        let b1 = (ty[0] - x[0]) * (xy[1] - tx[1]) - (xy[0] - ty[0]) * (ty[1] - x[1]);
        let c1 = (xy[0] - x[0]) * (tx[1] - xy[1]) - (tx[0] - xy[0]) * (xy[1] - x[1]);

        let a2 = (ty[0] - x[0]) * (tz[1] - ty[1]) - (tz[0] - ty[0]) * (ty[1] - x[1]);
        let b2 = (tz[0] - x[0]) * (yz[1] - tz[1]) - (yz[0] - tz[0]) * (tz[1] - x[1]);
        let c2 = (yz[0] - x[0]) * (ty[1] - yz[1]) - (ty[0] - yz[0]) * (yz[1] - x[1]);

        let a3 = (tz[0] - x[0]) * (tx[1] - tz[1]) - (tz[0] - tx[0]) * (tx[1] - x[1]);
        let b3 = (tx[0] - x[0]) * (xz[1] - tz[1]) - (xz[0] - tx[0]) * (tx[1] - x[1]);
        let c3 = (xz[0] - x[0]) * (tz[1] - xz[1]) - (tz[0] - xz[0]) * (xz[1] - x[1]);

        if a1.signum() == b1.signum() && b1.signum() == c1.signum() {
            InOrOut::Out(res)
        } else if a2.signum() == b2.signum() && b2.signum() == c2.signum() {
            InOrOut::Out(res)
        } else if a3.signum() == b3.signum() && b3.signum() == c3.signum() {
            InOrOut::Out(res)
        } else {
            InOrOut::In(res)
        }*/
        res
    }

    fn get_from_local(&self, x: na::Vector3<f64>) -> na::Vector3<f64> {
        self.matrix_a * x + self.old_x0
    }
}

impl From<&Polygon> for LocalPolygon {
    fn from(poly: &Polygon) -> Self {
        let e_1 = poly.triangle.y - poly.triangle.x;
        let e_2 = poly.triangle.z - poly.triangle.x;
        let e_3 = poly.triangle.z - poly.triangle.y;
        let e1 = e_1 / e_1.norm();
        let e2 = poly.normal.cross(&e1);
        let e3 = poly.normal;
        let a = na::Matrix3::from_columns(&[e1, e2, e3]);
        let at = a.try_inverse().unwrap();
        let n = at * poly.normal;
        let nx = /*at * */poly.nx;
        let ny = /*at * */poly.ny;
        let nz = /*at * */poly.nz;
        let x0 = -at * poly.triangle.x;
        let t = Triangle::new(
            at * poly.triangle.x + x0,
            at * poly.triangle.y + x0,
            at * poly.triangle.z + x0,
        );
        let local_poly = Polygon::new_all(n, t, nx, ny, nz);
        LocalPolygon {
            old_poly: poly.clone(),
            poly: local_poly,
            e1: e1,
            e2: e2,
            e3: e3,
            matrix_a: a,
            e_1: /*at * */e_1,
            e_2: /*at * */e_2,
            e_3: /*at * */e_3,
            x0: x0,
            old_x0: poly.triangle.x,
            matrix_a_1: at,
        }
    }
}

struct Points {
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

    fn get_fi(
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

impl From<&LocalPolygon> for Points {
    fn from(poly: &LocalPolygon) -> Self {
        let triangle = &poly.poly.triangle;
        Points {
            p1: triangle.x,
            p2: triangle.y,
            p3: triangle.z,
            p4: (triangle.x * 2. + triangle.y) / 3.,
            p5: (triangle.x + triangle.y * 2.) / 3.,
            p6: (triangle.y * 2. + triangle.z) / 3.,
            p7: (triangle.y + triangle.z * 2.) / 3.,
            p8: (triangle.x + triangle.z * 2.) / 3.,
            p9: (triangle.x * 2. + triangle.z) / 3.,
            p10: (triangle.x + triangle.y + triangle.z) / 3.,
        }
    }
}

struct U {
    u_1: f64,
    u_2: f64,
    u_3: f64,
    u_4: f64,
    u_5: f64,
    u_6: f64,
    u_7: f64,
    u_8: f64,
    u_9: f64,
    u_10: f64,
}

impl U {
    fn get(&self, i: i32) -> f64 {
        match i {
            1 => self.u_1,
            2 => self.u_2,
            3 => self.u_3,
            4 => self.u_4,
            5 => self.u_5,
            6 => self.u_6,
            7 => self.u_7,
            8 => self.u_8,
            9 => self.u_9,
            10 => self.u_10,
            _ => unreachable!(),
        }
    }
}

impl From<&LocalPolygon> for U {
    fn from(poly: &LocalPolygon) -> Self {
        let p = &poly.poly;
        let q11 = -p.nx.dot(&poly.e1) / (p.nx.dot(&poly.e3));
        let q12 = -p.ny.dot(&poly.e1) / (p.ny.dot(&poly.e3));
        let q21 = -p.nx.dot(&poly.e_2) / (poly.e_2.norm() * p.nx.dot(&poly.e3));
        let q23 = -p.nz.dot(&poly.e_2) / (poly.e_2.norm() * p.nz.dot(&poly.e3));
        let q32 = -p.ny.dot(&poly.e_3) / (poly.e_3.norm() * p.ny.dot(&poly.e3));
        let q33 = -p.nz.dot(&poly.e_3) / (poly.e_3.norm() * p.nz.dot(&poly.e3));
        let u_4 = 2. * poly.e_1.norm() / 27. * (2. * q11 - q12);
        let u_5 = 2. * poly.e_1.norm() / 27. * (q11 - 2. * q12);
        let u_6 = 2. * poly.e_3.norm() / 27. * (2. * q32 - q33);
        let u_7 = 2. * poly.e_3.norm() / 27. * (q32 - 2. * q33);
        let u_8 = 2. * poly.e_2.norm() / 27. * (q21 - 2. * q23);
        let u_9 = 2. * poly.e_2.norm() / 27. * (2. * q21 - q23);
        U {
            u_1: 0.,
            u_2: 0.,
            u_3: 0.,
            u_4: u_4,
            u_5: u_5,
            u_6: u_6,
            u_7: u_7,
            u_8: u_8,
            u_9: u_9,
            u_10: 0.25 * (u_4 + u_5 + u_6 + u_7 + u_8 + u_9),
        }
    }
}

fn gamma(a: f64, b: f64, p: &Points, u: &U) -> f64 {
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
    u.u_1 * fi1
        + u.u_2 * fi2
        + u.u_3 * fi3
        + u.u_4 * fi4
        + u.u_5 * fi5
        + u.u_6 * fi6
        + u.u_7 * fi7
        + u.u_8 * fi8
        + u.u_9 * fi9
        + u.u_10 * fi10
}

// size_edge - count of squares on edge
struct Cube {
    x_0: na::Vector3<f64>,
    size: f64,
    size_edge: usize,
    mesh: Mesh,
}

impl Cube {
    fn new_edge(size_edge: usize) -> Self {
        Self::new(ZEROS, 1., size_edge)
    }

    fn new(x0: na::Vector3<f64>, size: f64, size_edge: usize) -> Self {
        static NORM_ABCD: na::Vector3<f64> = na::vector![0., 0., -1.];
        static NORM_A2B2C2D2: na::Vector3<f64> = na::vector![0., 0., 1.];
        static NORM_AA2D2D: na::Vector3<f64> = na::vector![1., 0., 0.];
        static NORM_BB2C2C: na::Vector3<f64> = na::vector![-1., 0., 0.];
        static NORM_AA2B2B: na::Vector3<f64> = na::vector![0., -1., 0.];
        static NORM_DD2C2C: na::Vector3<f64> = na::vector![0., 1., 0.];

        let dl = size / 2.;
        let a = x0 + na::vector![dl, -dl, -dl];
        let b = x0 + na::vector![-dl, -dl, -dl];
        let c = x0 + na::vector![-dl, dl, -dl];
        let d = x0 + na::vector![dl, dl, -dl];
        let a2 = x0 + na::vector![dl, -dl, dl];
        let b2 = x0 + na::vector![-dl, -dl, dl];
        let c2 = x0 + na::vector![-dl, dl, dl];
        let d2 = x0 + na::vector![dl, dl, dl];
        let mut poly = Vec::with_capacity(size_edge * size_edge * 6 * 2);
        let dl = size / size_edge as f64;
        for i in 0..size_edge {
            for j in 0..size_edge {
                //abcd
                let x0 = b + na::vector![dl * i as f64, dl * j as f64, 0.];
                poly.push(Polygon::new_all(
                    NORM_ABCD,
                    Triangle::new(
                        x0,
                        x0 + na::vector![dl, 0., 0.],
                        x0 + na::vector![dl, dl, 0.],
                    ),
                    NORM_ABCD,
                    NORM_ABCD,
                    NORM_ABCD,
                ));
                poly.push(Polygon::new_all(
                    NORM_ABCD,
                    Triangle::new(
                        x0,
                        x0 + na::vector![0., dl, 0.],
                        x0 + na::vector![dl, dl, 0.],
                    ),
                    NORM_ABCD,
                    NORM_ABCD,
                    NORM_ABCD,
                ));
                //a2b2c2d2
                let x0 = b2 + na::vector![dl * i as f64, dl * j as f64, 0.];
                poly.push(Polygon::new_all(
                    NORM_A2B2C2D2,
                    Triangle::new(
                        x0,
                        x0 + na::vector![dl, 0., 0.],
                        x0 + na::vector![dl, dl, 0.],
                    ),
                    NORM_A2B2C2D2,
                    NORM_A2B2C2D2,
                    NORM_A2B2C2D2,
                ));
                poly.push(Polygon::new_all(
                    NORM_A2B2C2D2,
                    Triangle::new(
                        x0,
                        x0 + na::vector![0., dl, 0.],
                        x0 + na::vector![dl, dl, 0.],
                    ),
                    NORM_A2B2C2D2,
                    NORM_A2B2C2D2,
                    NORM_A2B2C2D2,
                ));
                //aa2d2d
                let x0 = a + na::vector![0., dl * i as f64, dl * j as f64];
                poly.push(Polygon::new_all(
                    NORM_AA2D2D,
                    Triangle::new(
                        x0,
                        x0 + na::vector![0., dl, 0.],
                        x0 + na::vector![0., dl, dl],
                    ),
                    NORM_AA2D2D,
                    NORM_AA2D2D,
                    NORM_AA2D2D,
                ));
                poly.push(Polygon::new_all(
                    NORM_AA2D2D,
                    Triangle::new(
                        x0,
                        x0 + na::vector![0., 0., dl],
                        x0 + na::vector![0., dl, dl],
                    ),
                    NORM_AA2D2D,
                    NORM_AA2D2D,
                    NORM_AA2D2D,
                ));
                //bb2c2c
                let x0 = b + na::vector![0., dl * i as f64, dl * j as f64];
                poly.push(Polygon::new_all(
                    NORM_BB2C2C,
                    Triangle::new(
                        x0,
                        x0 + na::vector![0., dl, 0.],
                        x0 + na::vector![0., dl, dl],
                    ),
                    NORM_BB2C2C,
                    NORM_BB2C2C,
                    NORM_BB2C2C,
                ));
                poly.push(Polygon::new_all(
                    NORM_BB2C2C,
                    Triangle::new(
                        x0,
                        x0 + na::vector![0., 0., dl],
                        x0 + na::vector![0., dl, dl],
                    ),
                    NORM_BB2C2C,
                    NORM_BB2C2C,
                    NORM_BB2C2C,
                ));
                //aa2b2b
                let x0 = b + na::vector![dl * i as f64, 0., dl * j as f64];
                poly.push(Polygon::new_all(
                    NORM_AA2B2B,
                    Triangle::new(
                        x0,
                        x0 + na::vector![dl, 0., 0.],
                        x0 + na::vector![dl, 0., dl],
                    ),
                    NORM_AA2B2B,
                    NORM_AA2B2B,
                    NORM_AA2B2B,
                ));
                poly.push(Polygon::new_all(
                    NORM_AA2B2B,
                    Triangle::new(
                        x0,
                        x0 + na::vector![0., 0., dl],
                        x0 + na::vector![dl, 0., dl],
                    ),
                    NORM_AA2B2B,
                    NORM_AA2B2B,
                    NORM_AA2B2B,
                ));
                //dd2c2c
                let x0 = c + na::vector![dl * i as f64, 0., dl * j as f64];
                poly.push(Polygon::new_all(
                    NORM_DD2C2C,
                    Triangle::new(
                        x0,
                        x0 + na::vector![dl, 0., 0.],
                        x0 + na::vector![dl, 0., dl],
                    ),
                    NORM_DD2C2C,
                    NORM_DD2C2C,
                    NORM_DD2C2C,
                ));
                poly.push(Polygon::new_all(
                    NORM_DD2C2C,
                    Triangle::new(
                        x0,
                        x0 + na::vector![0., 0., dl],
                        x0 + na::vector![dl, 0., dl],
                    ),
                    NORM_DD2C2C,
                    NORM_DD2C2C,
                    NORM_DD2C2C,
                ));
            }
        }
        Cube {
            x_0: x0,
            size: size,
            size_edge: size_edge,
            mesh: Mesh::new(poly)/*.set_normal_vertex()*/,
        }
    }

    fn new_threads(x0: na::Vector3<f64>, size: f64, size_edge: usize) -> Self {
        use std::thread as thd;

        static NORM_ABCD: na::Vector3<f64> = na::vector![0., 0., -1.];
        static NORM_A2B2C2D2: na::Vector3<f64> = na::vector![0., 0., 1.];
        static NORM_AA2D2D: na::Vector3<f64> = na::vector![1., 0., 0.];
        static NORM_BB2C2C: na::Vector3<f64> = na::vector![-1., 0., 0.];
        static NORM_AA2B2B: na::Vector3<f64> = na::vector![0., -1., 0.];
        static NORM_DD2C2C: na::Vector3<f64> = na::vector![0., 1., 0.];

        let dl = size / 2.;
        let a = x0 + na::vector![dl, -dl, -dl];
        let b = x0 + na::vector![-dl, -dl, -dl];
        let c = x0 + na::vector![-dl, dl, -dl];
        let d = x0 + na::vector![dl, dl, -dl];
        let a2 = x0 + na::vector![dl, -dl, dl];
        let b2 = x0 + na::vector![-dl, -dl, dl];
        let c2 = x0 + na::vector![-dl, dl, dl];
        let d2 = x0 + na::vector![dl, dl, dl];
        let mut poly = Vec::with_capacity(size_edge * size_edge * 6 * 2);
        let dl = size / size_edge as f64;
        let mut threads = Vec::new();
        for i in 0..size_edge {
            let i = i;
            threads.push(thd::spawn(move || {
                let mut poly = Vec::new();
                for j in 0..size_edge {
                    //abcd
                    let x0 = b + na::vector![dl * i as f64, dl * j as f64, 0.];
                    poly.push(Polygon::new_all(
                        NORM_ABCD,
                        Triangle::new(
                            x0,
                            x0 + na::vector![dl, 0., 0.],
                            x0 + na::vector![dl, dl, 0.],
                        ),
                        NORM_ABCD,
                        NORM_ABCD,
                        NORM_ABCD,
                    ));
                    poly.push(Polygon::new_all(
                        NORM_ABCD,
                        Triangle::new(
                            x0,
                            x0 + na::vector![0., dl, 0.],
                            x0 + na::vector![dl, dl, 0.],
                        ),
                        NORM_ABCD,
                        NORM_ABCD,
                        NORM_ABCD,
                    ));
                    //a2b2c2d2
                    let x0 = b2 + na::vector![dl * i as f64, dl * j as f64, 0.];
                    poly.push(Polygon::new_all(
                        NORM_A2B2C2D2,
                        Triangle::new(
                            x0,
                            x0 + na::vector![dl, 0., 0.],
                            x0 + na::vector![dl, dl, 0.],
                        ),
                        NORM_A2B2C2D2,
                        NORM_A2B2C2D2,
                        NORM_A2B2C2D2,
                    ));
                    poly.push(Polygon::new_all(
                        NORM_A2B2C2D2,
                        Triangle::new(
                            x0,
                            x0 + na::vector![0., dl, 0.],
                            x0 + na::vector![dl, dl, 0.],
                        ),
                        NORM_A2B2C2D2,
                        NORM_A2B2C2D2,
                        NORM_A2B2C2D2,
                    ));
                    //aa2d2d
                    let x0 = a + na::vector![0., dl * i as f64, dl * j as f64];
                    poly.push(Polygon::new_all(
                        NORM_AA2D2D,
                        Triangle::new(
                            x0,
                            x0 + na::vector![0., dl, 0.],
                            x0 + na::vector![0., dl, dl],
                        ),
                        NORM_AA2D2D,
                        NORM_AA2D2D,
                        NORM_AA2D2D,
                    ));
                    poly.push(Polygon::new_all(
                        NORM_AA2D2D,
                        Triangle::new(
                            x0,
                            x0 + na::vector![0., 0., dl],
                            x0 + na::vector![0., dl, dl],
                        ),
                        NORM_AA2D2D,
                        NORM_AA2D2D,
                        NORM_AA2D2D,
                    ));
                    //bb2c2c
                    let x0 = b + na::vector![0., dl * i as f64, dl * j as f64];
                    poly.push(Polygon::new_all(
                        NORM_BB2C2C,
                        Triangle::new(
                            x0,
                            x0 + na::vector![0., dl, 0.],
                            x0 + na::vector![0., dl, dl],
                        ),
                        NORM_BB2C2C,
                        NORM_BB2C2C,
                        NORM_BB2C2C,
                    ));
                    poly.push(Polygon::new_all(
                        NORM_BB2C2C,
                        Triangle::new(
                            x0,
                            x0 + na::vector![0., 0., dl],
                            x0 + na::vector![0., dl, dl],
                        ),
                        NORM_BB2C2C,
                        NORM_BB2C2C,
                        NORM_BB2C2C,
                    ));
                    //aa2b2b
                    let x0 = b + na::vector![dl * i as f64, 0., dl * j as f64];
                    poly.push(Polygon::new_all(
                        NORM_AA2B2B,
                        Triangle::new(
                            x0,
                            x0 + na::vector![dl, 0., 0.],
                            x0 + na::vector![dl, 0., dl],
                        ),
                        NORM_AA2B2B,
                        NORM_AA2B2B,
                        NORM_AA2B2B,
                    ));
                    poly.push(Polygon::new_all(
                        NORM_AA2B2B,
                        Triangle::new(
                            x0,
                            x0 + na::vector![0., 0., dl],
                            x0 + na::vector![dl, 0., dl],
                        ),
                        NORM_AA2B2B,
                        NORM_AA2B2B,
                        NORM_AA2B2B,
                    ));
                    //dd2c2c
                    let x0 = c + na::vector![dl * i as f64, 0., dl * j as f64];
                    poly.push(Polygon::new_all(
                        NORM_DD2C2C,
                        Triangle::new(
                            x0,
                            x0 + na::vector![dl, 0., 0.],
                            x0 + na::vector![dl, 0., dl],
                        ),
                        NORM_DD2C2C,
                        NORM_DD2C2C,
                        NORM_DD2C2C,
                    ));
                    poly.push(Polygon::new_all(
                        NORM_DD2C2C,
                        Triangle::new(
                            x0,
                            x0 + na::vector![0., 0., dl],
                            x0 + na::vector![dl, 0., dl],
                        ),
                        NORM_DD2C2C,
                        NORM_DD2C2C,
                        NORM_DD2C2C,
                    ));
                }
                poly
            }));
            let size = threads.len();
            for _ in 0..size {
                let mut r = threads.pop().unwrap().join().unwrap();
                poly.append(&mut r);
            }
        }
        Cube {
            x_0: x0,
            size: size,
            size_edge: size_edge,
            mesh: Mesh::new(poly)/*.set_normal_vertex()*/,
        }
    }

    fn run(&self, mesh: &Mesh) -> Mesh {
        let mut res = Vec::with_capacity(self.size_edge * self.size_edge * 6 * 2);
        for p in self.mesh.mesh.iter() {
            let (triangle, poly) = Self::found_triangle(&p, &mesh).unwrap();
            //println!("===============================================================");
            //println!("{:?}", triangle);
            //println!("===============================================================");
            //println!("{:?}", poly.triangle);
            let local_poly = LocalPolygon::from(&poly);
            let points = local_poly.get_points();
            let u = U::from(&local_poly);
            let local_triangle = Triangle::new(
                local_poly.get_local_point(triangle.x),
                local_poly.get_local_point(triangle.y),
                local_poly.get_local_point(triangle.z),
            );
            //println!("===============================================================");
            //println!("{:?}", local_triangle);
            let new_local_triangle = Triangle::new(
                na::vector![
                    local_triangle.x[0],
                    local_triangle.x[1],
                    gamma(local_triangle.x[0], local_triangle.x[1], &points, &u)
                ],
                na::vector![
                    local_triangle.y[0],
                    local_triangle.y[1],
                    gamma(local_triangle.y[0], local_triangle.y[1], &points, &u)
                ],
                na::vector![
                    local_triangle.z[0],
                    local_triangle.z[1],
                    gamma(local_triangle.z[0], local_triangle.z[1], &points, &u)
                ],
            );
            //println!("new ===============================================================");
            //println!("{:?}", new_local_triangle);
            let new_triangle = Triangle::new(
                local_poly.get_from_local(new_local_triangle.x),
                local_poly.get_from_local(new_local_triangle.y),
                local_poly.get_from_local(new_local_triangle.z),
            );
            //println!("new +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
            //println!("{:?}", new_triangle);
            res.push(Polygon::new(&ZEROS, new_triangle));
        }
        Mesh::new(res)/*.set_normal_vertex()*/
    }

    fn _run_threads(&self, mesh: &Mesh) -> Mesh {
        use std::thread as thd;
        use std::os::unix::thread::JoinHandleExt;
        //let mut res = Vec::with_capacity(self.size_edge * self.size_edge * 6 * 2);
        let mut m = self.mesh.mesh.clone();
        let mut m_vec = Vec::new();
        let sss = 3 * 3 * 6 * 2;
        while !m.is_empty() {
            m_vec.push(m.split_off(sss));
            if m.len() - sss < sss {
                m_vec.push(m.clone());
                m.clear();
            }
        }

        let size = m_vec.len();

        let arc = std::sync::Arc::new(mesh.mesh.clone());
        let mut arc_vec = Vec::new();

        for _ in 0..size {
            arc_vec.push(arc.clone());
        }

        println!("{}", size);
        let mut threads = Vec::new();
        let mut chanel = Vec::new();
        for i in 0..size {
            let (sender, receiver) = std::sync::mpsc::channel();
            chanel.push(receiver);
            let arc = arc_vec.pop().unwrap();
            //let arc = mesh.mesh.clone();
            let m = m_vec.pop().unwrap();
            threads.push(thd::Builder::new().name(format!("pthread_th{}_np", i)).spawn(move || {
                let mesh = arc.clone();
                let mut res = Vec::with_capacity(m.len());
                for p in m.iter() {
                    let (triangle, poly) = Self::found_triangle_thd(&p, &mesh).unwrap();
                    //println!("===============================================================");
                    //println!("{:?}", triangle);
                    //println!("===============================================================");
                    //println!("{:?}", poly.triangle);
                    let local_poly = LocalPolygon::from(&poly);
                    let points = local_poly.get_points();
                    let u = U::from(&local_poly);
                    let local_triangle = Triangle::new(
                        local_poly.get_local_point(triangle.x),
                        local_poly.get_local_point(triangle.y),
                        local_poly.get_local_point(triangle.z),
                    );
                    //println!("===============================================================");
                    //println!("{:?}", local_triangle);
                    let new_local_triangle = Triangle::new(
                        na::vector![
                            local_triangle.x[0],
                            local_triangle.x[1],
                            gamma(local_triangle.x[0], local_triangle.x[1], &points, &u)
                        ],
                        na::vector![
                            local_triangle.y[0],
                            local_triangle.y[1],
                            gamma(local_triangle.y[0], local_triangle.y[1], &points, &u)
                        ],
                        na::vector![
                            local_triangle.z[0],
                            local_triangle.z[1],
                            gamma(local_triangle.z[0], local_triangle.z[1], &points, &u)
                        ],
                    );
                    //println!("new ===============================================================");
                    //println!("{:?}", new_local_triangle);
                    let new_triangle = Triangle::new(
                        local_poly.get_from_local(new_local_triangle.x),
                        local_poly.get_from_local(new_local_triangle.y),
                        local_poly.get_from_local(new_local_triangle.z),
                    );
                    //println!("new +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
                    //println!("{:?}", new_triangle);
                    res.push(Polygon::new(&ZEROS, new_triangle));
                }
                sender.send(res).unwrap();
            }).unwrap().into_pthread_t())
        }

        let mut res = Vec::new();

        for i in 0..size {
            //println!("{:?}", threads[size - i - 1].thread().name());
            /*let mut r = threads.pop().unwrap().join().unwrap();
            res.append(&mut r);*/
            let mut r = chanel.pop().unwrap().recv().unwrap();
            unsafe {
                println!("{}", threads[i]);
                libc::pthread_join(threads[i], std::ptr::null_mut());
            }
            res.append(&mut r);
        }

        /*let mut m = self.mesh.mesh.clone();
        let mut m1 = m.split_off(m.len() / 4);
        let mut m2 = m.split_off(m.len() / 4);
        let mut m3 = m.split_off(m.len() / 4);

        /*let mesh1 = mesh.clone();
        let mesh2 = mesh.clone();
        let mesh3 = mesh.clone();
        let mesh4 = mesh.clone();*/
        let arc = std::sync::Arc::new(mesh.mesh.clone());
        let arc1 = arc.clone();
        let arc2 = arc.clone();
        let arc3 = arc.clone();
        let arc4 = arc.clone();

        let thread1 = thd::spawn(move || {
            let mesh = arc1.clone();
            let mut res = Vec::with_capacity(m.len());
            for p in m.iter() {
                let (triangle, poly) = Self::found_triangle_thd(&p, &mesh).unwrap();
                //println!("===============================================================");
                //println!("{:?}", triangle);
                //println!("===============================================================");
                //println!("{:?}", poly.triangle);
                let local_poly = LocalPolygon::from(&poly);
                let points = local_poly.get_points();
                let u = U::from(&local_poly);
                let local_triangle = Triangle::new(
                    local_poly.get_local_point(triangle.x),
                    local_poly.get_local_point(triangle.y),
                    local_poly.get_local_point(triangle.z),
                );
                //println!("===============================================================");
                //println!("{:?}", local_triangle);
                let new_local_triangle = Triangle::new(
                    na::vector![
                        local_triangle.x[0],
                        local_triangle.x[1],
                        gamma(local_triangle.x[0], local_triangle.x[1], &points, &u)
                    ],
                    na::vector![
                        local_triangle.y[0],
                        local_triangle.y[1],
                        gamma(local_triangle.y[0], local_triangle.y[1], &points, &u)
                    ],
                    na::vector![
                        local_triangle.z[0],
                        local_triangle.z[1],
                        gamma(local_triangle.z[0], local_triangle.z[1], &points, &u)
                    ],
                );
                //println!("new ===============================================================");
                //println!("{:?}", new_local_triangle);
                let new_triangle = Triangle::new(
                    local_poly.get_from_local(new_local_triangle.x),
                    local_poly.get_from_local(new_local_triangle.y),
                    local_poly.get_from_local(new_local_triangle.z),
                );
                //println!("new +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
                //println!("{:?}", new_triangle);
                res.push(Polygon::new(&ZEROS, new_triangle));
            }
            res
        });
        let thread2 = thd::spawn(move || {
            let mesh = arc2.clone();
            let mut res = Vec::with_capacity(m1.len());
            for p in m1.iter() {
                let (triangle, poly) = Self::found_triangle_thd(&p, &mesh).unwrap();
                //println!("===============================================================");
                //println!("{:?}", triangle);
                //println!("===============================================================");
                //println!("{:?}", poly.triangle);
                let local_poly = LocalPolygon::from(&poly);
                let points = local_poly.get_points();
                let u = U::from(&local_poly);
                let local_triangle = Triangle::new(
                    local_poly.get_local_point(triangle.x),
                    local_poly.get_local_point(triangle.y),
                    local_poly.get_local_point(triangle.z),
                );
                //println!("===============================================================");
                //println!("{:?}", local_triangle);
                let new_local_triangle = Triangle::new(
                    na::vector![
                        local_triangle.x[0],
                        local_triangle.x[1],
                        gamma(local_triangle.x[0], local_triangle.x[1], &points, &u)
                    ],
                    na::vector![
                        local_triangle.y[0],
                        local_triangle.y[1],
                        gamma(local_triangle.y[0], local_triangle.y[1], &points, &u)
                    ],
                    na::vector![
                        local_triangle.z[0],
                        local_triangle.z[1],
                        gamma(local_triangle.z[0], local_triangle.z[1], &points, &u)
                    ],
                );
                //println!("new ===============================================================");
                //println!("{:?}", new_local_triangle);
                let new_triangle = Triangle::new(
                    local_poly.get_from_local(new_local_triangle.x),
                    local_poly.get_from_local(new_local_triangle.y),
                    local_poly.get_from_local(new_local_triangle.z),
                );
                //println!("new +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
                //println!("{:?}", new_triangle);
                res.push(Polygon::new(&ZEROS, new_triangle));
            }
            res
        });
        let thread3 = thd::spawn(move || {
            let mesh = arc3.clone();
            let mut res = Vec::with_capacity(m2.len());
            for p in m2.iter() {
                let (triangle, poly) = Self::found_triangle_thd(&p, &mesh).unwrap();
                //println!("===============================================================");
                //println!("{:?}", triangle);
                //println!("===============================================================");
                //println!("{:?}", poly.triangle);
                let local_poly = LocalPolygon::from(&poly);
                let points = local_poly.get_points();
                let u = U::from(&local_poly);
                let local_triangle = Triangle::new(
                    local_poly.get_local_point(triangle.x),
                    local_poly.get_local_point(triangle.y),
                    local_poly.get_local_point(triangle.z),
                );
                //println!("===============================================================");
                //println!("{:?}", local_triangle);
                let new_local_triangle = Triangle::new(
                    na::vector![
                        local_triangle.x[0],
                        local_triangle.x[1],
                        gamma(local_triangle.x[0], local_triangle.x[1], &points, &u)
                    ],
                    na::vector![
                        local_triangle.y[0],
                        local_triangle.y[1],
                        gamma(local_triangle.y[0], local_triangle.y[1], &points, &u)
                    ],
                    na::vector![
                        local_triangle.z[0],
                        local_triangle.z[1],
                        gamma(local_triangle.z[0], local_triangle.z[1], &points, &u)
                    ],
                );
                //println!("new ===============================================================");
                //println!("{:?}", new_local_triangle);
                let new_triangle = Triangle::new(
                    local_poly.get_from_local(new_local_triangle.x),
                    local_poly.get_from_local(new_local_triangle.y),
                    local_poly.get_from_local(new_local_triangle.z),
                );
                //println!("new +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
                //println!("{:?}", new_triangle);
                res.push(Polygon::new(&ZEROS, new_triangle));
            }
            res
        });
        let thread4 = thd::spawn(move || {
            let mesh = arc4.clone();
            let mut res = Vec::with_capacity(m3.len());
            for p in m3.iter() {
                let (triangle, poly) = Self::found_triangle_thd(&p, &mesh).unwrap();
                //println!("===============================================================");
                //println!("{:?}", triangle);
                //println!("===============================================================");
                //println!("{:?}", poly.triangle);
                let local_poly = LocalPolygon::from(&poly);
                let points = local_poly.get_points();
                let u = U::from(&local_poly);
                let local_triangle = Triangle::new(
                    local_poly.get_local_point(triangle.x),
                    local_poly.get_local_point(triangle.y),
                    local_poly.get_local_point(triangle.z),
                );
                //println!("===============================================================");
                //println!("{:?}", local_triangle);
                let new_local_triangle = Triangle::new(
                    na::vector![
                        local_triangle.x[0],
                        local_triangle.x[1],
                        gamma(local_triangle.x[0], local_triangle.x[1], &points, &u)
                    ],
                    na::vector![
                        local_triangle.y[0],
                        local_triangle.y[1],
                        gamma(local_triangle.y[0], local_triangle.y[1], &points, &u)
                    ],
                    na::vector![
                        local_triangle.z[0],
                        local_triangle.z[1],
                        gamma(local_triangle.z[0], local_triangle.z[1], &points, &u)
                    ],
                );
                //println!("new ===============================================================");
                //println!("{:?}", new_local_triangle);
                let new_triangle = Triangle::new(
                    local_poly.get_from_local(new_local_triangle.x),
                    local_poly.get_from_local(new_local_triangle.y),
                    local_poly.get_from_local(new_local_triangle.z),
                );
                //println!("new +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
                //println!("{:?}", new_triangle);
                res.push(Polygon::new(&ZEROS, new_triangle));
            }
            res
        });

        let mut m = thread1.join().unwrap();
        let mut m1 = thread2.join().unwrap();
        let mut m2 = thread3.join().unwrap();
        let mut m3 = thread4.join().unwrap();

        let mut res = Vec::with_capacity(self.size_edge * self.size_edge * 6 * 2);
        res.append(&mut m);
        res.append(&mut m1);
        res.append(&mut m2);
        res.append(&mut m3);*/

        Mesh::new(res)/*.set_normal_vertex()*/
    }

    fn found_triangle(poly: &Polygon, oldmesh: &Mesh) -> Option<(Triangle, Polygon)> {
        let mut res: Option<(Triangle, Polygon)> = None;
        for p in oldmesh.mesh.iter() {
            let x = poly.triangle.x;
            let y = poly.triangle.y;
            let z = poly.triangle.z;
            let normal = p.normal;
            let d = -normal.dot(&p.triangle.get_center());
            /*println!("{:?}, {:?}, {:?}", d, normal, p.triangle.x);
            println!(
                "{:?}, {:?}, {:?}",
                -normal.dot(&p.triangle.y),
                normal,
                p.triangle.y
            );
            println!(
                "{:?}, {:?}, {:?}",
                -normal.dot(&p.triangle.z),
                normal,
                p.triangle.z
            );*/
            let lambda_x = -d / (normal[0] * x[0] + normal[1] * x[1] + normal[2] * x[2]);
            let x0 = na::vector![x[0] * lambda_x, x[1] * lambda_x, x[2] * lambda_x];
            let lambda_y = -d / (normal[0] * y[0] + normal[1] * y[1] + normal[2] * y[2]);
            let y0 = na::vector![y[0] * lambda_y, y[1] * lambda_y, y[2] * lambda_y];
            let lambda_z = -d / (normal[0] * z[0] + normal[1] * z[1] + normal[2] * z[2]);
            let z0 = na::vector![z[0] * lambda_z, z[1] * lambda_z, z[2] * lambda_z];
            /*if let Some(triangle) = res.clone() {
                if triangle.0.x.abs() < x0.abs()
                    && triangle.0.y.abs() < y0.abs()
                    && triangle.0.z.abs() < z0.abs()
                {
                    res = Some((Triangle::new(x0, y0, z0), p.clone()));
                }
            } else {
                res = Some((Triangle::new(x0, y0, z0), p.clone()));
            }*/
            /*if let Some(triangle) = res.clone() {
                if triangle.0.x.abs() >= x0.abs()
                    && triangle.0.y.abs() >= y0.abs()
                    && triangle.0.z.abs() >= z0.abs()
                {
                    res = Some((Triangle::new(x0, y0, z0), p.clone()));
                }
            } else {
                res = Some((Triangle::new(x0, y0, z0), p.clone()));
            }*/
            let p_center = p.triangle.get_center();
            let poly_center = poly.triangle.get_center();
            if let Some(triangle) = res.clone() {
                if p_center.metric_distance(&poly_center)
                    < triangle
                        .1
                        .triangle
                        .get_center()
                        .metric_distance(&poly_center)
                {
                    res = Some((Triangle::new(x0, y0, z0), p.clone()))
                }
            } else {
                res = Some((Triangle::new(x0, y0, z0), p.clone()));
            }
            /*let local = LocalPolygon::from(p);
            let new_tr = Triangle::new(
                local.get_local_point(x0),
                local.get_local_point(y0),
                local.get_local_point(z0),
            );
            let mx = (new_tr.x[0] * local.poly.triangle.y[1]
                - local.poly.triangle.y[0] * new_tr.x[1])
                / (local.poly.triangle.z[0] * local.poly.triangle.y[1]
                    - local.poly.triangle.y[0] * local.poly.triangle.z[1]);
            let my = (new_tr.y[0] * local.poly.triangle.y[1]
                - local.poly.triangle.y[0] * new_tr.y[1])
                / (local.poly.triangle.z[0] * local.poly.triangle.y[1]
                    - local.poly.triangle.y[0] * local.poly.triangle.z[1]);
            let mz = (new_tr.z[0] * local.poly.triangle.y[1]
                - local.poly.triangle.y[0] * new_tr.z[1])
                / (local.poly.triangle.z[0] * local.poly.triangle.y[1]
                    - local.poly.triangle.y[0] * local.poly.triangle.z[1]);
            if mx >= 0. && mx <= 1. && my >= 0. && my <= 1. && mz >= 0. && mz <= 1. {
                let lx = (new_tr.x[0] - mx * local.poly.triangle.z[0]) / local.poly.triangle.y[0];
                let ly = (new_tr.y[0] - my * local.poly.triangle.z[0]) / local.poly.triangle.y[0];
                let lz = (new_tr.z[0] - mz * local.poly.triangle.z[0]) / local.poly.triangle.y[0];
                if lx >= 0.
                    && (mx + lx) <= 1.
                    && ly >= 0.
                    && (my + ly) <= 1.
                    && lz >= 0.
                    && (mz + lz) <= 1.
                {
                    if let Some(triangle) = res.clone() {
                        if p_center.metric_distance(&poly_center)
                            < triangle
                                .1
                                .triangle
                                .get_center()
                                .metric_distance(&poly_center)
                        {
                            res = Some((Triangle::new(x0, y0, z0), p.clone()));
                        }
                    } else {
                        res = Some((Triangle::new(x0, y0, z0), p.clone()));
                    }
                }
            }*/
        }
        res
    }

    fn found_triangle_thd(poly: &Polygon, oldmesh: &Vec<Polygon>) -> Option<(Triangle, Polygon)> {
        Self::found_triangle(poly, &Mesh::new(oldmesh.clone()))
    }
}

enum InOrOut {
    //находится в зоне 1
    In(na::Vector1<f64>),
    //находится в зоне 2
    Out(na::Vector1<f64>, na::Vector3<f64>, na::Vector3<f64>),
}

struct Sphere {
    x0: na::Vector3<f64>,
    r: f64,
}

impl std::cmp::PartialEq for Sphere {
    fn eq(&self, other: &Self) -> bool {
        self.x0 == other.x0 && self.r == other.r
    }
}

impl Sphere {
    fn new() -> Self {
        Sphere {
            x0: ZEROS,
            r: 4.,
        }
    }

    fn new_with_x0_r(x0: na::Vector3<f64>, r: f64) -> Self {
        Sphere {
            x0: x0,
            r: r,
        }
    }

    fn run(&self, mesh: &Mesh) -> Mesh {
        let mut res = Vec::with_capacity(32);
        for p in mesh.mesh.iter() {
            let x = p.triangle.x;
            let y = p.triangle.y;
            let z = p.triangle.z;
            let mut xy = (x + y);
            let mut yz = (y + z);
            let mut xz = (x + z);
            xy = xy / xy.norm() * self.r;
            yz = yz / yz.norm() * self.r;
            xz = xz / xz.norm() * self.r;
            res.push(Polygon::new(
                &ZEROS,
                Triangle::new(
                    x,
                    xy,
                    xz
                )
            ));
            res.push(Polygon::new(
                &ZEROS,
                Triangle::new(
                    xy,
                    y,
                    yz
                )
            ));
            res.push(Polygon::new(
                &ZEROS,
                Triangle::new(
                    yz,
                    z,
                    xz
                )
            ));
            res.push(Polygon::new(
                &ZEROS,
                Triangle::new(
                    yz,
                    xy,
                    xz
                )
            ));

        }
        Mesh::new(res).set_normal_vertex()
    }
}

fn main() {
    let mut in_file = File::open("../123.stl").unwrap();
    let stl = stl_io::read_stl(&mut in_file).unwrap();
    let mesh = Mesh::from(&stl);

    let sphere = Sphere::new();

    let mesh = sphere.run(&mesh);

    /*let mut out_file = File::create("../2.stl").unwrap();
    stl_io::write_stl(
        &mut out_file,
        stl.faces.iter().map(|x| stl_io::Triangle {
            normal: x.normal,
            vertices: [
                stl.vertices[x.vertices[0]],
                stl.vertices[x.vertices[1]],
                stl.vertices[x.vertices[2]],
            ],
        }),
    )
    .unwrap();*/

    println!("make cube");
    let cube = Cube::new(ZEROS, 1., 24);
    //let cube = Cube::new_threads(ZEROS, 1., 2000);
    println!("start run");
    let res = cube.run(&mesh);
    //let res = cube.run_threads(&mesh);
    println!("end run");

    let mut out_file = File::create("../cube.stl").unwrap();
    stl_io::write_stl(
        &mut out_file,
        res.mesh.iter().map(|x| stl_io::Triangle {
            normal: stl_io::Normal::new([
                x.normal[0] as f32,
                x.normal[1] as f32,
                x.normal[2] as f32,
            ]),
            vertices: [
                stl_io::Vertex::new([
                    x.triangle.x[0] as f32,
                    x.triangle.x[1] as f32,
                    x.triangle.x[2] as f32,
                ]),
                stl_io::Vertex::new([
                    x.triangle.y[0] as f32,
                    x.triangle.y[1] as f32,
                    x.triangle.y[2] as f32,
                ]),
                stl_io::Vertex::new([
                    x.triangle.z[0] as f32,
                    x.triangle.z[1] as f32,
                    x.triangle.z[2] as f32,
                ]),
            ],
        }),
    )
    .unwrap();
}
