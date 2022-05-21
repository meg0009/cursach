mod cube;
mod join;
mod local;
mod mesh;
mod sphere;
mod triangle;
mod vertex;

use nalgebra as na;

use crate::sphere::Sphere;
use std::convert::From;
use std::fs::File;

const ZEROS: na::Vector3<f64> = na::vector![0., 0., 0.];

static ENABLE_JOIN: bool = true;

/*enum InOrOut {
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
        Sphere { x0: ZEROS, r: 4. }
    }

    fn new_with_x0_r(x0: na::Vector3<f64>, r: f64) -> Self {
        Sphere { x0: x0, r: r }
    }

    fn run(&self, mesh: &Mesh) -> Mesh {
        let mut res = Vec::with_capacity(32);
        for p in mesh.mesh.iter() {
            let x = p.triangle.x;
            let y = p.triangle.y;
            let z = p.triangle.z;
            let mut xy = x + y;
            let mut yz = y + z;
            let mut xz = x + z;
            xy = xy / xy.norm() * self.r;
            yz = yz / yz.norm() * self.r;
            xz = xz / xz.norm() * self.r;
            res.push(Polygon::new(&ZEROS, Triangle::new(x, xy, xz)));
            res.push(Polygon::new(&ZEROS, Triangle::new(xy, y, yz)));
            res.push(Polygon::new(&ZEROS, Triangle::new(yz, z, xz)));
            res.push(Polygon::new(&ZEROS, Triangle::new(yz, xy, xz)));
        }
        Mesh::new(res).set_normal_vertex()
    }
}*/

fn main() {
    // let mut in_file = File::open("../source in.stl").unwrap();
    let mut in_file = File::open("../source in.stl").unwrap();
    let stl = stl_io::read_stl(&mut in_file).unwrap();
    let mesh = mesh::Mesh::from(&stl);

    /*let sphere = Sphere::new(&ZEROS, 4.);
    let mesh = sphere.run(&mesh);*/

    println!("make cube");
    let cube = cube::Cube::new(ZEROS /* + na::Vector3::new(0.5, 0., 0.)*/, 1., 25);
    println!("start run");
    let res = cube.run(&mesh);
    println!("end run");

    /*let sphere = Sphere::new();

    let mesh = sphere.run(&mesh);*/

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

    let mut out_file = File::create("../cube.stl").unwrap();
    stl_io::write_stl(&mut out_file, Vec::<_>::from(&res).iter()).unwrap();
}
