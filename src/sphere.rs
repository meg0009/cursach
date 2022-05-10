use crate::mesh::Mesh;
use crate::triangle::IndexedTriangle;
use crate::vertex::Vertex;
use nalgebra as na;
use std::collections::btree_map::BTreeMap;
use std::collections::HashMap;

pub struct Sphere {
    x0: na::Vector3<f64>,
    r: f64,
}

impl Sphere {
    pub fn new(x0: &na::Vector3<f64>, r: f64) -> Self {
        Sphere { x0: x0.clone(), r }
    }

    pub fn run(&self, mesh: &Mesh) -> Mesh {
        let mut ver_res = Vec::with_capacity(20);
        let mut faces_res = Vec::with_capacity(32);
        let mut vert_map = HashMap::with_capacity(20);
        let mut new_map = HashMap::with_capacity(40);
        for p in mesh.faces().iter() {
            let triangle = p.triangle();
            let x = triangle.x(mesh.vertexes());
            let y = triangle.y(mesh.vertexes());
            let z = triangle.z(mesh.vertexes());
            let mut xy = x + y;
            let mut yz = y + z;
            let mut xz = x + z;
            xy = xy / xy.norm() * self.r;
            yz = yz / yz.norm() * self.r;
            xz = xz / xz.norm() * self.r;
            if !vert_map.contains_key(&triangle.x_index()) {
                ver_res.push(Vertex::new(&x));
                vert_map.insert(triangle.x_index(), ver_res.len() - 1);
            }
            if !vert_map.contains_key(&triangle.y_index()) {
                ver_res.push(Vertex::new(&y));
                vert_map.insert(triangle.y_index(), ver_res.len() - 1);
            }
            if !vert_map.contains_key(&triangle.z_index()) {
                ver_res.push(Vertex::new(&z));
                vert_map.insert(triangle.z_index(), ver_res.len() - 1);
            }
            let xy = Vertex::new(&xy);
            if !new_map.contains_key(&(triangle.x_index(), triangle.y_index())) {
                ver_res.push(xy);
                new_map.insert((triangle.x_index(), triangle.y_index()), ver_res.len() - 1);
                new_map.insert((triangle.y_index(), triangle.x_index()), ver_res.len() - 1);
            }
            let xz = Vertex::new(&xz);
            if !new_map.contains_key(&(triangle.x_index(), triangle.z_index())) {
                ver_res.push(xz);
                new_map.insert((triangle.x_index(), triangle.z_index()), ver_res.len() - 1);
                new_map.insert((triangle.z_index(), triangle.x_index()), ver_res.len() - 1);
            }
            let yz = Vertex::new(&yz);
            if !new_map.contains_key(&(triangle.y_index(), triangle.z_index())) {
                ver_res.push(yz);
                new_map.insert((triangle.y_index(), triangle.z_index()), ver_res.len() - 1);
                new_map.insert((triangle.z_index(), triangle.y_index()), ver_res.len() - 1);
            }
            faces_res.push(IndexedTriangle::new(
                *vert_map.get(&triangle.x_index()).unwrap(),
                *new_map
                    .get(&(triangle.x_index(), triangle.y_index()))
                    .unwrap(),
                *new_map
                    .get(&(triangle.x_index(), triangle.z_index()))
                    .unwrap(),
            ));
            faces_res.push(IndexedTriangle::new(
                *new_map
                    .get(&(triangle.x_index(), triangle.y_index()))
                    .unwrap(),
                *vert_map.get(&triangle.y_index()).unwrap(),
                *new_map
                    .get(&(triangle.y_index(), triangle.z_index()))
                    .unwrap(),
            ));
            faces_res.push(IndexedTriangle::new(
                *new_map
                    .get(&(triangle.y_index(), triangle.z_index()))
                    .unwrap(),
                *vert_map.get(&triangle.z_index()).unwrap(),
                *new_map
                    .get(&(triangle.x_index(), triangle.z_index()))
                    .unwrap(),
            ));
            faces_res.push(IndexedTriangle::new(
                *new_map
                    .get(&(triangle.y_index(), triangle.z_index()))
                    .unwrap(),
                *new_map
                    .get(&(triangle.x_index(), triangle.y_index()))
                    .unwrap(),
                *new_map
                    .get(&(triangle.x_index(), triangle.z_index()))
                    .unwrap(),
            ));
        }
        Mesh::new(&ver_res, &faces_res)
    }
}
