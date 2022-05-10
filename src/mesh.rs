use crate::na;
use crate::triangle::{IndexedPolygon, IndexedTriangle, Polygon};
use crate::vertex::Vertex;
use stl_io::IndexedMesh;

pub struct Mesh {
    vertexes: Vec<Vertex>,
    faces: Vec<IndexedPolygon>,
}

impl Mesh {
    pub fn new(vertexes: &Vec<Vertex>, triangles: &Vec<IndexedTriangle>) -> Self {
        Mesh {
            vertexes: vertexes.clone(),
            faces: triangles
                .iter()
                .map(|t| IndexedPolygon::new(t, vertexes.as_slice()))
                .collect(),
        }
        .set_vertex_norm()
    }

    pub fn set_vertex_norm(&self) -> Self {
        Mesh {
            vertexes: self
                .vertexes
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let polygons: Vec<_> = self
                        .faces
                        .iter()
                        .filter(|p| {
                            let triangle = p.triangle_indexes();
                            i == triangle.x_index()
                                || i == triangle.y_index()
                                || i == triangle.z_index()
                        })
                        .map(|p| Polygon::from_indexed_polygon(p, &self.vertexes))
                        .collect();
                    v.set_norm(&polygons)
                })
                .collect(),
            faces: self.faces.clone(),
        }
    }

    pub fn faces(&self) -> &Vec<IndexedPolygon> {
        &self.faces
    }

    pub fn vertexes(&self) -> &Vec<Vertex> {
        &self.vertexes
    }

    pub fn find_neighbor(
        &self,
        poly: &IndexedPolygon,
        i: usize,
        j: usize,
    ) -> Option<IndexedPolygon> {
        for p in &self.faces {
            if *p != *poly && p.contains_index(i) && p.contains_index(j) {
                return Some(p.clone());
            }
        }
        None
    }
}

impl From<&stl_io::IndexedMesh> for Mesh {
    fn from(in_mesh: &IndexedMesh) -> Self {
        let vertexes: Vec<_> = in_mesh
            .vertices
            .iter()
            .map(|v| Vertex::new(&na::Vector3::new(v[0] as f64, v[1] as f64, v[2] as f64)))
            .collect();
        let faces: Vec<_> = in_mesh
            .faces
            .iter()
            .map(|f| IndexedTriangle::new(f.vertices[0], f.vertices[1], f.vertices[2]))
            .collect();
        Mesh::new(&vertexes, &faces)
    }
}

impl From<&Mesh> for Vec<stl_io::Triangle> {
    fn from(mesh: &Mesh) -> Self {
        mesh.faces
            .iter()
            .map(|f| stl_io::Triangle {
                normal: stl_io::Normal::new([
                    f.norm()[0] as f32,
                    f.norm()[1] as f32,
                    f.norm()[2] as f32,
                ]),
                vertices: [
                    stl_io::Vertex::new({
                        let x = f.triangle_indexes().x(&mesh.vertexes);
                        [x[0] as f32, x[1] as f32, x[2] as f32]
                    }),
                    stl_io::Vertex::new({
                        let x = f.triangle_indexes().y(&mesh.vertexes);
                        [x[0] as f32, x[1] as f32, x[2] as f32]
                    }),
                    stl_io::Vertex::new({
                        let x = f.triangle_indexes().z(&mesh.vertexes);
                        [x[0] as f32, x[1] as f32, x[2] as f32]
                    }),
                ],
            })
            .collect()
    }
}
