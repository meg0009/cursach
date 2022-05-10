use crate::vertex::Vertex;
use nalgebra as na;

#[derive(Clone, Debug)]
pub struct Triangle {
    x: na::Vector3<f64>,
    y: na::Vector3<f64>,
    z: na::Vector3<f64>,
}

impl Triangle {
    pub fn new(x: &na::Vector3<f64>, y: &na::Vector3<f64>, z: &na::Vector3<f64>) -> Self {
        Triangle {
            x: x.clone(),
            y: y.clone(),
            z: z.clone(),
        }
    }

    pub fn get_center(&self) -> na::Vector3<f64> {
        (self.x + self.y + self.z) / 3.
    }

    pub fn x(&self) -> &na::Vector3<f64> {
        &self.x
    }

    pub fn y(&self) -> &na::Vector3<f64> {
        &self.y
    }

    pub fn z(&self) -> &na::Vector3<f64> {
        &self.z
    }

    pub fn x_mut(&mut self) -> &mut na::Vector3<f64> {
        &mut self.x
    }

    pub fn y_mut(&mut self) -> &mut na::Vector3<f64> {
        &mut self.y
    }

    pub fn z_mut(&mut self) -> &mut na::Vector3<f64> {
        &mut self.z
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedTriangle {
    x: usize,
    y: usize,
    z: usize,
}

impl IndexedTriangle {
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        IndexedTriangle { x, y, z }
    }

    pub fn get_center(&self, vertexes: &[Vertex]) -> na::Vector3<f64> {
        Triangle::new(
            vertexes[self.x].pos(),
            &vertexes[self.y].pos(),
            &vertexes[self.z].pos(),
        )
        .get_center()
    }

    pub fn x(&self, vertexes: &[Vertex]) -> na::Vector3<f64> {
        vertexes[self.x].pos().clone()
    }

    pub fn y(&self, vertexes: &[Vertex]) -> na::Vector3<f64> {
        vertexes[self.y].pos().clone()
    }

    pub fn z(&self, vertexes: &[Vertex]) -> na::Vector3<f64> {
        vertexes[self.z].pos().clone()
    }

    pub fn x_index(&self) -> usize {
        self.x
    }

    pub fn y_index(&self) -> usize {
        self.y
    }

    pub fn z_index(&self) -> usize {
        self.z
    }
}

#[derive(Clone, Debug)]
pub struct Polygon {
    norm: na::Vector3<f64>,
    triangle: Triangle,
}

impl Polygon {
    pub fn new(x: &na::Vector3<f64>, y: &na::Vector3<f64>, z: &na::Vector3<f64>) -> Self {
        let a = y - x;
        let b = z - x;
        let mut n = a.cross(&b);
        n /= n.norm();

        // направляем нормаль к поверхности от начала координат
        let center = Triangle::new(x, y, z).get_center();
        if (center + n).magnitude() < (center - n).magnitude() {
            n *= -1.;
        }

        Polygon {
            norm: n,
            triangle: Triangle::new(x, y, z),
        }
    }

    pub fn norm(&self) -> na::Vector3<f64> {
        self.norm.clone()
    }

    pub fn triangle(&self) -> Triangle {
        self.triangle.clone()
    }

    pub fn from_indexed_polygon(polygon: &IndexedPolygon, vertexes: &[Vertex]) -> Self {
        Polygon {
            norm: polygon.norm,
            triangle: Triangle::new(
                &polygon.triangle.x(vertexes),
                &polygon.triangle.y(vertexes),
                &polygon.triangle.z(vertexes),
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub struct IndexedPolygon {
    norm: na::Vector3<f64>,
    triangle: IndexedTriangle,
}

impl IndexedPolygon {
    pub fn new(triangle: &IndexedTriangle, vertexes: &[Vertex]) -> Self {
        let x = vertexes[triangle.x].pos().clone();
        let y = vertexes[triangle.y].pos().clone();
        let z = vertexes[triangle.z].pos().clone();

        IndexedPolygon {
            norm: Polygon::new(&x, &y, &z).norm(),
            triangle: triangle.clone(),
        }
    }

    pub fn triangle_indexes(&self) -> &IndexedTriangle {
        &self.triangle
    }

    pub fn norm(&self) -> na::Vector3<f64> {
        self.norm.clone()
    }

    pub fn triangle(&self) -> IndexedTriangle {
        self.triangle.clone()
    }

    pub fn contains_index(&self, i: usize) -> bool {
        self.triangle.x == i || self.triangle.y == i || self.triangle.z == i
    }
}

impl PartialEq for IndexedPolygon {
    fn eq(&self, other: &Self) -> bool {
        self.triangle == other.triangle
    }
}
