use crate::triangle::Polygon;
use crate::ZEROS;
use nalgebra as na;

#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pos: na::Vector3<f64>,
    norm: na::Vector3<f64>,
}

impl Vertex {
    pub fn new(pos: &na::Vector3<f64>) -> Self {
        Vertex {
            pos: pos.clone(),
            norm: ZEROS,
        }
    }

    pub fn set_norm(&self, polygons: &[Polygon]) -> Self {
        let mut n = ZEROS.clone();
        for pol in polygons.iter() {
            n += pol.norm() / self.pos.metric_distance(&pol.triangle().get_center());
        }
        Vertex {
            pos: self.pos.clone(),
            norm: n / n.norm(),
        }
    }

    pub fn pos(&self) -> &na::Vector3<f64> {
        &self.pos
    }

    pub fn norm(&self) -> &na::Vector3<f64> {
        &self.norm
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        static EPS: na::Vector3<f64> = na::Vector3::new(0.000001, 0.000001, 0.000001);
        (self.pos - other.pos).abs() < EPS || (other.pos - self.pos).abs() < EPS
    }
}
