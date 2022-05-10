use nalgebra as na;

pub struct JoinLocalPolygon {
    e1: na::Vector3<f64>,
    _e2: na::Vector3<f64>,
    e3: na::Vector3<f64>,
    /*e_1: na::Vector3<f64>,
    e_2: na::Vector3<f64>,
    e_3: na::Vector3<f64>,*/
    x0: na::Vector3<f64>,
    matrix_a: na::Matrix3<f64>,
    matrix_a_1: na::Matrix3<f64>,
    old_x0: na::Vector3<f64>,
}

impl JoinLocalPolygon {
    pub fn to_local(&self, a: &na::Vector3<f64>) -> na::Vector3<f64> {
        self.matrix_a_1 * a + self.x0
    }

    pub fn from_local(&self, a: &na::Vector3<f64>) -> na::Vector3<f64> {
        self.matrix_a * a + self.old_x0
    }
}

impl From<(&na::Vector3<f64>, &na::Vector3<f64>, &na::Vector3<f64>, &na::Vector3<f64>)> for JoinLocalPolygon {
    fn from((p, q, ni, nj): (&na::Vector3<f64>, &na::Vector3<f64>, &na::Vector3<f64>, &na::Vector3<f64>)) -> Self {
        let pq = p - q;
        let e1 = pq.normalize();
        let e3 = (ni + nj).normalize();
        let e2 = e3.cross(&e1);
        let a = na::Matrix3::from_columns(&[e1, e2, e3]);
        let at = a.try_inverse().unwrap();
        let x0 = -at * p;
        JoinLocalPolygon {
            e1, 
            _e2: e2, 
            e3, 
            /*e_1: (), 
            e_2: (), 
            e_3: (), */
            x0, 
            matrix_a: a, 
            matrix_a_1: at, 
            old_x0: p.clone(),
        }
    }
}
