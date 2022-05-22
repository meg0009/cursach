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

    pub fn get_e3(&self) -> na::Vector3<f64> {
        self.e3.clone()
    }
}

impl
    From<(
        &na::Vector3<f64>,
        &na::Vector3<f64>,
        &na::Vector3<f64>,
        &na::Vector3<f64>,
    )> for JoinLocalPolygon
{
    fn from(
        (p, q, ni, nj): (
            &na::Vector3<f64>,
            &na::Vector3<f64>,
            &na::Vector3<f64>,
            &na::Vector3<f64>,
        ),
    ) -> Self {
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

static mut A: i32 = 0;
static mut B: i32 = 0;

pub fn gamma(
    x: &na::Vector3<f64>,
    m: &na::Vector3<f64>,
    n: &na::Vector3<f64>,
    f_m: f64,
    f_n: f64,
    fd_m: f64,
    fd_n: f64,
) -> f64 {
    /*let res = f_m
        + fd_m * (a - m[0])
        + ((f_n - f_m) / (n[0] - m[0]) - fd_m) / (n[0] - m[0]) * (a - m[0]).powi(2)
        + (fd_n - 2. * (f_n - f_m) / (n[0] - m[0]) + fd_m) * (a - m[0]).powi(2) * (a - n[0]);
    let res2 = f_m
        + fd_m * (b - m[1])
        + ((f_n - f_m) / (n[1] - m[1]) - fd_m) / (n[1] - m[1]) * (b - m[1]).powi(2)
        + (fd_n - 2. * (f_n - f_m) / (n[1] - m[1]) + fd_m) * (b - m[1]).powi(2) * (b - n[1]);
    if res.is_nan() {
        res2
    } else {
        res
    }*/
    //f_m * (a - n[0]) * (b - n[1]) / ((m[0] - n[0]) * (m[1] - n[1])) + f_n * (a - m[0]) * (b - m[1]) / ((n[0] - m[0]) * (n[1] - m[1]))
    //f_m * (b - n[1]) / (m[1] - n[1]) + f_n * (b - m[1]) / (n[1] - m[1])
    /*if m[2].signum() != n[2].signum() {
        panic!("ну здарова");
    }*/

    let d = m.metric_distance(&n);

    /*let mn_x = ((n[0] - m[0]) * x[0] / (n[1] - m[1]) + x[1]
    - (m[1] + (n[1] - m[1]) * m[0] / (n[0] - m[0])))
    / ((n[1] - m[1]) / (n[0] - m[0]) + (n[0] - m[0]) / (n[1] - m[1]));*/
    let mn_x = (x[0] * (n[0] - m[0]) / (n[1] - m[1]) + x[1]
        - (m[1] - m[0] * (n[1] - m[1]) / (n[0] - m[0])))
        / ((n[1] - m[1]) / (n[0] - m[0]) + (n[0] - m[0]) / (n[1] - m[1]));
    let mn_y = -(n[0] - m[0]) * (mn_x - x[0]) / (n[1] - m[1]) + x[1];
    let mn_z = (mn_y - m[1]) * (n[2] - m[2]) / (n[1] - m[1]) + m[2];
    //let mn_y = mn_x * (n[1] - m[1]) / (n[0] - m[0]) + m[1] - m[0] * (n[1] - m[1]) / (n[0] - m[0]);

    if !mn_x.is_nan() {
        println!("1");
    }

    let xx = if mn_x.is_nan() {
        unsafe{
            A += 1;
        }
        na::Vector3::new(m[0], x[1], m[2])
    } else {
        unsafe{
            B += 1;
        }
        na::Vector3::new(mn_x, mn_y, mn_z)
    };
    unsafe{
        println!("A: {}, B: {}", A, B);
    }
    //let xx = na::Vector2::new(m[0], x[1]);
    //let dx = m.metric_distance(&xx);
    let dx = n.metric_distance(&xx);
    let t = if d > dx {
        dx / d
    } else {
        d / dx
    };

    /*let c0 = f_m;
    let c1 = fd_m;
    let c3 = fd_n - 2. * f_n + 2. * f_m + fd_m;
    let c2 = f_n - c3 - f_m - fd_m;

    c0 + c1 * t + c2 * t.powi(2) + c3 * t.powi(3)*/
    let c0 = f_n;
    let c1 = fd_n;
    let c3 = fd_m - 2. * f_m + 2. * f_n + fd_n;
    let c2 = f_m - c3 - f_n - fd_n;

    c0 + c1 * t + c2 * t.powi(2) + c3 * t.powi(3)
    /*let c0 = f_m;
    let c1 = f_n - f_m;
    c0 + c1 * t*/
}
