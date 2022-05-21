use crate::local::LocalPolygon;
use crate::mesh::Mesh;
use crate::triangle::{IndexedPolygon, IndexedTriangle, Polygon, Triangle};
use crate::ENABLE_JOIN;
use na::ComplexField;
use nalgebra as na;

const DELIMITER: f64 = 0.8;

pub enum InOrOut {
    //находится в зоне 1
    In,
    //находится в зоне 2
    Out(Points, Points, na::Vector3<f64>),
}

#[derive(PartialEq, Eq)]
pub enum Points {
    X,
    Y,
    Z,
}

impl From<(&na::Vector3<f64>, &LocalPolygon)> for InOrOut {
    fn from((x0, local): (&na::Vector3<f64>, &LocalPolygon)) -> Self {
        if ENABLE_JOIN {
            let poly = local.get_poly();
            let x = poly.triangle().x().clone();
            let y = poly.triangle().y().clone();
            let z = poly.triangle().z().clone();
            let xy = y - x; // стояли иначе
            let xz = z - x;
            let yz = z - y;
            let a_tmp = xy * 0.5 + x;
            let a_tmp2 = a_tmp - z;
            let a = a_tmp2 * DELIMITER + z;
            let c_tmp = xz * 0.5 + x;
            let c_tmp2 = c_tmp - y;
            let c = c_tmp2 * DELIMITER + y;
            let b_tmp = yz * 0.5 + y;
            let b_tmp2 = b_tmp - x;
            let b = b_tmp2 * DELIMITER + x;

            //println!("x: {}", x);
            //println!("y: {}", y);
            //println!("z: {}", z);
            //println!("a: {}", a);
            //println!("b: {}", b);
            //println!("c: {}", c);
            //println!("x0: {}", local.to_local(&x0));

            if inside_polygon(&x, &y, &a, &local.to_local(&x0)) {
                return InOrOut::Out(Points::X, Points::Y, a);
            }

            if inside_polygon(&x, &z, &c, &local.to_local(&x0)) {
                return InOrOut::Out(Points::X, Points::Z, c);
            }

            if inside_polygon(&y, &z, &b, &local.to_local(&x0)) {
                return InOrOut::Out(Points::Y, Points::Z, b);
            }

            /*let poly = local.get_poly();
            let x = poly.triangle().x().clone();
            let y = poly.triangle().y().clone();
            let z = poly.triangle().z().clone();
            let xy = y - x;
            let xz = z - x;
            let x_angle = xy.angle(&xz);
            let x_delimiter = x_angle / DELIMITER;
            let yx = x - y;
            let yz = z - y;
            let y_angle = yx.angle(&yz);
            let y_delimiter = y_angle / DELIMITER;
            let zx = x - z;
            let zy = y - z;
            let z_angle = zx.angle(&zy);
            let z_delimiter = z_angle / DELIMITER;

            //вращение против часовой стрелки
            let axisangle_anticlockwise = na::Vector3::z() * (-x_delimiter);
            let rot_x = na::Rotation3::new(axisangle_anticlockwise);
            let xa = rot_x * xy;

            let axisangle_anticlockwise = na::Vector3::z() * (-y_delimiter);
            let rot_y = na::Rotation3::new(axisangle_anticlockwise);
            let yc = rot_y * yz;

            let axisangle_anticlockwise = na::Vector3::z() * (-z_delimiter);
            let rot_z = na::Rotation3::new(axisangle_anticlockwise);
            let zb = rot_z * zx;

            // вращение по часовой стрелке
            let axisangle_clockwise = na::Vector3::z() * x_delimiter;
            let rot_x = na::Rotation3::new(axisangle_clockwise);
            let xb = rot_x * xz;

            let axisangle_clockwise = na::Vector3::z() * y_delimiter;
            let rot_y = na::Rotation3::new(axisangle_clockwise);
            let ya = rot_y * yx;

            let axisangle_clockwise = na::Vector3::z() * z_delimiter;
            let rot_z = na::Rotation3::new(axisangle_clockwise);
            let zc = rot_z * zy;

            let a = found_cross(&x, &xa, &y, &ya);
            let b = found_cross(&x, &xb, &z, &zb);
            let c = found_cross(&y, &yc, &z, &zc);

            //в xya
            if local.inside_triangle(x, y, a, &x0) {
                return InOrOut::Out(Points::X, Points::Y, a);
            }

            //в xzb
            let axisangle = na::Vector3::z() * (x_angle - x_delimiter);
            let rot_x = na::Rotation3::new(axisangle);
            if local.inside_triangle(x, rot_x * b, rot_x * z, &(rot_x * x0)) {
                return InOrOut::Out(Points::X, Points::Z, b);
            }

            //в yzc
            let axisangle = na::Vector3::z() * (y_delimiter - y_angle);
            let rot_y = na::Rotation3::new(axisangle);
            if local.inside_triangle(
                rot_y * (c - y),
                rot_y * (z - y),
                na::Vector3::zeros(),
                &(rot_y * (x0 - y)),
            ) {
                return InOrOut::Out(Points::Y, Points::Z, c);
            }*/
        }
        InOrOut::In
    }
}

fn triangle_square(x: f64, y: f64, z: f64) -> f64 {
    let p = (x + y + z) * 0.5;
    (p * (p - x) * (p - y) * (p - z)).sqrt()
}

fn sign(p1: &na::Vector3<f64>, p2: &na::Vector3<f64>, p3: &na::Vector3<f64>) -> f64 {
    (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
}

//*** возможно неопределённое поведение ***
//*** уже не должно быть никаких проблем https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle в файле local.rs он же ***
fn inside_polygon(
    x: &na::Vector3<f64>,
    y: &na::Vector3<f64>,
    z: &na::Vector3<f64>,
    a: &na::Vector3<f64>,
) -> bool {
    //static EPS: f64 = 0.000001;

    /*let xy = (x - y).magnitude();
        let xz = (x - z).magnitude();
        let yz = (y - z).magnitude();

        let ax = (x - a).magnitude();
        let ay = (y - a).magnitude();
        let az = (z - a).magnitude();

        (triangle_square(ax, ay, xy) + triangle_square(ax, az, xz) + triangle_square(ay, az, yz) - triangle_square(xy, xz, yz)).abs() < EPS
    */
    let d1 = sign(a, x, y);
    let d2 = sign(a, y, z);
    let d3 = sign(a, z, x);

    let has_neg = d1 < 0. || d2 < 0. || d3 < 0.;
    let has_pos = d1 > 0. || d2 > 0. || d3 > 0.;

    !(has_neg && has_pos)

    /*let xy = calculate_projection(&x, &y, &a);
    let xz = calculate_projection(&x, &z, &a);
    let yz = calculate_projection(&y, &z, &a);
    let max_y = if x[1].abs() > y[1].abs() && x[1].abs() > z[1].abs() {
        x[1]
    } else if y[1].abs() > z[1].abs() {
        y[1]
    } else {
        z[1]
    };
    if a[1].abs() < max_y.abs() && a[1].signum() == max_y.signum() {
        let mut res = 0;
        if a[0] <= xy && xy.is_finite() {
            res += 1;
        }
        if a[0] <= xz && xz.is_finite() {
            res += 1;
        }
        if a[0] <= yz && yz.is_finite() {
            res += 1;
        }
        res % 2 != 0 && res != 0 || (xy - a[0]).abs() < EPS || (xz - a[0]).abs() < EPS || (yz - a[0]).abs() < EPS
    } else {
        false
    }*/
}

fn calculate_projection(x: &na::Vector3<f64>, y: &na::Vector3<f64>, a: &na::Vector3<f64>) -> f64 {
    let y21 = x[1] - y[1];
    let x21 = x[0] - y[0];
    (a[1] - y[1]) / y21 * x21 + y[0]
}

fn found_cross(
    x1: &na::Vector3<f64>,
    x2: &na::Vector3<f64>,
    y1: &na::Vector3<f64>,
    y2: &na::Vector3<f64>,
) -> na::Vector3<f64> {
    let ay = (x2[1] * (x1[0] - x2[0]) / (x1[1] - x2[1])
        - y2[1] * (y1[0] - y2[0]) / (y1[1] - y2[1])
        + y2[0]
        - x2[0])
        / ((x1[0] - x2[0]) / (x1[1] - x2[1]) - (y1[0] - y2[0]) / (y1[1] - y2[1]));
    let ax = (ay - x2[1]) / (x1[1] - x2[1]) * (x1[0] - x2[0]) + x2[0];
    na::Vector3::new(ax, ay, 0.)
}

pub fn get_inside_point(
    poly: &IndexedPolygon,
    mesh: &Mesh,
    i: usize,
    j: usize,
    a: &na::Vector3<f64>,
) -> na::Vector3<f64> {
    let local = LocalPolygon::from(&Polygon::from_indexed_polygon(poly, mesh.vertexes()));

    let x = match i {
        p if p == poly.triangle().x_index() => poly.triangle().x(mesh.vertexes()),
        p if p == poly.triangle().y_index() => poly.triangle().y(mesh.vertexes()),
        p if p == poly.triangle().z_index() => poly.triangle().z(mesh.vertexes()),
        _ => unreachable!(),
    };

    let y = match j {
        p if p == poly.triangle().x_index() => poly.triangle().x(mesh.vertexes()),
        p if p == poly.triangle().y_index() => poly.triangle().y(mesh.vertexes()),
        p if p == poly.triangle().z_index() => poly.triangle().z(mesh.vertexes()),
        _ => unreachable!(),
    };

    let x = local.to_local(&x);
    let y = local.to_local(&y);

    let bx = ((x[0] - y[0]) * a[0] / (x[1] - y[1]) + a[1]
        - (y[1] - y[0] * (x[1] - y[1]) / (x[0] - y[0])))
        / ((x[1] - y[1]) / (x[0] - y[0]) + (x[0] - y[0]) / (x[1] - y[1]));
    let by = -(x[0] - y[0]) * (bx - a[0]) / (x[1] - y[1]) + a[1];

    if bx.is_nan() {
        local.from_local(&na::Vector3::new(a[0], 0., 0.))
    } else {
        local.from_local(&na::Vector3::new(bx, by, 0.))
    }

    /*let triangle = local.get_poly().triangle();
    let x = triangle.x();
    let y = triangle.y();
    let z = triangle.z();

    let xy = y - x;
    let xz = z - x;
    let x_angle = xy.angle(&xz);
    let x_delimiter = x_angle / DELIMITER;

    let yx = x - y;
    let yz = z - y;
    let y_angle = yx.angle(&yz);
    let y_delimiter = y_angle / DELIMITER;

    let y_delimiter = y_angle / DELIMITER;
    let zx = x - z;
    let zy = y - z;
    let z_angle = zx.angle(&zy);
    let z_delimiter = z_angle / DELIMITER;

    let triangle = poly.triangle();

    match (triangle.x_index(), triangle.y_index(), triangle.z_index()) {
        (xi, yi, _) if xi == i && yi == j || xi == j && yi == i => {
            let axisangle_anticlockwise = na::Vector3::z() * (-x_delimiter);
            let rot_x = na::Rotation3::new(axisangle_anticlockwise);
            let xa = rot_x * xy;
            let axisangle_clockwise = na::Vector3::z() * y_delimiter;
            let rot_y = na::Rotation3::new(axisangle_clockwise);
            let ya = rot_y * yx;
            found_cross(&x, &xa, &y, &ya)
        }
        (xi, _, zi) if xi == i && zi == j || xi == j && zi == i => {
            let axisangle_clockwise = na::Vector3::z() * x_delimiter;
            let rot_x = na::Rotation3::new(axisangle_clockwise);
            let xb = rot_x * xz;
            let axisangle_anticlockwise = na::Vector3::z() * (-z_delimiter);
            let rot_z = na::Rotation3::new(axisangle_anticlockwise);
            let zb = rot_z * zx;
            found_cross(&x, &xb, &z, &zb)
        }
        (_, yi, zi) if yi == i && zi == j || yi == j && zi == i => {
            let axisangle_anticlockwise = na::Vector3::z() * (-y_delimiter);
            let rot_y = na::Rotation3::new(axisangle_anticlockwise);
            let yc = rot_y * yz;
            let axisangle_clockwise = na::Vector3::z() * z_delimiter;
            let rot_z = na::Rotation3::new(axisangle_clockwise);
            let zc = rot_z * zy;
            found_cross(&y, &yc, &z, &zc)
        }
        _ => unreachable!(),
    }*/
}

pub fn get_mn(
    vec: &na::Vector3<f64>,
    b: &na::Vector3<f64>,
    poly_a: &IndexedPolygon,
    poly_b: &IndexedPolygon,
    mesh: &Mesh,
    i: usize,
    j: usize,
) -> (na::Vector3<f64>, na::Vector3<f64>) {
    (
        calculate_m(vec, b, poly_a, mesh, i, j),
        calculate_m(vec, b, poly_b, mesh, i, j),
    )
}

fn calculate_m(
    vec: &na::Vector3<f64>,
    b: &na::Vector3<f64>,
    poly_a: &IndexedPolygon,
    mesh: &Mesh,
    i: usize,
    j: usize,
) -> na::Vector3<f64> {
    let local = LocalPolygon::from(&Polygon::from_indexed_polygon(poly_a, mesh.vertexes()));

    let vec_local = local.to_local(vec);
    let b_local = local.to_local(b);

    let x = match i {
        x if x == poly_a.triangle().x_index() => poly_a.triangle().x(mesh.vertexes()),
        x if x == poly_a.triangle().y_index() => poly_a.triangle().y(mesh.vertexes()),
        x if x == poly_a.triangle().z_index() => poly_a.triangle().z(mesh.vertexes()),
        _ => unreachable!(),
    };

    let y = match j {
        x if x == poly_a.triangle().x_index() => poly_a.triangle().x(mesh.vertexes()),
        x if x == poly_a.triangle().y_index() => poly_a.triangle().y(mesh.vertexes()),
        x if x == poly_a.triangle().z_index() => poly_a.triangle().z(mesh.vertexes()),
        _ => unreachable!(),
    };

    let z = match (i, j) {
        (i, j) if poly_a.triangle().x_index() != i && poly_a.triangle().x_index() != j => {
            poly_a.triangle().x(mesh.vertexes())
        }
        (i, j) if poly_a.triangle().y_index() != i && poly_a.triangle().y_index() != j => {
            poly_a.triangle().y(mesh.vertexes())
        }
        (i, j) if poly_a.triangle().z_index() != i && poly_a.triangle().z_index() != j => {
            poly_a.triangle().z(mesh.vertexes())
        }
        _ => unreachable!(),
    };

    let x = local.to_local(&x);
    let y = local.to_local(&y);
    let z = local.to_local(&z);

    let xy = y - x;
    let a_tmp = xy * 0.5 + x;
    let a_tmp2 = a_tmp - z;
    let a = a_tmp2 * DELIMITER + z;

    local.from_local(&find_m(&x, &y, &a, &b_local))

    /*let triangle = local.get_poly().triangle();
    let x = triangle.x();
    let y = triangle.y();
    let z = triangle.z();

    let xy = y - x;
    let xz = z - x;
    let x_angle = xy.angle(&xz);
    let x_delimiter = x_angle / DELIMITER;

    let yx = x - y;
    let yz = z - y;
    let y_angle = yx.angle(&yz);
    let y_delimiter = y_angle / DELIMITER;

    let y_delimiter = y_angle / DELIMITER;
    let zx = x - z;
    let zy = y - z;
    let z_angle = zx.angle(&zy);
    let z_delimiter = z_angle / DELIMITER;*/

    /*let triangle = poly_a.triangle();

    match (triangle.x_index(), triangle.y_index(), triangle.z_index()) {
        (xi, yi, _) if xi == i && yi == j || xi == j && yi == i => {
            let p = triangle.x(mesh.vertexes());
            let q = triangle.y(mesh.vertexes());
            find_m(p, q, a, vec_local, &local)
        }
        (xi, _, zi) if xi == i && zi == j || xi == j && zi == i => {
            let q = triangle.x(mesh.vertexes());
            let p = triangle.z(mesh.vertexes());
            find_m(p, q, a, vec_local, &local)
        }
        (_, yi, zi) if yi == i && zi == j || yi == j && zi == i => {
            let p = triangle.y(mesh.vertexes());
            let q = triangle.z(mesh.vertexes());
            find_m(p, q, a, vec_local, &local)
        }
        _ => unreachable!(),
    }*/
}

fn find_m(
    p: &na::Vector3<f64>,
    q: &na::Vector3<f64>,
    a: &na::Vector3<f64>,
    b: &na::Vector3<f64>,
) -> na::Vector3<f64> {
    /*let mpx = ((q[0] - p[0]) * b[0] / (q[1] - p[1]) + b[1] + (p[1] - a[1]) * a[0] / (p[0] - a[0])
        - a[1])
        / ((p[1] - a[1]) / (p[0] - a[0]) + (q[0] - p[0]) / (q[1] - p[1]));
    let mpy = -(q[0] - p[0]) * (mpx - b[0]) / (q[1] - p[1]) + b[1];

    let mqx = ((q[0] - p[0]) * b[0] / (q[1] - p[1]) + b[1] + (q[1] - a[1]) * a[0] / (q[0] - a[0])
        - a[1])
        / ((q[1] - a[1]) / (q[0] - a[0]) + (q[0] - p[0]) / (q[1] - p[1]));
    let mqy = -(q[0] - p[0]) * (mqx - b[0]) / (q[1] - p[1]) + b[1];*/

    /*let mpy = (b[1] * (q[1] - p[1]) / (q[0] - p[0]) + b[0] + a[1] * (p[0] - a[0]) / (p[1] - a[1])
        - a[0])
        / ((p[0] - a[0]) / (p[1] - a[1]) + (q[1] - p[1]) / (q[0] - p[1]));
    let mpx = -(mpy - b[1]) * (q[1] - p[1]) / (q[0] - p[0]) + b[0];

    let mqy = (b[1] * (q[1] - p[1]) / (q[0] - p[0]) + b[0] + a[1] * (q[0] - a[0]) / (q[1] - a[1])
        - a[0])
        / ((q[0] - a[0]) / (q[1] - a[1]) + (q[1] - p[1]) / (q[0] - p[1]));
    let mqx = -(mqy - b[1]) * (q[1] - p[1]) / (q[0] - p[0]) + b[0];*/
    static EPS: f64 = 0.000001;

    let (mpx, mpy, mqx, mqy);

    if p[1].abs() > EPS || q[1].abs() > EPS {
        mpx = (b[0] * (q[0] - p[0]) / (q[1] - p[1]) + b[1]
            - (p[1] - p[0] * (a[1] - p[1]) / (a[0] - p[0])))
            / ((a[1] - p[1]) / (a[0] - p[0]) + (q[0] - p[0]) / (q[1] - p[1]));
        mpy = mpx * (a[1] - p[1]) / (a[0] - p[0]) + p[1] - p[0] * (a[1] - p[1]) / (a[0] - p[0]);

        mqx = (b[0] * (q[0] - p[0]) / (q[1] - p[1]) + b[1]
            - (q[1] - q[0] * (a[1] - q[1]) / (a[0] - q[0])))
            / ((a[1] - q[1]) / (a[0] - q[0]) + (q[0] - p[0]) / (q[1] - p[1]));
        mqy = mqx * (a[1] - q[1]) / (a[0] - q[0]) + q[1] - q[0] * (a[1] - q[1]) / (a[0] - q[0]);
    } else {
        mpx = b[0];
        mpy = (mpx - p[0]) * (a[1] - p[1]) / (a[0] - p[0]) + p[1];
        mqx = b[0];
        mqy = (mqx - q[0]) * (a[1] - q[1]) / (a[0] - q[0]) + q[1];
    }

    let mp = na::Vector3::new(mpx, mpy, 0.);
    let mq = na::Vector3::new(mqx, mqy, 0.);

    if (b - mp).magnitude() < (b - mq).magnitude() {
        mp
    } else {
        mq
    }

    /*let local_p = local.to_local(&p);
    let local_q = local.to_local(&q);
    let b_x = ((local_p[0] - local_q[0]) / (local_p[1] - local_q[1]) * vec_local[0] + vec_local[1]
        - local_q[1]
        + local_q[0] * (local_p[1] - local_q[1]) / (local_p[0] - local_q[0]))
        / ((local_p[1] - local_q[1]) / (local_p[0] - local_q[0])
            + (local_p[0] - local_q[0]) / (local_p[1] - local_q[1]));
    let b_y = (local_p[1] - local_q[1]) / (local_p[0] - local_q[0]) * b_x + local_q[1]
        - local_q[0] * (local_p[1] - local_q[1]) / (local_p[0] - local_q[0]);
    let b = na::Vector3::new(b_x, b_y, 0.);
    let mpx = (b_x * (vec_local[1] - b_y) / (vec_local[0] - b_x)
        - local_p[0] * (a[1] - local_p[1]) / (a[0] - local_p[0])
        + local_p[1])
        / ((vec_local[1] - b_y) / (vec_local[0] - b_x) - (a[1] - local_p[1]) / (a[0] - local_p[0]));
    let mpy = (mpx - b_x) / (vec_local[0] - b_x) * (vec_local[1] - b_y) + b_y;
    let mqx = (b_x * (vec_local[1] - b_y) / (vec_local[0] - b_x)
        - local_q[0] * (a[1] - local_q[1]) / (a[0] - local_q[0])
        + local_q[1])
        / ((vec_local[1] - b_y) / (vec_local[0] - b_x) - (a[1] - local_q[1]) / (a[0] - local_q[0]));
    let mqy = (mqx - b_x) / (vec_local[0] - b_x) * (vec_local[1] - b_y) + b_y;
    let mp = na::Vector3::new(mpx, mpy, 0.);
    let mq = na::Vector3::new(mqx, mqy, 0.);
    if (mp - b).magnitude() < (mq - b).magnitude() {
        mp
    } else {
        mq
    }*/
}
